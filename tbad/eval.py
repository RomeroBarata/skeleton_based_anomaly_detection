from copy import deepcopy
import os

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from tbad.data import load_trajectories, remove_short_trajectories, input_trajectories_missing_steps
from tbad.data import assemble_trajectories, reverse_trajectories, load_anomaly_masks
from tbad.data import uniquify_reconstructions, discard_steps_from_padded_frames
from tbad.data import write_all_reconstructed_trajectories, remove_missing_skeletons, extract_global_features
from tbad.data import concatenate_features, local_to_global_coordinates, scale_trajectories, inverse_scale_trajectories
from tbad.data import pull_global_features, compute_bounding_boxes_from_global_features, change_coordinate_system
from tbad.data import from_global_to_image_all_cameras, inverse_single_scale_trajectories
from tbad.data import compute_worst_mistakes, write_all_worst_mistakes
from tbad.rnn_autoencoder.rnn import load_pretrained_rnn_ae, reconstruct_trajectories
from tbad.autoencoder.autoencoder import load_ae_pretrained_models, compute_ae_reconstruction_errors
from tbad.autoencoder.autoencoder import reconstruct_skeletons
from tbad.combined_model.fusion import load_complete_rnn_ae_pretrained_models
from tbad.utils import select_scaler_model
from utils.metrics import compute_reconstruction_errors, summarise_reconstruction_errors
from utils.metrics import discard_errors_from_padded_frames, ground_truth_and_reconstructions


def eval_ae_models(args):
    # General
    video_resolution = [float(measurement) for measurement in args.video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)
    all_trajectories_path = args.trajectories
    all_pretrained_models_path = args.pretrained_models  # e.g. .../adam_bb-tl_mse
    frame_level_anomaly_masks_path = args.frame_level_anomaly_masks

    # Extract information about the models
    data_type = 'training' if 'training' in all_trajectories_path else 'testing'
    model_info = os.path.basename(all_pretrained_models_path)
    if 'bb-tl' in model_info:
        coordinate_system = 'bounding_box_top_left'
    elif 'bb-c' in model_info:
        coordinate_system = 'bounding_box_centre'
    else:
        coordinate_system = 'global'
    normalisation_strategy = 'three_stds' if '3stds' in model_info else 'zero_one'
    global_model = 'gm' in model_info

    pretrained_models, scalers = load_ae_pretrained_models(all_pretrained_models_path)
    camera_ids = set(pretrained_models.keys())

    skeletons = {}
    for camera_id in camera_ids:
        trajectories_path = os.path.join(all_trajectories_path, camera_id)
        trajectories_frames, trajectories_coordinates = load_trajectories(trajectories_path)
        if global_model:
            trajectories_coordinates = extract_global_features(trajectories_coordinates,
                                                               video_resolution=video_resolution)
            coordinate_system = 'global'
        trajectories_coordinates = change_coordinate_system(trajectories_coordinates,
                                                            video_resolution=video_resolution,
                                                            coordinate_system=coordinate_system,
                                                            invert=False)
        trajectories_coordinates, _ = scale_trajectories(trajectories_coordinates, scaler=scalers[camera_id],
                                                         strategy=normalisation_strategy)
        trajectories_frames, trajectories_coordinates = remove_missing_skeletons(trajectories_frames,
                                                                                 trajectories_coordinates)
        skeletons[camera_id] = [trajectories_frames, trajectories_coordinates]

    all_anomaly_masks, all_reconstruction_errors = [], []
    aurocs, auprs = {}, {}
    for camera_id in sorted(camera_ids):
        anomaly_model = pretrained_models[camera_id]
        trajectories_frames, trajectories_coordinates = skeletons[camera_id]

        trajectories_coordinates_reconstructed = reconstruct_skeletons(anomaly_model, trajectories_coordinates)
        reconstruction_errors = compute_ae_reconstruction_errors(trajectories_coordinates,
                                                                 trajectories_coordinates_reconstructed,
                                                                 loss=anomaly_model.loss)

        anomaly_masks = load_anomaly_masks(frame_level_anomaly_masks_path, camera_id=camera_id)
        y_true, y_hat = ground_truth_and_reconstructions(anomaly_masks, reconstruction_errors, trajectories_frames)
        if data_type == 'training':
            # This hack is necessary because the training set has no anomalies
            y_true[0] = 1
        aurocs[camera_id], auprs[camera_id] = roc_auc_score(y_true, y_hat), average_precision_score(y_true, y_hat)
        all_anomaly_masks.append(y_true)
        all_reconstruction_errors.append(y_hat)

    # Dump the reconstruction errors for computation of the AUROC/AUPR across all cameras
    reconstruction_errors_save_path = os.path.join(all_pretrained_models_path, data_type + '_reconstruction_errors')
    np.savez(reconstruction_errors_save_path, *all_reconstruction_errors)

    all_anomaly_masks = np.concatenate(all_anomaly_masks)
    anomaly_masks_save_path = os.path.join(all_pretrained_models_path, 'anomaly_masks.npy')
    if not os.path.exists(anomaly_masks_save_path) and data_type == 'testing':
        np.save(file=anomaly_masks_save_path[:-4], arr=all_anomaly_masks)

    for camera_id in sorted(aurocs.keys()):
        print('\nAUROC for camera %s: %.4f' % (camera_id, aurocs[camera_id]))
        print('AUPR for camera %s: %.4f' % (camera_id, auprs[camera_id]))

    return None


def eval_rnn_ae_models(args):
    # General
    video_resolution = [float(measurement) for measurement in args.video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)
    all_trajectories_path = args.trajectories
    all_pretrained_models_path = args.pretrained_models  # e.g. .../16_0_2_rrs_bb-c_3stds_mse
    frame_level_anomaly_masks_path = args.frame_level_anomaly_masks

    # Extract information about the models
    data_type = 'training' if 'training' in all_trajectories_path else 'testing'
    model_info = os.path.basename(all_pretrained_models_path)
    global_model = 'gm' in model_info
    reconstruct_reverse = 'rrs' in model_info
    input_missing = 'ims' in model_info
    # input_missing = False
    if 'bb-tl' in model_info:
        coordinate_system = 'bounding_box_top_left'
    elif 'bb-c' in model_info:
        coordinate_system = 'bounding_box_centre'
    else:
        coordinate_system = 'global'
    normalisation_strategy = 'three_stds' if '3stds' in model_info else 'zero_one'
    model_info = model_info.split('_')
    input_length, input_gap, pred_length = int(model_info[0]), int(model_info[1]), int(model_info[2])

    # A dictionary where the keys are the ids of the cameras and the values are the pre-trained models
    camera_ids = set()
    pretrained_models = {}
    scalers = {}
    for pretrained_model_name in os.listdir(all_pretrained_models_path):
        if pretrained_model_name.endswith('.npy') or pretrained_model_name.endswith('.npz'):
            continue
        camera_id = pretrained_model_name.split('_')[0]
        camera_ids.add(camera_id)
        pretrained_model_path = os.path.join(all_pretrained_models_path, pretrained_model_name)
        pretrained_models[camera_id], scalers[camera_id] = load_pretrained_rnn_ae(pretrained_model_path)

    # A dictionary where the keys are the ids of the cameras and the values are lists containing the trajectory's
    # frames and trajectory's coordinates
    trajectories = {}
    for camera_id in camera_ids:
        trajectories_path = os.path.join(all_trajectories_path, camera_id)
        trajectories_frames, trajectories_coordinates = load_trajectories(trajectories_path)

        # Remove short trajectories
        trajectories_coordinates = remove_short_trajectories(trajectories_coordinates,
                                                             input_length=input_length,
                                                             input_gap=input_gap,
                                                             pred_length=pred_length)
        for trajectory_id in set(trajectories_frames.keys()):
            if trajectory_id not in trajectories_coordinates.keys():
                del trajectories_frames[trajectory_id]

        # Input missing steps (optional)
        if input_missing:
            trajectories_coordinates = input_trajectories_missing_steps(trajectories_coordinates)

        if global_model:
            trajectories_coordinates = extract_global_features(trajectories_coordinates,
                                                               video_resolution=video_resolution)
            coordinate_system = 'global'

        trajectories_coordinates = change_coordinate_system(trajectories_coordinates,
                                                            video_resolution=video_resolution,
                                                            coordinate_system=coordinate_system, invert=False)

        trajectories_coordinates, _ = scale_trajectories(trajectories_coordinates, scaler=scalers[camera_id],
                                                         strategy=normalisation_strategy)

        trajectories[camera_id] = [trajectories_frames, trajectories_coordinates]

    all_anomaly_masks = []
    all_reconstruction_errors = []
    all_reconstructed_trajectories = {}
    all_original_lengths = {}
    aurocs, auprs = {}, {}
    for camera_id in sorted(camera_ids):
        anomaly_model = pretrained_models[camera_id]
        scaler = scalers[camera_id]
        trajectories_frames, trajectories_coordinates = trajectories[camera_id]
        original_lengths = {trajectory_id: len(trajectory_coordinates)
                            for trajectory_id, trajectory_coordinates in trajectories_coordinates.items()}
        all_original_lengths[camera_id] = original_lengths

        test_frames, test_coordinates = assemble_trajectories(trajectories_frames, trajectories_coordinates,
                                                              overlapping=args.overlapping_trajectories,
                                                              input_length=input_length,
                                                              input_gap=input_gap,
                                                              pred_length=pred_length)
        reconstructed_coordinates = reconstruct_trajectories(anomaly_model, test_coordinates)
        if reconstruct_reverse:
            reconstructed_coordinates = reverse_trajectories(reconstructed_coordinates)

        reconstruction_errors = compute_reconstruction_errors(test_coordinates, reconstructed_coordinates,
                                                              loss=anomaly_model.loss)
        reconstruction_errors = summarise_reconstruction_errors(reconstruction_errors, test_frames)
        reconstruction_errors = discard_errors_from_padded_frames(reconstruction_errors, original_lengths)

        reconstructed_coordinates = inverse_single_scale_trajectories(reconstructed_coordinates, scaler=scaler)
        all_reconstructed_trajectories[camera_id] = [test_frames, reconstructed_coordinates]

        anomaly_masks = load_anomaly_masks(frame_level_anomaly_masks_path, camera_id=camera_id)
        y_true, y_hat, _ = ground_truth_and_reconstructions(anomaly_masks, reconstruction_errors, trajectories_frames)
        if data_type == 'training':
            # This hack is necessary because the training set has no anomalies
            y_true[0] = 1
        aurocs[camera_id], auprs[camera_id] = roc_auc_score(y_true, y_hat), average_precision_score(y_true, y_hat)
        all_anomaly_masks.append(y_true)
        all_reconstruction_errors.append(y_hat)

    # Dump the reconstruction errors for computation of the AUROC across all cameras
    reconstruction_errors_save_path = os.path.join(all_pretrained_models_path, data_type + '_reconstruction_errors')
    np.savez(reconstruction_errors_save_path, *all_reconstruction_errors)

    all_anomaly_masks = np.concatenate(all_anomaly_masks)
    anomaly_masks_save_path = os.path.join(all_pretrained_models_path, 'anomaly_masks.npy')
    if not os.path.exists(anomaly_masks_save_path) and data_type == 'testing':
        np.save(file=anomaly_masks_save_path[:-4], arr=all_anomaly_masks)

    for camera_id in sorted(aurocs.keys()):
        print('\nAUROC for camera %s: %.4f' % (camera_id, aurocs[camera_id]))
        print('AUPR for camera %s: %.4f' % (camera_id, auprs[camera_id]))

    if args.write_reconstructions is not None and not global_model:
        all_reconstructed_trajectories = uniquify_reconstructions(all_reconstructed_trajectories)
        all_reconstructed_trajectories = discard_steps_from_padded_frames(all_reconstructed_trajectories,
                                                                          all_original_lengths)
        all_reconstructed_trajectories = from_global_to_image_all_cameras(all_reconstructed_trajectories,
                                                                          video_resolution=video_resolution)
        write_all_reconstructed_trajectories(all_reconstructed_trajectories, write_path=args.write_reconstructions)
        print('All reconstructed trajectories were written to %s.' % args.write_reconstructions)

    return None


def eval_complete_rnn_ae_models(args):
    # Extract command line arguments
    video_resolution = [float(measurement) for measurement in args.video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)
    all_trajectories_path = args.trajectories
    all_pretrained_models_path = args.pretrained_models  # e.g. .../16_0_2_rrs_mse
    frame_level_anomaly_masks_path = args.frame_level_anomaly_masks
    overlapping_trajectories = args.overlapping_trajectories

    # Extract information about the models
    data_type = 'training' if 'training' in all_trajectories_path else 'testing'
    model_info = os.path.basename(all_pretrained_models_path)
    reconstruct_reverse = 'rrs' in model_info
    global_normalisation_strategy = 'three_stds' if 'G3stds' in model_info else 'zero_one'
    local_normalisation_strategy = 'three_stds' if 'L3stds' in model_info else 'zero_one'
    # input_missing = 'ims' in model_info
    # input_missing = False
    model_info = model_info.split('_')
    input_length, input_gap, pred_length = int(model_info[0]), int(model_info[1]), int(model_info[2])

    # A dictionary where the keys are the ids of the cameras and the values are the pre-trained models
    pretrained_models, global_scalers, local_scalers = \
        load_complete_rnn_ae_pretrained_models(all_pretrained_models_path)
    camera_ids = set(pretrained_models.keys())

    trajectories = {}
    for camera_id in camera_ids:
        trajectories_path = os.path.join(all_trajectories_path, camera_id)
        trajectories_frames, trajectories_coordinates = load_trajectories(trajectories_path)

        trajectories_coordinates = remove_short_trajectories(trajectories_coordinates,
                                                             input_length=input_length,
                                                             input_gap=input_gap,
                                                             pred_length=pred_length)
        for trajectory_id in set(trajectories_frames.keys()):
            if trajectory_id not in trajectories_coordinates.keys():
                del trajectories_frames[trajectory_id]

        global_features = extract_global_features(trajectories_coordinates, video_resolution=video_resolution)
        global_features = change_coordinate_system(global_features, video_resolution=video_resolution,
                                                   coordinate_system='global', invert=False)
        global_features, _ = scale_trajectories(global_features, scaler=global_scalers[camera_id],
                                                strategy=global_normalisation_strategy)

        local_features = deepcopy(trajectories_coordinates)
        local_features = change_coordinate_system(local_features, video_resolution=video_resolution,
                                                  coordinate_system='bounding_box_centre', invert=False)
        local_features, _ = scale_trajectories(local_features, scaler=local_scalers[camera_id],
                                               strategy=local_normalisation_strategy)

        # out_features = trajectories_coordinates
        # out_features = change_coordinate_system(out_features, video_resolution=video_resolution,
        #                                         coordinate_system='global', invert=False)
        #
        # trajectories[camera_id] = [trajectories_frames, global_features, local_features, out_features]
        trajectories[camera_id] = [trajectories_frames, global_features, local_features]

    all_anomaly_masks = []
    all_reconstruction_errors = []
    all_reconstructed_trajectories = {}
    all_original_lengths = {}
    aurocs, auprs = {}, {}
    worst_false_positives, worst_false_negatives = {}, {}
    for camera_id in sorted(camera_ids):
        anomaly_model = pretrained_models[camera_id]
        global_scaler = global_scalers[camera_id]
        local_scaler = local_scalers[camera_id]
        # trajectories_frames, global_features, local_features, out_features = trajectories[camera_id]
        trajectories_frames, global_features, local_features = trajectories[camera_id]
        original_lengths = {trajectory_id: len(trajectory_coordinates)
                            for trajectory_id, trajectory_coordinates in local_features.items()}
        all_original_lengths[camera_id] = original_lengths

        test_frames, global_features_test = assemble_trajectories(trajectories_frames, global_features,
                                                                  overlapping=overlapping_trajectories,
                                                                  input_length=input_length,
                                                                  input_gap=input_gap,
                                                                  pred_length=pred_length)
        _, local_features_test = assemble_trajectories(trajectories_frames, local_features,
                                                       overlapping=overlapping_trajectories,
                                                       input_length=input_length,
                                                       input_gap=input_gap,
                                                       pred_length=pred_length)
        # _, out_features_test = assemble_trajectories(trajectories_frames, out_features,
        #                                              overlapping=overlapping_trajectories,
        #                                              input_length=input_length,
        #                                              input_gap=input_gap,
        #                                              pred_length=pred_length)
        features_test = concatenate_features(global_features_test, local_features_test)
        
        reconstructed_features = anomaly_model.reconstruct(global_features_test, local_features_test)
        if reconstruct_reverse:
            reconstructed_features = reverse_trajectories(reconstructed_features)
        # reconstruction_errors = compute_reconstruction_errors(out_features_test, reconstructed_features,
        #                                                       loss=anomaly_model.loss)
        reconstruction_errors = compute_reconstruction_errors(features_test, reconstructed_features,
                                                              loss=anomaly_model.loss)
        reconstruction_errors = summarise_reconstruction_errors(reconstruction_errors, test_frames)
        reconstruction_errors = discard_errors_from_padded_frames(reconstruction_errors, original_lengths)

        reconstructed_features = inverse_scale_trajectories(reconstructed_features,
                                                            global_scaler=global_scaler,
                                                            local_scaler=local_scaler)
        # reconstructed_features = from_global_to_image(reconstructed_features, video_resolution=video_resolution)
        all_reconstructed_trajectories[camera_id] = [test_frames, reconstructed_features]

        anomaly_masks = load_anomaly_masks(frame_level_anomaly_masks_path, camera_id=camera_id)
        y_true, y_hat, video_ids = ground_truth_and_reconstructions(anomaly_masks, reconstruction_errors,
                                                                    trajectories_frames)
        worst_false_positives[camera_id] = compute_worst_mistakes(y_true, y_hat, video_ids,
                                                                  type='false_positives', top=10)
        worst_false_negatives[camera_id] = compute_worst_mistakes(y_true, y_hat, video_ids,
                                                                  type='false_negatives', top=10)
        if data_type == 'training':
            # This hack is necessary because the training set has no anomalies
            y_true[0] = 1
        aurocs[camera_id], auprs[camera_id] = roc_auc_score(y_true, y_hat), average_precision_score(y_true, y_hat)
        all_anomaly_masks.append(y_true)
        all_reconstruction_errors.append(y_hat)

    # Dump the reconstruction errors for computation of the AUROC across all cameras
    reconstruction_errors_save_path = os.path.join(all_pretrained_models_path, data_type + '_reconstruction_errors')
    np.savez(reconstruction_errors_save_path, *all_reconstruction_errors)

    all_anomaly_masks = np.concatenate(all_anomaly_masks)
    anomaly_masks_save_path = os.path.join(all_pretrained_models_path, 'anomaly_masks.npy')
    if data_type == 'testing':
        np.save(file=anomaly_masks_save_path[:-4], arr=all_anomaly_masks)

    for camera_id in sorted(aurocs.keys()):
        print('\nAUROC for camera %s: %.4f' % (camera_id, aurocs[camera_id]))
        print('AUPR for camera %s: %.4f' % (camera_id, auprs[camera_id]))

    # Dump the worst mistakes in a .txt file
    if data_type == 'testing':
        write_all_worst_mistakes(all_pretrained_models_path, worst_false_positives, worst_false_negatives)
        print('All mistakes were written to %s.' % os.path.join(all_pretrained_models_path, 'mistakes.txt'))

    if args.write_reconstructions is not None:
        all_image_trajectories = local_to_global_coordinates(all_reconstructed_trajectories,
                                                             video_resolution=video_resolution)
        # all_image_trajectories = deepcopy(all_reconstructed_trajectories)
        all_image_trajectories = uniquify_reconstructions(all_image_trajectories)
        all_image_trajectories = discard_steps_from_padded_frames(all_image_trajectories,
                                                                  all_original_lengths)
        write_all_reconstructed_trajectories(all_image_trajectories, write_path=args.write_reconstructions)

        print('All reconstructed trajectories were written to %s.' % args.write_reconstructions)

    if args.write_bounding_boxes is not None:
        all_reconstructed_trajectories = uniquify_reconstructions(all_reconstructed_trajectories)
        all_reconstructed_trajectories = discard_steps_from_padded_frames(all_reconstructed_trajectories,
                                                                          all_original_lengths)
        all_reconstructed_trajectories = pull_global_features(all_reconstructed_trajectories)
        all_reconstructed_trajectories = from_global_to_image_all_cameras(all_reconstructed_trajectories,
                                                                          video_resolution=video_resolution)
        all_bounding_boxes = compute_bounding_boxes_from_global_features(all_reconstructed_trajectories)
        # all_bounding_boxes = compute_bounding_boxes_from_image_features(all_reconstructed_trajectories,
        #                                                                 video_resolution=video_resolution)
        write_all_reconstructed_trajectories(all_bounding_boxes, write_path=args.write_bounding_boxes)

        print('All reconstructed bounding boxes were written to %s.' % args.write_bounding_boxes)

    return None


def compute_all_cameras_performance_metrics(args):
    all_reconstruction_errors_path = args.reconstruction_errors_path
    train_with_nonzero_scores_only = args.train_with_nonzero_scores_only
    ignore_training = args.ignore_training
    ignore_scaler = args.ignore_scaler
    scaler_model = args.scaler_model

    all_anomaly_masks_file = os.path.join(all_reconstruction_errors_path, 'anomaly_masks.npy')
    all_anomaly_masks = np.load(all_anomaly_masks_file)
    training_reconstruction_errors_file = os.path.join(all_reconstruction_errors_path,
                                                       'training_reconstruction_errors.npz')
    training_reconstruction_errors = np.load(training_reconstruction_errors_file)
    testing_reconstruction_errors_file = os.path.join(all_reconstruction_errors_path,
                                                      'testing_reconstruction_errors.npz')
    testing_reconstruction_errors = np.load(testing_reconstruction_errors_file)

    all_testing_scores = []
    for camera_id in range(len(training_reconstruction_errors.files)):
        training_scores = training_reconstruction_errors['arr_%d' % camera_id]
        testing_scores = testing_reconstruction_errors['arr_%d' % camera_id]

        if not ignore_scaler:
            score_scaler_model = select_scaler_model(scaler_model)

            if train_with_nonzero_scores_only:
                is_non_zero_training_score = training_scores > 0.0
                non_zero_training_scores = training_scores[is_non_zero_training_score].reshape(-1, 1)

                is_non_zero_testing_score = testing_scores > 0.0
                non_zero_testing_scores = testing_scores[is_non_zero_testing_score].reshape(-1, 1)

                if ignore_training:
                    score_scaler_model.fit(non_zero_testing_scores)
                else:
                    score_scaler_model.fit(non_zero_training_scores)

                testing_scores[is_non_zero_testing_score] = score_scaler_model.transform(non_zero_testing_scores).ravel()
            else:
                if ignore_training:
                    score_scaler_model.fit(testing_scores.reshape(-1, 1))
                else:
                    score_scaler_model.fit(training_scores.reshape(-1, 1))

                testing_scores = score_scaler_model.transform(testing_scores.reshape(-1, 1)).ravel()
        
        all_testing_scores.append(testing_scores)

    all_testing_scores = np.concatenate(all_testing_scores)
    all_cameras_auroc = roc_auc_score(all_anomaly_masks, all_testing_scores)
    all_cameras_aupr = average_precision_score(all_anomaly_masks, all_testing_scores)

    print('\nAUROC for all cameras: %.4f' % all_cameras_auroc)
    print('AUPR for all cameras: %.4f' % all_cameras_aupr)

    return None


def combine_global_and_local_losses(args):
    global_model_losses_path = args.global_model_losses
    local_model_losses_path = args.local_model_losses
    anomaly_mask_path = args.anomaly_mask

    global_model_losses = np.load(global_model_losses_path)
    local_model_losses = np.load(local_model_losses_path)
    anomaly_mask = np.load(anomaly_mask_path)

    start_idx = 0
    for camera_id in range(len(global_model_losses.files)):
        global_errors = global_model_losses['arr_%d' % camera_id]
        local_errors = local_model_losses['arr_%d' % camera_id]
        end_idx = len(global_errors) + start_idx
        anomalies = anomaly_mask[start_idx:end_idx]
        start_idx = end_idx

        print('\n\nlocal error + 0 * global')
        combined_errors = local_errors
        print('AUROC for camera %d: %.4f' % (camera_id, roc_auc_score(anomalies, combined_errors)))
        print('AUPR for camera %d: %.4f' % (camera_id, average_precision_score(anomalies, combined_errors)))

        print('\nlocal error + 1 * global')
        combined_errors = local_errors + global_errors
        print('AUROC for camera %d: %.4f' % (camera_id, roc_auc_score(anomalies, combined_errors)))
        print('AUPR for camera %d: %.4f' % (camera_id, average_precision_score(anomalies, combined_errors)))

        print('\nlocal error + 10 * global')
        combined_errors = local_errors + 10 * global_errors
        print('AUROC for camera %d: %.4f' % (camera_id, roc_auc_score(anomalies, combined_errors)))
        print('AUPR for camera %d: %.4f' % (camera_id, average_precision_score(anomalies, combined_errors)))

        print('\nlocal error + 100 * global')
        combined_errors = local_errors + 100 * global_errors
        print('AUROC for camera %d: %.4f' % (camera_id, roc_auc_score(anomalies, combined_errors)))
        print('AUPR for camera %d: %.4f' % (camera_id, average_precision_score(anomalies, combined_errors)))

        print('\nlocal error + 1000 * global')
        combined_errors = local_errors + 1000 * global_errors
        print('AUROC for camera %d: %.4f' % (camera_id, roc_auc_score(anomalies, combined_errors)))
        print('AUPR for camera %d: %.4f' % (camera_id, average_precision_score(anomalies, combined_errors)))

        print('\nlocal error + 10000 * global')
        combined_errors = local_errors + 10000 * global_errors
        print('AUROC for camera %d: %.4f' % (camera_id, roc_auc_score(anomalies, combined_errors)))
        print('AUPR for camera %d: %.4f' % (camera_id, average_precision_score(anomalies, combined_errors)))

    return None
