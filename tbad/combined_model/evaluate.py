from collections import namedtuple
from copy import deepcopy
import os

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from tbad.autoencoder.data import load_trajectories, extract_global_features, change_coordinate_system
from tbad.autoencoder.data import scale_trajectories, load_anomaly_masks, assemble_ground_truth_and_reconstructions
from tbad.autoencoder.data import quantile_transform_errors
from tbad.rnn_autoencoder.data import remove_short_trajectories, aggregate_rnn_ae_evaluation_data
from tbad.rnn_autoencoder.data import compute_rnn_ae_reconstruction_errors, summarise_reconstruction_errors
from tbad.rnn_autoencoder.data import summarise_reconstruction, retrieve_future_skeletons
from tbad.rnn_autoencoder.data import discard_information_from_padded_frames
from tbad.combined_model.fusion import load_pretrained_combined_model
from tbad.combined_model.data import inverse_scale, restore_global_coordinate_system, restore_original_trajectory
from tbad.combined_model.data import write_reconstructed_trajectories, detect_most_anomalous_or_most_normal_frames
from tbad.combined_model.data import compute_num_frames_per_video, write_predicted_masks, compute_worst_mistakes
from tbad.combined_model.data import write_worst_mistakes, clip_trajectories, normalise_errors_by_bounding_box_area
from tbad.visualisation import compute_bounding_box


V_01 = [1] * 75 + [0] * 46 + [1] * 269 + [0] * 47 + [1] * 427 + [0] * 47 + [1] * 20 + [0] * 70 + [1] * 438  # 1439 Frames
V_02 = [1] * 272 + [0] * 48 + [1] * 403 + [0] * 41 + [1] * 447  # 1211 Frames
V_03 = [1] * 293 + [0] * 48 + [1] * 582  # 923 Frames
V_04 = [1] * 947  # 947 Frames
V_05 = [1] * 1007  # 1007 Frames
V_06 = [1] * 561 + [0] * 64 + [1] * 189 + [0] * 193 + [1] * 276  # 1283 Frames
V_07_to_15 = [1] * 6457
V_16 = [1] * 728 + [0] * 12  # 740 Frames
V_17_to_21 = [1] * 1317
AVENUE_MASK = np.array(V_01 + V_02 + V_03 + V_04 + V_05 + V_06 + V_07_to_15 + V_16 + V_17_to_21) == 1


def eval_combined_model(args):
    trajectories_path = args.trajectories  # e.g. .../optflow/alphapose/07
    camera_id = os.path.basename(trajectories_path)
    pretrained_model_path = args.pretrained_model  # e.g. .../16_0_2_rrs_mse/07_2018_09_20_13_15_13
    video_resolution = [float(measurement) for measurement in args.video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)
    frame_level_anomaly_masks_path = args.frame_level_anomaly_masks
    overlapping_trajectories = args.overlapping_trajectories
    # Logging
    write_reconstructions = args.write_reconstructions
    write_bounding_boxes = args.write_bounding_boxes
    write_predictions = args.write_predictions
    write_predictions_bounding_boxes = args.write_predictions_bounding_boxes
    write_anomaly_masks = args.write_anomaly_masks
    write_mistakes = args.write_mistakes

    model_info = os.path.basename(os.path.split(pretrained_model_path)[0])
    message_passing = 'mp' in model_info
    is_avenue = 'Avenue' in trajectories_path

    pretrained_combined_model, global_scaler, local_scaler, out_scaler = \
        load_pretrained_combined_model(pretrained_model_path, message_passing=message_passing)

    # Extract information about the models
    reconstruct_original_data = 'down' in model_info
    global_normalisation_strategy = 'zero_one'
    if '_G3stds_' in model_info:
        global_normalisation_strategy = 'three_stds'
    elif '_Grobust_' in model_info:
        global_normalisation_strategy = 'robust'

    local_normalisation_strategy = 'zero_one'
    if '_L3stds_' in model_info:
        local_normalisation_strategy = 'three_stds'
    elif '_Lrobust_' in model_info:
        local_normalisation_strategy = 'robust'

    out_normalisation_strategy = 'zero_one'
    if '_O3stds_' in model_info:
        out_normalisation_strategy = 'three_stds'
    elif '_Orobust_' in model_info:
        out_normalisation_strategy = 'robust'

    multiple_outputs = pretrained_combined_model.multiple_outputs
    input_length, rec_length = pretrained_combined_model.input_length, pretrained_combined_model.reconstruction_length
    input_gap, pred_length = 0, pretrained_combined_model.prediction_length
    reconstruct_reverse = pretrained_combined_model.reconstruct_reverse
    loss = pretrained_combined_model.loss

    # Data
    trajectories = load_trajectories(trajectories_path)

    trajectories = remove_short_trajectories(trajectories, input_length=input_length,
                                             input_gap=input_gap, pred_length=pred_length)

    global_trajectories = extract_global_features(deepcopy(trajectories), video_resolution=video_resolution)
    global_trajectories = change_coordinate_system(global_trajectories, video_resolution=video_resolution,
                                                   coordinate_system='global', invert=False)
    trajectories_ids, frames, X_global = \
        aggregate_rnn_ae_evaluation_data(global_trajectories,
                                         input_length=input_length,
                                         input_gap=input_gap,
                                         pred_length=pred_length,
                                         overlapping_trajectories=overlapping_trajectories)
    X_global, _ = scale_trajectories(X_global, scaler=global_scaler, strategy=global_normalisation_strategy)

    local_trajectories = deepcopy(trajectories)
    local_trajectories = change_coordinate_system(local_trajectories, video_resolution=video_resolution,
                                                  coordinate_system='bounding_box_centre', invert=False)
    _, _, X_local = aggregate_rnn_ae_evaluation_data(local_trajectories, input_length=input_length,
                                                     input_gap=input_gap, pred_length=pred_length,
                                                     overlapping_trajectories=overlapping_trajectories)
    X_local, _ = scale_trajectories(X_local, scaler=local_scaler, strategy=local_normalisation_strategy)

    original_trajectories = deepcopy(trajectories)
    _, _, X_original = aggregate_rnn_ae_evaluation_data(original_trajectories, input_length=input_length,
                                                        input_gap=input_gap, pred_length=pred_length,
                                                        overlapping_trajectories=overlapping_trajectories)

    if reconstruct_original_data:
        out_trajectories = trajectories
        out_trajectories = change_coordinate_system(out_trajectories, video_resolution=video_resolution,
                                                    coordinate_system='global', invert=False)
        _, _, X_out = aggregate_rnn_ae_evaluation_data(out_trajectories, input_length=input_length,
                                                       input_gap=input_gap, pred_length=pred_length,
                                                       overlapping_trajectories=overlapping_trajectories)
        X_out, _ = scale_trajectories(X_out, scaler=out_scaler, strategy=out_normalisation_strategy)

    # Reconstruct
    X_input = [X_global, X_local]
    if pred_length == 0:
        if multiple_outputs:
            _, _, reconstructed_X = pretrained_combined_model.predict(X_input, batch_size=1024)
        else:
            reconstructed_X = pretrained_combined_model.predict(X_input, batch_size=1024)
    else:
        if multiple_outputs:
            _, _, reconstructed_X, _, _, predicted_y = \
                pretrained_combined_model.predict(X_input, batch_size=1024)
        else:
            reconstructed_X, predicted_y = pretrained_combined_model.predict(X_input, batch_size=1024)

    if reconstruct_reverse:
        reconstructed_X = reconstructed_X[:, ::-1, :]

    X = X_out if reconstruct_original_data else np.concatenate((X_global, X_local), axis=-1)
    reconstruction_errors = compute_rnn_ae_reconstruction_errors(X[:, :rec_length, :], reconstructed_X, loss)
    reconstruction_ids, reconstruction_frames, reconstruction_errors = \
        summarise_reconstruction_errors(reconstruction_errors, frames[:, :rec_length], trajectories_ids[:, :rec_length])

    # Evaluate performance
    anomaly_masks = load_anomaly_masks(frame_level_anomaly_masks_path)
    y_true, y_hat, video_ids = assemble_ground_truth_and_reconstructions(anomaly_masks, reconstruction_ids,
                                                                         reconstruction_frames, reconstruction_errors,
                                                                         return_video_ids=True)

    if is_avenue:
        auroc, aupr = roc_auc_score(y_true[AVENUE_MASK], y_hat[AVENUE_MASK]), average_precision_score(
            y_true[AVENUE_MASK], y_hat[AVENUE_MASK])
    else:
        auroc, aupr = roc_auc_score(y_true, y_hat), average_precision_score(y_true, y_hat)

    print('Reconstruction Based:')
    print('Camera %s:\tAUROC\tAUPR' % camera_id)
    print('          \t%.4f\t%.4f\n' % (auroc, aupr))

    if pred_length > 0:
        predicted_frames = frames[:, :pred_length] + input_length
        predicted_ids = trajectories_ids[:, :pred_length]

        y = retrieve_future_skeletons(trajectories_ids, X, pred_length)

        pred_errors = compute_rnn_ae_reconstruction_errors(y, predicted_y, loss)

        pred_ids, pred_frames, pred_errors = discard_information_from_padded_frames(predicted_ids,
                                                                                    predicted_frames,
                                                                                    pred_errors, pred_length)

        pred_ids, pred_frames, pred_errors = summarise_reconstruction_errors(pred_errors, pred_frames, pred_ids)

        y_true_pred, y_hat_pred = assemble_ground_truth_and_reconstructions(anomaly_masks, pred_ids,
                                                                            pred_frames, pred_errors)
        if is_avenue:
            auroc, aupr = roc_auc_score(y_true_pred[AVENUE_MASK], y_hat_pred[AVENUE_MASK]), average_precision_score(
                y_true_pred[AVENUE_MASK], y_hat_pred[AVENUE_MASK])
        else:
            auroc, aupr = roc_auc_score(y_true_pred, y_hat_pred), average_precision_score(y_true_pred, y_hat_pred)

        print('Prediction Based:')
        print('Camera %s:\tAUROC\tAUPR' % camera_id)
        print('          \t%.4f\t%.4f\n' % (auroc, aupr))

        y_true_comb, y_hat_comb = y_true, y_hat + y_hat_pred
        if is_avenue:
            auroc, aupr = roc_auc_score(y_true_comb[AVENUE_MASK], y_hat_comb[AVENUE_MASK]), average_precision_score(
                y_true_comb[AVENUE_MASK], y_hat_comb[AVENUE_MASK])
        else:
            auroc, aupr = roc_auc_score(y_true_comb, y_hat_comb), average_precision_score(y_true_comb, y_hat_comb)

        print('Reconstruction + Prediction Based:')
        print('Camera %s:\tAUROC\tAUPR' % camera_id)
        print('          \t%.4f\t%.4f\n' % (auroc, aupr))

        if reconstruct_original_data:
            predicted_y_traj = inverse_scale(predicted_y, scaler=out_scaler)
            predicted_y_traj = restore_global_coordinate_system(predicted_y_traj, video_resolution=video_resolution)
        else:
            predicted_y_global = inverse_scale(predicted_y[..., :4], scaler=global_scaler)
            predicted_y_local = inverse_scale(predicted_y[..., 4:], scaler=local_scaler)
            predicted_y_global = restore_global_coordinate_system(predicted_y_global, video_resolution=video_resolution)
            predicted_y_traj = restore_original_trajectory(predicted_y_global, predicted_y_local)

        prediction_ids, prediction_frames, predicted_y_traj = \
            summarise_reconstruction(predicted_y_traj, predicted_frames, predicted_ids)

        predicted_bounding_boxes = np.apply_along_axis(compute_bounding_box, axis=1, arr=predicted_y_traj,
                                                       video_resolution=video_resolution)

    # Post-Processing
    if reconstruct_original_data:
        reconstructed_X_traj = inverse_scale(reconstructed_X, scaler=out_scaler)
        reconstructed_X_traj = restore_global_coordinate_system(reconstructed_X_traj, video_resolution=video_resolution)
    else:
        reconstructed_X_global = inverse_scale(reconstructed_X[..., :4], scaler=global_scaler)
        reconstructed_X_local = inverse_scale(reconstructed_X[..., 4:], scaler=local_scaler)

        reconstructed_X_global = restore_global_coordinate_system(reconstructed_X_global,
                                                                  video_resolution=video_resolution)
        reconstructed_X_traj = restore_original_trajectory(reconstructed_X_global, reconstructed_X_local)

    reconstruction_ids, reconstruction_frames, reconstructed_X_traj = \
        summarise_reconstruction(reconstructed_X_traj, frames[:, :rec_length], trajectories_ids[:, :rec_length])

    reconstructed_bounding_boxes = np.apply_along_axis(compute_bounding_box, axis=1, arr=reconstructed_X_traj,
                                                       video_resolution=video_resolution)

    worst_false_positives = compute_worst_mistakes(y_true=y_true_pred, y_hat=y_hat_pred, video_ids=video_ids,
                                                   error_type='false_positives', top=25)
    worst_false_negatives = compute_worst_mistakes(y_true=y_true_pred, y_hat=y_hat_pred, video_ids=video_ids,
                                                   error_type='false_negatives', top=25)

    # Logging
    if write_reconstructions:
        write_reconstructed_trajectories(pretrained_model_path, reconstructed_X_traj,
                                         reconstruction_ids, reconstruction_frames, trajectory_type='skeleton')

    if write_bounding_boxes:
        write_reconstructed_trajectories(pretrained_model_path, reconstructed_bounding_boxes,
                                         reconstruction_ids, reconstruction_frames, trajectory_type='bounding_box')

    if write_predictions:
        write_reconstructed_trajectories(pretrained_model_path, predicted_y_traj, prediction_ids, prediction_frames,
                                         trajectory_type='predicted_skeleton')

    if write_predictions_bounding_boxes:
        write_reconstructed_trajectories(pretrained_model_path, predicted_bounding_boxes, prediction_ids,
                                         prediction_frames, trajectory_type='predicted_bounding_box')

    if write_anomaly_masks:
        anomalous_frames = detect_most_anomalous_or_most_normal_frames(reconstruction_errors,
                                                                       anomalous=True,
                                                                       fraction=0.20)
        normal_frames = detect_most_anomalous_or_most_normal_frames(reconstruction_errors,
                                                                    anomalous=False,
                                                                    fraction=0.20)
        num_frames_per_video = compute_num_frames_per_video(anomaly_masks)
        write_predicted_masks(pretrained_model_path, num_frames_per_video, anomalous_frames, normal_frames,
                              reconstructed_bounding_boxes, reconstruction_ids, reconstruction_frames, video_resolution)

    if write_mistakes:
        write_worst_mistakes(pretrained_model_path, worst_false_positives=worst_false_positives,
                             worst_false_negatives=worst_false_negatives)

    if pred_length > 0:
        return y_true, y_hat, y_true_pred, y_hat_pred, y_true_comb, y_hat_comb
    else:
        return y_true, y_hat, None, None, None, None


def eval_combined_models(args):
    all_trajectories_path = args.all_trajectories
    pretrained_models_path = args.pretrained_models  # e.g. .../
    video_resolution = args.video_resolution
    all_frame_level_anomaly_masks_path = args.all_frame_level_anomaly_masks
    overlapping_trajectories = args.overlapping_trajectories
    is_avenue = 'Avenue' in all_trajectories_path
    # Logging
    write_reconstructions = args.write_reconstructions
    write_bounding_boxes = args.write_bounding_boxes
    write_predictions = args.write_predictions
    write_predictions_bounding_boxes = args.write_predictions_bounding_boxes
    write_anomaly_masks = args.write_anomaly_masks
    write_mistakes = args.write_mistakes

    EvalCombinedModelArgs = namedtuple('EvalCombinedModelArgs',
                                       ['pretrained_model', 'trajectories', 'frame_level_anomaly_masks',
                                        'video_resolution', 'overlapping_trajectories', 'write_reconstructions',
                                        'write_bounding_boxes', 'write_predictions',
                                        'write_predictions_bounding_boxes', 'write_anomaly_masks', 'write_mistakes'])

    pretrained_models_dirs = sorted(os.listdir(pretrained_models_path))
    y_trues, y_hats, y_trues_pred, y_hats_pred, y_trues_comb, y_hats_comb = {}, {}, {}, {}, {}, {}
    for pretrained_model_dir in pretrained_models_dirs:
        camera_id = pretrained_model_dir.split('_')[0]
        pretrained_model_path = os.path.join(pretrained_models_path, pretrained_model_dir)
        trajectories_path = os.path.join(all_trajectories_path, camera_id)
        frame_level_anomaly_masks_path = os.path.join(all_frame_level_anomaly_masks_path, camera_id)
        eval_combined_model_args = EvalCombinedModelArgs(pretrained_model_path, trajectories_path,
                                                         frame_level_anomaly_masks_path, video_resolution,
                                                         overlapping_trajectories, write_reconstructions,
                                                         write_bounding_boxes, write_predictions,
                                                         write_predictions_bounding_boxes, write_anomaly_masks,
                                                         write_mistakes)
        y_true, y_hat, y_true_pred, y_hat_pred, y_true_comb, y_hat_comb = eval_combined_model(eval_combined_model_args)
        y_trues[camera_id], y_hats[camera_id] = y_true, y_hat
        if y_true_pred is not None:
            y_trues_pred[camera_id], y_hats_pred[camera_id] = y_true_pred, y_hat_pred
            y_trues_comb[camera_id], y_hats_comb[camera_id] = y_true_comb, y_hat_comb

    y_hats = quantile_transform_errors(y_hats)
    camera_ids = sorted(y_hats.keys())
    y_trues_flat, y_hats_flat = [], []
    for camera_id in camera_ids:
        y_trues_flat.append(y_trues[camera_id])
        y_hats_flat.append(y_hats[camera_id])

    y_trues_flat, y_hats_flat = np.concatenate(y_trues_flat), np.concatenate(y_hats_flat)
    if is_avenue:
        auroc, aupr = roc_auc_score(y_trues_flat[AVENUE_MASK], y_hats_flat[AVENUE_MASK]), \
                      average_precision_score(y_trues_flat[AVENUE_MASK], y_hats_flat[AVENUE_MASK])
    else:
        auroc, aupr = roc_auc_score(y_trues_flat, y_hats_flat), average_precision_score(y_trues_flat, y_hats_flat)

    print('Reconstruction Based:')
    print('All Cameras\tAUROC\tAUPR')
    print('           \t%.4f\t%.4f\n' % (auroc, aupr))

    if y_trues_pred:
        y_hats_pred = quantile_transform_errors(y_hats_pred)
        y_trues_pred_flat, y_hats_pred_flat = [], []
        for camera_id in camera_ids:
            y_trues_pred_flat.append(y_trues_pred[camera_id])
            y_hats_pred_flat.append(y_hats_pred[camera_id])

        y_trues_pred_flat, y_hats_pred_flat = np.concatenate(y_trues_pred_flat), np.concatenate(y_hats_pred_flat)
        if is_avenue:
            auroc_pred = roc_auc_score(y_trues_pred_flat[AVENUE_MASK], y_hats_pred_flat[AVENUE_MASK])
            aupr_pred = average_precision_score(y_trues_pred_flat[AVENUE_MASK], y_hats_pred_flat[AVENUE_MASK])
        else:
            auroc_pred = roc_auc_score(y_trues_pred_flat, y_hats_pred_flat)
            aupr_pred = average_precision_score(y_trues_pred_flat, y_hats_pred_flat)

        print('Prediction Based:')
        print('All Cameras\tAUROC\tAUPR')
        print('           \t%.4f\t%.4f\n' % (auroc_pred, aupr_pred))

        y_hats_comb = quantile_transform_errors(y_hats_comb)
        y_trues_comb_flat, y_hats_comb_flat = [], []
        for camera_id in camera_ids:
            y_trues_comb_flat.append(y_trues_comb[camera_id])
            y_hats_comb_flat.append(y_hats_comb[camera_id])

        y_trues_comb_flat, y_hats_comb_flat = np.concatenate(y_trues_comb_flat), np.concatenate(y_hats_comb_flat)
        if is_avenue:
            auroc_comb = roc_auc_score(y_trues_comb_flat[AVENUE_MASK], y_hats_comb_flat[AVENUE_MASK])
            aupr_comb = average_precision_score(y_trues_comb_flat[AVENUE_MASK], y_hats_comb_flat[AVENUE_MASK])
        else:
            auroc_comb = roc_auc_score(y_trues_comb_flat, y_hats_comb_flat)
            aupr_comb = average_precision_score(y_trues_comb_flat, y_hats_comb_flat)

        print('Reconstruction + Prediction Based:')
        print('All Cameras\tAUROC\tAUPR')
        print('           \t%.4f\t%.4f\n' % (auroc_comb, aupr_comb))
