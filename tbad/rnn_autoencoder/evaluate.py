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
from tbad.rnn_autoencoder.data import retrieve_future_skeletons, discard_information_from_padded_frames
from tbad.rnn_autoencoder.rnn import load_pretrained_rnn_ae


def eval_rnn_ae(args):
    # General
    trajectories_path = args.trajectories  # e.g. .../optflow/alphapose/08
    camera_id = os.path.basename(trajectories_path)
    pretrained_model_path = args.pretrained_model  # e.g. .../16_0_2_rrs_bb-c_3stds_mse/08_2018_10_02_11_39_20
    video_resolution = [float(measurement) for measurement in args.video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)
    frame_level_anomaly_masks_path = args.frame_level_anomaly_masks
    overlapping_trajectories = args.overlapping_trajectories

    # Load Pre-Trained Model
    pretrained_rnn_ae, scaler = load_pretrained_rnn_ae(pretrained_model_path)

    # Extract information about the model
    model_info = os.path.basename(os.path.split(pretrained_model_path)[0])
    global_model = '_gm_' in model_info
    extract_delta = '_ed_' in model_info
    use_first_step_as_reference = '_ufsar_' in model_info
    concatenate_model = '_cm_' in model_info

    coordinate_system = 'global'
    if '_bb-tl_' in model_info:
        coordinate_system = 'bounding_box_top_left'
    elif '_bb-c_' in model_info:
        coordinate_system = 'bounding_box_centre'

    normalisation_strategy = 'zero_one'
    if '_3stds_' in model_info:
        normalisation_strategy = 'three_stds'
    elif '_robust_' in model_info:
        normalisation_strategy = 'robust'

    input_length, input_gap = pretrained_rnn_ae.input_length, pretrained_rnn_ae.input_gap
    rec_length, pred_length = pretrained_rnn_ae.reconstruction_length, pretrained_rnn_ae.prediction_length
    reconstruct_reverse = pretrained_rnn_ae.reconstruct_reverse
    loss = pretrained_rnn_ae.loss

    # Data pre-processing
    trajectories = load_trajectories(trajectories_path)

    trajectories = remove_short_trajectories(trajectories, input_length=input_length,
                                             input_gap=input_gap, pred_length=pred_length)

    if concatenate_model:
        trajectories_global = extract_global_features(deepcopy(trajectories), video_resolution=video_resolution,
                                                      extract_delta=extract_delta,
                                                      use_first_step_as_reference=use_first_step_as_reference)
        trajectories_global = change_coordinate_system(trajectories_global, video_resolution=video_resolution,
                                                       coordinate_system='global', invert=False)
        trajectories_ids, frames, X_global = \
            aggregate_rnn_ae_evaluation_data(trajectories_global,
                                             input_length=input_length,
                                             input_gap=input_gap,
                                             pred_length=pred_length,
                                             overlapping_trajectories=overlapping_trajectories)

        trajectories_local = trajectories
        trajectories_local = change_coordinate_system(trajectories_local, video_resolution=video_resolution,
                                                      coordinate_system='bounding_box_centre', invert=False)
        _, _, X_local = aggregate_rnn_ae_evaluation_data(trajectories_local, input_length=input_length,
                                                         input_gap=input_gap, pred_length=pred_length,
                                                         overlapping_trajectories=overlapping_trajectories)

        X = np.concatenate((X_global, X_local), axis=-1)
        X, _ = scale_trajectories(X, scaler=scaler, strategy=normalisation_strategy)
    else:
        if global_model:
            trajectories = extract_global_features(trajectories, video_resolution=video_resolution,
                                                   extract_delta=extract_delta,
                                                   use_first_step_as_reference=use_first_step_as_reference)
            coordinate_system = 'global'

        trajectories = change_coordinate_system(trajectories, video_resolution=video_resolution,
                                                coordinate_system=coordinate_system, invert=False)

        trajectories_ids, frames, X = aggregate_rnn_ae_evaluation_data(trajectories, input_length=input_length,
                                                                       input_gap=input_gap, pred_length=pred_length,
                                                                       overlapping_trajectories=overlapping_trajectories)

        X, _ = scale_trajectories(X, scaler=scaler, strategy=normalisation_strategy)

    # Reconstruct
    if pred_length == 0:
        reconstructed_X = pretrained_rnn_ae.predict(X)
    else:
        reconstructed_X, predicted_y = pretrained_rnn_ae.predict(X)

    if reconstruct_reverse:
        reconstructed_X = reconstructed_X[:, ::-1, :]

    reconstruction_errors = compute_rnn_ae_reconstruction_errors(X[:, :rec_length, :], reconstructed_X, loss)
    reconstruction_ids, reconstruction_frames, reconstruction_errors = \
        summarise_reconstruction_errors(reconstruction_errors, frames[:, :rec_length], trajectories_ids[:, :rec_length])

    # Evaluate performance
    anomaly_masks = load_anomaly_masks(frame_level_anomaly_masks_path)
    y_true, y_hat = assemble_ground_truth_and_reconstructions(anomaly_masks, reconstruction_ids,
                                                              reconstruction_frames, reconstruction_errors)

    auroc, aupr = roc_auc_score(y_true, y_hat), average_precision_score(y_true, y_hat)

    print('Reconstruction Based:')
    print('Camera %s:\tAUROC\tAUPR' % camera_id)
    print('          \t%.4f\t%.4f\n' % (auroc, aupr))

    # Future Prediction
    if pred_length > 0:
        predicted_frames = frames[:, :pred_length] + input_length
        predicted_ids = trajectories_ids[:, :pred_length]

        y = retrieve_future_skeletons(trajectories_ids, X, pred_length)

        pred_errors = compute_rnn_ae_reconstruction_errors(y, predicted_y, loss)

        predicted_ids, predicted_frames, pred_errors = discard_information_from_padded_frames(predicted_ids,
                                                                                              predicted_frames,
                                                                                              pred_errors, pred_length)

        pred_ids, pred_frames, pred_errors = \
            summarise_reconstruction_errors(pred_errors, predicted_frames, predicted_ids)

        y_true_pred, y_hat_pred = assemble_ground_truth_and_reconstructions(anomaly_masks, pred_ids,
                                                                            pred_frames, pred_errors)
        auroc, aupr = roc_auc_score(y_true_pred, y_hat_pred), average_precision_score(y_true_pred, y_hat_pred)

        print('Prediction Based:')
        print('Camera %s:\tAUROC\tAUPR' % camera_id)
        print('          \t%.4f\t%.4f\n' % (auroc, aupr))

        y_true_comb, y_hat_comb = y_true, y_hat + y_hat_pred
        auroc, aupr = roc_auc_score(y_true_comb, y_hat_comb), average_precision_score(y_true_comb, y_hat_comb)

        print('Reconstruction + Prediction Based:')
        print('Camera %s:\tAUROC\tAUPR' % camera_id)
        print('          \t%.4f\t%.4f\n' % (auroc, aupr))

    # Logging

    if pred_length > 0:
        return y_true, y_hat, y_true_pred, y_hat_pred, y_true_comb, y_hat_comb
    else:
        return y_true, y_hat, None, None, None, None


def eval_rnn_aes(args):
    all_trajectories_path = args.all_trajectories
    pretrained_models_path = args.pretrained_models  # e.g. .../
    video_resolution = args.video_resolution
    all_frame_level_anomaly_masks_path = args.all_frame_level_anomaly_masks
    overlapping_trajectories = args.overlapping_trajectories

    EvalRNNAeArgs = namedtuple('EvalRNNAeArgs',
                               ['pretrained_model', 'trajectories', 'frame_level_anomaly_masks', 'video_resolution',
                                'overlapping_trajectories'])

    pretrained_models_dirs = sorted(os.listdir(pretrained_models_path))
    y_trues, y_hats, y_trues_pred, y_hats_pred, y_trues_comb, y_hats_comb = {}, {}, {}, {}, {}, {}
    for pretrained_model_dir in pretrained_models_dirs:
        camera_id = pretrained_model_dir.split('_')[0]
        pretrained_model_path = os.path.join(pretrained_models_path, pretrained_model_dir)
        trajectories_path = os.path.join(all_trajectories_path, camera_id)
        frame_level_anomaly_masks_path = os.path.join(all_frame_level_anomaly_masks_path, camera_id)
        eval_rnn_ae_args = EvalRNNAeArgs(pretrained_model_path, trajectories_path, frame_level_anomaly_masks_path,
                                         video_resolution, overlapping_trajectories)
        y_true, y_hat, y_true_pred, y_hat_pred, y_true_comb, y_hat_comb = eval_rnn_ae(eval_rnn_ae_args)
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
        auroc_comb = roc_auc_score(y_trues_comb_flat, y_hats_comb_flat)
        aupr_comb = average_precision_score(y_trues_comb_flat, y_hats_comb_flat)

        print('Reconstruction + Prediction Based:')
        print('All Cameras\tAUROC\tAUPR')
        print('           \t%.4f\t%.4f\n' % (auroc_comb, aupr_comb))
