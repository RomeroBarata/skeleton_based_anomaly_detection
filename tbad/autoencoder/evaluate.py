from collections import namedtuple
import os

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from tbad.autoencoder.autoencoder import load_pretrained_ae
from tbad.autoencoder.data import load_trajectories, extract_global_features, change_coordinate_system
from tbad.autoencoder.data import aggregate_autoencoder_evaluation_data, scale_trajectories, remove_missing_skeletons
from tbad.autoencoder.data import compute_ae_reconstruction_errors, load_anomaly_masks
from tbad.autoencoder.data import assemble_ground_truth_and_reconstructions, quantile_transform_errors


def eval_ae(args):
    # General
    trajectories_path = args.trajectories
    camera_id = os.path.basename(trajectories_path)
    pretrained_model_path = args.pretrained_model  # e.g. .../adam_bb-tl_mse/06_2018_09_29_00_35_20
    video_resolution = [float(measurement) for measurement in args.video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)
    frame_level_anomaly_masks_path = args.frame_level_anomaly_masks

    # Extract information about the models
    model_info = os.path.basename(os.path.split(pretrained_model_path)[0])
    global_model = 'gm' in model_info

    coordinate_system = 'global'
    if 'bb-tl' in model_info:
        coordinate_system = 'bounding_box_top_left'
    elif 'bb-c' in model_info:
        coordinate_system = 'bounding_box_centre'

    normalisation_strategy = 'zero_one'
    if '_3stds_' in model_info:
        normalisation_strategy = 'three_stds'
    elif '_robust_' in model_info:
        normalisation_strategy = 'robust'

    pretrained_ae, scaler = load_pretrained_ae(pretrained_model_path)

    # Load data
    trajectories = load_trajectories(trajectories_path)

    if global_model:
        trajectories = extract_global_features(trajectories, video_resolution=video_resolution)
        coordinate_system = 'global'

    trajectories = change_coordinate_system(trajectories, video_resolution=video_resolution,
                                            coordinate_system=coordinate_system, invert=False)

    trajectories_ids, frames, X = aggregate_autoencoder_evaluation_data(trajectories)

    X, (trajectories_ids, frames) = remove_missing_skeletons(X, trajectories_ids, frames)

    X, _ = scale_trajectories(X, scaler=scaler, strategy=normalisation_strategy)

    # Reconstruct
    reconstructed_X = pretrained_ae.predict(X)
    reconstruction_errors = compute_ae_reconstruction_errors(X, reconstructed_X, loss=pretrained_ae.loss)

    # Evaluate Performance
    anomaly_masks = load_anomaly_masks(frame_level_anomaly_masks_path)
    y_true, y_hat = assemble_ground_truth_and_reconstructions(anomaly_masks, trajectories_ids,
                                                              frames, reconstruction_errors)

    auroc, aupr = roc_auc_score(y_true, y_hat), average_precision_score(y_true, y_hat)

    print('Camera %s:\tAUROC\tAUPR' % camera_id)
    print('          \t%.4f\t%.4f\n' % (auroc, aupr))

    # Logging

    return y_true, y_hat


def eval_aes(args):
    all_trajectories_path = args.all_trajectories
    pretrained_models_path = args.pretrained_models  # e.g. .../adam_bb-tl_mse
    video_resolution = args.video_resolution
    all_frame_level_anomaly_masks_path = args.all_frame_level_anomaly_masks

    EvalAeArgs = namedtuple('EvalAeArgs',
                            ['trajectories', 'pretrained_model', 'frame_level_anomaly_masks', 'video_resolution'])

    pretrained_models_dirs = sorted(os.listdir(pretrained_models_path))
    y_trues, y_hats = {}, {}
    for pretrained_model_dir in pretrained_models_dirs:
        camera_id = pretrained_model_dir.split('_')[0]
        pretrained_model_path = os.path.join(pretrained_models_path, pretrained_model_dir)
        trajectories_path = os.path.join(all_trajectories_path, camera_id)
        frame_level_anomaly_masks_path = os.path.join(all_frame_level_anomaly_masks_path, camera_id)
        eval_ae_args = EvalAeArgs(trajectories_path, pretrained_model_path,
                                  frame_level_anomaly_masks_path, video_resolution)
        y_true, y_hat = eval_ae(eval_ae_args)
        y_trues[camera_id], y_hats[camera_id] = y_true, y_hat

    y_hats = quantile_transform_errors(y_hats)
    camera_ids = sorted(y_hats.keys())
    y_trues_flat, y_hats_flat = [], []
    for camera_id in camera_ids:
        y_trues_flat.append(y_trues[camera_id])
        y_hats_flat.append(y_hats[camera_id])

    y_trues_flat, y_hats_flat = np.concatenate(y_trues_flat), np.concatenate(y_hats_flat)
    auroc, aupr = roc_auc_score(y_trues_flat, y_hats_flat), average_precision_score(y_trues_flat, y_hats_flat)

    print('All Cameras\tAUROC\tAUPR')
    print('           \t%.4f\t%.4f\n' % (auroc, aupr))
