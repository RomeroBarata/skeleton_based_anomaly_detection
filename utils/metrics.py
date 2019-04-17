import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from tbad.data import input2table
from tbad.losses import binary_crossentropy, mean_squared_error, mean_absolute_error


def frame_level_metrics(anomaly_masks, reconstruction_errors, reconstruction_frames):
    y_true, y_hat = {}, {}
    for full_id in anomaly_masks:
        _, video_id = full_id.split('_')
        y_true[video_id] = anomaly_masks[full_id].astype(np.int64)
        y_hat[video_id] = np.zeros_like(y_true[video_id], dtype=np.float64)

    for trajectory_id in reconstruction_errors:
        video_id, _ = trajectory_id.split('_')
        frames = reconstruction_frames[trajectory_id].astype(np.int64)
        y_hat[video_id][frames] = np.maximum(y_hat[video_id][frames], reconstruction_errors[trajectory_id])

    y_true_, y_hat_ = [], []
    for video_id in sorted(y_true.keys()):
        y_true_.append(y_true[video_id])
        y_hat_.append(y_hat[video_id])

    y_true_, y_hat_ = np.concatenate(y_true_), np.concatenate(y_hat_)

    return roc_auc_score(y_true_, y_hat_), average_precision_score(y_true_, y_hat_)


def ground_truth_and_reconstructions(anomaly_masks, reconstruction_errors, reconstruction_frames):
    y_true, y_hat = {}, {}
    for full_id in anomaly_masks:
        _, video_id = full_id.split('_')
        y_true[video_id] = anomaly_masks[full_id].astype(np.int64)
        y_hat[video_id] = np.zeros_like(y_true[video_id], dtype=np.float64)

    for trajectory_id in reconstruction_errors:
        video_id, _ = trajectory_id.split('_')
        frames = reconstruction_frames[trajectory_id].astype(np.int64)
        y_hat[video_id][frames] = np.maximum(y_hat[video_id][frames], reconstruction_errors[trajectory_id])

    y_true_, y_hat_, video_ids = [], [], []
    for video_id in sorted(y_true.keys()):
        y_true_.append(y_true[video_id])
        y_hat_.append(y_hat[video_id])
        video_ids.extend([video_id] * len(y_true_[-1]))

    y_true_, y_hat_ = np.concatenate(y_true_), np.concatenate(y_hat_)

    return y_true_, y_hat_, video_ids


def summarise_errors_per_frame(trajectory_errors_per_frame, trajectory_frames, summary_fn=np.mean):
    """Summarises the error of a frame.

    Argument(s):
        trajectory_errors_per_frame -- A numpy array of shape (trajectory_length,).
        trajectory_frames -- A numpy array of shape (trajectory_length // input_seq_len, input_seq_len).
        summary_fn -- A function to summarise errors that belong to the same frame.
    """
    trajectory_frames_reshaped = trajectory_frames.reshape(-1).astype(np.int64)
    frame_ids = np.unique(trajectory_frames_reshaped)
    summarised_errors = np.empty_like(frame_ids, dtype=np.float64)
    for frame_index, frame_id in enumerate(frame_ids):
        summarised_errors[frame_index] = summary_fn(trajectory_errors_per_frame[trajectory_frames_reshaped == frame_id])

    return summarised_errors


def compute_reconstruction_errors(trajectories_coordinates, reconstructed_trajectories_coordinates, loss='mse'):
    loss_fn = {'log_loss': binary_crossentropy, 'mae': mean_absolute_error, 'mse': mean_squared_error}[loss]
    reconstruction_errors = {}
    for trajectory_id in trajectories_coordinates:
        trajectory_coordinates_tbl = input2table(trajectories_coordinates[trajectory_id])
        reconstructed_trajectory_coordinates_tbl = input2table(reconstructed_trajectories_coordinates[trajectory_id])
        reconstruction_error_per_frame = loss_fn(trajectory_coordinates_tbl,
                                                 reconstructed_trajectory_coordinates_tbl)
        reconstruction_errors[trajectory_id] = reconstruction_error_per_frame

    return reconstruction_errors


def summarise_reconstruction_errors(reconstruction_errors, trajectories_frames, summary_fn=np.mean):
    for trajectory_id in reconstruction_errors:
        reconstruction_error_per_frame = reconstruction_errors[trajectory_id]
        reconstruction_errors[trajectory_id] = summarise_errors_per_frame(reconstruction_error_per_frame,
                                                                          trajectories_frames[trajectory_id],
                                                                          summary_fn=summary_fn)

    return reconstruction_errors


def discard_errors_from_padded_frames(reconstruction_errors, original_trajectory_lengths):
    """Discard errors from padded frames.

    Argument(s):
        reconstruction_errors: A dictionary where the keys identify the trajectory (video + person) and the values are
            numpy arrays of shape (trajectory_length,) containing the reconstruction error of each step.
        original_trajectory_lengths: A dictionary where the keys identify the trajectory (video + person) and the
            values are integers identifying the original length of the trajectory.

    Return(s):
        A dictionary similar to reconstruction_errors but the trajectories have the right length.
    """
    for trajectory_id in reconstruction_errors:
        original_length = original_trajectory_lengths[trajectory_id]
        reconstruction_errors[trajectory_id] = reconstruction_errors[trajectory_id][:original_length]

    return reconstruction_errors
