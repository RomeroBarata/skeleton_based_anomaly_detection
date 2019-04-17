# Temporary data manipulating routines for test
import numpy as np

from tbad.losses import binary_crossentropy, mean_absolute_error, mean_squared_error, balanced_mean_squared_error
from tbad.losses import balanced_mean_absolute_error


def remove_short_trajectories(trajectories, input_length, input_gap, pred_length=0):
    filtered_trajectories = {}
    for trajectory_id, trajectory in trajectories.items():
        if not trajectory.is_short(input_length=input_length, input_gap=input_gap, pred_length=pred_length):
            filtered_trajectories[trajectory_id] = trajectory

    return filtered_trajectories


def aggregate_rnn_autoencoder_data(trajectories, input_length, input_gap=0, pred_length=0):
    Xs, Xs_pred = [], []
    for trajectory in trajectories.values():
        X, X_pred = _aggregate_rnn_autoencoder_data(trajectory.coordinates, input_length, input_gap, pred_length)
        Xs.append(X)
        if X_pred is not None:
            Xs_pred.append(X_pred)

    Xs = np.vstack(Xs)
    if not Xs_pred:
        Xs_pred = None
    else:
        Xs_pred = np.vstack(Xs_pred)

    return Xs, Xs_pred


def _aggregate_rnn_autoencoder_data(coordinates, input_length, input_gap=0, pred_length=0):
    input_trajectories, future_trajectories = [], None
    total_input_seq_len = input_length + input_gap * (input_length - 1)
    step = input_gap + 1
    if pred_length > 0:
        future_trajectories = []
        stop = len(coordinates) - pred_length - total_input_seq_len + 1
        for start_index in range(0, stop):
            stop_index = start_index + total_input_seq_len
            input_trajectories.append(coordinates[start_index:stop_index:step, :])
            future_trajectories.append(coordinates[stop_index:(stop_index + pred_length), :])
        input_trajectories = np.stack(input_trajectories, axis=0)
        future_trajectories = np.stack(future_trajectories, axis=0)
    else:
        stop = len(coordinates) - total_input_seq_len + 1
        for start_index in range(0, stop):
            stop_index = start_index + total_input_seq_len
            input_trajectories.append(coordinates[start_index:stop_index:step, :])
        input_trajectories = np.stack(input_trajectories, axis=0)

    return input_trajectories, future_trajectories


def aggregate_rnn_ae_evaluation_data(trajectories, input_length, input_gap, pred_length, overlapping_trajectories):
    trajectories_ids, frames, X = [], [], []
    for trajectory in trajectories.values():
        traj_ids, traj_frames, traj_X = _aggregate_rnn_ae_evaluation_data(trajectory, input_length)
        trajectories_ids.append(traj_ids)
        frames.append(traj_frames)
        X.append(traj_X)

    trajectories_ids, frames, X = np.vstack(trajectories_ids), np.vstack(frames), np.vstack(X)

    return trajectories_ids, frames, X


def _aggregate_rnn_ae_evaluation_data(trajectory, input_length):
    traj_frames, traj_X = [], []
    coordinates = trajectory.coordinates
    frames = trajectory.frames

    total_input_seq_len = input_length
    stop = len(coordinates) - total_input_seq_len + 1
    for start_index in range(stop):
        stop_index = start_index + total_input_seq_len
        traj_X.append(coordinates[start_index:stop_index, :])
        traj_frames.append(frames[start_index:stop_index])
    traj_frames, traj_X = np.stack(traj_frames, axis=0), np.stack(traj_X, axis=0)

    trajectory_id = trajectory.trajectory_id
    traj_ids = np.full(traj_frames.shape, fill_value=trajectory_id)

    return traj_ids, traj_frames, traj_X


def compute_rnn_ae_reconstruction_errors(X, reconstructed_X, loss):
    num_examples, input_length, input_dim = X.shape
    X = X.reshape(-1, input_dim)
    reconstructed_X = reconstructed_X.reshape(-1, input_dim)

    loss_fn = {'log_loss': binary_crossentropy, 'mae': mean_absolute_error,
               'mse': mean_squared_error, 'balanced_mse': balanced_mean_squared_error,
               'balanced_mae': balanced_mean_absolute_error}[loss]
    reconstruction_errors = loss_fn(X, reconstructed_X)

    return reconstruction_errors.reshape(num_examples, input_length)


def summarise_reconstruction_errors(reconstruction_errors, frames, trajectory_ids):
    unique_ids = np.unique(trajectory_ids)
    all_trajectory_ids, all_summarised_frames, all_summarised_errors = [], [], []
    for trajectory_id in unique_ids:
        mask = trajectory_ids == trajectory_id
        current_frames = frames[mask]
        current_errors = reconstruction_errors[mask]
        summarised_frames, summarised_errors = summarise_reconstruction_errors_per_frame(current_errors, current_frames)
        all_summarised_frames.append(summarised_frames)
        all_summarised_errors.append(summarised_errors)
        all_trajectory_ids.append([trajectory_id] * len(summarised_frames))

    all_trajectory_ids = np.concatenate(all_trajectory_ids)
    all_summarised_frames = np.concatenate(all_summarised_frames)
    all_summarised_errors = np.concatenate(all_summarised_errors)

    return all_trajectory_ids, all_summarised_frames, all_summarised_errors


def summarise_reconstruction_errors_per_frame(errors, frames):
    unique_frames = np.unique(frames)
    unique_errors = np.empty(unique_frames.shape, dtype=np.float32)
    for idx, frame in enumerate(unique_frames):
        mask = frames == frame
        unique_errors[idx] = np.mean(errors[mask])

    return unique_frames, unique_errors


def summarise_reconstruction(reconstructed_X, frames, trajectory_ids):
    unique_ids = np.unique(trajectory_ids)
    num_examples, input_length, input_dim = reconstructed_X.shape
    reconstructed_X = reconstructed_X.reshape(-1, input_dim)
    frames = frames.reshape(-1)
    trajectory_ids = trajectory_ids.reshape(-1)

    all_trajectory_ids, all_summarised_frames, all_summarised_recs = [], [], []
    for trajectory_id in unique_ids:
        mask = trajectory_ids == trajectory_id
        current_frames = frames[mask]
        current_reconstructions = reconstructed_X[mask, :]
        summarised_frames, summarised_recs = summarise_reconstruction_per_frame(current_reconstructions, current_frames)
        all_summarised_frames.append(summarised_frames)
        all_summarised_recs.append(summarised_recs)
        all_trajectory_ids.append([trajectory_id] * len(summarised_frames))

    all_trajectory_ids = np.concatenate(all_trajectory_ids)
    all_summarised_frames = np.concatenate(all_summarised_frames)
    all_summarised_recs = np.vstack(all_summarised_recs)

    reconstructed_X = reconstructed_X.reshape(num_examples, input_length, input_dim)
    frames = frames.reshape(num_examples, input_length)
    trajectory_ids = trajectory_ids.reshape(num_examples, input_length)

    return all_trajectory_ids, all_summarised_frames, all_summarised_recs


def summarise_reconstruction_per_frame(recs, frames):
    unique_frames = np.unique(frames)
    unique_recs = np.empty((len(unique_frames), recs.shape[-1]), dtype=np.float32)
    for idx, frame in enumerate(unique_frames):
        mask = frames == frame
        unique_recs[idx, :] = np.mean(recs[mask, :], axis=0, keepdims=True)

    return unique_frames, unique_recs


def retrieve_future_skeletons(trajectories_ids, X, pred_length):
    input_dim = X.shape[-1]
    traj_id_per_example = trajectories_ids[:, 0]
    indices = np.unique(traj_id_per_example, return_index=True)[1]
    unique_ids = [traj_id_per_example[idx] for idx in sorted(indices)]

    y = []
    for unique_id in unique_ids:
        current_ids = unique_id == traj_id_per_example
        current_X = X[current_ids, :, :]
        future_X = current_X[pred_length:, -pred_length:, :]
        padding = np.zeros(shape=(pred_length, pred_length, input_dim), dtype=np.float32)
        future_X = np.concatenate((future_X, padding), axis=0)
        y.append(future_X)

    y = np.vstack(y)

    return y


def discard_information_from_padded_frames(pred_ids, pred_frames, pred_errors, pred_length):
    id_per_example = pred_ids[:, 0]
    indices = np.unique(id_per_example, return_index=True)[1]
    unique_ids = [id_per_example[idx] for idx in sorted(indices)]

    all_ids, all_frames, all_errors = [], [], []
    for unique_id in unique_ids:
        current_ids = unique_id == id_per_example
        actual_ids = pred_ids[current_ids][:-pred_length]
        actual_frames = pred_frames[current_ids][:-pred_length]
        actual_errors = pred_errors[current_ids][:-pred_length]

        all_ids.append(actual_ids)
        all_frames.append(actual_frames)
        all_errors.append(actual_errors)

    all_ids = np.vstack(all_ids)
    all_frames = np.vstack(all_frames)
    all_errors = np.vstack(all_errors)

    return all_ids, all_frames, all_errors
