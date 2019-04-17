# Temporary data manipulating routines for test
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, quantile_transform

from tbad.data import StdScaler
from tbad.visualisation import compute_bounding_box
from tbad.losses import binary_crossentropy, mean_absolute_error, mean_squared_error


class Trajectory:
    def __init__(self, trajectory_id, frames, coordinates):
        self.trajectory_id = trajectory_id
        self.person_id = trajectory_id.split('_')[1]
        self.frames = frames
        self.coordinates = coordinates
        self.is_global = False

    def __len__(self):
        return len(self.frames)

    def use_global_features(self, video_resolution, extract_delta=False, use_first_step_as_reference=False):
        self.coordinates = self._extract_global_features(video_resolution=video_resolution, extract_delta=extract_delta,
                                                         use_first_step_as_reference=use_first_step_as_reference)
        self.is_global = True

    def use_size_features(self, video_resolution):
        self.coordinates = self._extract_size_features(video_resolution=video_resolution)

    def _extract_size_features(self, video_resolution):
        bbs = np.apply_along_axis(compute_bounding_box, axis=1, arr=self.coordinates, video_resolution=video_resolution)
        bbs_measures = np.apply_along_axis(self._extract_bounding_box_measurements, axis=1, arr=bbs)
        return bbs_measures

    def _extract_global_features(self, video_resolution, extract_delta=False, use_first_step_as_reference=False):
        bounding_boxes = np.apply_along_axis(compute_bounding_box, axis=1, arr=self.coordinates,
                                             video_resolution=video_resolution)
        bbs_measures = np.apply_along_axis(self._extract_bounding_box_measurements, axis=1, arr=bounding_boxes)
        bbs_centre = np.apply_along_axis(self._extract_bounding_box_centre, axis=1, arr=bounding_boxes)
        if extract_delta:
            bbs_delta = np.vstack((np.full((1, 2), fill_value=1e-7), np.diff(bbs_centre, axis=0)))

        if use_first_step_as_reference:
            bbs_centre -= bbs_centre[0]
            # bbs_centre /= np.where(bbs_measures == 0.0, 1.0, bbs_measures)
            bbs_centre[0] += 1e-6

        if extract_delta:
            return np.hstack((bbs_centre, bbs_delta, bbs_measures))

        return np.hstack((bbs_centre, bbs_measures))

    @staticmethod
    def _extract_bounding_box_centre(bb):
        x = (bb[0] + bb[1]) / 2
        y = (bb[2] + bb[3]) / 2

        return np.array([x, y], dtype=np.float32)

    @staticmethod
    def _extract_bounding_box_measurements(bb):
        width = bb[1] - bb[0]
        height = bb[3] - bb[2]

        return np.array([width, height], dtype=np.float32)

    def change_coordinate_system(self, video_resolution, coordinate_system='global', invert=False):
        if invert:
            if coordinate_system == 'global':
                self.coordinates = self._from_global_to_image(self.coordinates, video_resolution=video_resolution)
            else:
                raise ValueError('Unknown coordinate system. Only global is available for inversion.')
        else:
            if coordinate_system == 'global':
                self.coordinates = self._from_image_to_global(self.coordinates, video_resolution=video_resolution)
            elif coordinate_system == 'bounding_box_top_left':
                self.coordinates = self._from_image_to_bounding_box(self.coordinates,
                                                                    video_resolution=video_resolution,
                                                                    location='top_left')
            elif coordinate_system == 'bounding_box_centre':
                self.coordinates = self._from_image_to_bounding_box(self.coordinates,
                                                                    video_resolution=video_resolution,
                                                                    location='centre')
            else:
                raise ValueError('Unknown coordinate system. Please select one of: global, bounding_box_top_left, or '
                                 'bounding_box_centre.')

    @staticmethod
    def _from_global_to_image(coordinates, video_resolution):
        original_shape = coordinates.shape
        coordinates = coordinates.reshape(-1, 2) * video_resolution

        return coordinates.reshape(original_shape)

    @staticmethod
    def _from_image_to_global(coordinates, video_resolution):
        original_shape = coordinates.shape
        coordinates = coordinates.reshape(-1, 2) / video_resolution

        return coordinates.reshape(original_shape)

    @staticmethod
    def _from_image_to_bounding_box(coordinates, video_resolution, location='centre'):
        if location == 'top_left':
            fn = Trajectory._from_image_to_top_left_bounding_box
        elif location == 'centre':
            fn = Trajectory._from_image_to_centre_bounding_box
        else:
            raise ValueError('Unknown location for the bounding box. Please select either top_left or centre.')

        coordinates = fn(coordinates, video_resolution=video_resolution)

        return coordinates

    @staticmethod
    def _from_image_to_top_left_bounding_box(coordinates, video_resolution):
        for idx, kps in enumerate(coordinates):
            if any(kps):
                left, right, top, bottom = compute_bounding_box(kps, video_resolution=video_resolution)
                xs, ys = np.hsplit(kps.reshape(-1, 2), indices_or_sections=2)
                xs, ys = np.where(xs == 0.0, float(left), xs), np.where(ys == 0.0, float(top), ys)
                xs, ys = (xs - left) / (right - left), (ys - top) / (bottom - top)
                kps = np.hstack((xs, ys)).ravel()

            coordinates[idx] = kps

        return coordinates

    @staticmethod
    def _from_image_to_centre_bounding_box(coordinates, video_resolution):
        # TODO: Better implementation
        # coordinates = np.where(coordinates == 0, np.nan, coordinates)
        # bounding_boxes = np.apply_along_axis(compute_bounding_box, axis=1, arr=coordinates,
        #                                      video_resolution=video_resolution)
        # centre_x = (bounding_boxes[:, 0] + bounding_boxes[:, 1]) / 2
        # centre_y = (bounding_boxes[:, 2] + bounding_boxes[:, 3]) / 2
        for idx, kps in enumerate(coordinates):
            if any(kps):
                left, right, top, bottom = compute_bounding_box(kps, video_resolution=video_resolution)
                centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
                xs, ys = np.hsplit(kps.reshape(-1, 2), indices_or_sections=2)
                xs, ys = np.where(xs == 0.0, centre_x, xs) - centre_x, np.where(ys == 0.0, centre_y, ys) - centre_y
                left, right, top, bottom = left - centre_x, right - centre_x, top - centre_y, bottom - centre_y
                width, height = right - left, bottom - top
                xs, ys = xs / width, ys / height
                kps = np.hstack((xs, ys)).ravel()

            coordinates[idx] = kps

        return coordinates

    def is_short(self, input_length, input_gap, pred_length=0):
        min_trajectory_length = input_length + input_gap * (input_length - 1) + pred_length

        return len(self) < min_trajectory_length

    def input_missing_steps(self):
        """Fill missing steps with a weighted average of the closest non-missing steps."""
        trajectory_length, input_dim = self.coordinates.shape
        last_step_non_missing = 0
        consecutive_missing_steps = 0
        while last_step_non_missing < trajectory_length - 1:
            step_is_missing = np.sum(self.coordinates[last_step_non_missing + 1, :] == 0) == input_dim
            while step_is_missing:
                consecutive_missing_steps += 1
                step_is_missing = \
                    np.sum(self.coordinates[last_step_non_missing + 1 + consecutive_missing_steps, :] == 0) == input_dim

            if consecutive_missing_steps:
                start_trajectory = self.coordinates[last_step_non_missing, :]
                end_trajectory = self.coordinates[last_step_non_missing + 1 + consecutive_missing_steps, :]
                for n in range(1, consecutive_missing_steps + 1):
                    a = ((consecutive_missing_steps + 1 - n) / (consecutive_missing_steps + 1)) * start_trajectory
                    b = (n / (consecutive_missing_steps + 1)) * end_trajectory
                    fill_step = a + b
                    fill_step = np.where((start_trajectory == 0) | (end_trajectory == 0), 0, fill_step)
                    self.coordinates[last_step_non_missing + n, :] = fill_step

            last_step_non_missing += consecutive_missing_steps + 1
            consecutive_missing_steps = 0


def load_trajectories(trajectories_path):
    trajectories = {}
    folder_names = os.listdir(trajectories_path)
    for folder_name in folder_names:
        csv_file_names = os.listdir(os.path.join(trajectories_path, folder_name))
        for csv_file_name in csv_file_names:
            trajectory_file_path = os.path.join(trajectories_path, folder_name, csv_file_name)
            trajectory = np.loadtxt(trajectory_file_path, dtype=np.float32, delimiter=',', ndmin=2)
            trajectory_frames, trajectory_coordinates = trajectory[:, 0].astype(np.int32), trajectory[:, 1:]
            person_id = csv_file_name.split('.')[0]
            trajectory_id = folder_name + '_' + person_id
            trajectories[trajectory_id] = Trajectory(trajectory_id=trajectory_id,
                                                     frames=trajectory_frames,
                                                     coordinates=trajectory_coordinates)

    return trajectories


def extract_global_features(trajectories, video_resolution, extract_delta=False, use_first_step_as_reference=False):
    for trajectory in trajectories.values():
        trajectory.use_global_features(video_resolution=video_resolution, extract_delta=extract_delta,
                                       use_first_step_as_reference=use_first_step_as_reference)

    return trajectories


def extract_size_features(trajectories, video_resolution):
    for trajectory in trajectories.values():
        trajectory.use_size_features(video_resolution=video_resolution)

    return trajectories


def change_coordinate_system(trajectories, video_resolution, coordinate_system='global', invert=False):
    for trajectory in trajectories.values():
        trajectory.change_coordinate_system(video_resolution, coordinate_system=coordinate_system, invert=invert)

    return trajectories


def split_into_train_and_test(trajectories, train_ratio=0.8, seed=42):
    np.random.seed(seed)

    trajectories_ids = []
    trajectories_lengths = []
    for trajectory_id, trajectory in trajectories.items():
        trajectories_ids.append(trajectory_id)
        trajectories_lengths.append(len(trajectory))

    sorting_indices = np.argsort(trajectories_lengths)
    q1_idx = round(len(sorting_indices) * 0.25)
    q2_idx = round(len(sorting_indices) * 0.50)
    q3_idx = round(len(sorting_indices) * 0.75)

    sorted_ids = np.array(trajectories_ids)[sorting_indices]
    train_ids = []
    val_ids = []
    quantiles_indices = [0, q1_idx, q2_idx, q3_idx, len(sorting_indices)]
    for idx, q_idx in enumerate(quantiles_indices[1:], 1):
        q_ids = sorted_ids[quantiles_indices[idx - 1]:q_idx]
        q_ids = np.random.permutation(q_ids)
        train_idx = round(len(q_ids) * train_ratio)
        train_ids.extend(q_ids[:train_idx])
        val_ids.extend(q_ids[train_idx:])

    trajectories_train = {}
    for train_id in train_ids:
        trajectories_train[train_id] = trajectories[train_id]

    trajectories_val = {}
    for val_id in val_ids:
        trajectories_val[val_id] = trajectories[val_id]

    return trajectories_train, trajectories_val


def scale_trajectories(X, scaler=None, strategy='zero_one'):
    original_shape = X.shape
    input_dim = original_shape[-1]
    X = X.reshape(-1, input_dim)

    if strategy == 'zero_one':
        X_scaled, scaler = scale_trajectories_zero_one(X, scaler=scaler)
    elif strategy == 'three_stds':
        X_scaled, scaler = scale_trajectories_three_stds(X, scaler=scaler)
    elif strategy == 'robust':
        X_scaled, scaler = scale_trajectories_robust(X, scaler=scaler)
    else:
        raise ValueError('Unknown strategy. Please select either zero_one or three_stds.')

    X, X_scaled = X.reshape(original_shape), X_scaled.reshape(original_shape)

    return X_scaled, scaler


def scale_trajectories_zero_one(X, scaler=None):
    if scaler is None:
        X = np.where(X == 0.0, np.nan, X)
        X_min = np.nanmin(X, axis=0, keepdims=True)
        X_min = np.where(np.isnan(X_min), 0.0, X_min)
        X_min = np.tile(X_min, reps=[X.shape[0], 1])

        eps = 1e-3
        X = np.where(np.isnan(X), X_min - eps, X)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X)

    num_examples = X.shape[0]
    X_scaled = np.where(X == 0.0, np.tile(scaler.data_min_, reps=[num_examples, 1]), X)
    X_scaled = scaler.transform(X_scaled)

    return X_scaled, scaler


def scale_trajectories_three_stds(X, scaler=None):
    if scaler is None:
        X = np.where(X == 0.0, np.nan, X)

        scaler = StdScaler(stds=3)
        scaler.fit(X)

    X_scaled = np.where(X == 0.0, np.nan, X)
    X_scaled = scaler.transform(X_scaled)
    X_scaled = np.where(np.isnan(X_scaled), 0.0, X_scaled)

    return X_scaled, scaler


def scale_trajectories_robust(X, scaler=None):
    X_scaled = np.where(X == 0.0, np.nan, X)
    if scaler is None:
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        scaler.fit(X_scaled)

    X_scaled = scaler.transform(X_scaled)
    X_scaled = np.where(np.isnan(X_scaled), 0.0, X_scaled)

    return X_scaled, scaler


def aggregate_autoencoder_data(trajectories):
    X = []
    for trajectory in trajectories.values():
        X.append(trajectory.coordinates)

    return np.vstack(X)


def aggregate_autoencoder_evaluation_data(trajectories):
    trajectories_ids, frames, X = [], [], []
    for trajectory_id, trajectory in trajectories.items():
        frames.append(trajectory.frames)
        X.append(trajectory.coordinates)
        trajectories_ids.append(np.repeat(trajectory_id, repeats=len(trajectory.frames)))

    return np.concatenate(trajectories_ids), np.concatenate(frames), np.vstack(X)


def remove_missing_skeletons(X, *arrs):
    non_missing_skeletons = np.sum(np.abs(X), axis=1) > 0.0
    X = X[non_missing_skeletons]
    filtered_arrs = []
    for idx, arr in enumerate(arrs):
        filtered_arrs.append(arr[non_missing_skeletons])

    return X, filtered_arrs


def compute_ae_reconstruction_errors(X, reconstructed_X, loss):
    loss_fn = {'log_loss': binary_crossentropy, 'mae': mean_absolute_error, 'mse': mean_squared_error}[loss]
    return loss_fn(X, reconstructed_X)


def load_anomaly_masks(anomaly_masks_path):
    file_names = os.listdir(anomaly_masks_path)
    masks = {}
    for file_name in file_names:
        full_id = file_name.split('.')[0]
        file_path = os.path.join(anomaly_masks_path, file_name)
        masks[full_id] = np.load(file_path)

    return masks


def assemble_ground_truth_and_reconstructions(anomaly_masks, trajectory_ids,
                                              reconstruction_frames, reconstruction_errors,
                                              return_video_ids=False):
    y_true, y_hat = {}, {}
    for full_id in anomaly_masks.keys():
        _, video_id = full_id.split('_')
        y_true[video_id] = anomaly_masks[full_id].astype(np.int32)
        y_hat[video_id] = np.zeros_like(y_true[video_id], dtype=np.float32)

    unique_ids = np.unique(trajectory_ids)
    for trajectory_id in unique_ids:
        video_id, _ = trajectory_id.split('_')
        indices = trajectory_ids == trajectory_id
        frames = reconstruction_frames[indices]
        y_hat[video_id][frames] = np.maximum(y_hat[video_id][frames], reconstruction_errors[indices])

    y_true_, y_hat_, video_ids = [], [], []
    for video_id in sorted(y_true.keys()):
        y_true_.append(y_true[video_id])
        y_hat_.append(y_hat[video_id])
        video_ids.extend([video_id] * len(y_true_[-1]))

    y_true_, y_hat_ = np.concatenate(y_true_), np.concatenate(y_hat_)

    if return_video_ids:
        return y_true_, y_hat_, video_ids
    else:
        return y_true_, y_hat_


def quantile_transform_errors(y_hats):
    for camera_id, y_hat in y_hats.items():
        y_hats[camera_id] = quantile_transform(y_hat.reshape(-1, 1)).reshape(-1)

    return y_hats


def input_trajectories_missing_steps(trajectories):
    for trajectory in trajectories.values():
        trajectory.input_missing_steps()

    return trajectories
