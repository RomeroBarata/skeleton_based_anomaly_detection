import os

import numpy as np

from tbad.visualisation import insert_anomaly_mask_from_bounding_box, compute_bounding_box


def inverse_scale(X, scaler):
    original_shape = X.shape
    input_dim = original_shape[-1]
    X = X.reshape(-1, input_dim)
    X = scaler.inverse_transform(X)
    X = X.reshape(original_shape)

    return X


def restore_global_coordinate_system(X, video_resolution):
    original_shape = X.shape
    X = X.reshape(-1, 2) * video_resolution
    X = X.reshape(original_shape)

    return X


def restore_original_trajectory(reconstructed_X_global, reconstructed_X_local):
    # X_global is already in image coordinates
    # X_local is in bounding_box_coordinates
    num_examples, input_length, local_input_dim = reconstructed_X_local.shape
    global_input_dim = reconstructed_X_global.shape[-1]
    reconstructed_X_global = reconstructed_X_global.reshape(-1, global_input_dim)
    reconstructed_X_local = reconstructed_X_local.reshape(-1, local_input_dim)
    reps = local_input_dim // 2
    reconstructed_X_traj = reconstructed_X_local * np.tile(reconstructed_X_global[:, -2:], reps=reps)
    reconstructed_X_traj += np.tile(reconstructed_X_global[:, :2], reps=reps)
    reconstructed_X_traj = reconstructed_X_traj.reshape(num_examples, input_length, local_input_dim)

    return reconstructed_X_traj


def write_reconstructed_trajectories(pretrained_model_path, reconstructed_traj,
                                     reconstruction_ids, reconstruction_frames, trajectory_type='skeleton'):
    if trajectory_type == 'skeleton':
        top_dir = 'reconstructed_skeletons'
    elif trajectory_type == 'bounding_box':
        top_dir = 'reconstructed_bounding_boxes'
    elif trajectory_type == 'predicted_skeleton':
        top_dir = 'predicted_skeletons'
    elif trajectory_type == 'predicted_bounding_box':
        top_dir = 'predicted_bounding_boxes'
    else:
        raise ValueError('Unknown trajectory type. Please choose one of skeleton, bounding_box, '
                         'predicted_skeleton or predicted_bounding_box.')

    video_ids, skeleton_ids = extract_video_and_skeleton_ids(reconstruction_ids)
    unique_video_ids = np.unique(video_ids)

    writing_dir = os.path.join(pretrained_model_path, top_dir)
    if not os.path.isdir(writing_dir):
        os.makedirs(writing_dir)

    for video_id in unique_video_ids:
        video_writing_dir = os.path.join(writing_dir, video_id)
        if not os.path.isdir(video_writing_dir):
            os.makedirs(video_writing_dir)

        mask = video_ids == video_id
        current_skeleton_ids = skeleton_ids[mask]
        current_frames = reconstruction_frames[mask]
        current_recs = reconstructed_traj[mask, :]

        unique_current_skeleton_ids = np.unique(current_skeleton_ids)
        for skeleton_id in unique_current_skeleton_ids:
            skeleton_writing_file = os.path.join(video_writing_dir, skeleton_id) + '.csv'
            mask = current_skeleton_ids == skeleton_id
            current_skeleton_frames = current_frames[mask].reshape(-1, 1)
            current_skeleton_recs = current_recs[mask, :]
            trajectory = np.hstack((current_skeleton_frames, current_skeleton_recs))
            np.savetxt(skeleton_writing_file, trajectory, fmt='%.4f', delimiter=',')


def extract_video_and_skeleton_ids(reconstruction_ids):
    split_ids = np.core.defchararray.split(reconstruction_ids, sep='_')
    video_ids, skeleton_ids = [], []
    for ids in split_ids:
        video_id, skeleton_id = ids
        video_ids.append(video_id)
        skeleton_ids.append(skeleton_id)

    return np.array(video_ids), np.array(skeleton_ids)


def detect_most_anomalous_or_most_normal_frames(reconstruction_errors, anomalous=True, fraction=0.20):
    reconstruction_errors_sorted = np.sort(reconstruction_errors)
    num_frames_to_blame = round(len(reconstruction_errors_sorted) * fraction)
    if anomalous:
        threshold = np.round(reconstruction_errors_sorted[-num_frames_to_blame], decimals=8)
        anomalous_or_normal_frames = reconstruction_errors >= threshold
    else:
        threshold = np.round(reconstruction_errors_sorted[num_frames_to_blame - 1], decimals=8)
        anomalous_or_normal_frames = (0 < reconstruction_errors) & (reconstruction_errors <= threshold)

    return anomalous_or_normal_frames


def compute_num_frames_per_video(anomaly_masks):
    num_frames_per_video = {}
    for full_id, anomaly_mask in anomaly_masks.items():
        _, video_id = full_id.split('_')
        num_frames_per_video[video_id] = len(anomaly_mask)

    return num_frames_per_video


def write_predicted_masks(pretrained_model_path, num_frames_per_video, anomalous_frames, normal_frames,
                          reconstructed_bounding_boxes, reconstruction_ids, reconstruction_frames, video_resolution):
    video_ids, skeleton_ids = extract_video_and_skeleton_ids(reconstruction_ids)
    unique_video_ids = np.unique(video_ids)

    width, height = video_resolution
    width, height = int(width), int(height)

    anomaly_path = os.path.join(pretrained_model_path, 'predicted_pixel_level_anomaly_masks')
    if not os.path.isdir(anomaly_path):
        os.makedirs(anomaly_path)

    normal_path = os.path.join(pretrained_model_path, 'predicted_pixel_level_normal_masks')
    if not os.path.isdir(normal_path):
        os.makedirs(normal_path)

    for video_id in unique_video_ids:
        num_frames = num_frames_per_video[video_id]
        anomaly_mask = np.zeros((num_frames, height, width), dtype=np.uint8)
        normal_mask = np.zeros((num_frames, height, width), dtype=np.uint8)

        mask = video_ids == video_id
        current_anomalous_frames, current_normal_frames = anomalous_frames[mask], normal_frames[mask]
        current_bounding_boxes, current_frames = reconstructed_bounding_boxes[mask, :], reconstruction_frames[mask]
        for idx, frame in enumerate(current_frames):
            bounding_box = current_bounding_boxes[idx, :]
            if current_anomalous_frames[idx]:
                insert_anomaly_mask_from_bounding_box(anomaly_mask[frame], bounding_box)

            if current_normal_frames[idx]:
                insert_anomaly_mask_from_bounding_box(normal_mask[frame], bounding_box)

        np.save(os.path.join(anomaly_path, video_id), arr=anomaly_mask)
        np.save(os.path.join(normal_path, video_id), arr=normal_mask)


def compute_worst_mistakes(y_true, y_hat, video_ids, error_type='false_positives', top=10):
    positive_indices = y_hat > 0
    video_ids_ = np.array(video_ids)
    frames = generate_array_of_frames(video_ids_)

    y_true_ = y_true[positive_indices]
    y_hat_ = y_hat[positive_indices]
    video_ids_ = video_ids_[positive_indices]
    frames = frames[positive_indices]

    if error_type == 'false_positives':
        true_negatives = y_true_ == 0
        y_hat_ = y_hat_[true_negatives]
        video_ids_ = video_ids_[true_negatives]
        frames_ = frames[true_negatives]

        sorting_indices = np.argsort(y_hat_)

        indices = sorting_indices[-top:]
    elif error_type == 'false_negatives':
        true_positives = y_true_ == 1
        y_hat_ = y_hat_[true_positives]
        video_ids_ = video_ids_[true_positives]
        frames_ = frames[true_positives]

        sorting_indices = np.argsort(y_hat_)

        indices = sorting_indices[:top]
    else:
        raise ValueError('Unknown mistake type. Please choose either false_positives or false_negatives.')

    return video_ids_[indices], frames_[indices], y_hat_[indices]


def generate_array_of_frames(x):
    """x is already sorted."""
    _, counts = np.unique(x, return_counts=True)
    result = []
    for count in counts:
        result.append(np.arange(count))

    return np.concatenate(result)


def write_worst_mistakes(pretrained_model_path, worst_false_positives, worst_false_negatives):
    camera_id = os.path.basename(pretrained_model_path).split('_')[0]
    file_path = os.path.join(pretrained_model_path, 'mistakes.txt')
    with open(file_path, mode='w') as file:
        print('\nCamera ID: %s' % camera_id, file=file)
        print('\nWorst False Positives:', file=file)
        video_ids, frames, scores = worst_false_positives
        for video_id, frame, score in zip(video_ids[::-1], frames[::-1], scores[::-1]):
            print('Video ID: %s\tFrame: %d\tRec. Error: %.4f' % (video_id, frame, score), file=file)

        print('\nWorst False Negatives:', file=file)
        video_ids, frames, scores = worst_false_negatives
        for video_id, frame, score in zip(video_ids, frames, scores):
            print('Video ID: %s\tFrame: %d\tRec. Error: %.4f' % (video_id, frame, score), file=file)

    return None


def clip_trajectories(trajectories, video_resolution, margin=0.05):
    clipped_trajectories = {}
    for trajectory_id, trajectory in trajectories.items():
        clipped_trajectory = clip_trajectory(trajectory, video_resolution, margin)
        clipped_trajectories[trajectory_id] = clipped_trajectory

    return clipped_trajectories


def clip_trajectory(trajectory, video_resolution, margin=0.05):
    width, height = video_resolution
    left_margin, top_margin = margin * width, margin * height
    right_margin, bottom_margin = width - left_margin, height - top_margin
    frames, coordinates = trajectory.frames, trajectory.coordinates
    steps_to_include = np.ones_like(frames, dtype=np.bool)
    for idx, kps in enumerate(coordinates):
        x, y = np.hsplit(kps.reshape(-1, 2), indices_or_sections=2)
        x, y = x[x != 0.0], y[y != 0.0]
        if np.all(x < left_margin) or np.all(x > right_margin) or np.all(y < top_margin) or np.all(y > bottom_margin):
            steps_to_include[idx] = False
        else:
            break

    for idx, kps in enumerate(coordinates[::-1, :], start=1):
        x, y = np.hsplit(kps.reshape(-1, 2), indices_or_sections=2)
        x, y = x[x != 0.0], y[y != 0.0]
        if np.all(x < left_margin) or np.all(x > right_margin) or np.all(y < top_margin) or np.all(y > bottom_margin):
            steps_to_include[-idx] = False
        else:
            break

    frames, coordinates = frames[steps_to_include], coordinates[steps_to_include]
    trajectory.frames, trajectory.coordinates = frames, coordinates

    return trajectory


def normalise_errors_by_bounding_box_area(errors, X, video_resolution):
    original_shape = X.shape
    input_dim = original_shape[-1]
    errors = errors.reshape(-1)
    X = X.reshape(-1, input_dim)
    bbs = np.apply_along_axis(compute_bounding_box, axis=1, arr=X, video_resolution=video_resolution)
    widths, heights = bbs[:, 1] - bbs[:, 0], bbs[:, 3] - bbs[:, 2]
    bbs_areas = np.float_power(widths * heights, 1)
    bbs_areas = np.where(bbs_areas == 0, 1, bbs_areas)
    errors_normalised = (errors / bbs_areas).reshape(original_shape[:2])
    errors = errors.reshape(original_shape[:2])
    X = X.reshape(original_shape)

    return errors_normalised
