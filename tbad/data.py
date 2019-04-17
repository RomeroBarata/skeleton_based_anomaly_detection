import os
from copy import deepcopy

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tbad.visualisation import compute_bounding_box


# For future refactoring
class Trajectory:
    def __init__(self, trajectory_id, frames, coordinates):
        self.trajectory_id = trajectory_id
        self.person_id = trajectory_id.split('_')[1]
        self.frames = frames
        self.coordinates = coordinates

    def __len__(self):
        return len(self.frames)


def load_trajectories(trajectories_path):
    """Load trajectory data into memory.

    The trajectory data contains information about the trajectory of each person detected in the video(s). More
    specifically, each person's trajectory is a .csv file, where each row contains the frame number followed by 17 pairs
    of x and y coordinates. Any missing detections are represented by pairs of zeros.

    The directory contains a folder for each video analysed, and each folder contains a .csv file for each person
    tracked. In summary, the directory is structured like trajectories_path/{<video_id>}/{<person_id>}.csv.

    Argument(s):
        trajectories_path -- Path to directory containing the trajectories to load.

    Return(s):
        Two dictionaries. Both dictionaries contain the same set of keys, which uniquely identify each trajectory
        loaded. Each value of the first dictionary is a numpy array of integers, of shape (trajectory_length,),
        containing the frame numbers of each trajectory. The second dictionary contains the actual trajectories, numpy
        arrays of shape (trajectory_length, input_dim).
    """
    trajectories_frames, trajectories_coordinates = {}, {}
    folder_names = os.listdir(trajectories_path)
    for folder_name in folder_names:
        csv_file_names = os.listdir(os.path.join(trajectories_path, folder_name))
        for csv_file_name in csv_file_names:
            trajectory_file_path = os.path.join(trajectories_path, folder_name, csv_file_name)
            trajectory = np.loadtxt(trajectory_file_path, dtype=np.float32, delimiter=',', ndmin=2)
            trajectory_frames, trajectory_coordinates = trajectory[:, 0].astype(np.int32), trajectory[:, 1:]
            person_id = csv_file_name.split('.')[0]
            trajectory_id = folder_name + '_' + person_id
            trajectories_frames[trajectory_id] = trajectory_frames
            trajectories_coordinates[trajectory_id] = trajectory_coordinates

    return trajectories_frames, trajectories_coordinates


def is_short_trajectory(trajectory_coordinates, input_length, input_gap=0, pred_length=0):
    """Identify whether a trajectory is short or not.

    To reconstruct a sequence of frames and predict a number of frames into the future, the trajectory of a person must
    have a minimum length of input_length + input_gap * (input_length - 1) + pred_length.

    Argument(s):
        trajectory_coordinates -- A numpy array of shape (trajectory_length, input_dim).
        input_length -- Number of timesteps for the encoder-decoder RNN.
        input_gap -- Number of timesteps to skip during encoding of adjacent timesteps.
        pred_length -- Number of future timesteps for the encoder-decoder RNN to predict.

    Return(s):
        True if the input trajectory is short. Otherwise, False.
    """
    min_trajectory_length = input_length + input_gap * (input_length - 1) + pred_length

    return len(trajectory_coordinates) < min_trajectory_length


def remove_short_trajectories(trajectories_coordinates, input_length, input_gap=0, pred_length=0):
    """Remove short trajectories.

    Argument(s):
        trajectories_coordinates -- A dictionary of numpy arrays, each of shape (trajectory_length, input_dim).
        input_length -- Number of input timesteps for the encoder-decoder RNN.
        input_gap -- Number of timesteps to skip during the encoding of adjacent timesteps.
        pred_length -- Number of future timesteps for the encoder-decoder RNN to predict.

    Return(s):
        A dictionary of numpy arrays, each of shape (trajectory_length, input_dim). The arrays have
        a trajectory_length greater than a computed threshold.
    """
    filtered_trajectories_coordinates = {trajectory_id: trajectory_coordinates
                                         for trajectory_id, trajectory_coordinates in trajectories_coordinates.items()
                                         if not is_short_trajectory(trajectory_coordinates,
                                                                    input_length=input_length,
                                                                    input_gap=input_gap,
                                                                    pred_length=pred_length)}

    return filtered_trajectories_coordinates


def input_trajectory_missing_steps(trajectory_coordinates):
    """Fill missing steps with a weighted average of the closest non-missing steps.

    Argument(s):
        trajectory_coordinates: A numpy array of shape (trajectory_length, input_dim).

    Return(s):
        A numpy array of shape (trajectory_length, input_dim) with no missing steps.
    """
    trajectory_length, input_dim = trajectory_coordinates.shape
    last_step_non_missing = 0
    consecutive_missing_steps = 0
    while last_step_non_missing < trajectory_length - 1:
        step_is_missing = np.sum(trajectory_coordinates[last_step_non_missing + 1, :] == 0) == input_dim
        while step_is_missing:
            consecutive_missing_steps += 1
            step_is_missing = np.sum(trajectory_coordinates[last_step_non_missing + 1 + consecutive_missing_steps, :] == 0) == input_dim

        if consecutive_missing_steps:
            start_trajectory = trajectory_coordinates[last_step_non_missing, :]
            end_trajectory = trajectory_coordinates[last_step_non_missing + 1 + consecutive_missing_steps, :]
            for n in range(1, consecutive_missing_steps + 1):
                a = ((consecutive_missing_steps + 1 - n) / (consecutive_missing_steps + 1)) * start_trajectory
                b = (n / (consecutive_missing_steps + 1)) * end_trajectory
                fill_step = a + b
                fill_step = np.where((start_trajectory == 0) | (end_trajectory == 0), 0, fill_step)
                trajectory_coordinates[last_step_non_missing + n, :] = fill_step

        last_step_non_missing += consecutive_missing_steps + 1
        consecutive_missing_steps = 0

    return trajectory_coordinates


def input_trajectories_missing_steps(trajectories_coordinates):
    trajectories_coordinates = {trajectory_id: input_trajectory_missing_steps(trajectory_coordinates)
                                for trajectory_id, trajectory_coordinates in trajectories_coordinates.items()}

    return trajectories_coordinates


def normalise_trajectory_video_resolution(trajectory_coordinates, video_resolution):
    """Normalise the coordinate-values of a trajectory into the [0, 1] range.

    Argument(s):
        trajectory_coordinates -- A numpy array of shape (trajectory_length, input_dim).
        video_resolution -- A list containing the width and height of the trajectories' original videos.

    Return(s):
        The input trajectory normalised into the [0, 1] range.
    """
    video_resolution = np.array(video_resolution, dtype=np.float32)
    normalised_trajectory_coordinates = trajectory_coordinates.reshape(-1, 2) / video_resolution

    return normalised_trajectory_coordinates.reshape(*trajectory_coordinates.shape)


def normalise_trajectories_video_resolution(trajectories_coordinates, video_resolution):
    """Normalise trajectories' coordinates into the [0, 1] range.

    Convert all x and y coordinates of the trajectories into the [0, 1] range. All x coordinates are divided by the
    video's width and y coordinates are divided by the video's height.

    Argument(s):
        trajectories -- A dictionary of numpy arrays, each of shape (trajectory_length, input_dim).
        video_resolution -- A list containing the width and height of the trajectories' original videos.

    Return(s):
        A dictionary of numpy arrays, each of shape (trajectory_length, input_dim). All trajectories' coordinates
        values are in the [0, 1] range.
    """
    normalised_coordinates = {trajectory_id: normalise_trajectory_video_resolution(trajectory_coordinates,
                                                                                   video_resolution=video_resolution)
                              for trajectory_id, trajectory_coordinates in trajectories_coordinates.items()}

    return normalised_coordinates


def normalise_trajectories(trajectories_coordinates, video_resolution, strategy='video_resolution'):
    if strategy == 'bounding_box':
        normalised_coordinates = normalise_bounding_boxes(trajectories_coordinates, video_resolution=video_resolution)
        print('\nTrajectories normalised into the [0, 1] range relative to the bounding boxes.')
    elif strategy == 'video_resolution':
        normalised_coordinates = normalise_trajectories_video_resolution(trajectories_coordinates,
                                                                         video_resolution=video_resolution)
        print('\nTrajectories normalised into the [0, 1] range relative to the video resolution.')
    else:
        raise ValueError('Unknown normalisation strategy. Please use either bounding_box or video_resolution.')

    return normalised_coordinates


def normalise_bounding_box(trajectory_coordinates, video_resolution):
    trajectory_length = trajectory_coordinates.shape[0]
    for step_idx in range(trajectory_length):
        coordinates = trajectory_coordinates[step_idx, :]
        if any(coordinates):
            left, right, top, bottom = compute_bounding_box(coordinates, video_resolution=video_resolution)
            coordinates_reshaped = coordinates.reshape(-1, 2)
            xs, ys = coordinates_reshaped[:, 0], coordinates_reshaped[:, 1]
            xs, ys = np.where(xs == 0.0, float(left), xs), np.where(ys == 0.0, float(top), ys)
            xs = (xs - left) / (right - left)
            ys = (ys - top) / (bottom - top)
            coordinates_reshaped[:, 0], coordinates_reshaped[:, 1] = xs, ys
            coordinates = coordinates_reshaped.reshape(-1)
        trajectory_coordinates[step_idx, :] = coordinates

    return trajectory_coordinates


def normalise_bounding_boxes(trajectories_coordinates, video_resolution):
    normalised_coordinates = {trajectory_id: normalise_bounding_box(trajectory_coordinates,
                                                                    video_resolution=video_resolution)
                              for trajectory_id, trajectory_coordinates in trajectories_coordinates.items()}

    return normalised_coordinates


def normalise_joints(X, min_max_values=None):
    num_examples, input_length, input_dim = X.shape
    X = X.reshape(-1, input_dim)
    X = np.where(X == 0.0, np.nan, X)
    if min_max_values is not None:
        min_values = min_max_values[0]
        max_values = min_max_values[1]
    else:
        min_values = np.nanmin(X, axis=0)
        max_values = np.nanmax(X, axis=0)
    X = (X - min_values) / (max_values - min_values)
    X = np.where(np.isnan(X), 0.0, X)
    X = X.reshape(num_examples, input_length, input_dim)

    return X, (min_values, max_values)


def denormalise_trajectory(trajectory_coordinates, video_resolution):
    video_resolution = np.array(video_resolution, dtype=np.float32)
    denormalised_trajectory_coordinates = trajectory_coordinates.reshape(-1, 2) * video_resolution
    return denormalised_trajectory_coordinates.reshape(*trajectory_coordinates.shape)


def denormalise_trajectories(trajectories_coordinates, video_resolution):
    """Denormalise trajectories' coordinates into the original video range.

    Convert all x and y coordinates of the trajectories from the [0, 1] range into the video's original range. All x
    coordinates are multiplied by the video's width and y coordinates are multiplied by the video's height.

    Argument(s):
        trajectories -- A list of numpy arrays, each of shape (trajectory_length, 34).
        video_resolution -- A list containing the width and height of the trajectories' original videos.

    Return(s):
        A list of numpy arrays, each of shape (trajectory_length, 34). All trajectories' coordinates values are in the
        video's original range.
    """
    video_resolution = np.array(video_resolution, dtype=np.float32)
    denormalised_trajectories_coordinates = \
        {trajectory_id: denormalise_trajectory(trajectory_coordinates,
                                               video_resolution=video_resolution)
         for trajectory_id, trajectory_coordinates in trajectories_coordinates.items()}

    return denormalised_trajectories_coordinates


def collect_trajectories(trajectory_coordinates, input_length, input_gap=0, pred_length=0):
    input_trajectories = []
    total_input_seq_len = input_length + input_gap * (input_length - 1)
    step = input_gap + 1
    if pred_length > 0:
        future_trajectories = []
        stop = len(trajectory_coordinates) - pred_length - total_input_seq_len + 1
        for start_index in range(0, stop):
            stop_index = start_index + total_input_seq_len
            input_trajectories.append(trajectory_coordinates[start_index:stop_index:step, :])
            future_trajectories.append(trajectory_coordinates[stop_index:(stop_index + pred_length), :])
        return np.stack(input_trajectories, axis=0), np.stack(future_trajectories, axis=0)
    else:
        stop = len(trajectory_coordinates) - total_input_seq_len + 1
        for start_index in range(0, stop):
            stop_index = start_index + total_input_seq_len
            input_trajectories.append(trajectory_coordinates[start_index:stop_index:step, :])
        return np.stack(input_trajectories, axis=0)


# def gather_trajectories(trajectories, input_seq_len, gap_between_frames=0, pred_seq_len=None):
#     """Assemble all possible trajectories into tensors.
#
#     Collect all possible trajectories into a single tensor for training of the RNN encoder-decoder. If pred_seq_len is
#     None only a single tensor of shape (num_unique_trajectories, input_seq_len, 34) is returned; otherwise, another
#     tensor containing the future frames, of shape (num_unique_trajectories, pred_seq_len, 34), is also returned.
#
#     The number of unique trajectories is dependent on the values of input_seq_len, gap_between_frames and pred_seq_len.
#
#     Argument(s):
#         trajectories -- A list of numpy arrays, each of shape(trajectory_length, 35).
#         input_seq_len -- Number of input timesteps for the encoder-decoder RNN.
#         gap_between_frames -- Number of timesteps to skip during the encoding of adjacent timesteps.
#         pred_seq_len -- Number of future timesteps for the encoder-decoder RNN to predict.
#
#     Return(s):
#         If pred_seq_len is None, a single tensor of shape (num_unique_trajectories, input_seq_len, 34). Otherwise,
#         a tensor of shape (num_unique_trajectories, pred_seq_len, 34) is returned in addition to the aforementioned one.
#     """
#     encoding_trajectories = []
#     if pred_seq_len is not None:
#         pred_trajectories = []
#     total_input_seq_len = input_seq_len + gap_between_frames * (input_seq_len - 1)
#     encoding_step = gap_between_frames + 1
#     for trajectory in trajectories:
#         if pred_seq_len is not None:
#             stop = len(trajectory) - pred_seq_len - total_input_seq_len + 1
#             for start_index in range(0, stop):
#                 stop_index = start_index + total_input_seq_len
#                 encoding_trajectories.append(trajectory[start_index:stop_index:encoding_step, 1:])
#                 pred_trajectories.append(trajectory[stop_index:(stop_index + pred_seq_len), 1:])
#         else:
#             stop = len(trajectory) - total_input_seq_len + 1
#             for start_index in range(0, stop):
#                 stop_index = start_index + total_input_seq_len
#                 encoding_trajectories.append(trajectory[start_index:stop_index:encoding_step, 1:])
#
#     if pred_seq_len is not None:
#         return np.stack(encoding_trajectories), np.stack(pred_trajectories)
#     else:
#         return np.stack(encoding_trajectories)


def collect_test_trajectories(trajectory_coordinates, trajectory_frames, input_length, input_gap=0, pred_length=0):
    """Collect all possible trajectories into a single tensor.

    Argument(s):
        trajectory_coordinates -- A numpy array of shape (trajectory_length, input_dim) containing the keypoints'
            coordinates at each step of the trajectory.
        trajectory_frames -- A numpy array of shape (trajectory_length,) containing the frame ids of each trajectory
            step in trajectory_coordinates.
        input_length -- .
        input_gap -- .
        pred_length -- .

    Return(s):
    """
    input_trajectories_coordinates = []
    input_trajectories_frames = []

    last_frame_id = trajectory_frames[-1]
    trajectory_original_length = len(trajectory_frames)
    padding_length = ((trajectory_original_length // input_length) + 1) * input_length - trajectory_original_length

    padded_trajectory_coordinates = pad_trajectory(trajectory_coordinates, padding_length=padding_length)
    padded_trajectory_frames = np.concatenate((trajectory_frames, last_frame_id + np.arange(1, padding_length + 1)))

    stop = len(padded_trajectory_coordinates) - input_length + 1
    for start_index in range(0, stop, input_length):
        stop_index = start_index + input_length
        input_trajectories_frames.append(padded_trajectory_frames[start_index:stop_index])
        input_trajectories_coordinates.append(padded_trajectory_coordinates[start_index:stop_index, :])

    return np.stack(input_trajectories_frames, axis=0), np.stack(input_trajectories_coordinates, axis=0)


def collect_overlapping_trajectories(trajectory_coordinates, trajectory_frames,
                                     input_length=12, input_gap=0, pred_length=0):
    input_trajectories_coordinates = []
    input_trajectories_frames = []
    total_input_seq_len = input_length
    stop = len(trajectory_coordinates) - total_input_seq_len + 1
    for start_index in range(0, stop):
        stop_index = start_index + total_input_seq_len
        input_trajectories_frames.append(trajectory_frames[start_index:stop_index])
        input_trajectories_coordinates.append(trajectory_coordinates[start_index:stop_index, :])

    return np.stack(input_trajectories_frames, axis=0), np.stack(input_trajectories_coordinates, axis=0)


# def gather_test_trajectories(trajectories, trajectories_ids, input_seq_len, gap_between_frames=0, pred_seq_len=None):
#     """Assemble necessary trajectories into tensors for testing.
#
#     During test time, only unique trajectories are needed for prediction. In addition, it is necessary to keep track of
#     which videos the trajectories come from. The number of unique trajectories depend on the values of input_seq_len,
#     gap_between_frames and pred_seq_len.
#
#     WARNING: For now, this function only works correctly for gap_between_frames=0 and pred_seq_len=None.
#
#     Argument(s):
#         trajectories -- A list of numpy arrays, each of shape (trajectory_length, 35).
#         trajectories_ids -- A list of ids for each trajectory in trajectories.
#         input_seq_len -- Number of input timesteps for the encoder-decoder RNN.
#         gap_between_frames -- Number of timesteps to skip during the encoding of adjacent timesteps.
#         pred_seq_len -- Number of future timesteps for the encoder-decoder RNN to predict.
#
#     Return(s):
#         A dictionary containing the test tensors of each video. In addition, a second dictionary containing information
#         about the frame id of each trajectory is returned.
#     """
#     encoding_trajectories = {}
#     encoding_frame_ids = {}
#     for trajectory, trajectory_id in zip(trajectories, trajectories_ids):
#         encoding_trajectories.setdefault(trajectory_id, [])
#         encoding_frame_ids.setdefault(trajectory_id, [])
#         padded_trajectory = pad_trajectory(trajectory, input_seq_len, gap_between_frames)
#         stop = len(padded_trajectory) - input_seq_len + 1
#         for start_index in range(0, stop, input_seq_len):
#             stop_index = start_index + input_seq_len
#             frame_ids = padded_trajectory[start_index:stop_index, 0]
#             coordinates = padded_trajectory[start_index:stop_index, 1:]
#             encoding_trajectories[trajectory_id].append(coordindenormalise_trajectoriesates)
#             encoding_frame_ids[trajectory_id].append(frame_ids)
#         encoding_trajectories[trajectory_id] = np.stack(encoding_trajectories[trajectory_id])
#         encoding_frame_ids[trajectory_id] = np.stack(encoding_frame_ids[trajectory_id])
#
#     return encoding_trajectories, encoding_frame_ids


def pad_trajectory(trajectory_coordinates, padding_length):
    zeros = np.zeros((padding_length, trajectory_coordinates.shape[1]), dtype=np.float32)
    return np.vstack((trajectory_coordinates, zeros))


# def pad_trajectory(trajectory, input_seq_len, gap_between_frames=0):
#     """Append null frames to the end of trajectory.
#
#     Argument(s):
#         trajectory -- A numpy array of shape (trajectory_length, 35).
#         input_seq_len -- Number of input timesteps for the encoder-decoder RNN.
#         gap_between_frames -- Number of timesteps to skip during the encoding of adjacent timesteps.
#
#     Return(s):
#         The input trajectory with null frames appended to the end.
#     """
#     last_frame_id = trajectory[-1, 0]
#     trajectory_length = len(trajectory)
#     padding_size = ((trajectory_length // input_seq_len) + 1 ) * input_seq_len - trajectory_length
#
#     new_frame_ids = last_frame_id + np.arange(1, padding_size + 1).reshape(-1, 1)
#     new_coords = np.zeros((padding_size, trajectory.shape[1] - 1))
#     new_trajectories = np.hstack((new_frame_ids, new_coords))
#
#     return np.vstack((trajectory, new_trajectories))


def shuffle_data(X, y=None, seed=None):
    """Shuffle the input data.

    Argument(s):
        X -- numpy array of rank >= 2 where the first axis represents the examples.
        y -- numpy array of rank >= 2 where the first axis represents the target values of the examples in X.
        seed -- integer value to seed numpy's random number generator and make the shuffle reproducible.

    Return(s):
        The input data shuffled.
    """
    if seed is not None:
        np.random.seed(seed)

    indexes = np.random.permutation(X.shape[0])
    if y is not None:
        return X[indexes], y[indexes]
    else:
        return X[indexes]


def input2table(reconstructed_trajectory):
    """Convert reconstructed trajectories to original format.

    Argument(s):
        reconstructed_trajectory -- a numpy array of shape (num_unique_trajectories, input_seq_len, 34),
            containing the trajectory reconstructed by the Encoder-Decoder RNN.

    Return(s):
        A dictionary ...
    """
    input_dim = reconstructed_trajectory.shape[-1]
    return reconstructed_trajectory.reshape(-1, input_dim)


def write_trajectories(write_path, trajectories):
    """Write reconstructed or predicted trajectories to disk.

    For each video identified in trajectories' keys, a sub-directory will be created to write the .csv
    files containing the reconstructed or predicted trajectories of different people.

    Argument(s):
        write_path -- Root directory to write reconstructed trajectories.
        trajectories -- A dictionary where the keys contain the video id and the person id in the format
            <video_id>_<person_id> and the values are numpy arrays of shape (trajectory_length, input_dim + 1). The
            first column contains the frame number and the other columns the x and y coordinates of each keypoint.
    """
    for trajectory_id, trajectory in trajectories.items():
        video_id, person_id = trajectory_id.split('_')
        video_folder_path = os.path.join(write_path, video_id)
        if not os.path.exists(video_folder_path):
            os.makedirs(video_folder_path, exist_ok=True)

        csv_file_path = os.path.join(video_folder_path, person_id + '.csv')
        np.savetxt(csv_file_path, trajectory, fmt='%.4f', delimiter=',')


def load_anomaly_masks(masks_path, camera_id=None):
    """Load anomaly masks into memory.

    Argument(s):
        masks_path -- Path to directory containing all anomaly masks, either at the frame-level or pixel-level.
        camera_id -- If specified, only load the anomaly masks for this camera.

    Return(s):
        A dictionary where the file names are the keys and the values are the anomaly masks. The anomaly masks are
        numpy arrays of shape (num_frames,) or (num_frames, video_height, video_width), depending on the type of mask.
    """
    file_names = os.listdir(masks_path)
    if camera_id is not None:
        file_names = [file_name for file_name in file_names if file_name.startswith(camera_id)]

    masks = {}
    for file_name in file_names:
        full_id = file_name.split('.')[0]
        file_path = os.path.join(masks_path, file_name)
        masks[full_id] = np.load(file_path)

    return masks


def assemble_trajectories(trajectories_frames, trajectories_coordinates, overlapping=False,
                          input_length=12, input_gap=0, pred_length=None):
    collect_fn = collect_overlapping_trajectories if overlapping else collect_test_trajectories
    test_trajectories_frames, test_trajectories_coordinates = {}, {}
    for trajectory_id, trajectory_coordinates in trajectories_coordinates.items():
        frames, coordinates = collect_fn(trajectory_coordinates, trajectories_frames[trajectory_id],
                                         input_length=input_length, input_gap=input_gap, pred_length=pred_length)
        test_trajectories_frames[trajectory_id] = frames
        test_trajectories_coordinates[trajectory_id] = coordinates

    return test_trajectories_frames, test_trajectories_coordinates


def detect_anomalous_frames(reconstruction_errors, anomaly_threshold):
    anomalous_frames = {}
    for trajectory_id, reconstruction_error in reconstruction_errors.items():
        anomalous_frames[trajectory_id] = reconstruction_error > anomaly_threshold

    return anomalous_frames


def detect_most_anomalous_or_most_normal_frames(reconstruction_errors, anomalous=True, fraction=0.2):
    all_errors = [reconstruction_error for _, reconstruction_error in reconstruction_errors.items()]
    all_errors = np.concatenate(all_errors)
    all_errors_sorted = np.sort(all_errors)
    num_frames_to_blame = round(len(all_errors_sorted) * fraction)
    anomalous_or_normal_frames = {}
    if anomalous:
        threshold = np.round(all_errors_sorted[-num_frames_to_blame], decimals=8)
        for trajectory_id, reconstruction_error in reconstruction_errors.items():
            anomalous_or_normal_frames[trajectory_id] = reconstruction_error >= threshold
    else:
        threshold = np.round(all_errors_sorted[num_frames_to_blame - 1], decimals=8)
        for trajectory_id, reconstruction_error in reconstruction_errors.items():
            anomalous_or_normal_frames[trajectory_id] = (0 < reconstruction_error) & (reconstruction_error <= threshold)

    return anomalous_or_normal_frames


def uniquify_reconstruction(trajectory_frames, reconstructed_coordinates):
    """Summarise the reconstruction of repeated frames into a single one.


    Argument(s):
        trajectory_frames: A numpy array of shape (trajectory_length, input_length) containing the frames from which
            the skeletons in reconstructed_coordinates are from.
        reconstructed_coordinates: A numpy array of shape (trajectory_length, input_length, input_dim) containing
            the reconstructed skeletons.

    Return(s):
        A list containing two numpy arrays. The first array has shape (num_unique_frames,) and contains the ids of the
        frames reconstructed. The second array has shape (num_unique_frames, input_dim) and contains the mean reconstructed
        skeleton of each unique frame.
    """
    trajectory_length, input_length, input_dim = reconstructed_coordinates.shape
    reconstructed_coordinates_reshaped = reconstructed_coordinates.reshape(-1, input_dim)
    trajectory_frames = trajectory_frames.astype(np.int64)
    trajectory_frames_reshaped = trajectory_frames.reshape(-1)
    unique_frames = np.unique(trajectory_frames_reshaped)
    unique_reconstructed_coordinates = np.empty(shape=(len(unique_frames), input_dim), dtype=np.float32)
    for idx, frame_id in enumerate(unique_frames):
        selected_reconstructions = reconstructed_coordinates_reshaped[trajectory_frames_reshaped == frame_id, :]
        unique_reconstructed_coordinates[idx, :] = np.mean(selected_reconstructions, axis=0)

    return [unique_frames, unique_reconstructed_coordinates]


def uniquify_reconstructions(reconstructed_trajectories):
    """Summarise the reconstruction of repeated frames into a single one.

    Argument(s):
        reconstructed_trajectories: A dictionary where the keys uniquely identify each camera and the values are lists
            containing two items. For each list, both items are dictionaries where the keys uniquely identify each
            trajectory (video + person) and, for the first item, the values are numpy arrays of shape
            (trajectory_length, input_length) containing the frame id of each possible skeleton's trajectory, and,
            for the second item, the values are numpy arrays of shape (trajectory_length, input_length, input_dim)
            containing the reconstructed trajectories.

    Return(s):
        A dictionary where the keys uniquely identifies the cameras and the values are lists of two items. For each
        list, both items are dictionaries where the keys uniquely identify the trajectory (video + person) and, for
        the first item, the values are numpy arrays of shape (num_unique_frames,) containing the frame id of the
        skeleton's trajectory, and, for the second item, the values are numpy arrays of shape
        (num_unique_frames, input_dim) containing the mean reconstruction of the skeletons for each frame.
    """
    for camera_id, (trajectories_frames, trajectories_coordinates) in reconstructed_trajectories.items():
        for trajectory_id in trajectories_frames.keys():
            frames, reconstructed_coordinates = uniquify_reconstruction(trajectories_frames[trajectory_id],
                                                                        trajectories_coordinates[trajectory_id])
            trajectories_frames[trajectory_id] = frames
            trajectories_coordinates[trajectory_id] = reconstructed_coordinates
        reconstructed_trajectories[camera_id] = [trajectories_frames, trajectories_coordinates]

    return reconstructed_trajectories


def discard_steps_from_padded_frames(reconstructed_trajectories, original_lengths):
    """Discard steps from padded frames.

    Argument(s):
        reconstructed_trajectories: A dictionary where the keys uniquely identifies the cameras and the values are
            lists of two items. For each list, both items are dictionaries where the keys uniquely identify the
            trajectory (video + person) and, for the first item, the values are numpy arrays of shape
            (num_unique_frames,) containing the frame id of the skeleton's trajectory, and, for the second item,
            the values are numpy arrays of shape (num_unique_frames, input_dim) containing the mean reconstruction of
            the skeletons for each frame.
        original_lengths: A dictionary where the keys uniquely identify the cameras and the values are dictionaries
            where the keys identify the trajectory (video + person) and the values are integers containing the
            original length of the associated trajectory.

    Return(s):

    """
    for camera_id in reconstructed_trajectories.keys():
        trajectories_frames, trajectories_coordinates = reconstructed_trajectories[camera_id]
        for trajectory_id in trajectories_frames.keys():
            original_length = original_lengths[camera_id][trajectory_id]
            trajectories_frames[trajectory_id] = trajectories_frames[trajectory_id][:original_length]
            trajectories_coordinates[trajectory_id] = trajectories_coordinates[trajectory_id][:original_length]
        reconstructed_trajectories[camera_id] = [trajectories_frames, trajectories_coordinates]

    return reconstructed_trajectories


def denormalise_all_trajectories(reconstructed_trajectories, video_resolution):
    """Bring the trajectories back to the original range.

    Argument(s):
        reconstructed_trajectories: A dictionary where the keys uniquely identifies the cameras and the values are
            lists of two items. For each list, both items are dictionaries where the keys uniquely identify the
            trajectory (video + person) and, for the first item, the values are numpy arrays of shape
            (num_unique_frames,) containing the frame id of the skeleton's trajectory, and, for the second item,
            the values are numpy arrays of shape (num_unique_frames, input_dim) containing the mean reconstruction of
            the skeletons for each frame.
        video_resolution: A list containing the video width and video height.

    Return(s):

    """
    video_resolution = np.array(video_resolution, dtype=np.float32)
    for camera_id in reconstructed_trajectories.keys():
        trajectories_frames, trajectories_coordinates = reconstructed_trajectories[camera_id]
        for trajectory_id in trajectories_coordinates.keys():
            trajectory_coordinates = trajectories_coordinates[trajectory_id]
            trajectory_coordinates_normalised = trajectory_coordinates.reshape(-1, 2) * video_resolution
            trajectory_coordinates = trajectory_coordinates_normalised.reshape(*trajectory_coordinates.shape)
            trajectories_coordinates[trajectory_id] = trajectory_coordinates
        reconstructed_trajectories[camera_id] = [trajectories_frames, trajectories_coordinates]

    return reconstructed_trajectories


def write_all_reconstructed_trajectories(reconstructed_trajectories, write_path):
    """Write all reconstructions to disk.

    Argument(s):
        reconstructed_trajectories: A dictionary where the keys uniquely identifies the cameras and the values are
            lists of two items. For each list, both items are dictionaries where the keys uniquely identify the
            trajectory (video + person) and, for the first item, the values are numpy arrays of shape
            (num_unique_frames,) containing the frame id of the skeleton's trajectory, and, for the second item,
            the values are numpy arrays of shape (num_unique_frames, input_dim) containing the mean reconstruction of
            the skeletons for each frame.
        write_path: Path to directory to save reconstructed skeletons. If not existent, a sub-directory for each
            camera is created and, for each camera, a sub-sub-directory is created for each video.
    """
    for camera_id in reconstructed_trajectories.keys():
        camera_path = os.path.join(write_path, camera_id)
        if not os.path.exists(camera_path):
            os.makedirs(camera_path)
        trajectories_frames, trajectories_coordinates = reconstructed_trajectories[camera_id]
        for trajectory_id in trajectories_frames.keys():
            video_id, person_id = trajectory_id.split('_')
            video_path = os.path.join(camera_path, video_id)
            if not os.path.exists(video_path):
                os.makedirs(video_path)
            skeleton_file = os.path.join(video_path, person_id) + '.csv'
            trajectory_frames = trajectories_frames[trajectory_id].reshape(-1, 1)
            trajectory_coordinates = trajectories_coordinates[trajectory_id]
            trajectory = np.hstack((trajectory_frames, trajectory_coordinates))
            np.savetxt(skeleton_file, trajectory, fmt='%.4f', delimiter=',')

    return None


def extract_input_dim(trajectories_coordinates):
    random_trajectory_id = list(trajectories_coordinates.keys())[0]
    return trajectories_coordinates[random_trajectory_id].shape[1]


def reverse_trajectories(trajectories_coordinates):
    trajectories_coordinates = {trajectory_id: trajectory_coordinates[:, ::-1, :]
                                for trajectory_id, trajectory_coordinates in trajectories_coordinates.items()}

    return trajectories_coordinates


def collect_skeletons(trajectories_frames, trajectories_coordinates):
    trajectories_ids = set(trajectories_coordinates.keys())
    frames = [trajectories_frames[trajectory_id] for trajectory_id in trajectories_ids]
    frames = np.concatenate(frames)
    skeletons = [trajectories_coordinates[trajectory_id] for trajectory_id in trajectories_ids]
    skeletons = np.vstack(skeletons)
    non_missing_skeletons = np.sum(skeletons, axis=1) > 0.0
    frames = frames[non_missing_skeletons]
    skeletons = skeletons[non_missing_skeletons, :]

    return frames, skeletons


def remove_missing_skeletons(trajectories_frames, trajectories_coordinates):
    for trajectory_id in trajectories_frames.keys():
        frames = trajectories_frames[trajectory_id]
        skeletons = trajectories_coordinates[trajectory_id]
        non_missing_skeletons = np.sum(skeletons, axis=1) > 0.0

        trajectories_frames[trajectory_id] = frames[non_missing_skeletons]
        trajectories_coordinates[trajectory_id] = skeletons[non_missing_skeletons, :]

    return trajectories_frames, trajectories_coordinates


def extract_center_of_mass(trajectory_coordinates):
    trajectory_features = np.where(trajectory_coordinates == 0.0, np.nan, trajectory_coordinates)
    # Left shoulder [10, 11], right shoulder [12, 13], left hip [22, 23], right hip [24, 25]
    selected_joints_x = [10, 12, 22, 24]
    selected_joints_y = [11, 13, 23, 25]

    trajectory_features_x = trajectory_features[:, selected_joints_x]
    trajectory_features_y = trajectory_features[:, selected_joints_y]

    center_of_mass_x = np.nanmean(trajectory_features_x, axis=1, keepdims=True)
    center_of_mass_y = np.nanmean(trajectory_features_y, axis=1, keepdims=True)

    center_of_mass = np.hstack((center_of_mass_x, center_of_mass_y))
    center_of_mass = np.where(np.isnan(center_of_mass), 0.0, center_of_mass)

    return center_of_mass


def extract_centre_of_bounding_box(trajectory_coordinates, video_resolution):
    bounding_box = np.apply_along_axis(compute_bounding_box, axis=1, arr=trajectory_coordinates,
                                       video_resolution=video_resolution)
    x = (bounding_box[:, 0] + bounding_box[:, 1]).reshape(-1, 1).astype(np.float32) / 2
    y = (bounding_box[:, 2] + bounding_box[:, 3]).reshape(-1, 1).astype(np.float32) / 2

    return np.hstack((x, y))


def extract_width_height(trajectory_coordinates, video_resolution):
    bounding_box = np.apply_along_axis(compute_bounding_box, axis=1, arr=trajectory_coordinates,
                                       video_resolution=video_resolution)
    width = (bounding_box[:, 1] - bounding_box[:, 0]).reshape(-1, 1).astype(np.float32)
    height = (bounding_box[:, 3] - bounding_box[:, 2]).reshape(-1, 1).astype(np.float32)

    return np.hstack((width, height))


def extract_global_features_from_trajectory(trajectory_coordinates, video_resolution):
    center_of_bounding_box = extract_centre_of_bounding_box(trajectory_coordinates, video_resolution=video_resolution)
    width_height = extract_width_height(trajectory_coordinates, video_resolution=video_resolution)
    trajectory_features = np.hstack((center_of_bounding_box, width_height))

    return trajectory_features


def extract_global_features(trajectories_coordinates, video_resolution):
    trajectories_features = {trajectory_id: extract_global_features_from_trajectory(trajectory_coordinates,
                                                                                    video_resolution=video_resolution)
                             for trajectory_id, trajectory_coordinates in trajectories_coordinates.items()}

    return trajectories_features


def concatenate_features(global_features, local_features):
    features = {trajectory_id: np.concatenate((global_features[trajectory_id], local_features[trajectory_id]),
                                              axis=-1)
                for trajectory_id in global_features.keys()}
    
    return features


def local_to_global_coordinates(reconstructed_features, video_resolution):
    video_resolution = np.tile(video_resolution, reps=2)
    reconstructed_features_global = {}
    for camera_id, (trajectories_frames, trajectories_features) in reconstructed_features.items():
        trajectories_features_global = {}
        for trajectory_id in trajectories_frames.keys():
            trajectory_features = trajectories_features[trajectory_id]
            input_examples, input_length, input_dim = trajectory_features.shape
            global_features = trajectory_features.reshape(-1, input_dim)[:, :4] * video_resolution
            local_features = trajectory_features.reshape(-1, input_dim)[:, 4:]
            reps = local_features.shape[1] // 2
            global_trajectory = local_features * np.tile(global_features[:, -2:], reps=reps)
            # global_trajectory += np.tile(global_features[:, :2] - global_features[:, -2:] / 2, reps=reps)
            global_trajectory += np.tile(global_features[:, :2], reps=reps)
            trajectories_features_global[trajectory_id] = global_trajectory.reshape(input_examples,
                                                                                    input_length,
                                                                                    input_dim - 4)

        reconstructed_features_global[camera_id] = [deepcopy(trajectories_frames), trajectories_features_global]

    return reconstructed_features_global


def discard_global_features(reconstructed_features):
    reconstructed_features_local = {}
    for camera_id, (trajectories_frames, trajectories_features) in reconstructed_features.items():
        trajectories_features_local = {}
        for trajectory_id in trajectories_frames.keys():
            trajectory_features = trajectories_features[trajectory_id]
            trajectories_features_local[trajectory_id] = trajectory_features[:, :, 4:]

        reconstructed_features_local[camera_id] = trajectories_features_local

    return reconstructed_features_local


class StdScaler:
    def __init__(self, stds=3):
        self.stds = stds
        self.mu = None
        self.sigma = None

    def fit(self, X):
        self.mu = np.nanmean(X, axis=0, keepdims=True)
        self.sigma = np.nanstd(X, axis=0, keepdims=True)

    def transform(self, X):
        reps = [X.shape[0], 1]
        mu = np.tile(self.mu, reps=reps)
        sigma = np.tile(self.sigma, reps=reps)
        X = (X - (mu - self.stds * sigma)) / (2 * self.stds * sigma)

        return X

    def inverse_transform(self, X):
        reps = [X.shape[0], 1]
        mu = np.tile(self.mu, reps=reps)
        sigma = np.tile(self.sigma, reps=reps)
        X = X * (2 * self.stds * sigma) + (mu - self.stds * sigma)

        return X


def scale_trajectories_three_stds(trajectories_coordinates, scaler=None):
    """Scale features to within three standard deviations.

    Argument(s):
        trajectories_coordinates -- A dictionary of numpy arrays, each of shape (trajectory_length, input_dim).
        scaler - If None, a scaler is trained on the input data; otherwise, the scaler is used to scale the input data.

    Return(s):
        The input data scaled between three standard deviations range.
    """
    if scaler is None:
        trajectories = [trajectory_coordinates for trajectory_coordinates in trajectories_coordinates.values()]
        trajectories = np.vstack(trajectories)

        trajectories = np.where(trajectories == 0.0, np.nan, trajectories)

        scaler = StdScaler(stds=3)
        scaler.fit(trajectories)

    trajectories_scaled = {}
    for trajectory_id, trajectory_coordinates in trajectories_coordinates.items():
        trajectory_scaled = scaler.transform(np.where(trajectory_coordinates == 0.0, np.nan, trajectory_coordinates))
        trajectory_scaled = np.where(np.isnan(trajectory_scaled), 0.0, trajectory_scaled)
        trajectories_scaled[trajectory_id] = trajectory_scaled

    return trajectories_scaled, scaler


def scale_trajectories_zero_one(trajectories_coordinates, scaler=None):
    """Scale features to the 0-1 range.

    Argument(s):
        trajectories_coordinates -- A dictionary of numpy arrays, each of shape (trajectory_length, input_dim).
        scaler - If None, a scaler is trained on the input data; otherwise, the scaler is used to scale the input data.

    Return(s):
        The input data scaled to the 0-1 range.
    """
    if scaler is None:
        trajectories = [trajectory_coordinates for trajectory_coordinates in trajectories_coordinates.values()]
        trajectories = np.vstack(trajectories)

        trajectories = np.where(trajectories == 0.0, np.nan, trajectories)
        trajectories_min = np.nanmin(trajectories, axis=0, keepdims=True)
        trajectories_min = np.where(np.isnan(trajectories_min), 0.0, trajectories_min)
        trajectories_min = np.tile(trajectories_min, reps=[trajectories.shape[0], 1])

        eps = 0.001
        trajectories = np.where(np.isnan(trajectories), trajectories_min - eps, trajectories)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(trajectories)

    trajectories_scaled = {}
    for trajectory_id, trajectory_coordinates in trajectories_coordinates.items():
        num_examples = trajectory_coordinates.shape[0]
        trajectory_scaled = scaler.transform(np.where(trajectory_coordinates == 0.0,
                                                      np.tile(scaler.data_min_, reps=[num_examples, 1]),
                                                      trajectory_coordinates))
        trajectories_scaled[trajectory_id] = trajectory_scaled

    return trajectories_scaled, scaler


def scale_trajectories(trajectories_coordinates, scaler=None, strategy='zero_one'):
    if strategy == 'zero_one':
        trajectories_scaled, scaler = scale_trajectories_zero_one(trajectories_coordinates, scaler=scaler)
    elif strategy == 'three_stds':
        trajectories_scaled, scaler = scale_trajectories_three_stds(trajectories_coordinates, scaler=scaler)
    else:
        raise ValueError('Unknown strategy. Please select either zero_one or three_stds.')

    return trajectories_scaled, scaler


def inverse_scale_trajectories(reconstructed_trajectories, global_scaler, local_scaler):
    rescaled_trajectories = {}
    for trajectory_id, trajectory_coordinates in reconstructed_trajectories.items():
        num_trajectories, input_length, input_dim = trajectory_coordinates.shape
        trajectory_coordinates = trajectory_coordinates.reshape(-1, input_dim)
        trajectory_coordinates[:, :4] = global_scaler.inverse_transform(trajectory_coordinates[:, :4])
        trajectory_coordinates[:, 4:] = local_scaler.inverse_transform(trajectory_coordinates[:, 4:])
        trajectory_coordinates = trajectory_coordinates.reshape(num_trajectories, input_length, input_dim)
        rescaled_trajectories[trajectory_id] = trajectory_coordinates

    return rescaled_trajectories


def inverse_single_scale_trajectories(reconstructed_trajectories, scaler):
    rescaled_trajectories = {}
    for trajectory_id, trajectory_coordinates in reconstructed_trajectories.items():
        shape, input_dim = trajectory_coordinates.shape, trajectory_coordinates.shape[-1]
        trajectory_coordinates = trajectory_coordinates.reshape(-1, input_dim)
        trajectory_coordinates = scaler.inverse_transform(trajectory_coordinates)
        trajectory_coordinates = trajectory_coordinates.reshape(shape)
        rescaled_trajectories[trajectory_id] = trajectory_coordinates

    return rescaled_trajectories


def _train_test_split_through_time(trajectory_id, trajectory_coordinates, input_length, pred_length, train_ratio=0.8):
    test_ratio = 1 - train_ratio
    trajectory_length = trajectory_coordinates.shape[0]
    test_slice_length = int(trajectory_length * test_ratio)

    if test_slice_length < input_length + pred_length:
        return {}, {trajectory_id: trajectory_coordinates}

    start_idx = np.random.randint(0, trajectory_length)
    end_idx = start_idx + test_slice_length

    if start_idx < input_length + pred_length:
        train = {trajectory_id: trajectory_coordinates[test_slice_length:, :]}
        test = {trajectory_id: trajectory_coordinates[:test_slice_length, :]}
    elif end_idx <= trajectory_length - 1 - input_length - pred_length:
        train = {trajectory_id + '_1': trajectory_coordinates[:start_idx, :],
                 trajectory_id + '_2': trajectory_coordinates[end_idx:, :]}
        test = {trajectory_id: trajectory_coordinates[start_idx:end_idx, :]}
    else:
        train = {trajectory_id: trajectory_coordinates[:-test_slice_length, :]}
        test = {trajectory_id: trajectory_coordinates[-test_slice_length:, :]}

    return train, test


def train_test_split_through_time(trajectories_coordinates, input_length, pred_length, train_ratio=0.8, seed=42):
    """Bla Bla.

    Argument(s):
        trajectories_coordinates -- A dictionary of numpy arrays, each of shape (trajectory_length, input_dim).
        input_length --
        pred_length --
        train_ratio --

    Return(s):
    """
    np.random.seed(seed)
    trajectories_coordinates_train = {}
    trajectories_coordinates_test = {}
    for trajectory_id, trajectory_coordinates in trajectories_coordinates.items():
        train, test = _train_test_split_through_time(trajectory_id, trajectory_coordinates,
                                                     input_length, pred_length, train_ratio)
        trajectories_coordinates_train.update(train)
        trajectories_coordinates_test.update(test)

    return trajectories_coordinates_train, trajectories_coordinates_test


def train_test_split_trajectories(trajectories_frames, trajectories_coordinates, train_ratio=0.8, seed=42):
    np.random.seed(seed)
    trajectories_frames_train, trajectories_coordinates_train = {}, {}
    trajectories_frames_test, trajectories_coordinates_test = {}, {}
    for trajectory_id, trajectory_coordinates in trajectories_coordinates.items():
        trajectory_length = trajectory_coordinates.shape[0]
        if trajectory_length == 1:
            trajectories_frames_test[trajectory_id] = trajectories_frames[trajectory_id]
            trajectories_coordinates_test[trajectory_id] = trajectory_coordinates[:1, :]
            continue
        train_size = int(train_ratio * trajectory_length)
        indices = np.random.permutation(trajectory_length)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        trajectories_frames_train[trajectory_id] = trajectories_frames[trajectory_id][train_indices]
        trajectories_coordinates_train[trajectory_id] = trajectory_coordinates[train_indices, :]
        trajectories_frames_test[trajectory_id] = trajectories_frames[trajectory_id][test_indices]
        trajectories_coordinates_test[trajectory_id] = trajectory_coordinates[test_indices, :]

    return trajectories_frames_train, trajectories_frames_test, \
        trajectories_coordinates_train, trajectories_coordinates_test


def pull_global_features(all_reconstructed_features):
    for camera_id, (trajectories_frames, trajectories_features) in all_reconstructed_features.items():
        for trajectory_id, trajectory_features in trajectories_features.items():
            trajectories_features[trajectory_id] = trajectory_features[..., :4]
        all_reconstructed_features[camera_id] = [trajectories_frames, trajectories_features]

    return all_reconstructed_features


def compute_bounding_boxes_from_global_features(all_reconstructed_features):
    all_bounding_boxes = {}
    for camera_id, (trajectories_frames, trajectories_features) in all_reconstructed_features.items():
        bounding_boxes = {}
        for trajectory_id, trajectory_features in trajectories_features.items():
            x, y, w, h = np.hsplit(trajectory_features, indices_or_sections=4)
            left, right, top, bottom = x - w / 2, x + w / 2, y - h / 2, y + h / 2
            bounding_boxes[trajectory_id] = np.round(np.hstack((left, right, top, bottom)), decimals=2).astype(np.int32)
        all_bounding_boxes[camera_id] = [trajectories_frames, bounding_boxes]

    return all_bounding_boxes


def compute_bounding_boxes_from_image_features(all_reconstructed_trajectories, video_resolution):
    all_bounding_boxes = {}
    for camera_id, (trajectories_frames, trajectories_features) in all_reconstructed_trajectories.items():
        bounding_boxes = {}
        for trajectory_id, trajectory_features in trajectories_features.items():
            bounding_box = np.apply_along_axis(compute_bounding_box, axis=1, arr=trajectory_features,
                                               video_resolution=video_resolution)
            bounding_boxes[trajectory_id] = bounding_box
        all_bounding_boxes[camera_id] = [trajectories_frames, bounding_boxes]

    return all_bounding_boxes


def change_coordinate_system(trajectories_coordinates, video_resolution, coordinate_system='global', invert=False):
    if invert:
        if coordinate_system == 'global':
            trajectories_coordinates = from_global_to_image(trajectories_coordinates, video_resolution=video_resolution)
        else:
            raise ValueError('Unknown coordinate system. Only global is available for inversion.')
    else:
        if coordinate_system == 'global':
            trajectories_coordinates = from_image_to_global(trajectories_coordinates, video_resolution=video_resolution)
        elif coordinate_system == 'bounding_box_top_left':
            trajectories_coordinates = from_image_to_bounding_box(trajectories_coordinates,
                                                                  video_resolution=video_resolution,
                                                                  location='top_left')
        elif coordinate_system == 'bounding_box_centre':
            trajectories_coordinates = from_image_to_bounding_box(trajectories_coordinates,
                                                                  video_resolution=video_resolution,
                                                                  location='centre')
        else:
            raise ValueError('Unknown coordinate system. Please select one of: global, bounding_box_top_left, or '
                             'bounding_box_centre.')

    return trajectories_coordinates


def from_image_to_global(trajectories_coordinates, video_resolution):
    for trajectory_id, trajectory_coordinates in trajectories_coordinates.items():
        trajectory_shape = trajectory_coordinates.shape
        trajectory_coordinates = trajectory_coordinates.reshape(-1, 2) / video_resolution
        trajectories_coordinates[trajectory_id] = trajectory_coordinates.reshape(trajectory_shape)

    return trajectories_coordinates


def from_global_to_image(trajectories_coordinates, video_resolution):
    for trajectory_id, trajectory_coordinates in trajectories_coordinates.items():
        trajectory_shape = trajectory_coordinates.shape
        trajectory_coordinates = trajectory_coordinates.reshape(-1, 2) * video_resolution
        trajectories_coordinates[trajectory_id] = trajectory_coordinates.reshape(trajectory_shape)

    return trajectories_coordinates


def from_global_to_image_all_cameras(all_reconstructed_trajectories, video_resolution):
    for camera_id, (trajectories_frames, trajectories_features) in all_reconstructed_trajectories.items():
        all_reconstructed_trajectories[camera_id] = [trajectories_frames,
                                                     from_global_to_image(trajectories_features, video_resolution)]

    return all_reconstructed_trajectories


def from_image_to_bounding_box(trajectories_coordinates, video_resolution, location='top_left'):
    if location == 'top_left':
        fn = from_image_to_top_left_bounding_box
    elif location == 'centre':
        fn = from_image_to_centre_bounding_box
    else:
        raise ValueError('Unknown location for the bounding box. Please select either top_left or centre.')

    trajectories_coordinates = {trajectory_id: fn(trajectory_coordinates, video_resolution=video_resolution)
                                for trajectory_id, trajectory_coordinates in trajectories_coordinates.items()}

    return trajectories_coordinates


def from_image_to_top_left_bounding_box(trajectory_coordinates, video_resolution):
    for idx, coordinates in enumerate(trajectory_coordinates):
        if any(coordinates):
            left, right, top, bottom = compute_bounding_box(coordinates, video_resolution=video_resolution)
            xs, ys = np.hsplit(coordinates.reshape(-1, 2), indices_or_sections=2)
            xs, ys = np.where(xs == 0.0, float(left), xs), np.where(ys == 0.0, float(top), ys)
            xs, ys = (xs - left) / (right - left), (ys - top) / (bottom - top)
            coordinates = np.hstack((xs, ys)).ravel()

        trajectory_coordinates[idx] = coordinates

    return trajectory_coordinates


def from_image_to_centre_bounding_box(trajectory_coordinates, video_resolution):
    for idx, coordinates in enumerate(trajectory_coordinates):
        if any(coordinates):
            left, right, top, bottom = compute_bounding_box(coordinates, video_resolution=video_resolution)
            centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
            xs, ys = np.hsplit(coordinates.reshape(-1, 2), indices_or_sections=2)
            xs, ys = np.where(xs == 0.0, centre_x, xs) - centre_x, np.where(ys == 0.0, centre_y, ys) - centre_y
            left, right, top, bottom = left - centre_x, right - centre_x, top - centre_y, bottom - centre_y
            width, height = right - left, bottom - top
            xs, ys = np.where(xs != 0.0, xs / width, xs), np.where(ys != 0.0, ys / height, ys)
            coordinates = np.hstack((xs, ys)).ravel()

        trajectory_coordinates[idx] = coordinates

    return trajectory_coordinates


def compute_worst_mistakes(y_true, y_hat, video_ids, type='false_positives', top=10):
    # sorting_indices = np.argsort(y_hat)
    frames = generate_array_of_frames(video_ids)
    video_ids = np.array(video_ids)

    if type == 'false_positives':
        true_negatives = y_true == 0
        y_hat_ = y_hat[true_negatives]
        video_ids_ = video_ids[true_negatives]
        frames_ = frames[true_negatives]

        sorting_indices = np.argsort(y_hat_)

        indices = sorting_indices[-top:]
    elif type == 'false_negatives':
        true_positives = y_true == 1
        y_hat_ = y_hat[true_positives]
        video_ids_ = video_ids[true_positives]
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


def write_all_worst_mistakes(all_pretrained_models_path, worst_false_positives, worst_false_negatives):
    file_path = os.path.join(all_pretrained_models_path, 'mistakes.txt')
    camera_ids = sorted(worst_false_positives.keys())
    with open(file_path, mode='w') as file:
        for camera_id in camera_ids:
            print('\nCamera ID: %s' % camera_id, file=file)
            print('\nWorst False Positives:', file=file)
            video_ids, frames, scores = worst_false_positives[camera_id]
            for video_id, frame, score in zip(video_ids[::-1], frames[::-1], scores[::-1]):
                print('Video ID: %s\tFrame: %d\tRec. Error: %.4f' % (video_id, frame, score), file=file)

            print('\nWorst False Negatives:', file=file)
            video_ids, frames, scores = worst_false_negatives[camera_id]
            for video_id, frame, score in zip(video_ids, frames, scores):
                print('Video ID: %s\tFrame: %d\tRec. Error: %.4f' % (video_id, frame, score), file=file)

    return None
