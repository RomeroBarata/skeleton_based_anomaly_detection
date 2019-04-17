from copy import deepcopy
import os
import warnings

import numpy as np
from sklearn.externals import joblib
from sklearn.utils import shuffle

from tbad.data import load_trajectories, remove_short_trajectories, input_trajectories_missing_steps
from tbad.data import extract_input_dim, collect_trajectories, collect_skeletons
from tbad.data import extract_global_features, train_test_split_through_time, scale_trajectories
from tbad.data import train_test_split_trajectories, change_coordinate_system
from tbad.rnn_autoencoder.rnn import RNNEncoderDecoder
from tbad.autoencoder.autoencoder import Autoencoder
from tbad.combined_model.fusion import CombinedEncoderDecoder
from tbad.utils import set_up_logging, resume_training_from_last_epoch


warnings.simplefilter(action='ignore', category=FutureWarning)


def train_ae(args):
    # General
    trajectories_path = args.trajectories  # e.g. .../03
    camera_id = os.path.basename(trajectories_path)
    video_resolution = [float(measurement) for measurement in args.video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)
    # Architecture
    global_model = args.global_model
    hidden_dims = args.hidden_dims
    coordinate_system = args.coordinate_system
    normalisation_strategy = args.normalisation_strategy
    # Training
    optimiser = args.optimiser
    learning_rate = args.learning_rate
    loss = args.loss
    epochs = args.epochs
    batch_size = args.batch_size
    # Logging
    root_log_dir = args.root_log_dir
    resume_training = args.resume_training

    trajectories_frames, trajectories_coordinates = load_trajectories(trajectories_path)
    print('\nLoaded %d trajectories.' % len(trajectories_coordinates))

    if global_model:
        trajectories_coordinates = extract_global_features(trajectories_coordinates, video_resolution=video_resolution)
        coordinate_system = 'global'
        print('\nExtracted global features from input skeletons. In addition, the coordinate system has been set '
              'to global.')

    trajectories_coordinates = change_coordinate_system(trajectories_coordinates,
                                                        video_resolution=video_resolution,
                                                        coordinate_system=coordinate_system,
                                                        invert=False)
    print('\nChanged coordinate system to %s.' % coordinate_system)

    trajectories_frames_train, trajectories_frames_val, trajectories_coordinates_train, trajectories_coordinates_val = \
        train_test_split_trajectories(trajectories_frames, trajectories_coordinates, train_ratio=0.8, seed=42)

    trajectories_coordinates_train, scaler = scale_trajectories(trajectories_coordinates_train,
                                                                strategy=normalisation_strategy)
    trajectories_coordinates_val, _ = scale_trajectories(trajectories_coordinates_val, scaler=scaler,
                                                         strategy=normalisation_strategy)
    print('\nNormalised input features using the %s normalisation strategy.' % normalisation_strategy)

    input_dim = extract_input_dim(trajectories_coordinates_train)
    ae_model = Autoencoder(input_dim=input_dim, hidden_dims=hidden_dims, optimiser=optimiser,
                           learning_rate=learning_rate, loss=loss)

    log_dir = set_up_logging(camera_id=camera_id, root_log_dir=root_log_dir, resume_training=resume_training)
    last_epoch = resume_training_from_last_epoch(model=ae_model, resume_training=resume_training)

    _, X_train = collect_skeletons(trajectories_frames_train, trajectories_coordinates_train)
    _, X_val = collect_skeletons(trajectories_frames_val, trajectories_coordinates_val)

    X_train = shuffle(X_train, random_state=42)
    ae_model.train(X_train, X_train, epochs=epochs, initial_epoch=last_epoch, batch_size=batch_size,
                   val_data=(X_val, X_val), log_dir=log_dir)

    print('\nAutoencoder model successfully trained.')
    if log_dir is not None:
        joblib.dump(scaler, filename=os.path.join(log_dir, 'scaler.pkl'))
        print('log files were written to: %s' % log_dir)

    return None


def train_rnn_ae(args):
    # General
    trajectories_path = args.trajectories  # e.g. .../11
    camera_id = os.path.basename(args.trajectories_path)
    video_resolution = [float(measurement) for measurement in args.video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)
    # Architecture
    global_model = args.global_model
    input_length = args.input_length
    input_gap = args.input_gap
    pred_length = args.pred_length
    hidden_dims = args.hidden_dims
    cell_type = args.cell_type
    disable_reconstruction_branch = args.disable_reconstruction_branch
    reconstruct_reverse = args.reconstruct_reverse
    conditional_reconstruction = args.conditional_reconstruction
    conditional_prediction = args.conditional_prediction
    # Training
    optimiser = args.optimiser
    learning_rate = args.learning_rate
    loss = args.loss
    epochs = args.epochs
    batch_size = args.batch_size
    input_missing_steps = args.input_missing_steps
    coordinate_system = args.coordinate_system
    normalisation_strategy = args.normalisation_strategy
    # Logging
    root_log_dir = args.root_log_dir
    resume_training = args.resume_training

    # trajectories_coordinates is a dictionary where the keys uniquely identify each person in each video and the values
    # are float32 tensors. Each tensor represents the detected trajectory of a single person and has shape
    # (trajectory_length, input_dim). trajectory_length is the total number of frames for which the person was tracked
    # and each detection is composed of k key points (17 for now), each represented by a pair of (x, y) coordinates.
    _, trajectories_coordinates = load_trajectories(trajectories_path)
    print('\nLoaded %d trajectories.' % len(trajectories_coordinates))

    # Filter-out short trajectories
    trajectories_coordinates = remove_short_trajectories(trajectories_coordinates, input_length=input_length,
                                                         input_gap=input_gap, pred_length=pred_length)
    print('\nRemoved short trajectories. Number of trajectories left: %d.' % len(trajectories_coordinates))

    # Global model (optional)
    if global_model:
        trajectories_coordinates = extract_global_features(trajectories_coordinates, video_resolution=video_resolution)
        coordinate_system = 'global'
        print('\nExtracted global features from input trajectories. In addition, the coordinate system has been set '
              'to global.')

    # Change coordinate system
    trajectories_coordinates = change_coordinate_system(trajectories_coordinates, video_resolution=video_resolution,
                                                        coordinate_system=coordinate_system, invert=False)
    print('\nChanged coordinate system to %s.' % coordinate_system)

    # Split into training and validation sets
    trajectories_coordinates_train, trajectories_coordinates_val = \
        train_test_split_through_time(trajectories_coordinates, input_length=input_length, pred_length=pred_length,
                                      train_ratio=0.8)

    # Input missing steps (optional)
    if input_missing_steps:
        trajectories_coordinates_train = input_trajectories_missing_steps(trajectories_coordinates_train)

    # Normalise the data
    trajectories_coordinates_train, scaler = scale_trajectories(trajectories_coordinates_train,
                                                                strategy=normalisation_strategy)
    trajectories_coordinates_val, _ = scale_trajectories(trajectories_coordinates_val, scaler=scaler,
                                                         strategy=normalisation_strategy)
    print('\nInput features normalised using the %s normalisation strategy.' % normalisation_strategy)

    print('\nInstantiating anomaly model ...')
    input_dim = extract_input_dim(trajectories_coordinates)
    anomaly_model = RNNEncoderDecoder(input_length=input_length, input_dim=input_dim, prediction_length=pred_length,
                                      hidden_dims=hidden_dims, cell_type=cell_type,
                                      reconstruction_branch=disable_reconstruction_branch,
                                      reconstruct_reverse=reconstruct_reverse,
                                      conditional_reconstruction=conditional_reconstruction,
                                      conditional_prediction=conditional_prediction,
                                      optimiser=optimiser, learning_rate=learning_rate, loss=loss)

    # Set up training logging (optional)
    log_dir = set_up_logging(camera_id=camera_id, root_log_dir=root_log_dir, resume_training=resume_training)

    # Resume training (optional)
    last_epoch = resume_training_from_last_epoch(model=anomaly_model, resume_training=resume_training)

    # Train the anomaly model
    if pred_length > 0:
        X_train, y_train = list(zip(*[collect_trajectories(trajectory_coordinates, input_length, input_gap, pred_length)
                                      for trajectory_coordinates in trajectories_coordinates_train.values()]))
        X_train, y_train = np.vstack(X_train), np.vstack(y_train)
        print('\nTrain trajectories\' shape: (%d, %d, %d).' % (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        print('Train future trajectories\' shape: (%d, %d, %d).' % (y_train.shape[0], y_train.shape[1],
                                                                    y_train.shape[2]))

        X_val, y_val = list(zip(*[collect_trajectories(trajectory_coordinates, input_length, input_gap, pred_length)
                                  for trajectory_coordinates in trajectories_coordinates_val.values()]))
        X_val, y_val = np.vstack(X_val), np.vstack(y_val)
        print('\nVal trajectories\' shape: (%d, %d, %d).' % (X_val.shape[0], X_val.shape[1], X_val.shape[2]))
        print('Val future trajectories\' shape: (%d, %d, %d).' % (y_val.shape[0], y_val.shape[1], y_val.shape[2]))

        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        anomaly_model.train(X_train, y_train, epochs=epochs, initial_epoch=last_epoch,
                            batch_size=batch_size, val_data=(X_val, y_val), log_dir=log_dir)
    else:
        X_train = [collect_trajectories(trajectory_coordinates, input_length, input_gap, pred_length)
                   for trajectory_coordinates in trajectories_coordinates_train.values()]
        X_train = np.vstack(X_train)
        print('\nTrain trajectories\' shape: (%d, %d, %d).' % (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

        X_val = [collect_trajectories(trajectory_coordinates, input_length, input_gap, pred_length)
                 for trajectory_coordinates in trajectories_coordinates_val.values()]
        X_val = np.vstack(X_val)
        print('\nVal trajectories\' shape: (%d, %d, %d).' % (X_val.shape[0], X_val.shape[1], X_val.shape[2]))

        X_train = shuffle(X_train, random_state=42)
        anomaly_model.train(X_train, epochs=epochs, initial_epoch=last_epoch, batch_size=batch_size,
                            val_data=(X_val,), log_dir=log_dir)

    print('\nRNN Autoencoder model successfully trained.')
    if log_dir is not None:
        joblib.dump(scaler, filename=os.path.join(log_dir, 'scaler.pkl'))
        print('log files were written to: %s' % log_dir)

    return None


def train_complete_rnn_ae(args):
    # General
    trajectories_path = args.trajectories
    camera_id = os.path.basename(trajectories_path)
    video_resolution = [float(measurement) for measurement in args.video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)
    # Architecture
    input_length = args.input_length
    pred_length = args.pred_length
    global_hidden_dims = args.global_hidden_dims
    local_hidden_dims = args.local_hidden_dims
    extra_hidden_dims = args.extra_hidden_dims
    cell_type = args.cell_type
    reconstruct_reverse = args.reconstruct_reverse
    # Training
    optimiser = args.optimiser
    learning_rate = args.learning_rate
    loss = args.loss
    epochs = args.epochs
    batch_size = args.batch_size
    global_normalisation_strategy = args.global_normalisation_strategy
    local_normalisation_strategy = args.local_normalisation_strategy
    # Logging
    root_log_dir = args.root_log_dir
    resume_training = args.resume_training

    _, trajectories_coordinates = load_trajectories(trajectories_path)
    print('\nLoaded %d trajectories.' % len(trajectories_coordinates))

    trajectories_coordinates = remove_short_trajectories(trajectories_coordinates, input_length=input_length, 
                                                         input_gap=0, pred_length=pred_length)
    print('\nRemoved short trajectories. Number of trajectories left: %d.' % len(trajectories_coordinates))

    trajectories_coordinates_train, trajectories_coordinates_val = \
        train_test_split_through_time(trajectories_coordinates, input_length=input_length, pred_length=pred_length,
                                      train_ratio=0.8)

    # Global pre-processing
    global_features_train = extract_global_features(trajectories_coordinates_train, video_resolution=video_resolution)
    global_features_val = extract_global_features(trajectories_coordinates_val, video_resolution=video_resolution)

    global_features_train = change_coordinate_system(global_features_train, video_resolution=video_resolution,
                                                     coordinate_system='global', invert=False)
    global_features_val = change_coordinate_system(global_features_val, video_resolution=video_resolution,
                                                   coordinate_system='global', invert=False)

    global_features_train, global_scaler = scale_trajectories(global_features_train,
                                                              strategy=global_normalisation_strategy)
    global_features_val, _ = scale_trajectories(global_features_val, scaler=global_scaler,
                                                strategy=global_normalisation_strategy)

    # Local pre-processing
    local_features_train = deepcopy(trajectories_coordinates_train)
    local_features_val = deepcopy(trajectories_coordinates_val)

    local_features_train = change_coordinate_system(local_features_train, video_resolution=video_resolution,
                                                    coordinate_system='bounding_box_centre', invert=False)
    local_features_val = change_coordinate_system(local_features_val, video_resolution=video_resolution,
                                                  coordinate_system='bounding_box_centre', invert=False)

    local_features_train, local_scaler = scale_trajectories(local_features_train, strategy=local_normalisation_strategy)
    local_features_val, _ = scale_trajectories(local_features_val, scaler=local_scaler,
                                               strategy=local_normalisation_strategy)

    # # Output
    # out_train = trajectories_coordinates_train
    # out_val = trajectories_coordinates_val
    #
    # out_train = change_coordinate_system(out_train,
    #                                      video_resolution=video_resolution,
    #                                      coordinate_system='global', invert=False)
    # out_val = change_coordinate_system(out_val,
    #                                    video_resolution=video_resolution,
    #                                    coordinate_system='global', invert=False)

    # Anomaly Model
    global_input_dim = extract_input_dim(global_features_train)
    local_input_dim = extract_input_dim(local_features_train)
    anomaly_model = CombinedEncoderDecoder(input_length=input_length, global_input_dim=global_input_dim,
                                           local_input_dim=local_input_dim, prediction_length=pred_length,
                                           global_hidden_dims=global_hidden_dims, local_hidden_dims=local_hidden_dims,
                                           extra_hidden_dims=extra_hidden_dims, cell_type=cell_type,
                                           reconstruct_reverse=reconstruct_reverse, optimiser=optimiser,
                                           learning_rate=learning_rate, loss=loss)

    # Set up training logging (optional)
    log_dir = set_up_logging(camera_id=camera_id, root_log_dir=root_log_dir, resume_training=resume_training)

    # Resume training (optional)
    last_epoch = resume_training_from_last_epoch(model=anomaly_model, resume_training=resume_training)

    if pred_length > 0:
        # Global
        X_global_train, y_global_train = list(zip(*[collect_trajectories(global_trajectory,
                                                                         input_length, 0, pred_length)
                                                    for global_trajectory in global_features_train.values()]))
        X_global_train, y_global_train = np.vstack(X_global_train), np.vstack(y_global_train)

        X_global_val, y_global_val = list(zip(*[collect_trajectories(global_trajectory,
                                                                     input_length, 0, pred_length)
                                                for global_trajectory in global_features_val.values()]))
        X_global_val, y_global_val = np.vstack(X_global_val), np.vstack(y_global_val)

        # Local
        X_local_train, y_local_train = list(zip(*[collect_trajectories(local_trajectory, input_length, 0, pred_length)
                                                  for local_trajectory in local_features_train.values()]))
        X_local_train, y_local_train = np.vstack(X_local_train), np.vstack(y_local_train)

        X_local_val, y_local_val = list(zip(*[collect_trajectories(local_trajectory, input_length, 0, pred_length)
                                              for local_trajectory in local_features_val.values()]))
        X_local_val, y_local_val = np.vstack(X_local_val), np.vstack(y_local_val)

        # # Output
        # X_out_train, y_out_train = list(zip(*[collect_trajectories(out_trajectory,
        #                                                            input_length, 0, pred_length)
        #                                       for out_trajectory in out_train.values()]))
        # X_out_train, y_out_train = np.vstack(X_out_train), np.vstack(y_out_train)
        #
        # X_out_val, y_out_val = list(zip(*[collect_trajectories(out_trajectory,
        #                                                        input_length, 0, pred_length)
        #                                   for out_trajectory in out_val.values()]))
        # X_out_val, y_out_val = np.vstack(X_out_val), np.vstack(y_out_val)
        #
        # X_global_train, X_local_train, X_out_train, y_global_train, y_local_train, y_out_train = \
        #     shuffle(X_global_train, X_local_train, X_out_train, y_global_train, y_local_train, y_out_train,
        #             random_state=42)
        X_global_train, X_local_train, y_global_train, y_local_train = \
            shuffle(X_global_train, X_local_train, y_global_train, y_local_train, random_state=42)

        # X_train = [X_global_train, X_local_train, X_out_train]
        # y_train = [y_global_train, y_local_train, y_out_train]
        # val_data = ([X_global_val, X_local_val, X_out_val], [y_global_val, y_local_val, y_out_val])
        X_train = [X_global_train, X_local_train]
        y_train = [y_global_train, y_local_train]
        val_data = ([X_global_val, X_local_val], [y_global_val, y_local_val])
        anomaly_model.train(X_train, y_train, epochs=epochs, initial_epoch=last_epoch, batch_size=batch_size,
                            val_data=val_data, log_dir=log_dir)
    else:
        # Global
        X_global_train = [collect_trajectories(global_trajectory, input_length, 0, pred_length)
                          for global_trajectory in global_features_train.values()]
        X_global_train = np.vstack(X_global_train)

        X_global_val = [collect_trajectories(global_trajectory, input_length, 0, pred_length)
                        for global_trajectory in global_features_val.values()]
        X_global_val = np.vstack(X_global_val)

        # Local
        X_local_train = [collect_trajectories(local_trajectory, input_length, 0, pred_length)
                         for local_trajectory in local_features_train.values()]
        X_local_train = np.vstack(X_local_train)

        X_local_val = [collect_trajectories(local_trajectory, input_length, 0, pred_length)
                       for local_trajectory in local_features_val.values()]
        X_local_val = np.vstack(X_local_val)

        # # Output
        # X_out_train = [collect_trajectories(out_trajectory, input_length, 0, pred_length)
        #                for out_trajectory in out_train.values()]
        # X_out_train = np.vstack(X_out_train)
        #
        # X_out_val = [collect_trajectories(out_trajectory, input_length, 0, pred_length)
        #              for out_trajectory in out_val.values()]
        # X_out_val = np.vstack(X_out_val)

        # X_global_train, X_local_train, X_out_train = shuffle(X_global_train, X_local_train, X_out_train,
        #                                                      random_state=42)
        X_global_train, X_local_train = shuffle(X_global_train, X_local_train, random_state=42)
        # X_train = [X_global_train, X_local_train, X_out_train]
        # val_data = ([X_global_val, X_local_val, X_out_val],)
        X_train = [X_global_train, X_local_train]
        val_data = ([X_global_val, X_local_val],)
        anomaly_model.train(X_train, epochs=epochs, initial_epoch=last_epoch, batch_size=batch_size,
                            val_data=val_data, log_dir=log_dir)

    print('Combined global and local anomaly model successfully trained.')
    if log_dir is not None:
        joblib.dump(global_scaler, filename=os.path.join(log_dir, 'global_scaler.pkl'))
        joblib.dump(local_scaler, filename=os.path.join(log_dir, 'local_scaler.pkl'))
        print('log files were written to: %s' % log_dir)

    return None
