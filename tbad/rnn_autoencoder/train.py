from copy import deepcopy
import os

import numpy as np
from sklearn.externals import joblib
from sklearn.utils import shuffle

from tbad.autoencoder.data import load_trajectories, extract_global_features, change_coordinate_system
from tbad.autoencoder.data import split_into_train_and_test, scale_trajectories, aggregate_autoencoder_data
from tbad.autoencoder.data import input_trajectories_missing_steps
from tbad.rnn_autoencoder.data import remove_short_trajectories, aggregate_rnn_autoencoder_data
from tbad.rnn_autoencoder.rnn import RNNEncoderDecoder
from tbad.utils import set_up_logging, resume_training_from_last_epoch


def train_rnn_ae(args):
    # General
    trajectories_path = args.trajectories  # e.g. .../11
    camera_id = os.path.basename(trajectories_path)
    video_resolution = [float(measurement) for measurement in args.video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)
    # Architecture
    model_type = args.model_type
    extract_delta = args.extract_delta
    use_first_step_as_reference = args.use_first_step_as_reference
    input_length = args.input_length
    input_gap = args.input_gap
    rec_length = args.rec_length
    pred_length = args.pred_length
    hidden_dims = args.hidden_dims
    output_activation = args.output_activation
    cell_type = args.cell_type
    disable_reconstruction_branch = args.disable_reconstruction_branch
    reconstruct_reverse = args.reconstruct_reverse
    conditional_reconstruction = args.conditional_reconstruction
    conditional_prediction = args.conditional_prediction
    # Training
    optimiser = args.optimiser
    learning_rate = args.learning_rate
    loss = args.loss
    l1_reg = args.l1_reg
    l2_reg = args.l2_reg
    epochs = args.epochs
    batch_size = args.batch_size
    input_missing_steps = args.input_missing_steps
    coordinate_system = args.coordinate_system
    normalisation_strategy = args.normalisation_strategy
    # Logging
    root_log_dir = args.root_log_dir
    resume_training = args.resume_training

    trajectories = load_trajectories(trajectories_path)
    print('\nLoaded %d trajectories.' % len(trajectories))

    trajectories = remove_short_trajectories(trajectories, input_length=input_length,
                                             input_gap=input_gap, pred_length=pred_length)
    print('\nRemoved short trajectories. Number of trajectories left: %d.' % len(trajectories))

    X_train, y_train, X_val, y_val, scaler = \
        produce_data_from_model_type(model_type, trajectories, video_resolution=video_resolution,
                                     input_length=input_length, input_gap=input_gap, pred_length=pred_length,
                                     normalisation_strategy=normalisation_strategy, coordinate_system=coordinate_system,
                                     input_missing_steps=input_missing_steps, extract_delta=extract_delta,
                                     use_first_step_as_reference=use_first_step_as_reference)
    if model_type == 'concatenate':
        loss = 'balanced_' + loss

    print('\nInstantiating anomaly model ...')
    input_dim = X_train.shape[-1]
    rnn_ae = RNNEncoderDecoder(input_length=input_length, input_dim=input_dim, reconstruction_length=rec_length,
                               prediction_length=pred_length, hidden_dims=hidden_dims,
                               output_activation=output_activation, cell_type=cell_type,
                               reconstruction_branch=disable_reconstruction_branch,
                               reconstruct_reverse=reconstruct_reverse,
                               conditional_reconstruction=conditional_reconstruction,
                               conditional_prediction=conditional_prediction,
                               optimiser=optimiser, learning_rate=learning_rate, loss=loss,
                               l1_reg=l1_reg, l2_reg=l2_reg)

    log_dir = set_up_logging(camera_id=camera_id, root_log_dir=root_log_dir, resume_training=resume_training)
    last_epoch = resume_training_from_last_epoch(model=rnn_ae, resume_training=resume_training)

    rnn_ae.train(X_train, y_train, epochs=epochs, initial_epoch=last_epoch, batch_size=batch_size,
                 val_data=(X_val, y_val), log_dir=log_dir)

    print('\nRNN Autoencoder anomaly model successfully trained.')
    if log_dir is not None:
        file_name = os.path.join(log_dir, 'scaler.pkl')
        joblib.dump(scaler, filename=file_name)
        print('log files were written to: %s' % log_dir)

    return rnn_ae, scaler


def produce_data_from_model_type(model_type, trajectories, video_resolution, input_length, input_gap, pred_length,
                                 normalisation_strategy, coordinate_system, input_missing_steps=False,
                                 extract_delta=False, use_first_step_as_reference=False):
    trajectories_train, trajectories_val = split_into_train_and_test(trajectories, train_ratio=0.8, seed=42)

    if input_missing_steps:
        trajectories_train = input_trajectories_missing_steps(trajectories_train)
        print('\nInputted missing steps of trajectories.')

    if model_type == 'concatenate':
        trajectories_global_train = extract_global_features(deepcopy(trajectories_train),
                                                            video_resolution=video_resolution,
                                                            extract_delta=extract_delta,
                                                            use_first_step_as_reference=use_first_step_as_reference)
        trajectories_global_train = change_coordinate_system(trajectories_global_train,
                                                             video_resolution=video_resolution,
                                                             coordinate_system='global', invert=False)

        trajectories_global_val = extract_global_features(deepcopy(trajectories_val),
                                                          video_resolution=video_resolution,
                                                          extract_delta=extract_delta,
                                                          use_first_step_as_reference=use_first_step_as_reference)
        trajectories_global_val = change_coordinate_system(trajectories_global_val, video_resolution=video_resolution,
                                                           coordinate_system='global', invert=False)

        trajectories_local_train, trajectories_local_val = trajectories_train, trajectories_val
        trajectories_local_train = change_coordinate_system(trajectories_local_train, video_resolution=video_resolution,
                                                            coordinate_system='bounding_box_centre', invert=False)
        trajectories_local_val = change_coordinate_system(trajectories_local_val, video_resolution=video_resolution,
                                                          coordinate_system='bounding_box_centre', invert=False)

        X_ref = np.concatenate((aggregate_autoencoder_data(trajectories_global_train),
                                aggregate_autoencoder_data(trajectories_local_train)), axis=-1)
        _, scaler = scale_trajectories(X_ref, strategy=normalisation_strategy)

        X_global_train, y_global_train = aggregate_rnn_autoencoder_data(trajectories_global_train,
                                                                        input_length=input_length, input_gap=input_gap,
                                                                        pred_length=pred_length)
        X_global_val, y_global_val = aggregate_rnn_autoencoder_data(trajectories_global_val, input_length=input_length,
                                                                    input_gap=input_gap, pred_length=pred_length)

        X_local_train, y_local_train = aggregate_rnn_autoencoder_data(trajectories_local_train,
                                                                      input_length=input_length, input_gap=input_gap,
                                                                      pred_length=pred_length)
        X_local_val, y_local_val = aggregate_rnn_autoencoder_data(trajectories_local_val, input_length=input_length,
                                                                  input_gap=input_gap, pred_length=pred_length)

        X_train = np.concatenate((X_global_train, X_local_train), axis=-1)
        X_val = np.concatenate((X_global_val, X_local_val), axis=-1)

        X_train, _ = scale_trajectories(X_train, scaler=scaler, strategy=normalisation_strategy)
        X_val, _ = scale_trajectories(X_val, scaler=scaler, strategy=normalisation_strategy)

        if y_global_train is not None:
            y_train = np.concatenate((y_global_train, y_local_train), axis=-1)
            y_val = np.concatenate((y_global_val, y_local_val), axis=-1)

            y_train, _ = scale_trajectories(y_train, scaler=scaler, strategy=normalisation_strategy)
            y_val, _ = scale_trajectories(y_val, scaler=scaler, strategy=normalisation_strategy)
        else:
            y_train = y_val = None

        if y_train is not None:
            X_train, y_train = shuffle(X_train, y_train, random_state=42)
        else:
            X_train = shuffle(X_train, random_state=42)
    else:
        if model_type == 'global':
            trajectories_train = extract_global_features(trajectories_train, video_resolution=video_resolution,
                                                         extract_delta=extract_delta,
                                                         use_first_step_as_reference=use_first_step_as_reference)
            trajectories_val = extract_global_features(trajectories_val, video_resolution=video_resolution,
                                                       extract_delta=extract_delta,
                                                       use_first_step_as_reference=use_first_step_as_reference)
            coordinate_system = 'global'
            print('\nExtracted global features from input trajectories. In addition, the coordinate system has been '
                  'set to global.')

        trajectories_train = change_coordinate_system(trajectories_train, video_resolution=video_resolution,
                                                      coordinate_system=coordinate_system, invert=False)
        trajectories_val = change_coordinate_system(trajectories_val, video_resolution=video_resolution,
                                                    coordinate_system=coordinate_system, invert=False)
        print('\nChanged coordinate system to %s.' % coordinate_system)

        _, scaler = scale_trajectories(aggregate_autoencoder_data(trajectories_train), strategy=normalisation_strategy)

        X_train, y_train = aggregate_rnn_autoencoder_data(trajectories_train, input_length=input_length,
                                                          input_gap=input_gap, pred_length=pred_length)
        if y_train is not None:
            X_train, y_train = shuffle(X_train, y_train, random_state=42)
        else:
            X_train, y_train = shuffle(X_train, random_state=42), None
        X_val, y_val = aggregate_rnn_autoencoder_data(trajectories_val, input_length=input_length,
                                                      input_gap=input_gap, pred_length=pred_length)

        X_train, _ = scale_trajectories(X_train, scaler=scaler, strategy=normalisation_strategy)
        X_val, _ = scale_trajectories(X_val, scaler=scaler, strategy=normalisation_strategy)
        if y_train is not None and y_val is not None:
            y_train, _ = scale_trajectories(y_train, scaler=scaler, strategy=normalisation_strategy)
            y_val, _ = scale_trajectories(y_val, scaler=scaler, strategy=normalisation_strategy)
        print('\nNormalised input features using the %s normalisation strategy.' % normalisation_strategy)

    return X_train, y_train, X_val, y_val, scaler
