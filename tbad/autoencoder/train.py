import os

import numpy as np
from sklearn.externals import joblib
from sklearn.utils import shuffle

from tbad.autoencoder.autoencoder import Autoencoder
from tbad.autoencoder.data import load_trajectories, extract_global_features, change_coordinate_system
from tbad.autoencoder.data import split_into_train_and_test, aggregate_autoencoder_data, scale_trajectories
from tbad.utils import set_up_logging, resume_training_from_last_epoch


def train_ae(args):
    # General
    trajectories_path = args.trajectories  # e.g. .../03
    camera_id = os.path.basename(trajectories_path)
    video_resolution = [float(measurement) for measurement in args.video_resolution.split('x')]
    video_resolution = np.array(video_resolution, dtype=np.float32)
    # Architecture
    global_model = args.global_model
    hidden_dims = args.hidden_dims
    output_activation = args.output_activation
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

    trajectories = load_trajectories(trajectories_path)
    print('\nLoaded %d trajectories.' % len(trajectories))

    if global_model:
        trajectories = extract_global_features(trajectories, video_resolution=video_resolution)
        coordinate_system = 'global'
        print('\nExtracted global features from input skeletons. In addition, the coordinate system has been set '
              'to global.')

    trajectories = change_coordinate_system(trajectories, video_resolution=video_resolution,
                                            coordinate_system=coordinate_system, invert=False)
    print('\nChanged coordinate system to %s.' % coordinate_system)

    trajectories_train, trajectories_val = split_into_train_and_test(trajectories, train_ratio=0.8, seed=42)

    X_train = shuffle(aggregate_autoencoder_data(trajectories_train), random_state=42)
    X_val = aggregate_autoencoder_data(trajectories_val)

    X_train, scaler = scale_trajectories(X_train, strategy=normalisation_strategy)
    X_val, _ = scale_trajectories(X_val, scaler=scaler, strategy=normalisation_strategy)
    print('\nNormalised input features using the %s normalisation strategy.' % normalisation_strategy)

    input_dim = X_train.shape[-1]
    ae = Autoencoder(input_dim=input_dim, hidden_dims=hidden_dims, output_activation=output_activation,
                     optimiser=optimiser, learning_rate=learning_rate, loss=loss)

    log_dir = set_up_logging(camera_id=camera_id, root_log_dir=root_log_dir, resume_training=resume_training)
    last_epoch = resume_training_from_last_epoch(model=ae, resume_training=resume_training)

    ae.train(X_train, X_train, epochs=epochs, initial_epoch=last_epoch,
             batch_size=batch_size, val_data=(X_val, X_val), log_dir=log_dir)
    print('Autoencoder anomaly model successfully trained.')

    if log_dir is not None:
        file_name = os.path.join(log_dir, 'scaler.pkl')
        joblib.dump(scaler, filename=file_name)
        print('log files were written to: %s' % log_dir)

    return ae, scaler
