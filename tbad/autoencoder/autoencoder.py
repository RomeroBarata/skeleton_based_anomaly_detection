import os

import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.externals import joblib

from tbad.rnn_autoencoder.rnn import load_architecture_specification
from tbad.utils import select_optimiser, select_loss
from tbad.losses import binary_crossentropy, mean_absolute_error, mean_squared_error


class Autoencoder:
    def __init__(self, input_dim, hidden_dims=(16,), output_activation='sigmoid', optimiser='adam',
                 learning_rate=0.001, loss='mse'):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_activation = output_activation
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.loss = loss

        self.model = self.build()

    def build(self):
        ae_input = Input(shape=(self.input_dim,), name='ae_input', dtype='float32')
        encoded = Dense(self.hidden_dims[0], activation='relu')(ae_input)
        for hidden_dim in self.hidden_dims[1:]:
            encoded = Dense(hidden_dim, activation='relu')(encoded)
        decoded = Dense(self.input_dim, activation=self.output_activation)(encoded)

        return Model(inputs=ae_input, outputs=decoded)

    def compile(self):
        self.model.compile(optimizer=select_optimiser(self.optimiser, self.learning_rate),
                           loss=select_loss(self.loss))

    def train(self, X_train, y_train, epochs=5, initial_epoch=0, batch_size=256, val_data=None, log_dir=None):
        self.compile()

        callbacks_list = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        ]

        if log_dir is not None:
            callbacks_list += [
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(log_dir, 'weights_{epoch:03d}_{val_loss:.2f}.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True
                ),
                keras.callbacks.CSVLogger(
                    filename=os.path.join(log_dir, 'training_report.csv'),
                    append=True
                )
            ]

            self._maybe_write_architecture(log_dir)

        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list,
                       validation_data=val_data, shuffle=True, initial_epoch=initial_epoch)

        return None

    def predict(self, X_test):
        return self.model.predict(X_test, batch_size=256)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)
        return None

    def _maybe_write_architecture(self, log_dir):
        file_path = os.path.join(log_dir, 'architecture.txt')
        if os.path.isfile(file_path):
            return None

        with open(file_path, mode='w') as file:
            print('input_dim', self.input_dim, file=file)
            print('hidden_dims', *self.hidden_dims, file=file)
            print('output_activation', self.output_activation, file=file)
            print('optimiser', self.optimiser, file=file)
            print('learning_rate', self.learning_rate, file=file)
            print('loss', self.loss, file=file)

        return None


def load_ae_pretrained_models(all_pretrained_models_path):
    pretrained_models, scalers = {}, {}
    for pretrained_model_name in os.listdir(all_pretrained_models_path):
        if pretrained_model_name.endswith('.npy') or pretrained_model_name.endswith('.npz'):
            continue
        camera_id = pretrained_model_name.split('_')[0]
        pretrained_model_path = os.path.join(all_pretrained_models_path, pretrained_model_name)
        pretrained_models[camera_id], scalers[camera_id] = load_pretrained_ae(pretrained_model_path)

    return pretrained_models, scalers


def load_pretrained_ae(pretrained_model_path):
    model_files = os.listdir(pretrained_model_path)
    architecture_file = model_files[model_files.index('architecture.txt')]
    weight_files = [file_name for file_name in model_files if file_name.startswith('weights')]
    scaler_file = model_files[model_files.index('scaler.pkl')]
    best_weights = sorted(weight_files)[-1]

    architecture_path = os.path.join(pretrained_model_path, architecture_file)
    best_weights_path = os.path.join(pretrained_model_path, best_weights)

    architecture_specification = load_architecture_specification(architecture_path)
    ae_model = Autoencoder(**architecture_specification)
    ae_model.compile()
    ae_model.load_weights(best_weights_path)
    scaler = joblib.load(filename=os.path.join(pretrained_model_path, scaler_file))

    return ae_model, scaler


def _compute_ae_reconstruction_errors(X, X_reconstructed, loss):
    loss_fn = {'log_loss': binary_crossentropy, 'mae': mean_absolute_error, 'mse': mean_squared_error}[loss]
    return loss_fn(X, X_reconstructed)


def compute_ae_reconstruction_errors(coordinates, coordinates_reconstructed, loss):
    errors = {trajectory_id: _compute_ae_reconstruction_errors(coordinates[trajectory_id],
                                                               rec_trajectory_coordinates, loss=loss)
              for trajectory_id, rec_trajectory_coordinates in coordinates_reconstructed.items()}

    return errors


def reconstruct_skeletons(anomaly_model, trajectories_coordinates):
    reconstructed_coordinates = {trajectory_id: anomaly_model.predict(trajectory_coordinates)
                                 for trajectory_id, trajectory_coordinates in trajectories_coordinates.items()}

    return reconstructed_coordinates
