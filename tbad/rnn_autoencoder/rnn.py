import os

import keras
from keras.models import Model
from keras.layers import Input, RNN, Dense, Lambda
import numpy as np
from sklearn.externals import joblib

from tbad.utils import select_optimiser, select_loss, select_cell


class RNNEncoderDecoder:
    def __init__(self, input_length, input_dim, reconstruction_length, input_gap=0, prediction_length=0,
                 hidden_dims=(16,), output_activation='sigmoid', cell_type='lstm', reconstruction_branch=True,
                 reconstruct_reverse=True, conditional_reconstruction=False, conditional_prediction=False,
                 optimiser='rmsprop', learning_rate=0.001, loss='mse', l1_reg=0.0, l2_reg=0.0):
        self.input_length = input_length
        self.input_dim = input_dim
        self.input_gap = input_gap
        self.reconstruction_length = reconstruction_length
        self.prediction_length = prediction_length
        self.hidden_dims = hidden_dims
        self.output_activation = output_activation
        self.cell_type = cell_type
        self.reconstruction_branch = reconstruction_branch
        self.reconstruct_reverse = reconstruct_reverse
        self.conditional_reconstruction = conditional_reconstruction
        self.conditional_prediction = conditional_prediction
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.loss = loss
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        self.model = self.build()

    def build(self):
        all_inputs, all_outputs = [], []

        enc_input = Input(shape=(self.input_length, self.input_dim), name='enc_input', dtype='float32')
        all_inputs.append(enc_input)

        enc_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                     for hidden_dim in self.hidden_dims]
        enc_rnn = RNN(enc_cells, return_state=True, name='enc_rnn')
        _, *enc_states = enc_rnn(enc_input)

        if not (self.reconstruction_branch or self.prediction_length > 0):
            raise ValueError('At least one of reconstruction_branch or prediction_branch must be True.')

        if self.reconstruction_branch:
            rec_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                         for hidden_dim in self.hidden_dims]
            rec_rnn = RNN(rec_cells, return_sequences=True, return_state=True, name='rec_rnn')
            rec_dense = Dense(self.input_dim, activation=self.output_activation, name='rec_dense')

            if self.conditional_reconstruction:
                rec_input = Input(shape=(1, self.input_dim), name='rec_input', dtype='float32')
                rec_outputs = []
                inputs = rec_input
                states = enc_states
                for _ in range(self.reconstruction_length):
                    rec_output, *states = rec_rnn(inputs, initial_state=states)
                    rec_output = rec_dense(rec_output)
                    rec_outputs.append(rec_output)
                    inputs = rec_output
                rec_output = Lambda(lambda x: keras.backend.concatenate(x, axis=1))(rec_outputs)
            else:
                rec_input = Input(shape=(self.reconstruction_length, self.input_dim), name='rec_input', dtype='float32')
                rec_output, *_ = rec_rnn(rec_input, initial_state=enc_states)
                rec_output = rec_dense(rec_output)

            all_inputs.append(rec_input)
            all_outputs.append(rec_output)

        if self.prediction_length > 0:
            pred_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                          for hidden_dim in self.hidden_dims]
            pred_rnn = RNN(pred_cells, return_sequences=True, return_state=True, name='pred_rnn')
            pred_dense = Dense(self.input_dim, activation=self.output_activation, name='pred_dense')

            if self.conditional_prediction:
                pred_input = Input(shape=(1, self.input_dim), name='pred_input', dtype='float32')
                pred_outputs = []
                inputs = pred_input
                states = enc_states
                for _ in range(self.prediction_length):
                    pred_output, *states = pred_rnn(inputs, initial_state=states)
                    pred_output = pred_dense(pred_output)
                    pred_outputs.append(pred_output)
                    inputs = pred_output
                pred_output = Lambda(lambda x: keras.backend.concatenate(x, axis=1))(pred_outputs)
            else:
                pred_input = Input(shape=(self.prediction_length, self.input_dim), name='pred_input', dtype='float32')
                pred_output, *_ = pred_rnn(pred_input, initial_state=enc_states)
                pred_output = pred_dense(pred_output)

            all_inputs.append(pred_input)
            all_outputs.append(pred_output)

        return Model(inputs=all_inputs, outputs=all_outputs)

    def compile(self):
        self.model.compile(optimizer=select_optimiser(self.optimiser, self.learning_rate),
                           loss=select_loss(self.loss))

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def _create_reconstruction_input_zeros(self, n_examples):
        if self.conditional_reconstruction:
            return np.zeros((n_examples, 1, self.input_dim))
        else:
            return np.zeros((n_examples, self.reconstruction_length, self.input_dim))

    def _create_prediction_input_zeros(self, n_examples):
        if self.conditional_prediction:
            return np.zeros((n_examples, 1, self.input_dim))
        else:
            return np.zeros((n_examples, self.prediction_length, self.input_dim))

    def _construct_input_data(self, X):
        X_input = [X]

        if self.reconstruction_branch:
            X_input.append(self._create_reconstruction_input_zeros(X.shape[0]))

        if self.prediction_length > 0:
            X_input.append(self._create_prediction_input_zeros(X.shape[0]))

        return X_input

    def _construct_output_data(self, X, y=None):
        y_output = []

        if self.reconstruction_branch:
            if self.reconstruct_reverse:
                y_output.append(X[:, (self.reconstruction_length - 1)::-1, :])
            else:
                y_output.append(X[:, :self.reconstruction_length, :])

        if self.prediction_length > 0:
            y_output.append(y)

        return y_output

    def train(self, X_train, y_train=None, epochs=10, initial_epoch=0, batch_size=64, val_data=None, log_dir=None):
        self.compile()

        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3
            )
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

        X = self._construct_input_data(X_train)
        y = self._construct_output_data(X_train, y_train)
        X_val = self._construct_input_data(val_data[0])
        y_val = self._construct_output_data(*val_data)
        self.model.fit(X, y,
                       batch_size=batch_size, epochs=epochs,
                       callbacks=callbacks_list, validation_data=(X_val, y_val), initial_epoch=initial_epoch)

    def predict(self, X_test, batch_size=256):
        X = self._construct_input_data(X_test)
        return self.model.predict(X, batch_size=batch_size)

    def eval(self, X_test, batch_size=256):
        X = self._construct_input_data(X_test)
        y = self._construct_output_data(X_test)
        return self.model.evaluate(X, y, batch_size=batch_size)

    def _maybe_write_architecture(self, log_dir):
        file_path = os.path.join(log_dir, 'architecture.txt')
        if os.path.isfile(file_path):
            return None

        with open(file_path, mode='w') as file:
            print('input_length', self.input_length, file=file)
            print('input_dim', self.input_dim, file=file)
            print('reconstruction_length', self.reconstruction_length, file=file)
            print('input_gap', self.input_gap, file=file)
            print('prediction_length', self.prediction_length, file=file)
            print('reconstruct_reverse', self.reconstruct_reverse, file=file)
            print('cell_type', self.cell_type, file=file)
            print('hidden_dims', *self.hidden_dims, file=file)
            print('output_activation', self.output_activation, file=file)
            print('reconstruction_branch', self.reconstruction_branch, file=file)
            print('conditional_reconstruction', self.conditional_reconstruction, file=file)
            print('conditional_prediction', self.conditional_prediction, file=file)
            print('optimiser', self.optimiser, file=file)
            print('learning_rate', self.learning_rate, file=file)
            print('loss', self.loss, file=file)
            print('l1_reg', self.l1_reg, file=file)
            print('l2_reg', self.l2_reg, file=file)


def load_architecture_specification(model_architecture):
    model_spec = {}
    with open(model_architecture, mode='r') as file:
        for line in file:
            key, *value = line.split()

            if key in ('input_length', 'input_dim', 'input_gap', 'prediction_length', 'global_input_dim',
                       'local_input_dim', 'reconstruction_length'):
                value = int(value[0])
            elif key in ('learning_rate', 'l1_reg', 'l2_reg'):
                value = float(value[0])
            elif key in ('hidden_dims', 'input_dims', 'global_hidden_dims', 'local_hidden_dims', 'extra_hidden_dims'):
                value = tuple(map(int, value))
            elif key in ('reconstruction_branch', 'reconstruct_reverse', 'reconstruct_original_data',
                         'conditional_reconstruction', 'conditional_prediction', 'multiple_outputs',
                         'multiple_outputs_before_concatenation'):
                value = value[0]
                if value == 'True':
                    value = True
                else:
                    value = False
            else:
                value = value[0]

            model_spec[key] = value

    return model_spec


def model_from_architecture_specification(architecture_specification):
    return RNNEncoderDecoder(**architecture_specification)


def reconstruct_trajectories(anomaly_model, trajectories_coordinates):
    if anomaly_model.prediction_length > 0:
        reconstructed_trajectories = {trajectory_id: anomaly_model.predict(trajectory_coordinates, batch_size=256)[0]
                                      for trajectory_id, trajectory_coordinates in trajectories_coordinates.items()}
    else:
        reconstructed_trajectories = {trajectory_id: anomaly_model.predict(trajectory_coordinates, batch_size=256)
                                      for trajectory_id, trajectory_coordinates in trajectories_coordinates.items()}
    return reconstructed_trajectories


def load_pretrained_rnn_ae(pretrained_model_path):
    model_files = os.listdir(pretrained_model_path)
    architecture_file = model_files[model_files.index('architecture.txt')]
    scaler_file = model_files[model_files.index('scaler.pkl')]
    weight_files = [file_name for file_name in model_files if file_name.startswith('weights')]
    best_weight = sorted(weight_files)[-1]

    architecture_specification = load_architecture_specification(os.path.join(pretrained_model_path, architecture_file))
    anomaly_model = model_from_architecture_specification(architecture_specification)
    anomaly_model.compile()
    anomaly_model.load_weights(os.path.join(pretrained_model_path, best_weight))
    scaler = joblib.load(filename=os.path.join(pretrained_model_path, scaler_file))

    return anomaly_model, scaler
