import os

import keras
from keras.models import Model
from keras.layers import Input, RNN, Dense
import numpy as np
from sklearn.externals import joblib

from tbad.rnn_autoencoder.rnn import load_architecture_specification
from tbad.combined_model.message_passing import MessagePassingEncoderDecoder
from tbad.utils import select_cell, select_optimiser, select_loss


class CombinedEncoderDecoder:
    def __init__(self, input_length, global_input_dim, local_input_dim, reconstruction_length, prediction_length=0,
                 global_hidden_dims=(2,), local_hidden_dims=(16,), extra_hidden_dims=(), output_activation='sigmoid',
                 cell_type='gru', reconstruct_reverse=True, reconstruct_original_data=False, multiple_outputs=False,
                 multiple_outputs_before_concatenation=True, optimiser='adam', learning_rate=0.001, loss='mse',
                 l1_reg=0.0, l2_reg=0.0):
        self.input_length = input_length
        self.global_input_dim = global_input_dim
        self.local_input_dim = local_input_dim
        self.reconstruction_length = reconstruction_length
        self.prediction_length = prediction_length
        self.global_hidden_dims = global_hidden_dims
        self.local_hidden_dims = local_hidden_dims
        self.extra_hidden_dims = extra_hidden_dims
        self.output_activation = output_activation
        self.cell_type = cell_type
        self.reconstruct_reverse = reconstruct_reverse
        self.reconstruct_original_data = reconstruct_original_data
        self.multiple_outputs = multiple_outputs
        self.multiple_outputs_before_concatenation = multiple_outputs_before_concatenation
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.loss = loss
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        self.model = self.build()

    def build(self):
        all_inputs, all_outputs = [], []
        activation = 'relu' if self.extra_hidden_dims else self.output_activation
        if self.reconstruct_original_data:
            out_dim = self.local_input_dim
        else:
            out_dim = self.global_input_dim + self.local_input_dim

        # Global Model
        global_enc_input = Input(shape=(self.input_length, self.global_input_dim), name='global_enc_input',
                                 dtype='float32')
        all_inputs.append(global_enc_input)

        global_enc_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                            for hidden_dim in self.global_hidden_dims]
        global_enc_rnn = RNN(cell=global_enc_cells, return_state=True, name='global_enc_rnn')
        # global_embedding = Dense(units=16, activation='relu')
        _, *global_enc_states = global_enc_rnn(global_enc_input)

        # Global Model - Reconstruction Branch
        global_rec_input = Input(shape=(self.reconstruction_length, self.global_input_dim), name='global_rec_input',
                                 dtype='float32')
        all_inputs.append(global_rec_input)

        global_rec_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                            for hidden_dim in self.global_hidden_dims]
        global_rec_rnn = RNN(cell=global_rec_cells, return_sequences=True, name='global_rec_rnn')
        global_rec_output = global_rec_rnn(global_rec_input, initial_state=global_enc_states)
        global_rec_dense = Dense(units=self.global_input_dim, activation=activation, name='global_rec_dense')
        global_rec_output = global_rec_dense(global_rec_output)

        # Global Model - Prediction Branch
        if self.prediction_length > 0:
            global_pred_input = Input(shape=(self.prediction_length, self.global_input_dim), name='global_pred_input',
                                      dtype='float32')
            all_inputs.append(global_pred_input)

            global_pred_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                                 for hidden_dim in self.global_hidden_dims]
            global_pred_rnn = RNN(cell=global_pred_cells, return_sequences=True, name='global_pred_rnn')
            global_pred_output = global_pred_rnn(global_pred_input, initial_state=global_enc_states)
            global_pred_dense = Dense(units=self.global_input_dim, activation=activation, name='global_pred_dense')
            global_pred_output = global_pred_dense(global_pred_output)

        # Local Model
        local_enc_input = Input(shape=(self.input_length, self.local_input_dim), name='local_enc_input',
                                dtype='float32')
        all_inputs.append(local_enc_input)

        local_enc_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                           for hidden_dim in self.local_hidden_dims]
        local_enc_rnn = RNN(cell=local_enc_cells, return_state=True, name='local_enc_rnn')
        # local_embedding = Dense(units=136, activation='relu')
        _, *local_enc_states = local_enc_rnn(local_enc_input)

        # Local Model - Reconstruction Branch
        local_rec_input = Input(shape=(self.reconstruction_length, self.local_input_dim), name='local_rec_input',
                                dtype='float32')
        all_inputs.append(local_rec_input)

        local_rec_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                           for hidden_dim in self.local_hidden_dims]
        local_rec_rnn = RNN(cell=local_rec_cells, return_sequences=True, name='local_rec_rnn')
        local_rec_output = local_rec_rnn(local_rec_input, initial_state=local_enc_states)
        local_rec_dense = Dense(units=self.local_input_dim, activation=activation, name='local_rec_dense')
        local_rec_output = local_rec_dense(local_rec_output)

        # Local Model - Prediction Branch
        if self.prediction_length > 0:
            local_pred_input = Input(shape=(self.prediction_length, self.local_input_dim), name='local_pred_input',
                                     dtype='float32')
            all_inputs.append(local_pred_input)

            local_pred_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                                for hidden_dim in self.local_hidden_dims]
            local_pred_rnn = RNN(cell=local_pred_cells, return_sequences=True, name='local_pred_rnn')
            local_pred_output = local_pred_rnn(local_pred_input, initial_state=local_enc_states)
            local_pred_dense = Dense(units=self.local_input_dim, activation=activation, name='local_pred_dense')
            local_pred_output = local_pred_dense(local_pred_output)

        # Merge global and local reconstruction outputs
        rec_output = keras.layers.concatenate(inputs=[global_rec_output, local_rec_output], axis=-1)
        if self.multiple_outputs:
            if self.multiple_outputs_before_concatenation:
                all_outputs.append(global_rec_output)
                all_outputs.append(local_rec_output)
            else:
                final_global_rec_output = Dense(units=self.global_input_dim,
                                                activation=self.output_activation)(rec_output)
                all_outputs.append(final_global_rec_output)
                final_local_rec_output = Dense(units=self.local_input_dim,
                                               activation=self.output_activation)(rec_output)
                all_outputs.append(final_local_rec_output)

        if self.extra_hidden_dims:
            for hidden_dim in self.extra_hidden_dims:
                rec_output = Dense(units=hidden_dim, activation='relu')(rec_output)
            rec_output = Dense(units=out_dim, activation=self.output_activation)(rec_output)
        elif self.reconstruct_original_data:
            rec_output = Dense(units=out_dim, activation=self.output_activation)(rec_output)
        all_outputs.append(rec_output)

        if self.prediction_length > 0:
            pred_output = keras.layers.concatenate(inputs=[global_pred_output, local_pred_output], axis=-1)
            if self.multiple_outputs:
                if self.multiple_outputs_before_concatenation:
                    all_outputs.append(global_pred_output)
                    all_outputs.append(local_pred_output)
                else:
                    final_global_pred_output = Dense(units=self.global_input_dim,
                                                     activation=self.output_activation)(pred_output)
                    all_outputs.append(final_global_pred_output)
                    final_local_pred_output = Dense(units=self.local_input_dim,
                                                    activation=self.output_activation)(pred_output)
                    all_outputs.append(final_local_pred_output)

            if self.extra_hidden_dims:
                for hidden_dim in self.extra_hidden_dims:
                    pred_output = Dense(units=hidden_dim, activation='relu')(pred_output)
                pred_output = Dense(units=out_dim, activation=self.output_activation)(pred_output)
            elif self.reconstruct_original_data:
                pred_output = Dense(units=out_dim, activation=self.output_activation)(pred_output)
            all_outputs.append(pred_output)

        return Model(inputs=all_inputs, outputs=all_outputs)

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

        # Preparing the training data
        if self.reconstruct_original_data:
            X_global_train, X_local_train, X_out_train = X_train
            X_global_val, X_local_val, X_out_val = val_data[0]
        else:
            X_global_train, X_local_train = X_train
            X_global_val, X_local_val = val_data[0]

        X = self._construct_input_data(X_global_train, X_local_train)
        X_val = self._construct_input_data(X_global_val, X_local_val)

        if y_train is not None:
            if self.reconstruct_original_data:
                y_global_train, y_local_train, y_out_train = y_train
                y_global_val, y_local_val, y_out_val = val_data[1]
            else:
                y_global_train, y_local_train = y_train
                y_global_val, y_local_val = val_data[1]
                y_out_train = y_out_val = None
        else:
            y_global_train = y_local_train = y_out_train = y_global_val = y_local_val = y_out_val = None

        if self.reconstruct_original_data:
            y = self._construct_output_data_alt(X_out_train, y_out_train,
                                                X_global_train, y_global_train, X_local_train, y_local_train)
            y_val = self._construct_output_data_alt(X_out_val, y_out_val,
                                                    X_global_val, y_global_val, X_local_val, y_local_val)
        else:
            y = self._construct_output_data(X_global_train, X_local_train, y_global_train, y_local_train)
            y_val = self._construct_output_data(X_global_val, X_local_val, y_global_val, y_local_val)

        validation_data = (X_val, y_val)
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list,
                       validation_data=validation_data, initial_epoch=initial_epoch)
    
    def predict(self, X_test, batch_size=256):
        X_global_test, X_local_test = X_test
        X = self._construct_input_data(X_global_test, X_local_test)
        return self.model.predict(X, batch_size=batch_size)
    
    def reconstruct(self, global_features, local_features):
        if self.prediction_length > 0:
            reconstructed_features = {trajectory_id: self.predict([global_features[trajectory_id],
                                                                   local_features[trajectory_id]], batch_size=256)[0]
                                      for trajectory_id in global_features.keys()}
        else:
            reconstructed_features = {trajectory_id: self.predict([global_features[trajectory_id],
                                                                   local_features[trajectory_id]], batch_size=256)
                                      for trajectory_id in global_features.keys()}
        
        return reconstructed_features

    def compile(self):
        self.model.compile(optimizer=select_optimiser(self.optimiser, self.learning_rate),
                           loss=select_loss(self.loss))
    
    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)
        return None

    def _construct_input_data(self, X_global, X_local):
        X = [X_global]

        n_examples = X_global.shape[0]
        X.append(np.zeros((n_examples, self.reconstruction_length, self.global_input_dim), dtype=np.float32))

        if self.prediction_length > 0:
            X.append(np.zeros((n_examples, self.prediction_length, self.global_input_dim), dtype=np.float32))

        X.append(X_local)

        X.append(np.zeros((n_examples, self.reconstruction_length, self.local_input_dim), dtype=np.float32))

        if self.prediction_length > 0:
            X.append(np.zeros((n_examples, self.prediction_length, self.local_input_dim), dtype=np.float32))

        return X

    def _construct_output_data(self, X_global, X_local, y_global=None, y_local=None):
        y = []

        X = np.concatenate((X_global, X_local), axis=-1)
        if self.multiple_outputs:
            if self.reconstruct_reverse:
                y.append(X_global[:, (self.reconstruction_length - 1)::-1, :])
                y.append(X_local[:, (self.reconstruction_length - 1)::-1, :])
                y.append(X[:, (self.reconstruction_length - 1)::-1, :])
            else:
                y.append(X_global[:, :self.reconstruction_length, :])
                y.append(X_local[:, :self.reconstruction_length, :])
                y.append(X[:, :self.reconstruction_length, :])

            if self.prediction_length > 0:
                y.append(y_global)
                y.append(y_local)
                y.append(np.concatenate((y_global, y_local), axis=-1))
        else:
            if self.reconstruct_reverse:
                y.append(X[:, (self.reconstruction_length - 1)::-1, :])
            else:
                y.append(X[:, :self.reconstruction_length, :])

            if self.prediction_length > 0:
                y.append(np.concatenate((y_global, y_local), axis=-1))

        return y

    def _construct_output_data_alt(self, X_out, y_out=None, X_global=None, y_global=None, X_local=None, y_local=None):
        y = []

        if self.multiple_outputs:
            if self.reconstruct_reverse:
                y.append(X_global[:, (self.reconstruction_length - 1)::-1, :])
                y.append(X_local[:, (self.reconstruction_length - 1)::-1, :])
                y.append(X_out[:, (self.reconstruction_length - 1)::-1, :])
            else:
                y.append(X_global[:, :self.reconstruction_length, :])
                y.append(X_local[:, :self.reconstruction_length, :])
                y.append(X_out[:, :self.reconstruction_length, :])

            if self.prediction_length > 0:
                y.append(y_global)
                y.append(y_local)
                y.append(y_out)
        else:
            if self.reconstruct_reverse:
                y.append(X_out[:, (self.reconstruction_length - 1)::-1, :])
            else:
                y.append(X_out[:, :self.reconstruction_length, :])

            if self.prediction_length > 0:
                y.append(y_out)

        return y

    def _maybe_write_architecture(self, log_dir):
        file_path = os.path.join(log_dir, 'architecture.txt')
        if os.path.isfile(file_path):
            return None

        with open(file_path, mode='w') as file:
            print('input_length', self.input_length, file=file)
            print('global_input_dim', self.global_input_dim, file=file)
            print('local_input_dim', self.local_input_dim, file=file)
            print('reconstruction_length', self.reconstruction_length, file=file)
            print('prediction_length', self.prediction_length, file=file)
            print('global_hidden_dims', *self.global_hidden_dims, file=file)
            print('local_hidden_dims', *self.local_hidden_dims, file=file)
            print('extra_hidden_dims', *self.extra_hidden_dims, file=file)
            print('output_activation', self.output_activation, file=file)
            print('reconstruct_reverse', self.reconstruct_reverse, file=file)
            print('reconstruct_original_data', self.reconstruct_original_data, file=file)
            print('multiple_outputs', self.multiple_outputs, file=file)
            print('multiple_outputs_before_concatenation', self.multiple_outputs_before_concatenation, file=file)
            print('cell_type', self.cell_type, file=file)
            print('optimiser', self.optimiser, file=file)
            print('learning_rate', self.learning_rate, file=file)
            print('loss', self.loss, file=file)
            print('l1_reg', self.l1_reg, file=file)
            print('l2_reg', self.l2_reg, file=file)

        return None


def load_complete_rnn_ae_pretrained_models(all_pretrained_models_path):
    pretrained_models = {}
    global_scalers, local_scalers = {}, {}
    for pretrained_model_name in os.listdir(all_pretrained_models_path):
        if pretrained_model_name.endswith('.npy') or pretrained_model_name.endswith('.npz') or \
                pretrained_model_name.endswith('.txt'):
            continue
        camera_id = pretrained_model_name.split('_')[0]
        pretrained_model_path = os.path.join(all_pretrained_models_path, pretrained_model_name)
        pretrained_models[camera_id], global_scalers[camera_id], local_scalers[camera_id] = \
            load_pretrained_combined_model(pretrained_model_path)

    return pretrained_models, global_scalers, local_scalers


def load_pretrained_combined_model(pretrained_model_path, message_passing=False):
    model_files = os.listdir(pretrained_model_path)
    architecture_file = model_files[model_files.index('architecture.txt')]
    global_scaler_file = model_files[model_files.index('global_scaler.pkl')]
    local_scaler_file = model_files[model_files.index('local_scaler.pkl')]
    try:
        out_scaler_file = model_files[model_files.index('out_scaler.pkl')]
    except ValueError:
        out_scaler_file = None
    weight_files = [file_name for file_name in model_files if file_name.startswith('weights')]
    best_weights = sorted(weight_files)[-1]
    
    architecture_path = os.path.join(pretrained_model_path, architecture_file)
    best_weights_path = os.path.join(pretrained_model_path, best_weights)

    architecture_specification = load_architecture_specification(architecture_path)
    if message_passing:
        combined_model = MessagePassingEncoderDecoder(**architecture_specification)
    else:
        combined_model = CombinedEncoderDecoder(**architecture_specification)
    combined_model.compile()
    combined_model.load_weights(best_weights_path)
    global_scaler = joblib.load(filename=os.path.join(pretrained_model_path, global_scaler_file))
    local_scaler = joblib.load(filename=os.path.join(pretrained_model_path, local_scaler_file))
    if out_scaler_file is not None:
        out_scaler = joblib.load(filename=os.path.join(pretrained_model_path, out_scaler_file))
    else:
        out_scaler = None

    return combined_model, global_scaler, local_scaler, out_scaler


# TODO Finalise the implementation of the custom coordinate change layer.
def coordinate_change(x):
    xs_star_global, ys_star_global = x[:, :, :1], x[:, :, 1:2]
    widths, heights = x[:, :, 2:3], x[:, :, 3:4]

    y = x[:, :, 4::2] * widths + xs_star_global - widths / 2
    z = x[:, :, 5::2] * heights + ys_star_global - heights / 2
    
    return x
