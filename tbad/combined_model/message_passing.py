import os

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, RNN, Dense, Reshape, Lambda
import numpy as np

from tbad.utils import select_cell, select_optimiser, select_loss


class MessagePassingEncoderDecoder:
    def __init__(self, input_length, global_input_dim, local_input_dim, reconstruction_length, prediction_length=0,
                 global_hidden_dims=(2,), local_hidden_dims=(16,), extra_hidden_dims=(), output_activation='sigmoid',
                 cell_type='gru', reconstruct_reverse=True, reconstruct_original_data=False, multiple_outputs=False,
                 multiple_outputs_before_concatenation=True, optimiser='adam', learning_rate=0.001,
                 loss='balanced_mse', l1_reg=0.0, l2_reg=0.0):
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

        # Encoding
        global_enc_input = Input(shape=(self.input_length, self.global_input_dim), name='global_enc_input',
                                 dtype='float32')
        all_inputs.append(global_enc_input)
        local_enc_input = Input(shape=(self.input_length, self.local_input_dim), name='local_enc_input',
                                dtype='float32')
        all_inputs.append(local_enc_input)

        global_input_embedding = Dense(units=8, activation='relu')
        global_enc_message_embedding = Dense(units=8, activation='relu')

        local_input_embedding = Dense(units=68, activation='relu')
        local_enc_message_embedding = Dense(units=68, activation='relu')

        global_enc_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                            for hidden_dim in self.global_hidden_dims]
        global_enc_rnn = RNN(cell=global_enc_cells, return_state=True)

        local_enc_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                           for hidden_dim in self.local_hidden_dims]
        local_enc_rnn = RNN(cell=local_enc_cells, return_state=True)

        global_tilde = global_input_embedding(Lambda(lambda x: x[:, 0, :])(global_enc_input))
        local_initial_enc_hidden_state = Input(shape=self.local_hidden_dims, dtype='float32')
        local_hidden_tilde = global_enc_message_embedding(local_initial_enc_hidden_state)
        global_concatenated = Reshape((1, -1))(keras.layers.concatenate([global_tilde, local_hidden_tilde], axis=-1))
        _, global_hidden_state = global_enc_rnn(global_concatenated)

        local_tilde = local_input_embedding(Lambda(lambda x: x[:, 0, :])(local_enc_input))
        global_initial_enc_hidden_state = Input(shape=self.global_hidden_dims, dtype='float32')
        global_hidden_tilde = local_enc_message_embedding(global_initial_enc_hidden_state)
        local_concatenated = Reshape((1, -1))(keras.layers.concatenate([local_tilde, global_hidden_tilde], axis=-1))
        _, local_hidden_state = local_enc_rnn(local_concatenated)

        all_inputs.append(local_initial_enc_hidden_state)
        all_inputs.append(global_initial_enc_hidden_state)

        for idx in range(1, self.input_length):
            global_tilde = global_input_embedding(Lambda(lambda x: x[:, idx, :])(global_enc_input))
            local_hidden_tilde = global_enc_message_embedding(local_hidden_state)
            global_concatenated = Reshape((1, -1))(keras.layers.concatenate([global_tilde, local_hidden_tilde], axis=-1))

            local_tilde = local_input_embedding(Lambda(lambda x: x[:, idx, :])(local_enc_input))
            global_hidden_tilde = local_enc_message_embedding(global_hidden_state)
            local_concatenated = Reshape((1, -1))(keras.layers.concatenate([local_tilde, global_hidden_tilde], axis=-1))

            _, global_hidden_state = global_enc_rnn(global_concatenated, initial_state=global_hidden_state)
            _, local_hidden_state = local_enc_rnn(local_concatenated, initial_state=local_hidden_state)

        # Reconstruction
        global_rec_message_embedding = Dense(units=8, activation='relu')

        local_rec_message_embedding = Dense(units=68, activation='relu')

        global_rec_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                            for hidden_dim in self.global_hidden_dims]
        global_rec_rnn = RNN(cell=global_rec_cells, return_sequences=True, return_state=True)

        local_rec_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                           for hidden_dim in self.local_hidden_dims]
        local_rec_rnn = RNN(cell=local_rec_cells, return_sequences=True, return_state=True)

        global_rec_dense = Dense(units=self.global_input_dim, activation=activation)
        local_rec_dense = Dense(units=self.local_input_dim, activation=activation)

        global_rec_outputs = []
        local_rec_tilde = global_rec_message_embedding(local_hidden_state)
        global_rec_concatenated = Reshape((1, -1))(local_rec_tilde)
        global_rec_output, global_rec_hidden_state = global_rec_rnn(global_rec_concatenated,
                                                                    initial_state=global_hidden_state)
        global_rec_output = global_rec_dense(global_rec_output)
        global_rec_outputs.append(global_rec_output)

        local_rec_outputs = []
        global_rec_tilde = local_rec_message_embedding(global_hidden_state)
        local_rec_concatenated = Reshape((1, -1))(global_rec_tilde)
        local_rec_output, local_rec_hidden_state = local_rec_rnn(local_rec_concatenated,
                                                                 initial_state=local_hidden_state)
        local_rec_output = local_rec_dense(local_rec_output)
        local_rec_outputs.append(local_rec_output)

        for idx in range(1, self.reconstruction_length):
            local_rec_tilde = global_rec_message_embedding(local_rec_hidden_state)
            global_rec_concatenated = Reshape((1, -1))(local_rec_tilde)

            global_rec_tilde = local_rec_message_embedding(global_rec_hidden_state)
            local_rec_concatenated = Reshape((1, -1))(global_rec_tilde)

            global_rec_output, global_rec_hidden_state = global_rec_rnn(global_rec_concatenated,
                                                                        initial_state=global_rec_hidden_state)
            global_rec_output = global_rec_dense(global_rec_output)
            global_rec_outputs.append(global_rec_output)

            local_rec_output, local_rec_hidden_state = local_rec_rnn(local_rec_concatenated,
                                                                     initial_state=local_rec_hidden_state)
            local_rec_output = local_rec_dense(local_rec_output)
            local_rec_outputs.append(local_rec_output)

        global_rec_output = Lambda(lambda x: K.concatenate(x, axis=1))(global_rec_outputs)
        local_rec_output = Lambda(lambda x: K.concatenate(x, axis=1))(local_rec_outputs)
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

        # Prediction
        if self.prediction_length > 0:
            global_pred_message_embedding = Dense(units=8, activation='relu')

            local_pred_message_embedding = Dense(units=68, activation='relu')

            global_pred_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                                 for hidden_dim in self.global_hidden_dims]
            global_pred_rnn = RNN(cell=global_pred_cells, return_sequences=True, return_state=True)

            local_pred_cells = [select_cell(self.cell_type, hidden_dim, l1=self.l1_reg, l2=self.l2_reg)
                                for hidden_dim in self.local_hidden_dims]
            local_pred_rnn = RNN(cell=local_pred_cells, return_sequences=True, return_state=True)

            global_pred_dense = Dense(units=self.global_input_dim, activation=activation)
            local_pred_dense = Dense(units=self.local_input_dim, activation=activation)

            global_pred_outputs = []
            local_pred_tilde = global_pred_message_embedding(local_hidden_state)
            global_pred_concatenated = Reshape((1, -1))(local_pred_tilde)
            global_pred_output, global_pred_hidden_state = global_pred_rnn(global_pred_concatenated,
                                                                           initial_state=global_hidden_state)
            global_pred_output = global_pred_dense(global_pred_output)
            global_pred_outputs.append(global_pred_output)

            local_pred_outputs = []
            global_pred_tilde = local_pred_message_embedding(global_hidden_state)
            local_pred_concatenated = Reshape((1, -1))(global_pred_tilde)
            local_pred_output, local_pred_hidden_state = local_pred_rnn(local_pred_concatenated,
                                                                        initial_state=local_hidden_state)
            local_pred_output = local_pred_dense(local_pred_output)
            local_pred_outputs.append(local_pred_output)

            for idx in range(1, self.prediction_length):
                local_pred_tilde = global_pred_message_embedding(local_pred_hidden_state)
                global_pred_concatenated = Reshape((1, -1))(local_pred_tilde)

                global_pred_tilde = local_pred_message_embedding(global_pred_hidden_state)
                local_pred_concatenated = Reshape((1, -1))(global_pred_tilde)

                global_pred_output, global_pred_hidden_state = global_pred_rnn(global_pred_concatenated,
                                                                               initial_state=global_pred_hidden_state)
                global_pred_output = global_pred_dense(global_pred_output)
                global_pred_outputs.append(global_pred_output)

                local_pred_output, local_pred_hidden_state = local_pred_rnn(local_pred_concatenated,
                                                                            initial_state=local_pred_hidden_state)
                local_pred_output = local_pred_dense(local_pred_output)
                local_pred_outputs.append(local_pred_output)

            global_pred_output = Lambda(lambda x: K.concatenate(x, axis=1))(global_pred_outputs)
            local_pred_output = Lambda(lambda x: K.concatenate(x, axis=1))(local_pred_outputs)
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

        if self.reconstruct_original_data:
            X_global_train, X_local_train, X_out_train = X_train
            X_global_val, X_local_val, X_out_val = val_data[0]

            if y_train is not None:
                y_global_train, y_local_train, y_out_train = y_train
                y_global_val, y_local_val, y_out_val = val_data[1]
            else:
                y_global_train = y_local_train = y_out_train = y_global_val = y_local_val = y_out_val = None

            y = self._construct_output_data_alt(X_out_train, y_out_train,
                                                X_global_train, y_global_train, X_local_train, y_local_train)
            y_val = self._construct_output_data_alt(X_out_val, y_out_val,
                                                    X_global_val, y_global_val, X_local_val, y_local_val)
        else:
            X_global_train, X_local_train = X_train
            X_global_val, X_local_val = val_data[0]

            if y_train is not None:
                y_global_train, y_local_train = y_train
                y_global_val, y_local_val = val_data[1]
            else:
                y_global_train = y_local_train = y_global_val = y_local_val = None

            y = self._construct_output_data(X_global_train, X_local_train, y_global_train, y_local_train)
            y_val = self._construct_output_data(X_global_val, X_local_val, y_global_val, y_local_val)

        X = self._construct_input_data(X_global_train, X_local_train)
        X_val = self._construct_input_data(X_global_val, X_local_val)

        validation_data = (X_val, y_val)
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list,
                       validation_data=validation_data, initial_epoch=initial_epoch)

    def predict(self, X_test, batch_size=256):
        X_global_test, X_local_test = X_test
        X = self._construct_input_data(X_global_test, X_local_test)
        return self.model.predict(X, batch_size=batch_size)

    def compile(self):
        self.model.compile(optimizer=select_optimiser(self.optimiser, self.learning_rate),
                           loss=select_loss(self.loss))

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def _construct_input_data(self, X_global, X_local):
        X = [X_global, X_local]

        num_examples = X_global.shape[0]
        X.append(np.zeros((num_examples, self.local_hidden_dims[0]), dtype=np.float32))
        X.append(np.zeros((num_examples, self.global_hidden_dims[0]), dtype=np.float32))

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
