import os
from datetime import datetime

from keras.layers import SimpleRNNCell, GRUCell, LSTMCell
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l1_l2

from tbad.losses import modified_binary_crossentropy_2, modified_mean_absolute_error, modified_mean_squared_error_2
from tbad.losses import modified_mean_squared_error_3, modified_balanced_mean_absolute_error
from utils.score_scaling import ScoreNormalization
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, MinMaxScaler


def select_optimiser(optimiser, learning_rate):
    """Select an optimiser to train the RNN."""
    if optimiser == 'rmsprop':
        return RMSprop(lr=learning_rate)
    elif optimiser == 'adam':
        return Adam(lr=learning_rate)
    else:
        raise ValueError('Unknown optimiser. Please select either rmsprop or adam.')


def select_loss(loss_name):
    """Select a loss function for the model."""
    if loss_name == 'log_loss':
        return modified_binary_crossentropy_2
    elif loss_name == 'mae':
        return modified_mean_absolute_error
    elif loss_name == 'mse':
        return modified_mean_squared_error_2
    elif loss_name == 'balanced_mse':
        return modified_mean_squared_error_3
    elif loss_name == 'balanced_mae':
        return modified_balanced_mean_absolute_error
    else:
        raise ValueError('Unknown loss function. Please select one of: log_loss, mae or mse.')


def select_cell(cell_type, hidden_dim, l1=0.0, l2=0.0):
    """Select an RNN cell and initialises it with hidden_dim units."""
    if cell_type == 'vanilla':
        return SimpleRNNCell(units=hidden_dim, kernel_regularizer=l1_l2(l1=l1, l2=l2),
                             recurrent_regularizer=l1_l2(l1=l1, l2=l2))
    elif cell_type == 'gru':
        return GRUCell(units=hidden_dim, kernel_regularizer=l1_l2(l1=l1, l2=l2),
                       recurrent_regularizer=l1_l2(l1=l1, l2=l2))
    elif cell_type == 'lstm':
        return LSTMCell(units=hidden_dim, kernel_regularizer=l1_l2(l1=l1, l2=l2),
                        recurrent_regularizer=l1_l2(l1=l1, l2=l2))
    else:
        raise ValueError('Unknown cell type. Please select one of: vanilla, gru, or lstm.')


def set_up_logging(camera_id, root_log_dir=None, resume_training=None):
    log_dir = None
    if resume_training is not None:
        log_dir = os.path.dirname(resume_training)
    elif root_log_dir is not None:
        time_now = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(root_log_dir, camera_id + '_' + time_now)
        os.makedirs(log_dir)

    return log_dir


def resume_training_from_last_epoch(model, resume_training=None):
    last_epoch = 0
    if resume_training is not None:
        model.load_weights(resume_training)
        last_epoch = int(os.path.basename(resume_training).split('_')[1])

    return last_epoch


def select_scaler_model(scaler_name):
    if scaler_name == 'standard':
        return StandardScaler()
    elif scaler_name == 'robust':
        return RobustScaler(quantile_range=(0.00, 50.0))
    elif scaler_name == 'quantile':
        return QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=42)
    elif scaler_name == 'max_abs':
        return MaxAbsScaler()
    elif scaler_name == 'min_max':
        return MinMaxScaler()
    elif scaler_name == 'kde':
        return ScoreNormalization(method='KDE')
    elif scaler_name == 'gamma':
        return ScoreNormalization(method='gamma')
    elif scaler_name == 'chi2':
        return ScoreNormalization(method='chi2')
    else:
        raise ValueError('Unknown scaler. Please select one of: standard, robust, quantile, max_abs, min_max, '
                         'kde, gamma, chi2.')
    
    return None
