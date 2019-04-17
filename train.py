import argparse

from tbad.gpu import configure_gpu_resources
from tbad.autoencoder.train import train_ae
from tbad.rnn_autoencoder.train import train_rnn_ae
from tbad.combined_model.train import train_combined_model


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Training Functions for Trajectory-Based Anomaly Detection.')

    gp_gpu = parser.add_argument_group('GPU')
    gp_gpu.add_argument('--gpu_ids', default='0', type=str, help='Which GPUs to use.')
    gp_gpu.add_argument('--gpu_memory_fraction', default=0.20, type=float,
                        help='Fraction of the memory to grab from each GPU.')

    subparsers = parser.add_subparsers(title='sub-commands', description='Valid sub-commands.')

    # Create sub-parser for training of an Autoencoder
    parser_ae = subparsers.add_parser('autoencoder', help='Train an Autoencoder.')
    parser_ae.add_argument('trajectories', type=str,
                           help='Path to directory containing training trajectories. For each video in the '
                                'training set, there must be a folder inside this directory containing the '
                                'trajectories.')
    parser_ae.add_argument('--video_resolution', default='856x480', type=str,
                           help='Resolution of the trajectories\' original video(s). It should be specified '
                                'as WxH, where W is the width and H the height of the video.')
    parser_ae.add_argument('--resume_training', type=str, help='Pre-trained model weights.')

    gp_ae_arch = parser_ae.add_argument_group('Model Architecture')
    gp_ae_arch.add_argument('--global_model', action='store_true',
                            help='If this flag is specified, instead of training the Autoencoder '
                                 'on the joints of the skeletons, the model is trained on the center of '
                                 'mass (x*, y*) and on the width and height of the bounding box of the '
                                 'skeletons.')
    gp_ae_arch.add_argument('--hidden_dims', nargs='+', default=[16], type=int,
                            help='Number of hidden units per hidden layer.')
    gp_ae_arch.add_argument('--output_activation', default='sigmoid', type=str, choices=['linear', 'sigmoid'],
                            help='Activation function of the output layer.')

    gp_ae_hp = parser_ae.add_argument_group('Model Training Hyperparameters')
    gp_ae_hp.add_argument('--optimiser', default='adam', type=str, choices=['adam', 'rmsprop'],
                          help='Optimiser for model training.')
    gp_ae_hp.add_argument('--learning_rate', default=0.001, type=float,
                          help='Learning rate of the optimiser.')
    gp_ae_hp.add_argument('--loss', default='mse', type=str, choices=['log_loss', 'mae', 'mse'],
                          help='Loss function to be minimised by the optimiser.')
    gp_ae_hp.add_argument('--epochs', default=5, type=int, help='Maximum number of epochs for training.')
    gp_ae_hp.add_argument('--batch_size', default=256, type=int, help='Mini-batch size for model training.')
    gp_ae_hp.add_argument('--coordinate_system', default='global', type=str,
                          choices=['global', 'bounding_box_top_left', 'bounding_box_centre'],
                          help='Which coordinate system to use.')
    gp_ae_hp.add_argument('--normalisation_strategy', default='zero_one', type=str,
                          choices=['zero_one', 'three_stds', 'robust'],
                          help='Strategy for normalisation of the skeletons.')

    gp_ae_logging = parser_ae.add_argument_group('Model Logging')
    gp_ae_logging.add_argument('--root_log_dir', type=str,
                               help='TODO.')

    parser_ae.set_defaults(func=train_ae)

    # Create sub-parser for training of an RNN Autoencoder
    parser_rnn_ae = subparsers.add_parser('rnn_autoencoder', help='Train an RNN Autoencoder.')
    parser_rnn_ae.add_argument('trajectories', type=str, help='Path to training trajectories.')
    parser_rnn_ae.add_argument('--video_resolution', default='856x480', type=str,
                               help='Resolution of the trajectories\' original video(s).')
    parser_rnn_ae.add_argument('--resume_training', type=str, help='Pre-trained model weights.')

    gp_rnn_ae_arch = parser_rnn_ae.add_argument_group('Model Architecture')
    gp_rnn_ae_arch.add_argument('--model_type', default='plain', type=str, choices=['plain', 'global', 'concatenate'],
                                help='Select the model type. If plain, no modifications are done to the input data. '
                                     'If global, the RNN is trained on the center of mass (x*, y*) and on the width '
                                     'and height of the bounding box of the skeletons. If concatenate, the RNN is '
                                     'trained on the concatenation of the global model features with the joints of the '
                                     'skeletons centered at their bounding box (local model).')
    gp_rnn_ae_arch.add_argument('--extract_delta', action='store_true',
                                help='Only meaningful if model_type is global. If specified, include the difference '
                                     'between consecutive time-steps in addition to the absolute x and y coordinates.')
    gp_rnn_ae_arch.add_argument('--use_first_step_as_reference', action='store_true',
                                help='Only meaningful if model type is global. If specified, use the difference '
                                     'between all time-steps and the first time-step instead of the absolute x and '
                                     'y coordinates.')
    gp_rnn_ae_arch.add_argument('--input_length', default=8, type=int,
                                help='Number of input time-steps to encode.')
    gp_rnn_ae_arch.add_argument('--input_gap', default=0, type=int,
                                help='Number of input time-steps to skip during encoding.')
    gp_rnn_ae_arch.add_argument('--rec_length', default=8, type=int,
                                help='Number of time-steps to decode from the input sequence.')
    gp_rnn_ae_arch.add_argument('--pred_length', default=0, type=int,
                                help='Number of time-steps to predict into future. Ignored if 0.')
    gp_rnn_ae_arch.add_argument('--reconstruct_reverse', action='store_true',
                                help='Whether to reconstruct the reverse of the input sequence or not.')
    gp_rnn_ae_arch.add_argument('--cell_type', default='gru', type=str, choices=['vanilla', 'gru', 'lstm'],
                                help='Type of cell used by the RNN.')
    gp_rnn_ae_arch.add_argument('--hidden_dims', nargs='+', default=[16], type=int,
                                help='Number of hidden units per hidden layer.')
    gp_rnn_ae_arch.add_argument('--output_activation', default='sigmoid', type=str, choices=['linear', 'sigmoid'],
                                help='Activation function of the output layer.')
    gp_rnn_ae_arch.add_argument('--disable_reconstruction_branch', action='store_false',
                                help='')
    gp_rnn_ae_arch.add_argument('--conditional_reconstruction', action='store_true',
                                help='')
    gp_rnn_ae_arch.add_argument('--conditional_prediction', action='store_true',
                                help='')

    gp_rnn_ae_hp = parser_rnn_ae.add_argument_group('Model Training Hyperparameters')
    gp_rnn_ae_hp.add_argument('--optimiser', default='adam', type=str, choices=['adam', 'rmsprop'],
                              help='Optimiser to train model.')
    gp_rnn_ae_hp.add_argument('--learning_rate', default=0.001, type=float,
                              help='Learning rate of the optimiser.')
    gp_rnn_ae_hp.add_argument('--loss', default='mse', type=str, choices=['log_loss', 'mae', 'mse'],
                              help='Loss function to be minimised by the optimiser.')
    gp_rnn_ae_hp.add_argument('--l1_reg', default=0.0, type=float,
                              help='Amount of L1 regularisation added to the model weights.')
    gp_rnn_ae_hp.add_argument('--l2_reg', default=0.0, type=float,
                              help='Amount of L2 regularisation added to the model weights.')
    gp_rnn_ae_hp.add_argument('--epochs', default=5, type=int, help='Maximum number of epochs for training.')
    gp_rnn_ae_hp.add_argument('--batch_size', default=256, type=int,
                              help='Mini-batch size for model training.')
    gp_rnn_ae_hp.add_argument('--input_missing_steps', action='store_true',
                              help='Fill missing steps of trajectories with a weighted combination of '
                                   'the closest non-missing steps.')
    gp_rnn_ae_hp.add_argument('--coordinate_system', default='global', type=str,
                              choices=['global', 'bounding_box_top_left', 'bounding_box_centre'],
                              help='Which coordinate system to use.')
    gp_rnn_ae_hp.add_argument('--normalisation_strategy', default='zero_one', type=str,
                              choices=['zero_one', 'three_stds', 'robust'],
                              help='Strategy for normalisation of the trajectories.')

    gp_rnn_ae_logging = parser_rnn_ae.add_argument_group('Model Logging')
    gp_rnn_ae_logging.add_argument('--root_log_dir', type=str,
                                   help='Root directory to write: trained weights, training report and '
                                        'model architecture. A time-stamped sub-directory is created to save '
                                        'all files. Ignored if resume_training is specified.')

    parser_rnn_ae.set_defaults(func=train_rnn_ae)

    # Create sub-parser for training of the combined global and local RNN model
    parser_combined_model = subparsers.add_parser('combined_model',
                                                  help='Train a Global + Local RNN Autoencoder.')
    parser_combined_model.add_argument('trajectories', type=str, help='Path to training trajectories.')
    parser_combined_model.add_argument('--video_resolution', default='856x480', type=str,
                                       help='Video resolution of the videos from where the trajectories '
                                            'were extracted.')
    parser_combined_model.add_argument('--resume_training', type=str, help='Pre-trained model weights.')

    gp_combined_model_arch = parser_combined_model.add_argument_group('Model Architecture')
    gp_combined_model_arch.add_argument('--message_passing', action='store_true',
                                        help='Whether to perform message passing between the global and local branches '
                                             'or not.')
    gp_combined_model_arch.add_argument('--reconstruct_original_data', action='store_true',
                                        help='Whether to reconstruct the original trajectories or the concatenation '
                                             'of the output of the global and local models.')
    gp_combined_model_arch.add_argument('--multiple_outputs', action='store_true',
                                        help='If specified, the network also outputs the global and local '
                                             'reconstructions/predictions.')
    gp_combined_model_arch.add_argument('--multiple_outputs_before_concatenation', action='store_true',
                                        help='Only meaningful if multiple_outputs is specified as well. If specified,'
                                             'the global and local outputs are created before concatenation of the '
                                             'branches.')
    gp_combined_model_arch.add_argument('--input_length', default=16, type=int,
                                        help='Number of input time-steps to encode.')
    gp_combined_model_arch.add_argument('--rec_length', default=16, type=int,
                                        help='Number of time-steps to decode from the input sequence.')
    gp_combined_model_arch.add_argument('--pred_length', default=2, type=int,
                                        help='Number of time-steps to predict into future. Ignored if 0.')
    gp_combined_model_arch.add_argument('--reconstruct_reverse', action='store_true',
                                        help='Whether to reconstruct the reverse of the input sequence or '
                                             'not.')
    gp_combined_model_arch.add_argument('--cell_type', default='gru', type=str,
                                        choices=['vanilla', 'gru', 'lstm'],
                                        help='Type of cell used by the RNN.')
    gp_combined_model_arch.add_argument('--global_hidden_dims', nargs='+', default=[2], type=int,
                                        help='Number of hidden units per hidden layer of the global model.')
    gp_combined_model_arch.add_argument('--local_hidden_dims', nargs='+', default=[16], type=int,
                                        help='Number of hidden units per hidden layer of the local model.')
    gp_combined_model_arch.add_argument('--extra_hidden_dims', nargs='+', default=[], type=int,
                                        help='Number of hidden units per hidden layer after concatenation '
                                             'of the global and local models.')
    gp_combined_model_arch.add_argument('--output_activation', default='sigmoid', type=str,
                                        choices=['linear', 'sigmoid'],
                                        help='Activation function of the output layer.')

    gp_combined_model_hp = parser_combined_model.add_argument_group('Model Training Hyperparameters')
    gp_combined_model_hp.add_argument('--optimiser', default='adam', type=str, choices=['adam', 'rmsprop'],
                                      help='Optimiser to train model.')
    gp_combined_model_hp.add_argument('--learning_rate', default=0.001, type=float,
                                      help='Learning rate of the optimiser.')
    gp_combined_model_hp.add_argument('--loss', default='mse', type=str,
                                      choices=['log_loss', 'mae', 'mse', 'balanced_mse', 'balanced_mae'],
                                      help='Loss function to be minimised by the optimiser.')
    gp_combined_model_hp.add_argument('--l1_reg', default=0.0, type=float,
                                      help='Amount of L1 regularisation added to the model weights.')
    gp_combined_model_hp.add_argument('--l2_reg', default=0.0, type=float,
                                      help='Amount of L2 regularisation added to the model weights.')
    gp_combined_model_hp.add_argument('--epochs', default=5, type=int,
                                      help='Maximum number of epochs for training.')
    gp_combined_model_hp.add_argument('--batch_size', default=256, type=int,
                                      help='Mini-batch size for model training.')
    gp_combined_model_hp.add_argument('--input_missing_steps', action='store_true',
                                      help='Fill missing steps of trajectories with a weighted combination of '
                                           'the closest non-missing steps.')
    gp_combined_model_hp.add_argument('--global_normalisation_strategy', default='zero_one', type=str,
                                      choices=['zero_one', 'three_stds', 'robust'],
                                      help='Global normalisation strategy.')
    gp_combined_model_hp.add_argument('--local_normalisation_strategy', default='zero_one', type=str,
                                      choices=['zero_one', 'three_stds', 'robust'],
                                      help='Local normalisation strategy.')
    gp_combined_model_hp.add_argument('--out_normalisation_strategy', default='zero_one', type=str,
                                      choices=['zero_one', 'three_stds', 'robust'])

    gp_combined_model_logging = parser_combined_model.add_argument_group('Model Logging')
    gp_combined_model_logging.add_argument('--root_log_dir', type=str,
                                           help='Root directory to write: trained weights, training report '
                                                'and model architecture. A time-stamped sub-directory is '
                                                'created to save all files. Ignored if resume_training is '
                                                'specified.')

    parser_combined_model.set_defaults(func=train_combined_model)

    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    configure_gpu_resources(args.gpu_ids, args.gpu_memory_fraction)
    args.func(args)


if __name__ == '__main__':
    main()
