import argparse

from tbad.autoencoder.evaluate import eval_ae, eval_aes
from tbad.rnn_autoencoder.evaluate import eval_rnn_ae, eval_rnn_aes
from tbad.combined_model.evaluate import eval_combined_model, eval_combined_models
from tbad.gpu import configure_gpu_resources


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Functions for Evaluation of Trained Trajectory-Based Anomaly Models.')

    gp_gpu = parser.add_argument_group('GPU')
    gp_gpu.add_argument('--gpu_ids', default='0', type=str, help='Which GPUs to use.')
    gp_gpu.add_argument('--gpu_memory_fraction', default=0.20, type=float,
                        help='Fraction of the memory to grab from each GPU.')

    subparsers = parser.add_subparsers(title='sub-commands', description='Valid sub-commands.')

    # Create sub-parser for evaluation of a pre-trained Autoencoder model
    parser_ae = subparsers.add_parser('autoencoder',
                                      help='Evaluate a Trained Autoencoder Model.')
    parser_ae.add_argument('pretrained_model', type=str,
                           help='Directory containing pre-trained model architecture definition, model weights and '
                                'data scaler.')
    parser_ae.add_argument('trajectories', type=str,
                           help='Directory containing skeleton\'s trajectories.')
    parser_ae.add_argument('frame_level_anomaly_masks', type=str,
                           help='Directory containing .npy files for each video in the specific camera.')
    parser_ae.add_argument('--video_resolution', default='856x480', type=str,
                           help='Resolution of the trajectories\' original video(s). It should be specified '
                                'as WxH, where W is the width and H the height of the video.')

    parser_ae.set_defaults(func=eval_ae)

    # Create sub-parser for evaluation of multiple pre-trained Autoencoder models
    parser_aes = subparsers.add_parser('autoencoders',
                                       help='Evaluate Several Trained Autoencoder Models.')
    parser_aes.add_argument('pretrained_models', type=str,
                            help='Directory containing a folder for each pre-trained model.')
    parser_aes.add_argument('all_trajectories', type=str,
                            help='Directory containing a folder for each camera, where each folder contains the '
                                 'trajectories for the associated camera.')
    parser_aes.add_argument('all_frame_level_anomaly_masks', type=str,
                            help='Directory containing a folder for the anomaly masks of each camera.')
    parser_aes.add_argument('--video_resolution', default='856x480', type=str,
                            help='Resolution of the trajectories\' original video(s). It should be specified '
                                 'as WxH, where W is the width and H the height of the video.')

    parser_aes.set_defaults(func=eval_aes)

    # Create sub-parser for evaluation of a pre-trained RNN Autoencoder model
    parser_rnn_ae = subparsers.add_parser('rnn_autoencoder', help='Evaluate a Trained RNN Autoencoder model.')
    parser_rnn_ae.add_argument('pretrained_model', type=str,
                               help='Directory containing pre-trained model architecture definition, model weights and '
                                    'data scaler.')
    parser_rnn_ae.add_argument('trajectories', type=str,
                               help='Directory containing skeleton\'s trajectories.')
    parser_rnn_ae.add_argument('frame_level_anomaly_masks', type=str,
                               help='Directory containing .npy files for each video in the specific camera.')
    parser_rnn_ae.add_argument('--video_resolution', default='856x480', type=str,
                               help='Resolution of the trajectories\' original video(s). It should be specified '
                                    'as WxH, where W is the width and H the height of the video.')
    parser_rnn_ae.add_argument('--overlapping_trajectories', action='store_true')

    gp_rnn_ae_logging = parser_rnn_ae.add_argument_group('Evaluation Logging')
    gp_rnn_ae_logging.add_argument('--write_reconstructions', type=str,
                                   help='TO DO')

    parser_rnn_ae.set_defaults(func=eval_rnn_ae)

    # Create sub-parser for evaluation of multiple pre-trained RNN Autoencoder models
    parser_rnn_aes = subparsers.add_parser('rnn_autoencoders',
                                           help='Evaluate Multiple Pre-Trained RNN Autoencoder models.')
    parser_rnn_aes.add_argument('pretrained_models', type=str,
                                help='Directory containing a folder for each pre-trained model.')
    parser_rnn_aes.add_argument('all_trajectories', type=str,
                                help='Directory containing a folder for each camera, where each folder contains the '
                                     'trajectories for the associated camera.')
    parser_rnn_aes.add_argument('all_frame_level_anomaly_masks', type=str,
                                help='Directory containing a folder for the anomaly masks of each camera.')
    parser_rnn_aes.add_argument('--video_resolution', default='856x480', type=str,
                                help='Resolution of the trajectories\' original video(s). It should be specified '
                                     'as WxH, where W is the width and H the height of the video.')
    parser_rnn_aes.add_argument('--overlapping_trajectories', action='store_true')

    gp_rnn_aes_logging = parser_rnn_aes.add_argument_group('Evaluation Logging')
    gp_rnn_aes_logging.add_argument('--write_reconstructions', type=str,
                                    help='TO DO')

    parser_rnn_aes.set_defaults(func=eval_rnn_aes)

    # Create a sub-parser for evaluation of a trained Combined model
    parser_combined_model = subparsers.add_parser('combined_model',
                                                  help='Evaluate a Trained Combined Model.')
    parser_combined_model.add_argument('pretrained_model', type=str,
                                       help='Directory containing pre-trained model architecture definition, model '
                                            'weights and data scaler.')
    parser_combined_model.add_argument('trajectories', type=str,
                                       help='Directory containing skeleton\'s trajectories.')
    parser_combined_model.add_argument('frame_level_anomaly_masks', type=str,
                                       help='Directory containing .npy files for each video in the specific camera.')
    parser_combined_model.add_argument('--video_resolution', default='856x480', type=str,
                                       help='Resolution of the trajectories\' original video(s). It should be '
                                            'specified as WxH, where W is the width and H the height of the video.')
    parser_combined_model.add_argument('--overlapping_trajectories', action='store_true')

    gp_combined_model_logging = parser_combined_model.add_argument_group('Evaluation Logging')
    gp_combined_model_logging.add_argument('--write_reconstructions', action='store_true')
    gp_combined_model_logging.add_argument('--write_bounding_boxes', action='store_true')
    gp_combined_model_logging.add_argument('--write_predictions', action='store_true')
    gp_combined_model_logging.add_argument('--write_predictions_bounding_boxes', action='store_true')
    gp_combined_model_logging.add_argument('--write_anomaly_masks', action='store_true')
    gp_combined_model_logging.add_argument('--write_mistakes', action='store_true')

    parser_combined_model.set_defaults(func=eval_combined_model)

    # Create a sub-parser for evaluation of multiple pre-trained Combined models
    parser_combined_models = subparsers.add_parser('combined_models',
                                                   help='Evaluate Multiple Pre-Trained Combined Models.')
    parser_combined_models.add_argument('pretrained_models', type=str,
                                        help='Directory containing a folder for each pre-trained model.')
    parser_combined_models.add_argument('all_trajectories', type=str,
                                        help='Directory containing a folder for each camera, where each folder '
                                             'contains the trajectories for the associated camera.')
    parser_combined_models.add_argument('all_frame_level_anomaly_masks', type=str,
                                        help='Directory containing a folder for the anomaly masks of each camera.')
    parser_combined_models.add_argument('--video_resolution', default='856x480', type=str,
                                        help='Resolution of the trajectories\' original video(s). It should be '
                                             'specified as WxH, where W is the width and H the height of the video.')
    parser_combined_models.add_argument('--overlapping_trajectories', action='store_true')

    gp_combined_models_logging = parser_combined_models.add_argument_group('Evaluation Logging')
    gp_combined_models_logging.add_argument('--write_reconstructions', action='store_true')
    gp_combined_models_logging.add_argument('--write_bounding_boxes', action='store_true')
    gp_combined_models_logging.add_argument('--write_predictions', action='store_true')
    gp_combined_models_logging.add_argument('--write_predictions_bounding_boxes', action='store_true')
    gp_combined_models_logging.add_argument('--write_anomaly_masks', action='store_true')
    gp_combined_models_logging.add_argument('--write_mistakes', action='store_true')

    parser_combined_models.set_defaults(func=eval_combined_models)

    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    configure_gpu_resources(args.gpu_ids, args.gpu_memory_fraction)
    args.func(args)


if __name__ == '__main__':
    main()
