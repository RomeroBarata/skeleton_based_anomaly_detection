import argparse

from tbad.visualisation import render_trajectories_skeletons, render_video_diff_heatmaps
from tbad.visualisation import render_video_diff_heatmaps_hasan, render_video_heatmaps_mpedrnn


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Functions for Visualisation of Skeletons.')

    subparsers = parser.add_subparsers(title='sub-commands', description='Valida sub-commands.')

    # Visualisation of reconstructed/predicted skeletons and their bounding boxes
    parser_skeletons = subparsers.add_parser('skeletons',
                                             help='Visualise Reconstructed/Predicted Skeletons and '
                                                  'their Bounding Boxes.')
    parser_skeletons.add_argument('frames', type=str, help='Directory containing video frames.')
    parser_skeletons.add_argument('--ground_truth_trajectories', type=str,
                                  help='Directory containing the ground-truth trajectories of people in the video.')
    parser_skeletons.add_argument('--draw_ground_truth_trajectories_skeleton', action='store_true',
                                  help='Whether to draw the ground-truth skeletons or not.')
    parser_skeletons.add_argument('--draw_ground_truth_trajectories_bounding_box', action='store_true',
                                  help='Whether to draw the bounding box of the ground-truth skeletons or not.')
    parser_skeletons.add_argument('--trajectories', type=str,
                                  help='Directory containing the reconstructed/predicted trajectories of people in '
                                       'the video.')
    parser_skeletons.add_argument('--draw_trajectories_skeleton', action='store_true',
                                  help='Whether to draw the reconstructed/predicted skeleton or not.')
    parser_skeletons.add_argument('--draw_trajectories_bounding_box', action='store_true',
                                  help='Whether to draw the bounding box of the reconstructed/predicted trajectories '
                                       'or not.')
    parser_skeletons.add_argument('--person_id', type=int, help='Draw only a specific person in the video.')
    parser_skeletons.add_argument('--draw_local_skeleton', action='store_true',
                                  help='If specified, draw local skeletons on a white background. It must be used '
                                       'in conjunction with --person_id, since it is only possible to visualise '
                                       'one pair (ground-truth, reconstructed/predicted) of local skeletons.')
    parser_skeletons.add_argument('--ground_truth_trajectories_colour', type=str, choices=['black', 'red'],
                                  help='Draw the ground-truth skeletons and bounding boxes in either black or red.'
                                       ' If not specified, colours are automatic assigned to skeletons and bounding '
                                       'boxes.')
    parser_skeletons.add_argument('--trajectories_colour', type=str, choices=['black', 'red'],
                                  help='Draw the reconstructed/predicted skeletons and bounding boxes in either '
                                       'black or red. If not specified, colours are automatic assigned to skeletons '
                                       'and bounding boxes.')
    parser_skeletons.add_argument('--write_dir', type=str,
                                  help='Directory to write rendered frames. If the specified directory does not '
                                       'exist, it will be created.')

    parser_skeletons.set_defaults(func=render_trajectories_skeletons)

    # Heatmap of the difference between two videos when the inputs are the video's frames
    parser_video_diff = subparsers.add_parser('video_diff',
                                              help='Visualise heatmap of the differences between ground-truth and '
                                                   'reconstructed/predicted frames of a video.')
    parser_video_diff.add_argument('ground_truth_frames', type=str,
                                   help='Directory containing the ground-truth frames of a video.')
    parser_video_diff.add_argument('generated_frames', type=str,
                                   help='Directory containing the reconstructed/predicted frames of a video.')
    parser_video_diff.add_argument('--skip_first_n_frames', default=0, type=int,
                                   help='In case the reconstructed/predicted frames do not include some frames in the '
                                        'beginning, these can be skipped.')
    parser_video_diff.add_argument('--write_dir', type=str,
                                   help='Directory to write heatmaps. If the specified directory does not exist, '
                                        'it will be created.')

    parser_video_diff.set_defaults(func=render_video_diff_heatmaps)

    # Heatmap of the difference between two videos for Hasan's output (227x227 numpy arrays)
    parser_video_diff_hasan = subparsers.add_parser('video_diff_hasan',
                                                    help='Visualise heatmap of the differences between ground-truth '
                                                         'and reconstructed/predicted frames of a video.')
    parser_video_diff_hasan.add_argument('ground_truth_array', type=str,
                                         help='.npy file containing the grayscale intensity values of the ground-truth '
                                              'images.')
    parser_video_diff_hasan.add_argument('generated_array', type=str,
                                         help='.npy file containing the grayscale reconstructed/predicted intensity '
                                              'values of the generated image.')
    parser_video_diff_hasan.add_argument('--write_dir', type=str,
                                         help='Directory to write heatmaps. If the specified directory does not '
                                              'exist, it will be created.')

    parser_video_diff_hasan.set_defaults(func=render_video_diff_heatmaps_hasan)

    # Heatmap of our proposed method
    parser_heatmap_mpedrnn = subparsers.add_parser('video_heatmap_mpedrnn',
                                                   help='Visualise heatmap of the skeletons in the video.')
    parser_heatmap_mpedrnn.add_argument('frames', type=str,
                                        help='Directory containing the video frames.')
    parser_heatmap_mpedrnn.add_argument('ground_truth_trajectories', type=str,
                                        help='Directory containing ground-truth trajectories of all skeletons in '
                                             'every camera and every video in the test set.')
    parser_heatmap_mpedrnn.add_argument('generated_trajectories', type=str,
                                        help='Directory containing generated trajectories of all skeletons in every '
                                             'camera and every video in the test set.')
    parser_heatmap_mpedrnn.add_argument('--camera_id', default='01', type=str,
                                        help='Which camera to plot heatmap.')
    parser_heatmap_mpedrnn.add_argument('--video_id', default='0014', type=str,
                                        help='Which video to plot heatmap.')
    parser_heatmap_mpedrnn.add_argument('--write_dir', type=str,
                                        help='Directory to write heatmaps. If the specified directory does not '
                                             'exist, it will be created.')

    parser_heatmap_mpedrnn.set_defaults(func=render_video_heatmaps_mpedrnn)

    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
