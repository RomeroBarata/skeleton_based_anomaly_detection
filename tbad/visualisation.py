import os

import cv2 as cv
import numpy as np
from sklearn.preprocessing import MinMaxScaler


COLOURS = {0: (0, 0, 0),  # Black
           1: (255, 0, 0),  # Red
           2: (0, 255, 0),  # Lime
           3: (0, 0, 255),  # Blue
           4: (255, 255, 0),  # Yellow
           5: (0, 255, 255),  # Cyan / Aqua
           6: (255, 0, 255),  # Magenta / Fuchsia
           7: (128, 128, 128),  # Gray
           8: (128, 0, 0),  # Maroon
           9: (128, 128, 0),  # Olive
           10: (0, 128, 0),  # Green
           11: (128, 0, 128),  # Purple
           12: (0, 128, 128),  # Teal
           13: (0, 0, 128),  # Navy
           14: (0, 0, 0),  # White
           15: (192, 192, 192),  # Silver
           16: (220, 20, 60),  # Crimson
           17: (255, 140, 0),  # Dark Orange
           18: (184, 134, 11),  # Dark Golden Rod
           19: (189, 183, 107),  # Dark Khaki
           20: (0, 100, 0)}  # Dark Green


def compute_bounding_box(keypoints, video_resolution, return_discrete_values=True):
    """Compute the bounding box of a set of keypoints.

    Argument(s):
        keypoints -- A numpy array, of shape (num_keypoints * 2,), containing the x and y values of each
            keypoint detected.
        video_resolution -- A numpy array, of shape (2,) and dtype float32, containing the width and the height of
            the video.

    Return(s):
        The bounding box of the keypoints represented by a 4-uple of integers. The order of the corners is: left,
        right, top, bottom.
    """
    width, height = video_resolution
    keypoints_reshaped = keypoints.reshape(-1, 2)
    x, y = keypoints_reshaped[:, 0], keypoints_reshaped[:, 1]
    x, y = x[x != 0.0], y[y != 0.0]
    try:
        left, right, top, bottom = np.min(x), np.max(x), np.min(y), np.max(y)
    except ValueError:
        # print('All joints missing for input skeleton. Returning zeros for the bounding box.')
        return 0, 0, 0, 0

    extra_width, extra_height = 0.1 * (right - left + 1), 0.1 * (bottom - top + 1)
    left, right = np.clip(left - extra_width, 0, width - 1), np.clip(right + extra_width, 0, width - 1)
    top, bottom = np.clip(top - extra_height, 0, height - 1), np.clip(bottom + extra_height, 0, height - 1)
    # left, right = left - extra_width, right + extra_width
    # top, bottom = top - extra_height, bottom + extra_height

    if return_discrete_values:
        return int(round(left)), int(round(right)), int(round(top)), int(round(bottom))
    else:
        return left, right, top, bottom


def compute_chest_centred_bounding_box(keypoints, video_resolution):
    """Compute the bounding box of a set of keypoints.

    Argument(s):
        keypoints -- A numpy array, of shape (num_keypoints * 2,), containing the x and y values of each
            keypoint detected.
        video_resolution -- A list containing the width and the height of the video.

    Return(s):
        The bounding box of the keypoints represented by a 4-uple of integers. The order of the corners is: left,
        right, top, bottom.
    """
    selected_joints_x = [10, 12, 22, 24]
    selected_joints_y = [11, 13, 23, 25]

    width, height = [float(measurement) for measurement in video_resolution]
    keypoints = np.where(keypoints == 0.0, np.nan, keypoints)

    center_x = np.nanmean(keypoints[selected_joints_x])
    center_y = np.nanmean(keypoints[selected_joints_y])

    x, y = np.hsplit(keypoints.reshape(-1, 2), indices_or_sections=2)
    left, right, top, bottom = np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)

    extra_width, extra_height = 0.1 * (right - left + 1), 0.1 * (bottom - top + 1)
    left, right = np.clip(left - extra_width, 0, width - 1), np.clip(right + extra_width, 0, width - 1)
    top, bottom = np.clip(top - extra_height, 0, height - 1), np.clip(bottom + extra_height, 0, height - 1)

    left, right, top, bottom = [int(round(corner)) for corner in (left, right, top, bottom)]
    bb_width, bb_height = right - left, bottom - top
    bb_left, bb_right = center_x - bb_width / 2, center_x + bb_width / 2
    bb_top, bb_bottom = center_y - bb_height / 2, center_y + bb_height / 2

    bb_left, bb_right, bb_top, bb_bottom = [int(round(corner)) for corner in (bb_left, bb_right, bb_top, bb_bottom)]

    return bb_left, bb_right, bb_top, bb_bottom


def insert_anomaly_mask_from_bounding_box(mask, bounding_box):
    """Insert bounding box into anomaly mask.

    Argument(s):
        mask -- Pixel-mask identifying location of anomalies. It is a numpy array of dtype uint8 and shape
            (video_height, video_width).
        bounding_box -- A 4-uple of integers identifying the bounding box. The order of the corners is left, right,
            top, bottom.

    Return(s):
        The modified mask containing the inserted bounding box.
    """
    left, right, top, bottom = bounding_box
    anomaly = np.ones((bottom - top + 1, right - left + 1), dtype=np.uint8)
    mask[top:(bottom + 1), left:(right + 1)] = anomaly

    return mask


def render_article_main_figure(write_path, frames_path, gt_trajectories_path, bbs_path,
                               trajectories_path, video_resolution):
    frames_names = sorted(os.listdir(frames_path))  # 000.jpg, 001.jpg, ...
    bbs_files = sorted(os.listdir(bbs_path))[1:2]  # 001.csv, 002.csv, ...
    trajectories_files = sorted(os.listdir(trajectories_path))[1:2]  # 001.csv, 002.csv, ...

    for person_id, bb_file_name in enumerate(bbs_files):
        gt_trajectory_file = os.path.join(gt_trajectories_path, bb_file_name)
        gt_trajectory = np.loadtxt(gt_trajectory_file, delimiter=',', ndmin=2)
        gt_trajectory_frames = gt_trajectory[:, 0].astype(np.int32)
        gt_trajectory_coordinates = gt_trajectory[:, 1:]
        gt_skeletons = dict(zip(gt_trajectory_frames, gt_trajectory_coordinates))

        bb_file = os.path.join(bbs_path, bb_file_name)
        bb = np.loadtxt(bb_file, delimiter=',', ndmin=2)
        bb_frames = bb[:, 0].astype(np.int32)
        bb_coordinates = bb[:, 1:].astype(np.int32)
        skeletons_bb = dict(zip(bb_frames, bb_coordinates))

        trajectory_file = os.path.join(trajectories_path, bb_file_name)
        trajectory = np.loadtxt(trajectory_file, delimiter=',', ndmin=2)
        trajectory_coordinates = trajectory[:, 1:]
        skeletons = dict(zip(bb_frames, trajectory_coordinates))

        frame_id_start = 272
        frame_id_end = frame_id_start + 20
        frame = cv.imread(os.path.join(frames_path, frames_names[frame_id_start]))
        output_file = os.path.join(write_path, frames_names[frame_id_start])
        gt_colour = (0, 0, 0)
        colour = (0, 0, 255)
        for frame_id in range(frame_id_start, frame_id_end, 6):
            gt_skeleton = gt_skeletons[frame_id]
            draw_skeleton(frame, gt_skeleton.reshape(-1, 2), gt_colour)

            # bb_left, bb_right, bb_top, bb_bottom = tuple(skeletons_bb[frame_id])
            # bb_center = int(round((bb_left + bb_right) / 2)), int(round((bb_top + bb_bottom) / 2))
            # cv.circle(frame, center=bb_center, radius=4, color=colour, thickness=-1)
            skeleton = skeletons[frame_id]
            draw_skeleton(frame, skeleton.reshape(-1, 2), colour)

        cv.imwrite(output_file, frame)


def render_article_main_figure_2(write_path, frames_path, gt_trajectories_path):
    frames_names = sorted(os.listdir(frames_path))  # 000.jpg, 001.jpg, ...
    gt_trajectories_files = sorted(os.listdir(gt_trajectories_path))  # 001.csv, 002.csv, ...

    frame_id = 186
    frame = cv.imread(os.path.join(frames_path, frames_names[frame_id]))
    output_file = os.path.join(write_path, frames_names[frame_id])
    # colours = [(0, 255, 0), (0, 255, 0), (0, 0, 255)] + [(0, 255, 0)] * 7
    colours = [(0, 255, 0)] * 10
    for person_id, gt_trajectory_file_name in enumerate(gt_trajectories_files):
        gt_trajectory_file = os.path.join(gt_trajectories_path, gt_trajectory_file_name)
        gt_trajectory = np.loadtxt(gt_trajectory_file, delimiter=',', ndmin=2)
        gt_trajectory_frames = gt_trajectory[:, 0].astype(np.int32)
        gt_trajectory_coordinates = gt_trajectory[:, 1:]
        gt_skeletons = dict(zip(gt_trajectory_frames, gt_trajectory_coordinates))

        gt_skeleton = gt_skeletons.get(frame_id)
        if gt_skeleton is not None:
            draw_skeleton(frame, gt_skeleton.reshape(-1, 2), colours[person_id])

    cv.imwrite(output_file, frame)


def draw_skeleton(frame, keypoints, colour, dotted=False):
    connections = [(0, 1), (0, 2), (1, 3), (2, 4),
                   (5, 7), (7, 9), (6, 8), (8, 10),
                   (11, 13), (13, 15), (12, 14), (14, 16),
                   (3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)]

    for x, y in keypoints:
        if 0 in (x, y):
            continue
        center = int(round(x)), int(round(y))
        cv.circle(frame, center=center, radius=4, color=colour, thickness=-1)

    for keypoint_id1, keypoint_id2 in connections:
        x1, y1 = keypoints[keypoint_id1]
        x2, y2 = keypoints[keypoint_id2]
        if 0 in (x1, y1, x2, y2):
            continue
        pt1 = int(round(x1)), int(round(y1))
        pt2 = int(round(x2)), int(round(y2))
        if dotted:
            draw_line(frame, pt1=pt1, pt2=pt2, color=colour, thickness=2, gap=5)
        else:
            cv.line(frame, pt1=pt1, pt2=pt2, color=colour, thickness=2)

    return None


def draw_line(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv.line(img, s, e, color, thickness)
            i += 1


def draw_poly(img, pts, color, thickness=1, style='dotted'):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        draw_line(img, s, e, color, thickness, style)


def draw_rect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    draw_poly(img, pts, color, thickness, style)


def render_trajectories_skeletons(args):
    frames_path = args.frames
    gt_trajectories_path = args.ground_truth_trajectories
    draw_gt_skeleton = args.draw_ground_truth_trajectories_skeleton
    draw_gt_bounding_box = args.draw_ground_truth_trajectories_bounding_box
    trajectories_path = args.trajectories
    draw_trajectories_skeleton = args.draw_trajectories_skeleton
    draw_trajectories_bounding_box = args.draw_trajectories_bounding_box
    specific_person_id = args.person_id
    draw_local_skeleton = args.draw_local_skeleton
    gt_trajectories_colour = args.ground_truth_trajectories_colour
    trajectories_colour = args.trajectories_colour
    write_dir = args.write_dir

    if gt_trajectories_path is None and trajectories_path is None:
        raise ValueError('At least one of --ground_truth_trajectories or --trajectories must be specified.')

    if not any([draw_gt_skeleton, draw_gt_bounding_box, draw_trajectories_skeleton, draw_trajectories_bounding_box]):
        raise ValueError('At least one of --draw_ground_truth_trajectories_skeleton, '
                         '--draw_ground_truth_trajectories_bounding_box, --draw_trajectories_skeleton or '
                         '--draw_trajectories_bounding_box must be specified.')

    if draw_local_skeleton and specific_person_id is None:
        raise ValueError('If --draw_local_skeleton is specified, a --person_id must be chosen as well.')
    elif draw_local_skeleton:
        draw_gt_skeleton = draw_trajectories_skeleton = True
        draw_gt_bounding_box = draw_trajectories_bounding_box = False

    maybe_create_dir(write_dir)

    _render_trajectories_skeletons(write_dir, frames_path, gt_trajectories_path, trajectories_path,
                                   draw_gt_skeleton, draw_trajectories_skeleton,
                                   draw_gt_bounding_box, draw_trajectories_bounding_box, specific_person_id,
                                   draw_local_skeleton, gt_trajectories_colour, trajectories_colour)

    print('Visualisation successfully rendered to %s' % write_dir)

    return None


def _render_trajectories_skeletons(write_dir, frames_path, gt_trajectories_path, trajectories_path,
                                   draw_gt_skeleton, draw_trajectories_skeleton,
                                   draw_gt_bounding_box, draw_trajectories_bounding_box, specific_person_id=None,
                                   draw_local_skeleton=False, gt_trajectories_colour=None, trajectories_colour=None):
    frames_names = sorted(os.listdir(frames_path))  # 000.jpg, 001.jpg, ...
    max_frame_id = len(frames_names)

    rendered_frames = {}
    if trajectories_path is not None:
        trajectories_files_names = sorted(os.listdir(trajectories_path))  # 001.csv, 002.csv, ...
        for trajectory_file_name in trajectories_files_names:
            person_id = int(trajectory_file_name.split('.')[0])
            if specific_person_id is not None and specific_person_id != person_id:
                continue

            if trajectories_colour is None:
                colour = COLOURS[person_id % len(COLOURS)]
            else:
                colour = (0, 0, 0) if trajectories_colour == 'black' else (0, 0, 255)

            trajectory = np.loadtxt(os.path.join(trajectories_path, trajectory_file_name), delimiter=',', ndmin=2)
            trajectory_frames = trajectory[:, 0].astype(np.int32)
            trajectory_coordinates = trajectory[:, 1:]

            for frame_id, skeleton_coordinates in zip(trajectory_frames, trajectory_coordinates):
                if frame_id >= max_frame_id:
                    break

                frame = rendered_frames.get(frame_id)
                if frame is None:
                    frame = cv.imread(os.path.join(frames_path, frames_names[frame_id]))
                    if draw_local_skeleton:
                        frame = np.full_like(frame, fill_value=255)

                if draw_trajectories_skeleton:
                    if draw_local_skeleton:
                        height, width = frame.shape[:2]
                        left, right, top, bottom = compute_simple_bounding_box(skeleton_coordinates)
                        bb_center = np.array([(left + right) / 2, (top + bottom) / 2], dtype=np.float32)
                        target_center = np.array([3 * width / 4, height / 2], dtype=np.float32)
                        displacement_vector = target_center - bb_center
                        draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2) + displacement_vector,
                                      colour=colour, dotted=True)
                    else:
                        draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2), colour=colour, dotted=True)

                if draw_trajectories_bounding_box:
                    left, right, top, bottom = compute_simple_bounding_box(skeleton_coordinates)
                    bb_center = int(round((left + right) / 2)), int(round((top + bottom) / 2))
                    cv.circle(frame, center=bb_center, radius=4, color=colour, thickness=-1)
                    draw_rect(frame, pt1=(left, top), pt2=(right, bottom), color=colour, thickness=3, style='dotted')

                rendered_frames[frame_id] = frame

    if gt_trajectories_path is not None:
        gt_trajectories_files_names = sorted(os.listdir(gt_trajectories_path))
        for gt_trajectory_file_name in gt_trajectories_files_names:
            person_id = int(gt_trajectory_file_name.split('.')[0])
            if specific_person_id is not None and specific_person_id != person_id:
                continue

            if gt_trajectories_colour is None:
                colour = COLOURS[person_id % len(COLOURS)]
            else:
                colour = (0, 0, 0) if gt_trajectories_colour == 'black' else (0, 0, 255)

            gt_trajectory = np.loadtxt(os.path.join(gt_trajectories_path, gt_trajectory_file_name),
                                       delimiter=',', ndmin=2)
            gt_trajectory_frames = gt_trajectory[:, 0].astype(np.int32)
            gt_trajectory_coordinates = gt_trajectory[:, 1:]

            for frame_id, skeleton_coordinates in zip(gt_trajectory_frames, gt_trajectory_coordinates):
                frame = rendered_frames.get(frame_id)
                if frame is None:
                    frame = cv.imread(os.path.join(frames_path, frames_names[frame_id]))
                    if draw_local_skeleton:
                        frame = np.full_like(frame, fill_value=255)

                skeleton_is_not_null = np.any(skeleton_coordinates)
                if draw_gt_skeleton and skeleton_is_not_null:
                    if draw_local_skeleton:
                        height, width = frame.shape[:2]
                        left, right, top, bottom = compute_simple_bounding_box(skeleton_coordinates)
                        bb_center = np.array([(left + right) / 2, (top + bottom) / 2], dtype=np.float32)
                        target_center = np.array([width / 4, height / 2], dtype=np.float32)
                        displacement_vector = target_center - bb_center
                        keypoints = np.where(skeleton_coordinates == 0.0, np.nan, skeleton_coordinates).reshape(-1, 2)
                        keypoints += displacement_vector
                        keypoints = np.where(np.isnan(keypoints), 0.0, keypoints)
                        draw_skeleton(frame, keypoints=keypoints, colour=colour)
                    else:
                        draw_skeleton(frame, keypoints=skeleton_coordinates.reshape(-1, 2), colour=colour)

                if draw_gt_bounding_box and skeleton_is_not_null:
                    left, right, top, bottom = compute_simple_bounding_box(skeleton_coordinates)
                    bb_center = int(round((left + right) / 2)), int(round((top + bottom) / 2))
                    cv.circle(frame, center=bb_center, radius=4, color=colour, thickness=-1)
                    cv.rectangle(frame, pt1=(left, top), pt2=(right, bottom), color=colour, thickness=3)

                rendered_frames[frame_id] = frame

    for frame_id, frame_name in enumerate(frames_names):
        frame = rendered_frames.get(frame_id)
        if frame is None:
            frame = cv.imread(os.path.join(frames_path, frame_name))
            if draw_local_skeleton:
                frame = np.full_like(frame, fill_value=255)
        cv.imwrite(os.path.join(write_dir, frame_name), img=frame)


def maybe_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True

    return False


def compute_simple_bounding_box(skeleton):
    x = skeleton[::2]
    x = np.where(x == 0.0, np.nan, x)
    left, right = int(round(np.nanmin(x))), int(round(np.nanmax(x)))
    y = skeleton[1::2]
    y = np.where(y == 0.0, np.nan, y)
    top, bottom = int(round(np.nanmin(y))), int(round(np.nanmax(y)))

    return left, right, top, bottom


def render_video_diff_heatmaps(args):
    ground_truth_frames_dir = args.ground_truth_frames
    generated_frames_dir = args.generated_frames
    skip_first_n_frames = args.skip_first_n_frames
    write_dir = args.write_dir

    maybe_create_dir(write_dir)

    ground_truth_frames_names = sorted(os.listdir(ground_truth_frames_dir))[skip_first_n_frames:]
    generated_frames_names = sorted(os.listdir(generated_frames_dir))

    for ground_truth_frame_name, generated_frame_name in zip(ground_truth_frames_names, generated_frames_names):
        ground_truth_frame = cv.imread(os.path.join(ground_truth_frames_dir, ground_truth_frame_name))
        ground_truth_frame = cv.cvtColor(ground_truth_frame, code=cv.COLOR_BGR2GRAY)

        generated_frame = cv.imread(os.path.join(generated_frames_dir, generated_frame_name))
        generated_frame = cv.cvtColor(generated_frame, code=cv.COLOR_BGR2GRAY)

        diff_frame = np.abs(generated_frame - ground_truth_frame)

        jet_frame = cv.applyColorMap(diff_frame, colormap=cv.COLORMAP_JET)

        cv.imwrite(os.path.join(write_dir, generated_frame_name), img=jet_frame)

    print('Heatmaps successfully rendered and written to %s' % write_dir)


def render_video_diff_heatmaps_hasan(args):
    ground_truth_arr_file = args.ground_truth_array
    generated_arr_file = args.generated_array
    write_dir = args.write_dir

    maybe_create_dir(write_dir)

    ground_truth_arr = np.load(ground_truth_arr_file).astype(np.float32)
    ground_truth_arr = (ground_truth_arr + 1) * 127.5
    generated_arr = np.load(generated_arr_file).astype(np.float32)
    generated_arr = (generated_arr + 1) * 127.5

    diff_arr = np.abs(generated_arr - ground_truth_arr)
    diff_arr = diff_arr.astype(np.uint8)

    for frame_id, diff_frame in enumerate(diff_arr):
        jet_frame = cv.applyColorMap(diff_frame, colormap=cv.COLORMAP_JET)
        cv.imwrite(os.path.join(write_dir, '%.3d.jpg' % frame_id), img=jet_frame)

    print('Heatmaps successfully rendered and written to %s' % write_dir)


def render_video_heatmaps_mpedrnn(args):
    frames_path = args.frames
    ground_truth_trajectories_dir = args.ground_truth_trajectories
    generated_trajectories_dir = args.generated_trajectories
    camera_id = args.camera_id
    video_id = args.video_id
    write_dir = args.write_dir

    camera_dirs = sorted(os.listdir(generated_trajectories_dir))

    mses = []
    for camera_dir in camera_dirs:
        video_dirs = sorted(os.listdir(os.path.join(generated_trajectories_dir, camera_dir)))
        for video_dir in video_dirs:
            generated_trajectories_files = os.listdir(os.path.join(generated_trajectories_dir, camera_dir, video_dir))
            for generated_trajectory_file in generated_trajectories_files:
                generated_trajectory_file_path = os.path.join(generated_trajectories_dir, camera_dir,
                                                              video_dir, generated_trajectory_file)
                generated_trajectory = np.loadtxt(generated_trajectory_file_path, delimiter=',', ndmin=2)
                generated_trajectory_frames = generated_trajectory[:, 0].astype(np.int32)
                generated_trajectory_coords = generated_trajectory[:, 1:]
                generated_trajectory_coords = np.where(generated_trajectory_coords == 0.0, np.nan,
                                                       generated_trajectory_coords)
                generated_skeletons = dict(zip(generated_trajectory_frames, generated_trajectory_coords))

                ground_truth_trajectory_file_path = os.path.join(ground_truth_trajectories_dir, camera_dir,
                                                                 video_dir, generated_trajectory_file)
                ground_truth_trajectory = np.loadtxt(ground_truth_trajectory_file_path, delimiter=',', ndmin=2)
                ground_truth_trajectory_frames = ground_truth_trajectory[:, 0].astype(np.int32)
                ground_truth_trajectory_coords = ground_truth_trajectory[:, 1:]
                ground_truth_trajectory_coords = np.where(ground_truth_trajectory_coords == 0.0, np.nan,
                                                          ground_truth_trajectory_coords)
                ground_truth_skeletons = dict(zip(ground_truth_trajectory_frames, ground_truth_trajectory_coords))

                for generated_frame, generated_skeleton in generated_skeletons.items():
                    ground_truth_skeleton = ground_truth_skeletons.get(generated_frame)
                    if ground_truth_skeleton is None:
                        continue
                    num_non_null_coords = np.sum(np.logical_not(np.isnan(ground_truth_skeleton)))
                    if num_non_null_coords == 0:
                        continue
                    mse = np.nansum((generated_skeleton - ground_truth_skeleton) ** 2) / num_non_null_coords
                    mses.append(mse)

    mses = np.array(mses)
    mse_cutoff = np.quantile(mses, 0.95)
    mses[mses >= mse_cutoff] = mse_cutoff
    scaler = MinMaxScaler(feature_range=(50, 255))
    scaler.fit(mses.reshape(-1, 1))

    maybe_create_dir(write_dir)

    frames_names = os.listdir(frames_path)
    num_frames = len(frames_names)
    height, width = cv.imread(os.path.join(frames_path, frames_names[0])).shape[:2]
    canvas = np.full(shape=(num_frames, height, width), fill_value=0, dtype=np.uint8)

    generated_trajectories_files = os.listdir(os.path.join(generated_trajectories_dir, camera_id, video_id))
    for generated_trajectory_file in generated_trajectories_files:
        generated_trajectory_file_path = os.path.join(generated_trajectories_dir, camera_id,
                                                      video_id, generated_trajectory_file)
        generated_trajectory = np.loadtxt(generated_trajectory_file_path, delimiter=',', ndmin=2)
        generated_trajectory_frames = generated_trajectory[:, 0].astype(np.int32)
        generated_trajectory_coords = generated_trajectory[:, 1:]
        generated_trajectory_coords = np.where(generated_trajectory_coords == 0.0, np.nan,
                                               generated_trajectory_coords)
        generated_skeletons = dict(zip(generated_trajectory_frames, generated_trajectory_coords))

        ground_truth_trajectory_file_path = os.path.join(ground_truth_trajectories_dir, camera_id,
                                                         video_id, generated_trajectory_file)
        ground_truth_trajectory = np.loadtxt(ground_truth_trajectory_file_path, delimiter=',', ndmin=2)
        ground_truth_trajectory_frames = ground_truth_trajectory[:, 0].astype(np.int32)
        ground_truth_trajectory_coords = ground_truth_trajectory[:, 1:]
        ground_truth_trajectory_coords = np.where(ground_truth_trajectory_coords == 0.0, np.nan,
                                                  ground_truth_trajectory_coords)
        ground_truth_skeletons = dict(zip(ground_truth_trajectory_frames, ground_truth_trajectory_coords))

        for frame_id in ground_truth_skeletons.keys() & generated_skeletons.keys():
            ground_truth_skeleton = ground_truth_skeletons[frame_id]
            generated_skeleton = generated_skeletons[frame_id]

            num_non_null_coords = np.sum(np.logical_not(np.isnan(ground_truth_skeleton)))
            if num_non_null_coords == 0:
                continue
            mse = np.nansum((generated_skeleton - ground_truth_skeleton) ** 2) / num_non_null_coords
            mse = scaler.transform(mse.reshape(-1, 1)).reshape(-1)
            ground_truth_skeleton_ = np.where(np.isnan(ground_truth_skeleton), 0.0, ground_truth_skeleton)
            draw_skeleton(canvas[frame_id], ground_truth_skeleton_.reshape(-1, 2), colour=mse)

    for frame_id, frame in enumerate(canvas):
        jet_frame = cv.applyColorMap(frame, colormap=cv.COLORMAP_JET)
        cv.imwrite(os.path.join(write_dir, '%.3d.jpg' % frame_id), img=jet_frame)

    print('Heatmaps successfully rendered and written to %s' % write_dir)
