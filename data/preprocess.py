import os
import shutil
import numpy as np
import pickle
from tqdm import tqdm
from data import VIDEO_DIR, TRAIN_SPLIT_FILE, TEST_SPLIT_FILE, VIDEO_EXT, LABEL_EXT
from data import TRAIN_VID_DIR, TEST_VID_DIR, TRAIN_LABEL_DIR, TEST_LABEL_DIR
from data import TRAIN_SEGMENT_DIR, TRAIN_SEGMENT_DICT, TEST_SEGMENT_DIR, TEST_SEGMENT_DICT
from data import read_label_file, read_mapping_file, read_segment_from_video
from data import OLD_MAPPING_FILE, MAPPING_FILE


def _read_split_file(split_file):
    with open(split_file) as f:
        videos = f.readlines()

    videos = videos[1:]
    videos = [video.strip() for video in videos]
    videos = [(video.split('/'))[-1] for video in videos]
    videos = [video[:-4] for video in videos]
    videos = [video.split('_') for video in videos]
    return videos


def _get_split_dict(split_file):
    split_vids = _read_split_file(split_file)
    split_dict = dict()

    for vid in split_vids:
        if vid[0] not in split_dict.keys():
            split_dict[vid[0]] = dict()

        if vid[1] == 'stereo01':
            vid[1] = 'stereo'

        part_dict = split_dict[vid[0]]
        if vid[1] not in part_dict:
            part_dict[vid[1]] = []

        camera_dict = part_dict[vid[1]]
        camera_dict.append('_'.join(vid[2:]))
    return split_dict


def _get_video_list():
    print('Retrieving list of videos...')
    videos = []

    parts = os.listdir(VIDEO_DIR)
    for part in parts:
        part_dir = os.path.join(VIDEO_DIR, part)
        cameras = os.listdir(part_dir)

        for camera in cameras:
            camera_dir = os.path.join(part_dir, camera)
            files = os.listdir(camera_dir)
            files = [file for file in files if file[-7:] != '.labels']
            videos += files
    return videos


def _print_video_exts():
    """ prints the list of video annotations, video extension is .avi """
    videos = _get_video_list()
    exts = [vid.split('.')[-1] for vid in videos]
    exts = np.unique(exts)
    print(exts)


def _split_train_test_videos():
    def split_videos(video_dir, label_dir, split_dict):
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        pbar = tqdm(split_dict.items())
        for part, part_dict in pbar:
            pbar.set_postfix({'part': part})
            for camera, files in part_dict.items():
                for file in files:
                    if camera == 'stereo':
                        original_vid = os.path.join(VIDEO_DIR, part, camera, file + '_ch1' + VIDEO_EXT)
                        original_label = os.path.join(VIDEO_DIR, part, camera, file + '_ch1' + LABEL_EXT)

                        if not (os.path.exists(original_label) and os.path.exists(original_vid)):
                            original_vid = os.path.join(VIDEO_DIR, part, camera, file + '_ch0' + VIDEO_EXT)
                            original_label = os.path.join(VIDEO_DIR, part, camera, file + '_ch0' + LABEL_EXT)
                    else:
                        original_vid = os.path.join(VIDEO_DIR, part, camera, file + VIDEO_EXT)
                        original_label = os.path.join(VIDEO_DIR, part, camera, file + LABEL_EXT)

                    assert os.path.exists(original_vid), print(original_vid)
                    assert os.path.exists(original_label), print(original_label)

                    if camera == 'stereo':
                        copy_vid = os.path.join(video_dir, '.'.join([part, camera + '01', file + VIDEO_EXT]))
                        copy_label = os.path.join(label_dir, '.'.join([part, camera + '01', file + LABEL_EXT]))
                    else:
                        copy_vid = os.path.join(video_dir, '.'.join([part, camera, file + VIDEO_EXT]))
                        copy_label = os.path.join(label_dir, '.'.join([part, camera, file + LABEL_EXT]))

                    if not os.path.exists(copy_vid):
                        shutil.copy2(original_vid, copy_vid)
                    if not os.path.exists(copy_label):
                        shutil.copy2(original_label, copy_label)
    print('generating train videos and labels')
    train_split = _get_split_dict(TRAIN_SPLIT_FILE)
    split_videos(TRAIN_VID_DIR, TRAIN_LABEL_DIR, train_split)

    print('generating test videos and labels')
    test_split = _get_split_dict(TEST_SPLIT_FILE)
    split_videos(TEST_VID_DIR, TEST_LABEL_DIR, test_split)


def _extract_train_test_segments():
    def generate_segments(video_dir, label_dir, segment_dir):
        # the frames are in one-based indexing.
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir)

        files = sorted(os.listdir(video_dir))
        action_to_logit_dict, _ = read_mapping_file(MAPPING_FILE)

        all_dict = dict()
        pbar = tqdm(files)
        for file in pbar:
            pbar.set_postfix({'video': file})

            vid_file = os.path.join(video_dir, file)
            label_file = str(file)[:-len(VIDEO_EXT)] + LABEL_EXT
            label_file = os.path.join(label_dir, label_file)

            segment_windows, segment_actions = read_label_file(label_file)
            for i, segment_action in enumerate(segment_actions):
                segment_window = segment_windows[i]
                segment_logit = action_to_logit_dict[segment_action]
                segment_name = '.'.join([file[:-len(VIDEO_EXT)], str(i)])
                segment_file = os.path.join(segment_dir, segment_name + '.npy')
                if not os.path.exists(segment_file):
                    segment_frames = read_segment_from_video(vid_file, segment_window)
                    np.save(segment_file, segment_frames)

                segment_dict = {
                    'action': segment_action,
                    'label': segment_logit,
                    'window': segment_window,
                    'vid-file': vid_file,
                    'label-file': label_file,
                    'segment-file': segment_file
                }
                all_dict[segment_name] = segment_dict
        return all_dict
    train_segments_dict = generate_segments(TRAIN_VID_DIR, TRAIN_LABEL_DIR, TRAIN_SEGMENT_DIR)
    with open(TRAIN_SEGMENT_DICT, 'wb') as f:
        pickle.dump(train_segments_dict, f)
    test_segments_dict = generate_segments(TEST_VID_DIR, TEST_LABEL_DIR, TEST_SEGMENT_DIR)
    with open(TEST_SEGMENT_DICT, 'wb') as f:
        pickle.dump(test_segments_dict, f)


def _generate_mapping_file():
    def update_mapping(mapping, label_dir):
        label_files = os.listdir(label_dir)
        label_files = [os.path.join(label_dir, file) for file in label_files]
        for file in label_files:
            _, segment_labels = read_label_file(file)
            for label in segment_labels:
                if label not in mapping.keys():
                    mapping[label] = len(mapping.keys())
        return mapping
    updated_mapping, _ = read_mapping_file(OLD_MAPPING_FILE)
    updated_mapping = update_mapping(updated_mapping, label_dir=TRAIN_LABEL_DIR)
    updated_mapping = update_mapping(updated_mapping, label_dir=TEST_LABEL_DIR)
    with open(MAPPING_FILE, 'w') as f:
        for key, value in updated_mapping.items():
            f.write(' '.join([str(value), str(key)]) + '\n')


def main():
    _split_train_test_videos()
    _generate_mapping_file()
    _extract_train_test_segments()


if __name__ == '__main__':
    main()
