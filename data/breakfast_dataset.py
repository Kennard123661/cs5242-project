import os
from tqdm import tqdm
import numpy as np
import shutil

from data import get_n_video_frames, sample_video_clip, write_video_file, get_video_fps

DATA_DIR = '/mnt/Data/cs5242-dataset'
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')

SPLIT_DIR = os.path.join(DATA_DIR, 'splits')
TEST_SPLIT_FILE = os.path.join(SPLIT_DIR, 'test.split1.bundle')
TRAIN_SPLIT_FILE = os.path.join(SPLIT_DIR, 'train.split1.bundle')
OLD_MAPPING_FILE = os.path.join(SPLIT_DIR, 'mapping_bf.txt')
MAPPING_FILE = os.path.join(SPLIT_DIR, 'mapping.txt')

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_VID_DIR = os.path.join(TRAIN_DIR, 'videos')
TRAIN_LABEL_DIR = os.path.join(TRAIN_DIR, 'labels')
TRAIN_SEGMENT_DIR = os.path.join(TRAIN_DIR, 'segments')
TRAIN_SEGMENT_DICT = os.path.join(TRAIN_DIR, 'segments-dict.json')
BAD_TRAIN_SEGMENTS_FILE = os.path.join(TRAIN_DIR, 'bad-segments.txt')

TEST_DIR = os.path.join(DATA_DIR, 'test')
TEST_VID_DIR = os.path.join(TEST_DIR, 'videos')
TEST_LABEL_DIR = os.path.join(TEST_DIR, 'labels')
TEST_SEGMENT_DIR = os.path.join(TEST_DIR, 'segments')
TEST_SEGMENT_DICT = os.path.join(TEST_DIR, 'segments-dict.json')
BAD_TEST_SEGMENTS_FILE = os.path.join(TEST_DIR, 'bad-segments.txt')

VIDEO_EXT = '.avi'
LABEL_EXT = '.avi.labels'
N_CLASSES = 50


def read_mapping_file(mapping_file):
    with open(mapping_file, 'r') as f:
        content = f.readlines()
    content = [line.strip().split(' ') for line in content]

    logit_to_action_dict = np.array([line[1] for line in content], dtype=str)
    action_to_logit_dict = dict()
    for i, action in enumerate(logit_to_action_dict):
        action_to_logit_dict[action] = i
    return action_to_logit_dict, logit_to_action_dict


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


def _read_label_file(label_file):
    with open(label_file, 'r') as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]

    labels = [label.split(' ') for label in labels]
    segment_windows = [np.array(label[0].split('-'), dtype=int) for label in labels]
    segment_labels = [str(label[1]) for label in labels]
    return segment_windows, segment_labels


def _generate_video_segments(video_dir, label_dir, segment_dir):
    """ video segments are stored with the extension {name}.{segment no.}.{logit label}.avi inside segment dir """
    # the frames are in one-based indexing.
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)

    files = sorted(os.listdir(video_dir))
    action_to_logit_dict, _ = read_mapping_file(MAPPING_FILE)

    pbar = tqdm(files)
    for file in pbar:
        pbar.set_postfix({'video': file})

        video_file = os.path.join(video_dir, file)
        n_frames = get_n_video_frames(video_file)
        label_file = str(file)[:-len(VIDEO_EXT)] + LABEL_EXT
        label_file = os.path.join(label_dir, label_file)
        fps = get_video_fps(video_file=video_file)
        segment_windows, segment_actions = _read_label_file(label_file)
        for i, segment_action in enumerate(segment_actions):
            start, end = segment_windows[i]
            end = min(end, n_frames)
            if end <= start:
                continue  # no a valid segment

            frame_idxs = np.arange(start, end)
            segment_logit = action_to_logit_dict[segment_action]
            segment_name = '.'.join([file[:-len(VIDEO_EXT)], str(i), str(segment_logit)])
            segment_file = os.path.join(segment_dir, segment_name + VIDEO_EXT)
            segment_frames = sample_video_clip(video_file, n_frames, frame_idxs)
            write_video_file(segment_frames, fps, segment_file)


def _extract_train_test_segments():
    print('generating train segments')
    _generate_video_segments(TRAIN_VID_DIR, TRAIN_LABEL_DIR, TRAIN_SEGMENT_DIR)
    print('generating test segments')
    _generate_video_segments(TEST_VID_DIR, TEST_LABEL_DIR, TEST_SEGMENT_DIR)



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


def main():
    # _split_train_test_videos()
    # _extract_train_test_segments()
    pass


if __name__ == '__main__':
    main()
