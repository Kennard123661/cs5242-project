import os
import numpy as np
from tqdm import tqdm
if __name__ == '__main__':
    import sys
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(base_dir)

from data import DATA_DIR
import data.breakfast_data as breakfast_data
VMZ_DIR = os.path.join(DATA_DIR, 'vmz')
PRETRAINED_MODEL_DIR = os.path.join(VMZ_DIR, 'pretrained-models')

VMZ_BREAKFAST_DIR = os.path.join(VMZ_DIR, 'breakfast')
TRAIN_BREAKFAST_CSV = os.path.join(VMZ_BREAKFAST_DIR, 'train.csv')
TEST_BREAKFAST_CSV = os.path.join(VMZ_BREAKFAST_DIR, 'test.csv')

VMZ_FEATURE_DIR = os.path.join(VMZ_DIR, 'features')


def _generate_feature_extraction_file(video_dir, label_dir, video_len_dir, save_file, frame_stride=8, min_clip_len=8):
    print('INFO: generating feature extraction list csv {0} from {1}\' videos'.format(save_file, video_dir))
    if os.path.exists(save_file):
        print('WARNING: {0} exists. delete {0} and rerun to remake a new save file'.format(save_file))
        return

    videos = os.listdir(video_dir)
    video_files = [os.path.join(video_dir, video) for video in videos]
    label_files = [os.path.join(label_dir, video + '.labels') for video in videos]
    video_len_files = [os.path.join(video_len_dir, video + '.npy') for video in videos]
    action_to_logit_dict, _ = breakfast_data._read_mapping_file(breakfast_data.MAPPING_FILE)

    feature_extract_list = []
    pbar = tqdm(video_files)
    v_idx = 0
    for i, video_file in enumerate(pbar):
        label_file = label_files[i]
        video_len_file = video_len_files[i]

        video_len = np.load(video_len_file)
        segment_windows, segment_labels = breakfast_data.read_label_file(label_file)

        seg_idx = 0
        start_frames = np.arange(0, video_len-min_clip_len, frame_stride)
        for start_frame in start_frames:
            window_start, window_end = segment_windows[seg_idx]
            assert start_frame >= (window_start - 1), 'start frame at {0}, window start at {1}'.format(start_frame,
                                                                                                       window_start)
            if start_frame >= (window_end - 1):
                seg_idx += 1
            window_label = segment_labels[seg_idx]
            window_label = action_to_logit_dict[window_label]

            # format: video_file,label,start_frm,video_id
            out = ','.join([video_file, str(window_label), str(start_frame), str(v_idx)])
            feature_extract_list.append(out)
            v_idx += 1

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    with open(save_file, 'w') as f:
        f.write('org_video,label,start_frm,video_id\n')
        for line in feature_extract_list:
            f.write(line + '\n')


def _generate_train_test_csv_for_feature_extraction(frame_stride=8, min_clip_len=8):
    _generate_feature_extraction_file(breakfast_data.TRAIN_VIDEO_DIR, breakfast_data.TRAIN_LABEL_DIR,
                                      breakfast_data.TRAIN_VIDEO_N_FRAMES_DIR, TRAIN_BREAKFAST_CSV,
                                      frame_stride=frame_stride, min_clip_len=min_clip_len)
    _generate_feature_extraction_file(breakfast_data.TEST_VIDEO_DIR, breakfast_data.TEST_LABEL_DIR,
                                      breakfast_data.TEST_VIDEO_N_FRAMES_DIR, TEST_BREAKFAST_CSV,
                                      frame_stride=frame_stride, min_clip_len=min_clip_len)


def main():
    _generate_train_test_csv_for_feature_extraction()


if __name__ == '__main__':
    main()
