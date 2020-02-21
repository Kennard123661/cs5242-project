import os
import json
import numpy as np

KINETICS_DIR = '/mnt/Data/kinetics-dataset'
LABEL_MAPPING_FILE = os.path.join(KINETICS_DIR, 'label-mappings.csv')

TRAIN_JSON_FILE = os.path.join(KINETICS_DIR, 'train.json')
TRAIN_VIDEO_DIR = os.path.join(KINETICS_DIR, 'train-videos')
TRAIN_VIDEO_DL_SCRIPT = os.path.join(KINETICS_DIR, 'download-train-video.sh')
TRAIN_CLIP_DIR = os.path.join(KINETICS_DIR, 'train-clips')


def read_kinetics_json(json_file):
    assert os.path.exists(json_file)
    with open(json_file, 'r') as f:
        contents = json.load(f)
    return contents


def _generate_download_script(kinetics_json, save_file, video_dir, n_downloads=100):
    video_dict = read_kinetics_json(kinetics_json)

    scripts = []
    template = 'youtube-dl --postprocessor-args "-ss {0} -to {1}" -o "{2}" "{3}"'
    for i, [v_id, v_dict] in enumerate(video_dict.items(), 1):
        url = v_dict['url']
        annotations = v_dict['annotations']
        # label = str(annotations['label'])
        start, end = np.array(annotations['segment']).astype(float).astype(int)

        start_h = start // 3600
        start_m = (start % 3600) // 60
        start_s = (start % 60)

        end_h = end // 3600
        end_m = (end % 3600) // 60
        end_s = (end % 60)

        start_time = '{0:02d}:{1:02d}:{2:02d}'.format(start_h, start_m, start_s)
        end_time = '{0:02d}:{1:02d}:{2:02d}'.format(end_h, end_m, end_s)
        script = template.format(start_time, end_time, os.path.join(video_dir, v_id), url)
        scripts.append(script)

        if i == n_downloads:
            break
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    with open(save_file, 'w') as f:
        for script in scripts:
            f.write(script + '\n')


def _read_label_mappings():
    with open(LABEL_MAPPING_FILE, 'r') as f:
        mappings = f.readlines()[1:]
    mappings = np.array([mapping.strip().split(',') for mapping in mappings]).astype(str)
    words = mappings[:, 1]

    label_word_dict = mappings
    word_label_dict = dict()
    for i, word in enumerate(words):
        word_label_dict[word] = i
    return label_word_dict, word_label_dict


def get_train_data():
    _, word_label_mapping = _read_label_mappings()
    train_videos = sorted(os.listdir(TRAIN_VIDEO_DIR))
    train_ids = [vid.split('.')[0] for vid in train_videos]
    train_dict = read_kinetics_json(json_file=TRAIN_JSON_FILE)
    labels = []
    for v_id in train_ids:
        v_dict = train_dict[v_id]
        word = str(v_dict['annotations']['label'])
        label = word_label_mapping[word]
        labels.append(label)
    videos = [os.path.join(TRAIN_VIDEO_DIR, vid) for vid in train_videos]
    return videos, labels


def main():
    # train_dict = read_kinetics_json(TRAIN_JSON_FILE)
    # _generate_download_script(TRAIN_JSON_FILE, TRAIN_VIDEO_DL_SCRIPT, TRAIN_VIDEO_DIR)
    print(get_train_data())
    pass


if __name__ == '__main__':
    main()
