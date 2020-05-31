import os
import numpy as np
import argparse
import tarfile
# import data_process.extract_valid_positions


def create_img_list(root_path, meta_path, n_start, n_end, fps=5, ori_fps=60):
    save_file = os.path.join(root_path, meta_path, 'rgb_{}_{}.txt'.format(n_start, n_end))
    print('Create image list file: {}\n'.format(save_file))
    with open(save_file, 'w') as f:
        all_filename = [f for f in os.listdir(root_path) if f.endswith('.jpg')]
        all_filename.sort()
        interval = int(ori_fps / fps)
        if interval == 0:
            interval = 1
        all_filename = all_filename[n_start:n_end:interval]
        for filename in all_filename:
            tframe = float(filename[6:-4]) / ori_fps
            f.write('{0:.6f} {1}\n'.format(tframe, os.path.join(root_path, filename)))


def run_cpp(settings_file, timestamped_frames_file, save_prefix, use_viewer):
    vocabulary_file = 'Vocabulary/ORBvoc.txt'
    cmd = './Examples/Monocular/extract_mono_epic {}'.format(vocabulary_file)

    if not os.path.isfile(settings_file):
        raise IOError("Settings file doesn't exist: {}".format(settings_file))
    if not os.path.isfile(timestamped_frames_file):
        raise IOError("Timestamped frames file doesn't exist: {}".format(timestamped_frames_file))

    cmd += ' {} '.format(settings_file)
    cmd += ' {} '.format(timestamped_frames_file)
    cmd += ' {} '.format(save_prefix)
    cmd += ' {} '.format(use_viewer)
    print("Running command: '{}' \n".format(cmd))
    return os.system(cmd)


def get_original_fps(sub_id):
    ori_fps = 60
    if sub_id in ['P09_07', 'P09_08', 'P10_01', 'P10_04', 'P11_01', 'P18_02', 'P18_03']:
        ori_fps = 30
    elif sub_id in ['P17_01', 'P17_02', 'P17_03', 'P17_04']:
        ori_fps = 48
    elif sub_id in ['P18_09']:
        ori_fps = 90
    return ori_fps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", default="train", help="train or test")
    parser.add_argument("--data_id", help="PXX")
    parser.add_argument("--sub_data_id", default="R", help="XX")
    parser.add_argument("--metadata_path", default="pos_info")
    parser.add_argument("--fps", default=60, type=float, help="sample frames with fps")
    parser.add_argument("--ns", default=0, type=int, help="start frame")
    parser.add_argument("--ne", default=-1, type=int, help="end frame")
    parser.add_argument("--use_viewer", default=1, help="whether to use viewer")
    parser.add_argument("--only_extract_valid", help="only extract valid frames from existing files", action="store_true")
    args = parser.parse_args()

    data_path_prefix = '/media/hdd1/guanjq/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/'
    data_split = args.data_split
    data_path = os.path.join(data_path_prefix, data_split, args.data_id)
    print('Data Path: ', data_path)

    # unzip file if needed
    to_unzip_file = [f for f in os.listdir(data_path) if f.endswith('.tar')]
    if not to_unzip_file:
        print('There is no file to be unzipped!')
    else:
        print('unzip {} file(s): '.format(len(to_unzip_file)), to_unzip_file)
        for file in to_unzip_file:
            file_dir = file.split('.')[0]
            tar = tarfile.open(os.path.join(data_path, file), "r:")
            tar.extractall(os.path.join(data_path, file_dir))
            tar.close()
            print('unzip {} and remove it.'.format(file))
            os.remove(os.path.join(data_path, file))

    sub_id_list = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f)) and f.startswith('P')]
    sub_id_list.sort()
    print('Sub-dataset to be processed: ', sub_id_list)
    ns, ne = args.ns, args.ne

    if args.sub_data_id == "R":
        rand_id = np.random.randint(0, len(sub_id_list))
        sub_id = sub_id_list[rand_id]
        process_id_list = [sub_id]
    elif args.sub_data_id == "A":
        process_id_list = sub_id_list
    else:
        sub_id = args.data_id + '_' + args.sub_data_id
        if os.path.isdir(os.path.join(data_path, sub_id)):
            process_id_list = [sub_id]
        else:
            raise ValueError('Not found sub data id!')

    for idx, sub_id in enumerate(process_id_list):
        print('\nStart processing {} of {}: {}'.format(idx + 1, len(process_id_list), sub_id))
        sub_data_path = os.path.join(data_path, sub_id)
        metadata_path = args.metadata_path
        if not os.path.isdir(os.path.join(sub_data_path, metadata_path)):
            os.mkdir(os.path.join(sub_data_path, metadata_path))

        settings = "config.yaml"
        sub_ns, sub_ne = ns, ne
        if ne == -1:
            sub_ne = len([f for f in os.listdir(sub_data_path) if f.endswith('.jpg')])
        frames = os.path.join(sub_data_path, metadata_path, 'rgb_{}_{}.txt'.format(sub_ns, sub_ne))
        prefix = os.path.join(sub_data_path, metadata_path, 'PosInfo_{}_{}'.format(sub_ns, sub_ne))
        nframe = 5
        threshold = 0.01

        if not args.only_extract_valid:
            # create img list file
            ori_fps = get_original_fps(sub_id)
            create_img_list(sub_data_path, metadata_path, sub_ns, sub_ne, fps=args.fps, ori_fps=ori_fps)

            # run orb_slam to extract positions
            run_cpp(settings, frames, prefix, args.use_viewer)

        # extract valid frames
        # extract_valid_positions.extract(nframe, threshold, prefix, os.path.join(sub_data_path, 'pos_info'))
