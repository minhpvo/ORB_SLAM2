import os
import argparse
import tarfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", help="train or test")
    parser.add_argument("--data_id", help="PXX")
    args = parser.parse_args()

    data_path_prefix = '/media/hdd1/guanjq/EPIC_KITCHENS_2018/frames_rgb_flow/flow/'
    data_split = args.data_split
    root_path = os.path.join(data_path_prefix, data_split)
    if args.data_id == 'A':
        data_path_list = [os.path.join(root_path, f) for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    else:
        data_path_list = [os.path.join(root_path, args.data_id)]

    # unzip file if needed
    for data_path in data_path_list:
        print('Data Path: ', data_path)
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
