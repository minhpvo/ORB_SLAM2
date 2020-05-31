import os
import numpy as np
import pandas as pd
import dill
import argparse
from ego_data import EgoData
from extract_valid_positions import extract


def get_heading(q):
    hq = q.copy()
    hq[1] = 0
    hq[2] = 0
    hq /= np.linalg.norm(hq)
#     return 2 * math.acos(hq[0])
    return hq


def toRotMatrix(q):
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    R = np.zeros([3, 3])
    R[0, 0] = 1 - 2 * q2 ** 2 - 2 * q3 ** 2
    R[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    R[0, 2] = 2 * q1 * q3 - 2 * q0 * q2
    R[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    R[1, 1] = 1 - 2 * q1 ** 2 - 2 * q3 ** 2
    R[1, 2] = 2 * q2 * q3 + 2 * q0 * q1
    R[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    R[2, 1] = 2 * q2 * q3 - 2 * q0 * q1
    R[2, 2] = 1 - 2 * q1 ** 2 - 2 * q2 ** 2
    return R

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", help="train or test")
    parser.add_argument("--data_id", help="PXX")
    # parser.add_argument("--sub_data_id", default="R", help="XX")
    parser.add_argument("--make_fav", help="make frame action vector", action="store_true")
    parser.add_argument("--save_root", help="save root path")
    args = parser.parse_args()

    annotations_root = '/media/hdd1/guanjq/EPIC_KITCHENS_2018/annotations/'
    fav_save_path = os.path.join(args.save_root, "frame_action")

    if args.make_fav:
        print('Making frame action vectors ...')
        df = pd.read_pickle(os.path.join(annotations_root, 'EPIC_train_action_labels.pkl'))
        video_info = pd.read_csv(os.path.join(annotations_root, 'video_frames_info.csv'), index_col=0)

        frame_action = pd.DataFrame(columns=['action'])
        for video_id in video_info.index:
            frame_action.loc[video_id] = [[[[-1, -1]]] * video_info.loc[video_id, 'num_frames']]

        # construct frame action vectors
        for idx, row in df.iterrows():
            vec = frame_action.loc[row.video_id].action
            verb = row.verb_class
            noun = row.noun_class
            for frame_idx in range(row.start_frame, row.stop_frame + 1):
                frame_vec = vec[frame_idx]
                if [-1, -1] == frame_vec[0]:
                    frame_vec = [[verb, noun]]
                else:
                    frame_vec.append([verb, noun])
                vec[frame_idx] = frame_vec

        with open(fav_save_path, "wb") as f:
            dill.dump(frame_action, f)

    else:
        frame_action = dill.load(open(fav_save_path, "rb"))

    all_fps = pd.read_csv(os.path.join(annotations_root, 'EPIC_video_info.csv'))
    data_root_path = '/media/hdd1/guanjq/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/'

    if args.data_id == 'A':
        data_id_list = [f for f in os.listdir(os.path.join(data_root_path, args.data_split)) if f.startswith('P')]
        data_id_list.sort()
        # data_id_list = data_id_list[:16]
    else:
        data_id_list = [args.data_id]

    # parameters to be adjusted
    n_frame = 5
    threshold = 0.01
    example_past_time = 2
    example_future_time = 5
    example_time = example_past_time + example_future_time

    total_examples = 0
    for data_idx, data_id in enumerate(data_id_list):
        print('\n{} {} of {}'.format(data_id, data_idx + 1, len(data_id_list)))
        example_list = []
        data_path = os.path.join(data_root_path, args.data_split, data_id)
        sub_id_list = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
        sub_id_list.sort()

        for sub_idx, sub_id in enumerate(sub_id_list):
            print('\n{} {} of {}'.format(sub_id, sub_idx + 1, len(sub_id_list)))
            sub_data_path = os.path.join(data_path, sub_id)
            metadata_dir = os.path.join(sub_data_path, 'pos_info')

            if not os.path.exists(metadata_dir):
                print('{} not exists!'.format(metadata_dir))
                continue

            # num_frames  = len([f for f in os.listdir(sub_data_path) if f.endswith('.jpg')])
            num_frames = int([f for f in os.listdir(metadata_dir) if f.startswith('rgb')][0][6:-4])
            if len(all_fps.loc[all_fps.video == sub_id, 'fps']) == 0:
                ori_fps = 60
            else:
                ori_fps = int(round(all_fps.loc[all_fps.video == sub_id, 'fps'].values[0]))

            example_past_frames = example_past_time * ori_fps
            example_future_frames = example_future_time * ori_fps
            example_frames = example_time * ori_fps
            print('Number of frames: {} \t fps: {}'.format(num_frames, ori_fps))

            file_prefix = os.path.join(metadata_dir, 'PosInfo_0_%d' % num_frames)  # prefix of Frame txt and keyFrame txt
            if not os.path.exists(file_prefix + '_Frame.txt'):
                print('{} not exists!'.format(file_prefix + '_Frame.txt'))
                continue

            # extract valid positions
            if os.path.exists(os.path.join(metadata_dir, 'validFrame.csv')):
                print('valid frame file exists!')
            else:
                print('extract valid frame ...')
                extract(n_frame, threshold, file_prefix, metadata_dir)
            valid_frame = pd.read_csv(os.path.join(metadata_dir, 'validFrame.csv'), index_col=0)

            # consider tracking lost!
            n_examples = 0
            if len(valid_frame) == 0:
                print('The number of examples: ', n_examples)
                continue
            idx = 0
            s_idx = valid_frame.index[0]

            while s_idx <= valid_frame.index[-1] - example_frames + 1 and idx + example_frames - 1 < len(valid_frame):
                if s_idx + example_frames - 1 != valid_frame.index[idx + example_frames - 1]:
                    idx += 1
                    s_idx = valid_frame.index[idx]
                    continue
                else:
                    # example_df = valid_frame.loc[s_idx: s_idx + example_frames - 1]
                    # past_pos = example_df.iloc[0: example_past_frames, 1:4].values
                    # future_pos = example_df.iloc[example_past_frames:, 1:4].values

                    if valid_frame.index[idx] >= 0:
                        example_df = valid_frame.loc[s_idx: s_idx + example_frames - 1]
                        qwo = example_df.iloc[example_past_frames - 1, 4:].values
                        qwo = np.hstack([qwo[3], qwo[0:3]])

                        # Transformation 1:
                        # heading = get_heading(qwo)
                        # Rwo = toRotMatrix(heading)
                        # two = np.expand_dims(example_df.iloc[example_past_frames, 1:4].values, 1)
                        # Row = Rwo.T
                        # tow = np.matmul(-Rwo.T, two)
                        # assert tow.shape == (3, 1)
                        # past_pos_ori = example_df.iloc[0: example_past_frames, 1:4].values
                        # future_pos_ori = example_df.iloc[example_past_frames:, 1:4].values
                        # past_pos = (np.matmul(Row, np.expand_dims(past_pos_ori, -1)) + tow)[..., 0]
                        # future_pos = (np.matmul(Row, np.expand_dims(future_pos_ori, -1)) + tow)[..., 0]

                        # Transformation 2:
                        R = toRotMatrix(qwo)
                        t = example_df.iloc[example_past_frames - 1, 1:4].values
                        past_pos_ori = example_df.iloc[0: example_past_frames, 1:4].values
                        future_pos_ori = example_df.iloc[example_past_frames:, 1:4].values
                        past_pos = np.matmul(R.T, np.expand_dims(past_pos_ori - t, -1))[..., 0]
                        future_pos = np.matmul(R.T, np.expand_dims(future_pos_ori - t, -1))[..., 0]

                        past_imgs = np.array([os.path.join(sub_data_path, 'frame_{:010d}.jpg'.format(img_idx))
                                              for img_idx in range(s_idx, s_idx + example_past_frames)])
                        past_flow_u = np.array([os.path.join(sub_data_path.replace('rgb/', 'flow/'), 'u', 'frame_{:010d}.jpg'.format(img_idx))
                                                for img_idx in range(int(s_idx / 2), int(s_idx / 2) + int(example_past_frames / 2))])
                        past_flow_v = np.array([os.path.join(sub_data_path.replace('rgb/', 'flow/'), 'v', 'frame_{:010d}.jpg'.format(img_idx))
                                                for img_idx in range(int(s_idx / 2), int(s_idx / 2) + int(example_past_frames / 2))])

                        future_actions = frame_action.loc[sub_id, 'action'][s_idx: s_idx + example_future_frames]
                        example = EgoData(past_pos, past_imgs, past_flow_u, past_flow_v, future_pos, future_actions,
                                          data_id=total_examples + n_examples, video_id=sub_id, start_frame=s_idx)
                        example_list.append(example)
                        n_examples += 1

                    idx += example_frames
                    s_idx = valid_frame.index[idx]

                    # print('past pos: ', past_pos.shape)
                    # print('future pos: ', future_pos.shape)
                    # print('past img: ', len(past_imgs))
                    # print('future actions: ', len(future_actions))

            total_examples += n_examples
            print('The number of examples: ', n_examples)

        print('Total examples: ', total_examples)
        save_path = os.path.join(args.save_root, "EgoData_{}".format(data_id))
        with open(save_path, "wb") as dill_file:
            dill.dump(example_list, dill_file)

