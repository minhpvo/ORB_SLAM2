import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def extract(nframe, threshold, file_prefix, save_path):
    print("Extracting valid positions ...")
    frame_traj_file = '{}_Frame.txt'.format(file_prefix)
    keyframe_traj_file = '{}_keyFrame.txt'.format(file_prefix)
    if not os.path.isfile(frame_traj_file):
        raise IOError("Frames info file doesn't exist: {}".format(frame_traj_file))
    if not os.path.isfile(keyframe_traj_file):
        raise IOError("Keyframes info file doesn't exist: {}".format(keyframe_traj_file))

    full_traj = pd.DataFrame(columns=['t', 'x', 'y', 'z', 'q0', 'q1', 'q2', 'q3'])
    key_traj = pd.DataFrame(columns=['t', 'x', 'y', 'z', 'q0', 'q1', 'q2', 'q3'])
    with open(frame_traj_file, 'r') as f:
        for line in f.readlines():
            frame_info = [float(d) for d in line.split()]
            full_traj = full_traj.append({'t': frame_info[0],
                                          'x': frame_info[1],
                                          'y': frame_info[2],
                                          'z': frame_info[3],
                                          'q0': frame_info[4],
                                          'q1': frame_info[5],
                                          'q2': frame_info[6],
                                          'q3': frame_info[7]}, ignore_index=True)

    with open(keyframe_traj_file, 'r') as f:
        for line in f.readlines():
            frame_info = [float(d) for d in line.split()]
            key_traj = key_traj.append({'t': frame_info[0],
                                        'x': frame_info[1],
                                        'y': frame_info[2],
                                        'z': frame_info[3],
                                        'q0': frame_info[4],
                                        'q1': frame_info[5],
                                        'q2': frame_info[6],
                                        'q3': frame_info[7]}, ignore_index=True)

    full_traj.to_csv(os.path.join(save_path, 'Frame.csv'))
    key_traj.to_csv(os.path.join(save_path, 'keyFrame.csv'))

    # Calculate distances
    dist = []
    for frame_id in key_traj.t:
        frame_pos = (full_traj[full_traj.t == frame_id].iloc[0, 1:4]).values
        keyframe_pos = (key_traj[key_traj.t == frame_id].iloc[0, 1:4]).values
        # print('frame_id: %d' % frame_id, ' frame pos: ', frame_pos, '\t key frame pos: ', keyframe_pos)
        dist.append(np.linalg.norm(frame_pos - keyframe_pos))
    dist = np.array(dist)

    # If {nframe} consecutive distances between frames and keyframes < threshold, we believe the positions are stable
    stable_start_frame = []
    stable_end_frame = []
    stable_flag = False

    for nt in range(len(key_traj.t)):
        if not stable_flag:
            if np.all(dist[nt:nt+nframe] < threshold):
                stable_start_frame.append(key_traj.t[nt])
                stable_flag = True
        else:
            if dist[nt] > 2 * threshold:
                stable_end_frame.append(key_traj.t[nt])
                stable_flag = False

    if stable_flag:
        stable_end_frame.append(key_traj.t.values[-1])

    print("stable start frame: ", stable_start_frame)
    print("stable end frame: ", stable_end_frame)
    assert len(stable_start_frame) == len(stable_end_frame)

    # Incorporate all stable positions
    valid_full_traj = pd.DataFrame(columns=['t', 'x', 'y', 'z', 'q0', 'q1', 'q2', 'q3'])
    for s, e in zip(stable_start_frame, stable_end_frame):
        s_idx = full_traj[full_traj.t == s].index.values[0]
        e_idx = full_traj[full_traj.t == e].index.values[0]
        valid_full_traj = pd.concat([valid_full_traj, full_traj[s_idx:e_idx + 1]])

    # Remove tracking lost positions
    # valid frames may start at the first keyframe (0, 0, 0), which is removed
    if len(stable_start_frame) != 0:
        drop_index = valid_full_traj[(valid_full_traj.x == 0) & (valid_full_traj.t != stable_start_frame[0])].index
        total_drop_index = full_traj[full_traj.x == 0].index
        valid_full_traj = valid_full_traj.drop(drop_index)
        valid_full_traj.to_csv(os.path.join(save_path, 'validFrame.csv'))
        print('Total frames: {} \t Total keyframes: {} \t Total valid frames: {} ({}) \n'
              'Tracking lost or uninitialized frames: {} ({})'.
              format(len(full_traj), len(key_traj), len(valid_full_traj), len(valid_full_traj) / len(full_traj),
                     len(total_drop_index), len(total_drop_index) / len(full_traj)))
    else:
        total_drop_index = full_traj[full_traj.x == 0].index
        valid_full_traj.to_csv(os.path.join(save_path, 'validFrame.csv'))
        print('Total frames: {} \t Total keyframes: {} \t Total valid frames: {} ({}) \n '
              'Tracking lost or uninitialized frames: {} ({})'.
              format(len(full_traj), len(key_traj), len(valid_full_traj), len(valid_full_traj) / len(full_traj),
                     len(total_drop_index), len(total_drop_index) / len(full_traj)))

    # Visualization
    fig = plt.figure(figsize=(18, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # 1: plot frames
    drop_index = full_traj[full_traj.x == 0].index
    ax1.plot(-full_traj.drop(drop_index).y, full_traj.drop(drop_index).x)
    if len(key_traj) != 0:
        ax1.plot(-key_traj.y, key_traj.x, color='y')
    x_ticks = ax1.get_xticks()
    y_ticks = ax1.get_yticks()
    ax1.set_title('Frames and KeyFrames')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(['Frame', 'KeyFrame'])

    # 2: plot stable frames
    if len(stable_start_frame) != 0:
        for s, e in zip(stable_start_frame, stable_end_frame):
            s_idx = valid_full_traj[valid_full_traj.t == s].index.values[0]
            e_idx = valid_full_traj[valid_full_traj.t == e].index.values[0]
            ax2.plot(-valid_full_traj.loc[s_idx:e_idx].y, valid_full_traj.loc[s_idx:e_idx].x)

    ax2.set_xticks(x_ticks)
    ax2.set_yticks(y_ticks)
    ax2.set_title('Stable Frames')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # 3: plot distances between frames and keyframes
    ax3.plot(key_traj.t, dist)
    ax3.set_title('Distance between frames and keyframes')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Distance')

    plt.savefig(os.path.join(save_path, 'vis.png'))
    print('Save visualization result in vis.png!')
    plt.close()
    # plt.show()

    return stable_start_frame, stable_end_frame
