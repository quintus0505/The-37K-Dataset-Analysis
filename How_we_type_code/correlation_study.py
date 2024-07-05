import os
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import how_we_type_key_coordinate, HOW_WE_TYPE_TYPING_LOG_DATA_DIR, HOW_WE_TYPE_GAZE_DATA_DIR, \
    HOW_WE_TYPE_FINGER_DATA_DIR

# Provided configurations
original_gaze_columns = ['subject_id', 'block', 'sentence_id', 'trialtime', 'x', 'y']
original_finger_columns = ['optitime', 'subject_id', 'block', 'sentence_id', 'trialtime', 'x1', 'y1', 'z1', 'x2', 'y2',
                           'z2']
original_log_columns = ['systime', 'subject_id', 'block', 'sentence_id', 'trialtime', 'DATA', 'layout', 'INPUT',
                        'touchx', 'touchy']

gaze_data_dir = osp.join(HOW_WE_TYPE_GAZE_DATA_DIR, 'Gaze')
typing_log_dir = osp.join(HOW_WE_TYPE_TYPING_LOG_DATA_DIR, 'Typing_log')
finger_data_dir = osp.join(HOW_WE_TYPE_FINGER_DATA_DIR, 'Finger_Motion_Capture')

tail_offset = -300
head_offset = 300


def load_data(gaze_file, typing_file):
    gaze_df = pd.read_csv(gaze_file, names=original_gaze_columns)
    gaze_df = gaze_df.iloc[1:]
    gaze_df['x'] = gaze_df['x'].astype(float)
    gaze_df['y'] = gaze_df['y'].astype(float)
    gaze_df['trialtime'] = gaze_df['trialtime'].astype(float).astype(int)

    typinglog_df = pd.read_csv(typing_file, names=original_log_columns)
    typinglog_df = typinglog_df.iloc[1:]
    typinglog_df['touchx'] = typinglog_df['touchx'].astype(float)
    typinglog_df['touchy'] = typinglog_df['touchy'].astype(float)
    typinglog_df['trialtime'] = typinglog_df['trialtime'].astype(float).astype(int)

    return gaze_df, typinglog_df


def compute_distance(gaze_df, typinglog_df):
    distances = {}

    for sentence_id, group in typinglog_df.groupby('sentence_id'):
        gaze_group = gaze_df[gaze_df['sentence_id'] == sentence_id]

        for _, typing_row in group.iterrows():
            trialtime = typing_row['trialtime']
            window_gaze_df = gaze_group[(gaze_group['trialtime'] >= trialtime + tail_offset) &
                                        (gaze_group['trialtime'] <= trialtime + head_offset)]

            for _, gaze_row in window_gaze_df.iterrows():
                if gaze_row['y'] < 1500:
                    continue
                offset = gaze_row['trialtime'] - trialtime
                dist = np.linalg.norm([gaze_row['x'] - 500 - typing_row['touchx'], gaze_row['y'] - typing_row['touchy']])
                if offset not in distances:
                    distances[offset] = []
                distances[offset].append(dist)

    return distances


def plot_distances(avg_distances):
    offsets = sorted(avg_distances.keys())
    avg_dists = [avg_distances[offset] for offset in offsets]

    plt.figure(figsize=(10, 6))
    plt.plot(offsets, avg_dists, label='Average Distance', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.xlabel('Time Interval (ms)')
    plt.ylabel('Average Distance (pixels)')
    plt.legend()
    plt.title('Average Distance between Gaze Position and Typed Position')
    plt.grid(True)
    plt.show()


def process_all_files():
    gaze_files = [osp.join(gaze_data_dir, f) for f in os.listdir(gaze_data_dir) if
                  f.startswith('gaze') and f.endswith('_1.csv')]
    typing_files = [osp.join(typing_log_dir, f.replace('gaze', 'typinglog')) for f in os.listdir(gaze_data_dir) if
                    f.startswith('gaze') and f.endswith('_1.csv')]

    all_distances = {}

    for gaze_file, typing_file in zip(gaze_files, typing_files):
        if osp.exists(typing_file):
            print("Processing files: ", gaze_file, typing_file)
            gaze_df, typinglog_df = load_data(gaze_file, typing_file)
            distances = compute_distance(gaze_df, typinglog_df)

            for offset, dists in distances.items():
                if offset not in all_distances:
                    all_distances[offset] = []
                all_distances[offset].extend(dists)

    final_avg_distances = {offset: np.nanmean(all_distances[offset]) for offset in all_distances}
    plot_distances(final_avg_distances)


if __name__ == '__main__':
    process_all_files()
