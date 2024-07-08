from config import how_we_type_key_coordinate, HOW_WE_TYPE_TYPING_LOG_DATA_DIR, HOW_WE_TYPE_GAZE_DATA_DIR, \
    HOW_WE_TYPE_FINGER_DATA_DIR
import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt

original_gaze_columns = ['subject_id', 'block', 'sentence_id', 'trialtime', 'x', 'y']
original_gaze_columns_plus_position = original_gaze_columns + ['position']

original_finger_columns = ['optitime', 'subject_id', 'block', 'sentence_id', 'trialtime', 'x1', 'y1', 'z1', 'x2', 'y2',
                           'z2']
original_finger_columns_plus_position = original_finger_columns + ['position1, position2']
original_log_columns = ['systime', 'subject_id', 'block', 'sentence_id', 'trialtime', 'DATA', 'layout', 'INPUT',
                        'touchx', 'touchy']


# Function to filter out top and bottom 2.5% of values
def filter_percentiles(df, column, lower_percentile=2.5, upper_percentile=97.5):
    lower_bound = df[column].quantile(lower_percentile / 100)
    upper_bound = df[column].quantile(upper_percentile / 100)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# Desired ranges
x_min, x_max = 501.5, 1942.5
y_min, y_max = 100, 2760


# Scaling function
def scale_to_range(df, column, new_min, new_max):
    old_min = df[column].min()
    old_max = df[column].max()
    df[column] = ((df[column] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return df


# x_min, y_min = 0, 0

def get_key_position(x, y):
    for key, value in how_we_type_key_coordinate.items():
        if value[0] <= x <= value[2] and value[1] <= int(y) <= value[3]:
            return key
    return '-'


def visualize_data():
    GAZE_DATA_DIR = osp.join(HOW_WE_TYPE_GAZE_DATA_DIR, 'Gaze')
    TYPY_DATA_DIR = osp.join(HOW_WE_TYPE_FINGER_DATA_DIR, 'Finger_Motion_Capture')
    TYPING_LOG_DIR = osp.join(HOW_WE_TYPE_TYPING_LOG_DATA_DIR, 'Typing_log')

    for file in os.listdir(GAZE_DATA_DIR):
        print("Processing file: ", file)
        # if file.endswith("129_1.csv"):
        # get the number like this 101_1 as save_dir_name in gaze_101_1.csv
        save_dir_name = file.split('.')[0].split('_')[1] + '_' + file.split('.')[0].split('_')[2]
        if not os.path.exists(f'../figs/how_we_type/{save_dir_name}'):
            os.makedirs(f'../figs/how_we_type/{save_dir_name}')
        file_path = osp.join(GAZE_DATA_DIR, file)
        gaze_df = pd.read_csv(file_path, names=original_gaze_columns)
        gaze_df = gaze_df.iloc[1:]
        gaze_df['x'] = gaze_df['x'].astype(float)
        gaze_df['y'] = gaze_df['y'].astype(float)
        gaze_df['sentence_id'] = gaze_df['sentence_id'].astype(int)

        # Load the corresponding finger data
        finger_file = file.replace("gaze", "finger")
        finger_path = osp.join(TYPY_DATA_DIR, finger_file)
        if not osp.exists(finger_path):
            continue

        typing_file = file.replace("gaze", "typinglog")
        typing_path = osp.join(TYPING_LOG_DIR, typing_file)

        typinglog_df = pd.read_csv(typing_path, names=original_log_columns)
        typinglog_df = typinglog_df.iloc[1:]
        typinglog_df['touchx'] = typinglog_df['touchx'].astype(float)
        typinglog_df['touchy'] = typinglog_df['touchy'].astype(float)
        typinglog_df['trialtime'] = typinglog_df['trialtime'].astype(float).astype(int)
        typinglog_df['sentence_id'] = typinglog_df['sentence_id'].astype(int)

        typinglog_df.loc[:, 'touchx'] += 501.5 - typinglog_df['touchx'].min()
        typinglog_df.loc[:, 'touchy'] += 1840 - typinglog_df['touchy'].min()

        finger_df = pd.read_csv(finger_path, names=original_finger_columns)
        finger_df = finger_df.iloc[1:]
        finger_df[['x1', 'y1', 'x2', 'y2']] = finger_df[['x1', 'y1', 'x2', 'y2']].astype(float)
        finger_df['sentence_id'] = finger_df['sentence_id'].astype(int)

        sentence_groups = gaze_df.groupby('sentence_id')
        for sentence_id, group in sentence_groups:
            plt.figure(figsize=(9, 16))
            group = filter_percentiles(group, 'x', lower_percentile=5, upper_percentile=95)
            group = filter_percentiles(group, 'y', lower_percentile=5, upper_percentile=95)

            group = scale_to_range(group, 'x', x_min, x_max)
            group = scale_to_range(group, 'y', y_min, y_max)
            # Draw the keyboard layout
            for key, coord in how_we_type_key_coordinate.items():
                plt.gca().add_patch(plt.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1],
                                                  fill=None, edgecolor='blue', linewidth=1))
                plt.text((coord[0] + coord[2]) / 2, (coord[1] + coord[3]) / 2, key,
                         horizontalalignment='center', verticalalignment='center', color='blue')
            # Plot gaze trail
            plt.plot(group['x'], group['y'], color='red', marker='o', linestyle='-', markersize=2, linewidth=0.5)

            # Reshape and plot typing trail
            finger_group = finger_df[finger_df['sentence_id'] == sentence_id].copy()
            if not finger_group.empty:
                plt.plot(finger_group['x1'], finger_group['y1'], color='green', marker='o', linestyle='-',
                         markersize=2, linewidth=0.5, label='Finger 1')
                plt.plot(finger_group['x2'], finger_group['y2'], color='purple', marker='o', linestyle='-',
                         markersize=2, linewidth=0.5, label='Finger 2')

            typinglog_group = typinglog_df[typinglog_df['sentence_id'] == sentence_id].copy()
            # plot the touch points as dots
            plt.scatter(typinglog_group['touchx'], typinglog_group['touchy'], color='yellow', marker='o', s=40,
                        label='Touch Points')
            plt.title(f'Gaze and Typing Trail for Sentence {sentence_id}')
            plt.xlim(501.5, 1942.5)
            plt.ylim(0, 2760)
            plt.gca().invert_yaxis()
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
            # plt.savefig(f'../figs/how_we_type/gaze_typing_trail_sentence_{sentence_id}.png')
            plt.savefig(f'../figs/how_we_type/{save_dir_name}/gaze_typing_trail_sentence_{sentence_id}.png')
            plt.close()


if __name__ == '__main__':
    visualize_data()
