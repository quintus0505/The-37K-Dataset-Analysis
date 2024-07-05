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

x_max, y_max = 1942.5, 2645


# x_min, y_min = 0, 0

def get_key_position(x, y):
    for key, value in how_we_type_key_coordinate.items():
        if value[0] <= x <= value[2] and value[1] <= int(y) <= value[3]:
            return key
    return '-'


def visualize_data():
    GAZE_DATA_DIR = osp.join(HOW_WE_TYPE_GAZE_DATA_DIR, 'Gaze')
    TYPY_DATA_DIR = osp.join(HOW_WE_TYPE_FINGER_DATA_DIR, 'Finger_Motion_Capture')

    for file in os.listdir(GAZE_DATA_DIR):
        if file.endswith("101_2.csv"):
            file_path = osp.join(GAZE_DATA_DIR, file)
            gaze_df = pd.read_csv(file_path, names=original_gaze_columns)
            gaze_df = gaze_df.iloc[1:]
            gaze_df['x'] = gaze_df['x'].astype(float)
            gaze_df['y'] = gaze_df['y'].astype(float)

            # Load the corresponding finger data
            finger_file = file.replace("gaze", "finger")
            finger_path = osp.join(TYPY_DATA_DIR, finger_file)
            if not osp.exists(finger_path):
                continue

            finger_df = pd.read_csv(finger_path, names=original_finger_columns)
            finger_df = finger_df.iloc[1:]
            finger_df[['x1', 'y1', 'x2', 'y2']] = finger_df[['x1', 'y1', 'x2', 'y2']].astype(float)

            sentence_groups = gaze_df.groupby('sentence_id')
            for sentence_id, group in sentence_groups:
                plt.figure(figsize=(9, 16))

                # Draw the keyboard layout
                for key, coord in how_we_type_key_coordinate.items():
                    plt.gca().add_patch(plt.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1],
                                                      fill=None, edgecolor='blue', linewidth=1))
                    plt.text((coord[0] + coord[2]) / 2, (coord[1] + coord[3]) / 2, key,
                             horizontalalignment='center', verticalalignment='center', color='blue')
                # group['x'] = 501.5 + (group['x'] - group['x'].min()) * (1942.5 - 501.5) / (
                #             group['x'].max() - group['x'].min())
                # # group['x'] = group['x'] * 1942.5 / group['x'].max()
                # # group['y'] = group['y'] * 2760 / group['y'].max()
                # group['y'] = 500 + (group['y'] - group['y'].min()) * (2760 - 500) / (
                #             group['y'].max() - group['y'].min())
                # Plot gaze trail
                plt.plot(group['x'], group['y'], color='red', marker='o', linestyle='-', markersize=2, linewidth=0.5)

                # Reshape and plot typing trail
                finger_group = finger_df[finger_df['sentence_id'] == sentence_id].copy()
                if not finger_group.empty:
                    # finger_group.loc[:, 'x1'] = 501.5 + (finger_group['x1'] - finger_group['x1'].min()) * (
                    #             1942.5 - 501.5) / (finger_group['x1'].max() - finger_group['x1'].min())
                    # finger_group.loc[:, 'x2'] = 501.5 + (finger_group['x2'] - finger_group['x2'].min()) * (
                    #             1942.5 - 501.5) / (finger_group['x2'].max() - finger_group['x2'].min())
                    # finger_group.loc[:, 'y1'] = 1840 + (finger_group['y1'] - finger_group['y1'].min()) * (
                    #             2760 - 1840) / (finger_group['y1'].max() - finger_group['y1'].min())
                    # finger_group.loc[:, 'y2'] = 1840 + (finger_group['y2'] - finger_group['y2'].min()) * (
                    #             2760 - 1840) / (finger_group['y2'].max() - finger_group['y2'].min())

                    plt.plot(finger_group['x1'], finger_group['y1'], color='green', marker='o', linestyle='-',
                             markersize=2, linewidth=0.5, label='Finger 1')
                    plt.plot(finger_group['x2'], finger_group['y2'], color='purple', marker='o', linestyle='-',
                             markersize=2, linewidth=0.5, label='Finger 2')

                plt.title(f'Gaze and Typing Trail for Sentence {sentence_id}')
                plt.xlim(501.5, 1942.5)
                plt.ylim(0, 2760)
                plt.gca().invert_yaxis()
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.legend()
                plt.savefig(f'../figs/how_we_type/gaze_typing_trail_sentence_{sentence_id}.png')
                plt.close()

            break


if __name__ == '__main__':
    visualize_data()
