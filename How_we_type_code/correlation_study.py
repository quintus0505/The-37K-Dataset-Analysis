import os
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import how_we_type_key_coordinate, HOW_WE_TYPE_TYPING_LOG_DATA_DIR, HOW_WE_TYPE_GAZE_DATA_DIR, \
    HOW_WE_TYPE_FINGER_DATA_DIR, DEFAULT_FIGS_DIR
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# Provided configurations
original_gaze_columns = ['subject_id', 'block', 'sentence_id', 'trialtime', 'x', 'y']
original_finger_columns = ['optitime', 'subject_id', 'block', 'sentence_id', 'trialtime', 'x1', 'y1', 'z1', 'x2', 'y2',
                           'z2']
original_log_columns = ['systime', 'subject_id', 'block', 'sentence_id', 'trialtime', 'DATA', 'layout', 'INPUT',
                        'touchx', 'touchy']

gaze_data_dir = osp.join(HOW_WE_TYPE_GAZE_DATA_DIR, 'Gaze')
typing_log_dir = osp.join(HOW_WE_TYPE_TYPING_LOG_DATA_DIR, 'Typing_log')

tail_offset = -300
head_offset = 300

FIG_DIR = osp.join(DEFAULT_FIGS_DIR, 'how_we_type')


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


def load_data(gaze_file, typing_file):
    gaze_df = pd.read_csv(gaze_file, names=original_gaze_columns)
    gaze_df = gaze_df.iloc[1:]
    gaze_df['x'] = gaze_df['x'].astype(float)
    gaze_df['y'] = gaze_df['y'].astype(float)
    gaze_df['trialtime'] = gaze_df['trialtime'].astype(float).astype(int)
    gaze_df['sentence_id'] = gaze_df['sentence_id'].astype(int)

    typinglog_df = pd.read_csv(typing_file, names=original_log_columns)
    typinglog_df = typinglog_df.iloc[1:]
    typinglog_df['touchx'] = typinglog_df['touchx'].astype(float)
    typinglog_df['touchy'] = typinglog_df['touchy'].astype(float)
    typinglog_df['trialtime'] = typinglog_df['trialtime'].astype(float).astype(int)
    typinglog_df['sentence_id'] = typinglog_df['sentence_id'].astype(int)

    typinglog_df['touchx'] += 501.5 - typinglog_df['touchx'].min()
    typinglog_df['touchy'] += 1840 - typinglog_df['touchy'].min()

    return gaze_df, typinglog_df


def normalize_coordinates(df, x_col, y_col):
    df[[x_col, y_col]] = normalize(df[[x_col, y_col]])
    return df


def compute_distance_and_cosine_similarity(gaze_df, typinglog_df):
    # gaze_df = normalize_coordinates(gaze_df, 'x', 'y')
    # typinglog_df = normalize_coordinates(typinglog_df, 'touchx', 'touchy')
    distances = {}
    similarities = {}
    for sentence_id, group in typinglog_df.groupby('sentence_id'):
        gaze_group = gaze_df[gaze_df['sentence_id'] == sentence_id]
        gaze_group = filter_percentiles(gaze_group, 'x', lower_percentile=5, upper_percentile=95)
        gaze_group = filter_percentiles(gaze_group, 'y', lower_percentile=5, upper_percentile=95)

        gaze_group = scale_to_range(gaze_group, 'x', x_min, x_max)
        gaze_group = scale_to_range(gaze_group, 'y', y_min, y_max)
        # try:
        #     gaze_group = normalize_coordinates(gaze_group, 'x', 'y')
        # except:
        #     continue
        for _, typing_row in group.iterrows():
            trialtime = typing_row['trialtime']
            window_gaze_df = gaze_group[(gaze_group['trialtime'] >= trialtime + tail_offset) &
                                        (gaze_group['trialtime'] <= trialtime + head_offset)]

            for _, gaze_row in window_gaze_df.iterrows():
                if gaze_row['y'] < 0.6 * gaze_group['y'].max():
                    continue
                offset = gaze_row['trialtime'] - trialtime
                dist = np.linalg.norm([gaze_row['x'] - typing_row['touchx'], gaze_row['y'] - typing_row['touchy']])
                gaze_vec = np.array([gaze_row['x'], gaze_row['y']]).reshape(1, -1)
                touch_vec = np.array([typing_row['touchx'], typing_row['touchy']]).reshape(1, -1)
                try:
                    sim = cosine_similarity(gaze_vec, touch_vec)[0][0]
                except:
                    continue
                if offset not in distances:
                    distances[offset] = []
                distances[offset].append(dist)

                if offset not in similarities:
                    similarities[offset] = []
                similarities[offset].append(sim)

    return distances, similarities


def compute_pearson_correlation_and_cosine_similarity(gaze_df, typinglog_df):
    correlations = []
    cosine_similarities = []
    gaze_df = normalize_coordinates(gaze_df, 'x', 'y')
    typinglog_df = normalize_coordinates(typinglog_df, 'touchx', 'touchy')

    for sentence_id, group in typinglog_df.groupby('sentence_id'):
        gaze_group = gaze_df[gaze_df['sentence_id'] == sentence_id]

        for _, typing_row in group.iterrows():
            try:
                trialtime = typing_row['trialtime']
                closest_gaze_row = gaze_group.iloc[(gaze_group['trialtime'] - trialtime).abs().argmin()]
                correlations.append(
                    (closest_gaze_row['x'], closest_gaze_row['y'], typing_row['touchx'], typing_row['touchy']))
                gaze_vec = np.array([closest_gaze_row['x'], closest_gaze_row['y']]).reshape(1, -1)
                touch_vec = np.array([typing_row['touchx'], typing_row['touchy']]).reshape(1, -1)
                cosine_sim = cosine_similarity(gaze_vec, touch_vec)[0][0]
                cosine_similarities.append(cosine_sim)
            except:
                continue

    if correlations:
        correlations_df = pd.DataFrame(correlations, columns=['gaze_x', 'gaze_y', 'touchx', 'touchy'])
        corr_matrix = correlations_df[['gaze_x', 'gaze_y', 'touchx', 'touchy']].corr()
        corr_xy_touch = (corr_matrix.loc['gaze_x', 'touchx'] + corr_matrix.loc['gaze_y', 'touchy']) / 2
        corr_x = corr_matrix.loc['gaze_x', 'touchx']
        corr_y = corr_matrix.loc['gaze_y', 'touchy']
    else:
        corr_xy_touch = np.nan
        corr_x = np.nan
        corr_y = np.nan

    avg_cosine_similarity = np.nanmean(cosine_similarities) if cosine_similarities else np.nan

    return corr_xy_touch, corr_x, corr_y, avg_cosine_similarity


def plot_similarities(avg_similarities):
    offsets = sorted(avg_similarities.keys())
    avg_sims = [avg_similarities[offset] for offset in offsets]

    if not avg_sims:
        print("No valid data to plot.")
        return

    # Calculate rolling mean and standard deviation
    rolling_mean = pd.Series(avg_sims, dtype=float).rolling(window=10, min_periods=1).mean()
    rolling_std = pd.Series(avg_sims, dtype=float).rolling(window=10, min_periods=1).std()

    # Remove NaN values for plotting limits
    rolling_mean = rolling_mean.dropna()
    rolling_std = rolling_std.dropna()

    # Ensure no NaN values in the max calculation
    if rolling_mean.empty or rolling_std.empty:
        print("No valid data after rolling mean/std calculation.")
        return

    max_ylim = max((rolling_mean + rolling_std).dropna())

    plt.figure(figsize=(10, 6))

    # Plot the rolling mean
    sns.lineplot(x=offsets, y=rolling_mean, label='Average Similarities', color='blue')

    # Plot the confidence interval (rolling std deviation)
    plt.fill_between(offsets, rolling_mean - rolling_std, rolling_mean + rolling_std, color='blue', alpha=0.3)

    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.xlabel('Time Interval (ms)')
    plt.ylabel('Average Similarities')
    plt.ylim(0, max_ylim)
    plt.legend()
    plt.title('Average Similarities between Gaze Position and Typed Position')
    plt.show()


def plot_distances(avg_distances, save_dir=None):
    if save_dir:
        avg_distances = {offset: np.nanmean(avg_distances[offset]) for offset in avg_distances}
    offsets = sorted(avg_distances.keys())
    avg_dists = [avg_distances[offset] for offset in offsets]

    if not avg_dists:
        print("No valid data to plot.")
        return

    # Calculate rolling mean and standard deviation
    rolling_mean = pd.Series(avg_dists, dtype=float).rolling(window=10, min_periods=1).mean()
    rolling_std = pd.Series(avg_dists, dtype=float).rolling(window=10, min_periods=1).std()

    # Remove NaN values for plotting limits
    rolling_mean = rolling_mean.dropna()
    rolling_std = rolling_std.dropna()

    # Ensure no NaN values in the max calculation
    if rolling_mean.empty or rolling_std.empty:
        print("No valid data after rolling mean/std calculation.")
        return

    max_ylim = max((rolling_mean + rolling_std).dropna())

    plt.figure(figsize=(10, 6))

    # Plot the rolling mean
    sns.lineplot(x=offsets, y=rolling_mean, label='Average Normalized Distance', color='blue')

    # Plot the confidence interval (rolling std deviation)
    plt.fill_between(offsets, rolling_mean - rolling_std, rolling_mean + rolling_std, color='blue', alpha=0.3)

    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.xlabel('Time Interval (ms)')
    plt.ylabel('Average Normalized Distance')
    plt.ylim(0, max_ylim)
    plt.legend()
    if save_dir:
        plt.title('Average Normalized Distance between Gaze Position and Typed Position of log {}'.format(
            save_dir.split('/')[-1]))
    else:
        plt.title('Average Normalized Distance between Gaze Position and Typed Position')
    if save_dir:
        plt.savefig(osp.join(save_dir, 'average_normalized_distance.png'))
        plt.close()  # Close the plot to avoid displaying
    else:
        plt.show()


def get_files():
    gaze_files = [osp.join(gaze_data_dir, f) for f in os.listdir(gaze_data_dir) if
                  f.startswith('gaze') and f.endswith('.csv')]
    typing_files = [osp.join(typing_log_dir, f.replace('gaze', 'typinglog')) for f in os.listdir(gaze_data_dir) if
                    f.startswith('gaze') and f.endswith('.csv')]
    return gaze_files, typing_files


def process_all_distance_and_similarity():
    gaze_files, typing_files = get_files()
    all_distances = {}
    all_similarities = {}
    all_correlations_xy_touch = []
    all_correlations_x = []
    all_correlations_y = []
    all_cosine_similarities = []

    for gaze_file, typing_file in zip(gaze_files, typing_files):
        if osp.exists(typing_file):
            # gaze_129_1.csv, get 129_1
            csv_num = gaze_file.split('_')[-2] + '_' + gaze_file.split('_')[-1].split('.')[0]
            print("Processing files: ", csv_num)
            # print("Processing files: ", gaze_file, typing_file)
            gaze_df, typinglog_df = load_data(gaze_file, typing_file)
            distances, similarities = compute_distance_and_cosine_similarity(gaze_df, typinglog_df)
            corr_xy_touch, corr_x, corr_y, avg_cosine_similarity = compute_pearson_correlation_and_cosine_similarity(
                gaze_df, typinglog_df)

            print(f"Correlation XY Touch: {corr_xy_touch}")
            print(f"Correlation X: {corr_x}")
            print(f"Correlation Y: {corr_y}")
            print(f"Cosine Similarity: {avg_cosine_similarity}")

            all_correlations_xy_touch.append(corr_xy_touch)
            all_correlations_x.append(corr_x)
            all_correlations_y.append(corr_y)
            all_cosine_similarities.append(avg_cosine_similarity)

            for offset, dists in distances.items():
                if offset not in all_distances:
                    all_distances[offset] = []
                all_distances[offset].extend(dists)

            for offset, sims in similarities.items():
                if offset not in all_similarities:
                    all_similarities[offset] = []
                all_similarities[offset].extend(sims)

            plot_distances(distances, save_dir=osp.join(FIG_DIR, csv_num))

    final_avg_distances = {offset: np.nanmean(all_distances[offset]) for offset in all_distances}
    final_avg_similarities = {offset: np.nanmean(all_similarities[offset]) for offset in all_similarities}
    plot_distances(final_avg_distances)
    plot_similarities(final_avg_similarities)

    # Compute overall correlations
    overall_correlation_xy_touch = np.nanmean(all_correlations_xy_touch)
    overall_correlation_x = np.nanmean(all_correlations_x)
    overall_correlation_y = np.nanmean(all_correlations_y)
    overall_cosine_similarity = np.nanmean(all_cosine_similarities)
    print(f"Overall Correlation XY Touch: {overall_correlation_xy_touch}")
    print(f"Overall Correlation X: {overall_correlation_x}")
    print(f"Overall Correlation Y: {overall_correlation_y}")
    print(f"Overall Cosine Similarity: {overall_cosine_similarity}")


def get_key_vs_proofreading(gaze_df, typinglog_df):
    keys = {}
    for sentence_id, group in typinglog_df.groupby('sentence_id'):
        gaze_group = gaze_df[gaze_df['sentence_id'] == sentence_id]
        if gaze_group.empty:
            continue
        # gaze_group = normalize_coordinates(gaze_group, 'x', 'y')
        for _, typing_row in group.iterrows():
            try:
                trialtime = typing_row['trialtime']
                closest_gaze_row = gaze_group.iloc[(gaze_group['trialtime'] - trialtime).abs().argmin()]
                key = typing_row['DATA']
                if closest_gaze_row['y'] < 0.4 * gaze_group['y'].max():
                    is_proofreading = 1
                else:
                    is_proofreading = 0
                if key not in keys:
                    keys[key] = []
                keys[key].append(is_proofreading)
            except:
                continue
    return keys


def get_iki_vs_proofreading(gaze_df, typinglog_df):
    iki = {}
    for sentence_id, group in typinglog_df.groupby('sentence_id'):
        gaze_group = gaze_df[gaze_df['sentence_id'] == sentence_id]
        if gaze_group.empty:
            continue

        for i in range(1, len(group)):
            try:
                typing_row_prev = group.iloc[i - 1]
                typing_row_curr = group.iloc[i]

                trialtime_prev = typing_row_prev['trialtime']
                trialtime_curr = typing_row_curr['trialtime']

                # Calculate IKI
                inter_key_interval = trialtime_curr - trialtime_prev

                if inter_key_interval >= 3000:
                    continue

                closest_gaze_row = gaze_group.iloc[(gaze_group['trialtime'] - trialtime_curr).abs().argmin()]

                if closest_gaze_row['y'] < 0.4 * gaze_group['y'].max():
                    is_proofreading = 1
                else:
                    is_proofreading = 0
                if inter_key_interval not in iki:
                    iki[inter_key_interval] = []
                iki[inter_key_interval].append(is_proofreading)
            except:
                continue
    return iki


def process_all_key_vs_proofreading():
    gaze_files, typing_files = get_files()
    all_keys = {}
    for gaze_file, typing_file in zip(gaze_files, typing_files):
        if osp.exists(typing_file):
            # gaze_129_1.csv, get 129_1
            csv_num = gaze_file.split('_')[-2] + '_' + gaze_file.split('_')[-1].split('.')[0]
            print("Processing files: ", csv_num)
            # print("Processing files: ", gaze_file, typing_file)
            gaze_df, typinglog_df = load_data(gaze_file, typing_file)
            keys = get_key_vs_proofreading(gaze_df, typinglog_df)
            for key, proofreading in keys.items():
                if key not in all_keys:
                    all_keys[key] = []
                all_keys[key].extend(proofreading)

    for key, proofreading in all_keys.items():
        print(f"Key: {key}, Proofreading: {np.mean(proofreading)}")

    # sort the keys by ASCII
    all_keys = dict(sorted(all_keys.items(), key=lambda x: ord(x[0])))
    # plot the count of key
    keys = list(all_keys.keys())
    key_count = [len(all_keys[key]) for key in keys]
    keys = [key.replace(" ", "_") for key in keys]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=keys, y=key_count)
    plt.xlabel('Key')
    plt.ylabel('Count')
    plt.title('Count of Each Key')
    plt.xticks(rotation=45)
    plt.show()

    # plot proofreading rate with std
    keys = list(all_keys.keys())
    proofreading_rate = [np.mean(all_keys[key]) for key in keys]
    proofreading_std = [np.std(all_keys[key]) for key in keys]
    # Ensure error bars do not go below zero
    proofreading_rate = np.array(proofreading_rate)
    proofreading_std = np.array(proofreading_std)
    lower_error = np.clip(proofreading_rate - proofreading_std, 0, np.inf)
    upper_error = proofreading_rate + proofreading_std
    asymmetric_error = [proofreading_rate - lower_error, upper_error - proofreading_rate]
    # change " " to "_" for better visualization
    keys = [key.replace(" ", "_") for key in keys]
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=keys, y=proofreading_rate)
    ax.errorbar(x=np.arange(len(keys)), y=proofreading_rate, yerr=asymmetric_error, fmt='none', c='black', capsize=5)
    plt.xlabel('Key')
    plt.ylabel('Proofreading Rate')
    plt.title('Proofreading Rate for Each Key')
    plt.xticks(rotation=45)
    plt.show()

    # plot the proofreading rate of "_", "B" and other keys
    first_row_char = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'å']
    second_row_char = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'ö', 'ä']
    third_row_char = ['z', 'x', 'c', 'v', 'b', 'n', 'm']
    summary_keys = {"space": [], "backspace": [], "1st row": [], "2nd row": [], "3rd row": []}
    for key, proofreading in all_keys.items():
        if key == " ":
            summary_keys["space"].extend(proofreading)
        elif key == "B":
            summary_keys["backspace"].extend(proofreading)
        elif key in first_row_char:
            summary_keys["1st row"].extend(proofreading)
        elif key in second_row_char:
            summary_keys["2nd row"].extend(proofreading)
        elif key in third_row_char:
            summary_keys["3rd row"].extend(proofreading)
    for key, proofreading in summary_keys.items():
        print(f"Key: {key}, Proofreading: {np.mean(proofreading)}")
    keys = list(summary_keys.keys())
    proofreading_rate = [np.mean(summary_keys[key]) for key in keys]
    proofreading_std = [np.std(summary_keys[key]) for key in keys]
    # Ensure error bars do not go below zero
    proofreading_rate = np.array(proofreading_rate)
    proofreading_std = np.array(proofreading_std)
    lower_error = np.clip(proofreading_rate - proofreading_std, 0, np.inf)
    upper_error = np.clip(proofreading_rate + proofreading_std, -np.inf, 1)
    asymmetric_error = [proofreading_rate - lower_error, upper_error - proofreading_rate]
    plt.figure(figsize=(5, 6))
    ax = sns.barplot(x=keys, y=proofreading_rate)
    ax.errorbar(x=np.arange(len(keys)), y=proofreading_rate, yerr=asymmetric_error, fmt='none', c='black', capsize=5)
    plt.xlabel('Key')
    plt.ylabel('Proofreading Rate')
    plt.title('Proofreading Rate for space, backspace and other keys')
    plt.xticks(rotation=45)
    plt.show()


def process_all_iki_vs_proofreading():
    gaze_files, typing_files = get_files()
    all_ikis = {}
    for gaze_file, typing_file in zip(gaze_files, typing_files):
        if osp.exists(typing_file):
            # gaze_129_1.csv, get 129_1
            csv_num = gaze_file.split('_')[-2] + '_' + gaze_file.split('_')[-1].split('.')[0]
            print("Processing files: ", csv_num)
            # print("Processing files: ", gaze_file, typing_file)
            gaze_df, typinglog_df = load_data(gaze_file, typing_file)
            iki = get_iki_vs_proofreading(gaze_df, typinglog_df)
            for key, proofreading in iki.items():
                if key not in all_ikis:
                    all_ikis[key] = []
                all_ikis[key].extend(proofreading)

    # Group IKIs into bins of 200
    bin_size = 100
    max_iki = 3000
    bins = np.arange(0, max_iki + bin_size, bin_size)
    binned_ikis = {bin: [] for bin in bins}

    for iki, proofreading in all_ikis.items():
        bin = (iki // bin_size) * bin_size
        binned_ikis[bin].extend(proofreading)

    # Calculate the average proofreading rate for each bin
    binned_iki_keys = sorted(binned_ikis.keys())
    proofreading_rate = [np.mean(binned_ikis[bin]) if binned_ikis[bin] else 0 for bin in binned_iki_keys]
    proofreading_std = [np.std(binned_ikis[bin]) if binned_ikis[bin] else 0 for bin in binned_iki_keys]

    # Ensure error bars do not go below zero
    proofreading_rate = np.array(proofreading_rate)
    proofreading_std = np.array(proofreading_std)
    lower_error = np.clip(proofreading_rate - proofreading_std, 0, np.inf)
    upper_error = np.clip(proofreading_rate + proofreading_std, -np.inf, 1)
    asymmetric_error = [proofreading_rate - lower_error, upper_error - proofreading_rate]

    # Plot the bar chart with std
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=binned_iki_keys, y=proofreading_rate)
    ax.errorbar(x=np.arange(len(binned_iki_keys)), y=proofreading_rate, yerr=asymmetric_error, fmt='none', c='black',
                capsize=5)
    ax.set_xlabel('IKI (Grouped by 200 ms)')
    ax.set_ylabel('Proofreading Rate')
    ax.set_title('Average Proofreading Rate by IKI')
    ax.set_xticklabels(binned_iki_keys, rotation=45)
    plt.show()

    # Plot the iki count bar chart, for the same size of bins
    iki_count = [len(binned_ikis[bin]) for bin in binned_iki_keys]
    plt.figure(figsize=(12, 8))
    sns.barplot(x=binned_iki_keys, y=iki_count)
    plt.xlabel('IKI (Grouped by 200 ms)')
    plt.ylabel('Count')
    plt.title('IKI Count by Bins')
    plt.xticks(rotation=45)
    plt.show()


if __name__ == '__main__':
    process_all_distance_and_similarity()
    process_all_key_vs_proofreading()
    process_all_iki_vs_proofreading()
