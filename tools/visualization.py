import pandas as pd
from config import logdata_columns, DEFAULT_DATASETS_DIR, DEFAULT_VISUALIZATION_DIR, DEFAULT_CLEANED_DATASETS_DIR, \
    DEFAULT_FIGS_DIR
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt


def calculate_iki_intervals(df, interval_size=10, y_label='WMR'):
    # Define the intervals for IKI
    intervals = np.arange(145, 1045, interval_size)
    df['IKI_interval'] = pd.cut(df['IKI'], bins=intervals, right=False)

    # print 25%, 50%, 75% iki from the original df
    print("25% quantile (bottom 25% cutoff) IKI: {}".format(df['IKI'].quantile(0.25)))
    print("50% quantile (median, bottom 50% cutoff) IKI: {}".format(df['IKI'].quantile(0.50)))
    print("75% quantile (bottom 75% cutoff) IKI: {}".format(df['IKI'].quantile(0.75)))

    # Counting how many data points belong to each section
    # Below 25% quantile
    below_25 = df[df['IKI'] <= df['IKI'].quantile(0.25)].shape[0]
    # # Between 25% and 50% quantile
    # between_25_50 = df[(df['IKI'] > df['IKI'].quantile(0.25)) & (df['IKI'] <= df['IKI'].quantile(0.50))].shape[0]
    # # Between 50% and 75% quantile
    # between_50_75 = df[(df['IKI'] > df['IKI'].quantile(0.50)) & (df['IKI'] <= df['IKI'].quantile(0.75))].shape[0]
    # Between 25% and 75% quntile
    between_25_75 = df[(df['IKI'] > df['IKI'].quantile(0.25)) & (df['IKI'] <= df['IKI'].quantile(0.57))].shape[0]
    # Above 75% quantile
    above_75 = df[df['IKI'] > df['IKI'].quantile(0.75)].shape[0]

    print("Number of data points below 25% quantile: {}".format(below_25))
    # print("Number of data points between 25% and 50% quantile: {}".format(between_25_50))
    # print("Number of data points between 50% and 75% quantile: {}".format(between_50_75))
    print("Number of data points between 25% and 75% quantile: {}".format(between_25_75))
    print("Number of data points above 75% quantile: {}".format(above_75))

    # Group by the IKI_interval and calculate the WMR for each group
    if y_label == 'WMR':
        count_level_name = 'WORD_COUNT'
        compute_target_name = 'MODIFIED_WORD_COUNT'
        reset_index_name = 'WMR'
    elif y_label == 'AC':
        count_level_name = 'WORD_COUNT'
        compute_target_name = 'AC_WORD_COUNT'
        reset_index_name = 'AC'
    elif y_label == 'MODIFICATION':
        count_level_name = 'CHAR_COUNT'
        compute_target_name = 'MODIFICATION_COUNT'
        reset_index_name = 'MODIFICATION'
    elif y_label == 'AGE':
        reset_index_name = 'AGE'
    elif y_label == 'NUM':
        reset_index_name = 'NUM'
    else:
        raise ValueError("y_label must be either 'WMR' or 'AC' or 'MODIFICATION'")

    def calculate(x):
        if y_label == 'AGE':
            # return the mean of the age for each interval
            return x['AGE'].mean()
        elif y_label == 'NUM':
            return len(x['TEST_SECTION_ID'])
        else:
            # print how many rows in each interval, if none, pass
            try:
                print("Interval: ", x['IKI_interval'].values[0], " Count: ", len(x))
            except:
                pass

            word_count = x[count_level_name].sum()
            # do not compute those interval with less than 10 counts
            if word_count == 0 or len(x) < 5:
                return np.nan  # Return NaN if the word count is zero to avoid division by zero
            else:
                return x[compute_target_name].sum() / word_count

    grouped = df.groupby('IKI_interval').apply(calculate)

    return grouped.reset_index(name=reset_index_name)


def plot_age_vs_iki(age_intervals_df, save_file_name=None, origin_df=None, interval_size=10):
    iki_25 = origin_df['IKI'].quantile(0.25)
    iki_75 = origin_df['IKI'].quantile(0.75)
    # Convert interval to string and get the midpoint for the label
    plt.figure(figsize=(12, 6))
    midpoints = age_intervals_df['IKI_interval'].apply(lambda x: (x.left + x.right) / 2).astype(int)

    # Plot the WMR vs IKI intervals
    plt.bar(midpoints, age_intervals_df['AGE'], width=interval_size, edgecolor='black')

    # Set the title and labels
    plt.title('Age vs. Typing Interval')
    plt.xlabel('Typing Interval (ms)')
    plt.ylabel('Age')

    # Set x-ticks to be the midpoints of intervals, but only label every 50ms
    plt.xticks(ticks=midpoints, labels=['' if x % (500 / interval_size) != 0 else str(x) for x in midpoints])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Draw red vertical lines for 25% and 75% IKI
    plt.axvline(x=iki_25, color='red', linestyle='--', label='25% IKI')
    plt.axvline(x=iki_75, color='red', linestyle='--', label='75% IKI')

    # Text annotations
    plt.text(iki_25, max(age_intervals_df['AGE']), f'{int(iki_25)}', color='red', horizontalalignment='right')
    plt.text(iki_75, max(age_intervals_df['AGE']), f'{int(iki_75)}', color='red', horizontalalignment='right')
    # # Add legend
    # plt.legend()

    # save the plot
    save_path = osp.join(DEFAULT_FIGS_DIR, save_file_name)
    plt.savefig(save_path)

    # Show the plot
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()


# Plotting function for WMR vs IKI
def plot_wmr_vs_iki(wmr_intervals_df, save_file_name=None, origin_df=None, interval_size=10):
    iki_25 = origin_df['IKI'].quantile(0.25)
    iki_75 = origin_df['IKI'].quantile(0.75)
    iki_95 = origin_df['IKI'].quantile(0.9)
    # Convert interval to string and get the midpoint for the label
    plt.figure(figsize=(12, 6))
    midpoints = wmr_intervals_df['IKI_interval'].apply(lambda x: (x.left + x.right) / 2).astype(int)

    # Plot the WMR vs IKI intervals
    plt.bar(midpoints, wmr_intervals_df['WMR'] * 100, width=interval_size, edgecolor='black')

    # Set the title and labels
    plt.title('WMR vs. Typing Interval')
    plt.xlabel('Typing Interval (ms)')
    plt.ylabel('WMR (%)')

    # Set x-ticks to be the midpoints of intervals, but only label every 50ms
    plt.xticks(ticks=midpoints, labels=['' if x % (500 / interval_size) != 0 else str(x) for x in midpoints])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Draw red vertical lines for 25% and 75% IKI
    plt.axvline(x=iki_25, color='red', linestyle='--', label='25% IKI')
    plt.axvline(x=iki_75, color='red', linestyle='--', label='75% IKI')
    plt.axvline(x=iki_95, color='red', linestyle='--', label='95% IKI')

    # Text annotations
    plt.text(iki_25, max(wmr_intervals_df['WMR']) * 100, f'{int(iki_25)}', color='red', horizontalalignment='right')
    plt.text(iki_75, max(wmr_intervals_df['WMR']) * 100, f'{int(iki_75)}', color='red', horizontalalignment='right')
    plt.text(iki_95, max(wmr_intervals_df['WMR']) * 100, f'{int(iki_95)}', color='red', horizontalalignment='right')
    # # Add legend
    # plt.legend()

    # save the plot
    save_path = osp.join(DEFAULT_FIGS_DIR, save_file_name)
    plt.savefig(save_path)

    # Show the plot
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()


def plot_modification_vs_iki(modification_intervals_df, save_file_name=None, origin_df=None, interval_size=10):
    iki_25 = origin_df['IKI'].quantile(0.25)
    iki_75 = origin_df['IKI'].quantile(0.75)
    # Convert interval to string and get the midpoint for
    # Convert interval to string and get the midpoint for the label
    plt.figure(figsize=(12, 6))
    midpoints = modification_intervals_df['IKI_interval'].apply(lambda x: (x.left + x.right) / 2).astype(int)

    # Plot the WMR vs IKI intervals
    plt.bar(midpoints, modification_intervals_df['MODIFICATION'] * 100, width=interval_size, edgecolor='black')

    # Set the title and labels
    plt.title('Modification ratio vs. Typing Interval')
    plt.xlabel('Typing Interval (ms)')
    plt.ylabel('Modification ratio (%)')

    # Set x-ticks to be the midpoints of intervals, but only label every 50ms
    plt.xticks(ticks=midpoints, labels=['' if x % (500 / interval_size) != 0 else str(x) for x in midpoints])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Draw red vertical lines for 25% and 75% IKI
    plt.axvline(x=iki_25, color='red', linestyle='--', label='25% IKI')
    plt.axvline(x=iki_75, color='red', linestyle='--', label='75% IKI')

    # Text annotations
    plt.text(iki_25, max(modification_intervals_df['MODIFICATION']), f'{int(iki_25)}', color='red',
             horizontalalignment='right')
    plt.text(iki_75, max(modification_intervals_df['MODIFICATION']), f'{int(iki_75)}', color='red',
             horizontalalignment='right')
    # # Add legend
    # plt.legend()

    # save the plot
    save_path = osp.join(DEFAULT_FIGS_DIR, save_file_name)
    plt.savefig(save_path)

    # Show the plot
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()


def plot_ac_vs_iki(ac_intervals_df, save_file_name=None, origin_df=None, interval_size=10):
    iki_25 = origin_df['IKI'].quantile(0.25)
    iki_75 = origin_df['IKI'].quantile(0.75)
    # Convert interval to string and get the midpoint for the label
    plt.figure(figsize=(12, 6))
    midpoints = ac_intervals_df['IKI_interval'].apply(lambda x: (x.left + x.right) / 2).astype(int)

    # Plot the WMR vs IKI intervals
    plt.bar(midpoints, ac_intervals_df['AC'] * 100, width=interval_size, edgecolor='black')

    # Set the title and labels
    plt.title('Auto-corrected ratio vs. Typing Interval')
    plt.xlabel('Typing Interval (ms)')
    plt.ylabel('Auto-corrected ratio (%)')

    # Set x-ticks to be the midpoints of intervals, but only label every 50ms
    plt.xticks(ticks=midpoints, labels=['' if x % (500 / interval_size) != 0 else str(x) for x in midpoints])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Draw red vertical lines for 25% and 75% IKI
    plt.axvline(x=iki_25, color='red', linestyle='--', label='25% IKI')
    plt.axvline(x=iki_75, color='red', linestyle='--', label='75% IKI')

    # Text annotations
    plt.text(iki_25, max(ac_intervals_df['AC']), f'{int(iki_25)}', color='red', horizontalalignment='right')
    plt.text(iki_75, max(ac_intervals_df['AC']), f'{int(iki_75)}', color='red', horizontalalignment='right')
    # # Add legend
    # plt.legend()

    # save the plot
    save_path = osp.join(DEFAULT_FIGS_DIR, save_file_name)
    plt.savefig(save_path)

    # Show the plot
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()


def plot_num_vs_iki(num_intervals_df, save_file_name=None, origin_df=None, interval_size=10):
    # Convert interval to string and get the midpoint for the label
    iki_25 = origin_df['IKI'].quantile(0.25)
    iki_75 = origin_df['IKI'].quantile(0.75)
    iki_95 = origin_df['IKI'].quantile(0.95)
    iki_99 = origin_df['IKI'].quantile(0.99)


    plt.figure(figsize=(12, 6))
    midpoints = num_intervals_df['IKI_interval'].apply(lambda x: (x.left + x.right) / 2).astype(int)

    # Plot the WMR vs IKI intervals
    plt.bar(midpoints, num_intervals_df['NUM'], width=interval_size, edgecolor='black')

    # Set the title and labels
    plt.title('NUM vs. Typing Interval')
    plt.xlabel('Typing Interval (ms)')
    plt.ylabel('NUM')

    # Set x-ticks to be the midpoints of intervals, but only label every 50ms
    plt.xticks(ticks=midpoints, labels=['' if x % (500 / interval_size) != 0 else str(x) for x in midpoints])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Draw red vertical lines for 25% and 75% IKI
    plt.axvline(x=iki_25, color='red', linestyle='--', label='25% IKI')
    plt.axvline(x=iki_75, color='red', linestyle='--', label='75% IKI')
    plt.axvline(x=iki_95, color='red', linestyle='--', label='95% IKI')
    plt.axvline(x=iki_99, color='red', linestyle='--', label='99% IKI')

    # Text annotations
    plt.text(iki_25, max(num_intervals_df['NUM']), f'{int(iki_25)}', color='red', horizontalalignment='right')
    plt.text(iki_75, max(num_intervals_df['NUM']), f'{int(iki_75)}', color='red', horizontalalignment='right')
    plt.text(iki_95, max(num_intervals_df['NUM']), f'{int(iki_95)}', color='red', horizontalalignment='right')
    plt.text(iki_99, max(num_intervals_df['NUM']), f'{int(iki_99)}', color='red', horizontalalignment='right')

    # # Add legend
    # plt.legend()

    # save the plot
    save_path = osp.join(DEFAULT_FIGS_DIR, save_file_name)
    plt.savefig(save_path)

    # Show the plot
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()


def show_plot_info(df, save_file_name, y_label='WMR'):
    print("Total participants: ", len(df['PARTICIPANT_ID'].unique()))
    print("Total test sections: ", len(df['TEST_SECTION_ID'].unique()))
    # add all the AC world count and word count then calculate the ratio
    if y_label == 'WMR':
        print("Plotting Word Modified Ratio (WMR) vs. Typing Interval for file: ", save_file_name)
        print("Word Modified Ratio (WMR): ", df['MODIFIED_WORD_COUNT'].sum() / df['WORD_COUNT'].sum())
    elif y_label == 'AC':
        print("Plotting Auto-corrected ratio vs. Typing Interval for file: ", save_file_name)
        print("Auto-corrected ratio: ", df['AC_WORD_COUNT'].sum() / df['WORD_COUNT'].sum())
    elif y_label == 'MODIFICATION':
        print("Plotting Modification ratio vs. Typing Interval for file: ", save_file_name)
        print("Modification ratio: ", df['MODIFICATION_COUNT'].sum() / df['CHAR_COUNT'].sum())
    elif y_label == 'AGE':
        print("Plotting Age vs. Typing Interval for file: ", save_file_name)
        print("Average age: ", df['AGE'].mean())
    elif y_label == 'NUM':
        print("Plotting Number vs. Typing Interval for file: ", save_file_name)
        print("Total number: ", len(df))