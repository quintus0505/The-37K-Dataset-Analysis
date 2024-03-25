import pandas as pd
from config import logdata_columns, DEFAULT_DATASETS_DIR, DEFAULT_VISUALIZATION_DIR, DEFAULT_CLEANED_DATASETS_DIR, \
    DEFAULT_FIGS_DIR
import os.path as osp
import os
import numpy as np
import matplotlib.pyplot as plt


def calculate_iki_intervals(df, interval_size=10, y_label='WMR', intervals=(125, 1045)):
    # Define the intervals for IKI
    intervals = np.arange(intervals[0], intervals[1], interval_size)
    df['IKI_interval'] = pd.cut(df['IKI'], bins=intervals, right=False)

    # print 25%, 50%, 75% iki from the original df
    print("25% quantile (bottom 25% cutoff) IKI: {}".format(df['IKI'].quantile(0.25)))
    print("50% quantile (median, bottom 50% cutoff) IKI: {}".format(df['IKI'].quantile(0.50)))
    print("75% quantile (bottom 75% cutoff) IKI: {}".format(df['IKI'].quantile(0.75)))

    print(df['IKI'].quantile(0.99))
    # remove the bottom 1% data from df
    df = df[df['IKI'] < df['IKI'].quantile(0.99)]

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

    # print("Number of data points below 25% quantile: {}".format(below_25))
    # print("Number of data points between 25% and 75% quantile: {}".format(between_25_75))
    # print("Number of data points above 75% quantile: {}".format(above_75))

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
    elif y_label == 'EDIT_DISTANCE':
        reset_index_name = 'EDIT_DISTANCE'
    else:
        raise ValueError("y_label must be either 'WMR' or 'AC' or 'MODIFICATION'")

    def calculate(x):
        if y_label == 'AGE':
            # return the mean of the age for each interval
            return x['AGE'].mean()
        elif y_label == 'EDIT_DISTANCE':
            return x['EDIT_DISTANCE'].mean()
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


def plot_age_vs_iki(age_intervals_df, save_file_name=None, origin_df=None, interval_size=10, label_extra_info=''):
    iki_25 = origin_df['IKI'].quantile(0.25)
    iki_75 = origin_df['IKI'].quantile(0.75)
    # Convert interval to string and get the midpoint for the label
    avg = origin_df['AGE'].mean()
    total_participants = len(origin_df['PARTICIPANT_ID'].unique())
    total_test_sections = len(origin_df['TEST_SECTION_ID'].unique())

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

    # # Draw red vertical lines for 25% and 75% IKI
    # plt.axvline(x=iki_25, color='red', linestyle='--', label='25% IKI')
    # plt.axvline(x=iki_75, color='red', linestyle='--', label='75% IKI')

    # # Text annotations
    # plt.text(iki_25, max(age_intervals_df['AGE']), f'{int(iki_25)}', color='red', horizontalalignment='right')
    # plt.text(iki_75, max(age_intervals_df['AGE']), f'{int(iki_75)}', color='red', horizontalalignment='right')

    # Annotations for avg, total participants, and total test sections
    annotation_text = f'Avg Age: {avg:.2f}\nTotal Participants: {total_participants}\nTotal Test Sections: {total_test_sections}'
    plt.text(0.95, 0.95, annotation_text, transform=plt.gca().transAxes, horizontalalignment='right',
             verticalalignment='top')

    # # Add legend
    # plt.legend()

    # save the plot
    save_dir = osp.join(DEFAULT_FIGS_DIR, 'age_vs_iki')
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, save_file_name)
    plt.savefig(save_path)

    # Show the plot
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()


# Plotting function for WMR vs IKI
def plot_wmr_vs_iki(wmr_intervals_df, save_file_name=None, origin_df=None, interval_size=10, label_extra_info=''):
    iki_25 = origin_df['IKI'].quantile(0.25)
    iki_50 = origin_df['IKI'].quantile(0.5)
    iki_75 = origin_df['IKI'].quantile(0.75)
    iki_95 = origin_df['IKI'].quantile(0.9)
    origin_df['WMR'] = origin_df['MODIFIED_WORD_COUNT'] / origin_df['WORD_COUNT']
    avg = origin_df['WMR'].mean() * 100
    total_participants = len(origin_df['PARTICIPANT_ID'].unique())
    total_test_sections = len(origin_df['TEST_SECTION_ID'].unique())
    # Convert interval to string and get the midpoint for the label
    plt.figure(figsize=(12, 6))
    midpoints = wmr_intervals_df['IKI_interval'].apply(lambda x: (x.left + x.right) / 2).astype(int)

    # Plot the WMR vs IKI intervals
    plt.bar(midpoints, wmr_intervals_df['WMR'] * 100, width=interval_size, edgecolor='black')

    # Set the title and labels
    plt.title('WMR vs. Typing Interval' + label_extra_info)
    plt.xlabel('Typing Interval (ms)')
    plt.ylabel('WMR (%)')

    # Set x-ticks to be the midpoints of intervals, but only label every 50ms
    plt.xticks(ticks=midpoints, labels=['' if x % (500 / interval_size) != 0 else str(x) for x in midpoints])

    # Set the y-axis to start from 5.5
    y_bottom = 5.5
    y_top = 27.5
    plt.yticks(ticks=np.arange(y_bottom, y_top, 1.0),
               labels=[str(x) for x in np.arange(y_bottom, y_top, 1.0)])
    plt.ylim(bottom=y_bottom, top=y_top)  # Here's the key change
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Draw red horizontal lines for avg
    plt.axhline(y=avg, color='red', linestyle='--', label='Avg')
    plt.text(100, avg * 1.05, f'Avg: {avg:.2f}', color='black')

    # # Draw red vertical lines for 25% and 75% IKI
    # plt.axvline(x=iki_25, color='red', linestyle='--', label='25% IKI')
    # plt.axvline(x=iki_75, color='red', linestyle='--', label='75% IKI')
    # plt.axvline(x=iki_95, color='red', linestyle='--', label='95% IKI')
    #
    # # Text annotations
    # plt.text(iki_25, max(wmr_intervals_df['WMR']) * 100, f'{int(iki_25)}', color='red', horizontalalignment='right')
    # plt.text(iki_75, max(wmr_intervals_df['WMR']) * 100, f'{int(iki_75)}', color='red', horizontalalignment='right')
    # plt.text(iki_95, max(wmr_intervals_df['WMR']) * 100, f'{int(iki_95)}', color='red', horizontalalignment='right')
    # # Add legend
    # plt.legend()
    # Annotations for avg, total participants, and total test sections
    annotation_text = f'Avg WMR: {avg:.2f}\nTotal Participants: {total_participants}\nTotal Test Sections: {total_test_sections}'
    plt.text(0.95, 0.95, annotation_text, transform=plt.gca().transAxes, horizontalalignment='right',
             verticalalignment='top')

    # plt.text(iki_25, max(wmr_intervals_df['WMR']) * 90, "25%", color='black', horizontalalignment='right')

    # save the plot
    save_dir = osp.join(DEFAULT_FIGS_DIR, 'wmr_vs_iki')
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, save_file_name)
    plt.savefig(save_path)

    # Show the plot
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()


def plot_modification_vs_iki(modification_intervals_df, save_file_name=None, origin_df=None, interval_size=10,
                             label_extra_info=''):
    iki_25 = origin_df['IKI'].quantile(0.25)
    iki_75 = origin_df['IKI'].quantile(0.75)

    avg = origin_df['MODIFICATION_COUNT'].sum() / origin_df['CHAR_COUNT'].sum() * 100
    total_participants = len(origin_df['PARTICIPANT_ID'].unique())
    total_test_sections = len(origin_df['TEST_SECTION_ID'].unique())
    # Convert interval to string and get the midpoint for
    # Convert interval to string and get the midpoint for the label
    plt.figure(figsize=(12, 6))
    midpoints = modification_intervals_df['IKI_interval'].apply(lambda x: (x.left + x.right) / 2).astype(int)

    # Plot the WMR vs IKI intervals
    plt.bar(midpoints, modification_intervals_df['MODIFICATION'] * 100, width=interval_size, edgecolor='black')

    # Set the title and labels
    plt.title('Edit Before Commit vs. Typing Interval' + label_extra_info)
    plt.xlabel('Typing Interval (ms)')
    plt.ylabel('Edit Before Commits (%)')

    # Set x-ticks to be the midpoints of intervals, but only label every 50ms
    plt.xticks(ticks=midpoints, labels=['' if x % (500 / interval_size) != 0 else str(x) for x in midpoints])

    # Set the y-axis to start from 5.5
    y_bottom = 1.0
    y_top = 9.0
    plt.yticks(ticks=np.arange(y_bottom, y_top, 1.0),
               labels=[str(x) for x in np.arange(y_bottom, y_top, 1.0)])
    plt.ylim(bottom=y_bottom, top=y_top)  # Here's the key change

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Draw red horizontal lines for avg
    plt.axhline(y=avg, color='red', linestyle='--', label='Avg')
    plt.text(100, avg * 1.05, f'Avg: {avg:.2f}', color='black')

    # Draw red vertical lines for 25% and 75% IKI
    # plt.axvline(x=iki_25, color='red', linestyle='--', label='25% IKI')
    # plt.axvline(x=iki_75, color='red', linestyle='--', label='75% IKI')

    # Text annotations
    # plt.text(iki_25, max(modification_intervals_df['MODIFICATION']) * 100, f'{int(iki_25)}', color='red',
    #          horizontalalignment='right')
    # plt.text(iki_75, max(modification_intervals_df['MODIFICATION']), f'{int(iki_75)}', color='red',
    #          horizontalalignment='right')
    # # Add legend
    # plt.legend()
    # Annotations for avg, total participants, and total test sections
    annotation_text = f'Avg MODIFICATION: {avg:.2f}\nTotal Participants: {total_participants}\nTotal Test Sections: {total_test_sections}'
    plt.text(0.95, 0.95, annotation_text, transform=plt.gca().transAxes, horizontalalignment='right',
             verticalalignment='top')

    # plt.text(iki_25, max(modification_intervals_df['MODIFICATION']) * 90, "25%", color='black',
    #          horizontalalignment='right')

    # save the plot
    save_dir = osp.join(DEFAULT_FIGS_DIR, 'modification_vs_iki')
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, save_file_name)
    plt.savefig(save_path)

    # Show the plot
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()


def plot_ac_vs_iki(ac_intervals_df, save_file_name=None, origin_df=None, interval_size=10, label_extra_info=''):
    iki_25 = origin_df['IKI'].quantile(0.25)
    iki_75 = origin_df['IKI'].quantile(0.75)
    # Convert interval to string and get the midpoint for the label
    avg = origin_df['AC_WORD_COUNT'].sum() / origin_df['WORD_COUNT'].sum() * 100
    total_participants = len(origin_df['PARTICIPANT_ID'].unique())
    total_test_sections = len(origin_df['TEST_SECTION_ID'].unique())
    plt.figure(figsize=(12, 6))
    midpoints = ac_intervals_df['IKI_interval'].apply(lambda x: (x.left + x.right) / 2).astype(int)

    # Plot the WMR vs IKI intervals
    plt.bar(midpoints, ac_intervals_df['AC'] * 100, width=interval_size, edgecolor='black')

    # Set the title and labels
    plt.title('Auto-corrected ratio vs. Typing Interval' + label_extra_info)
    plt.xlabel('Typing Interval (ms)')
    plt.ylabel('Auto-corrected ratio (%)')

    # Set x-ticks to be the midpoints of intervals, but only label every 50ms
    plt.xticks(ticks=midpoints, labels=['' if x % (500 / interval_size) != 0 else str(x) for x in midpoints])
    y_bottom = 0.0
    y_top = 20.0
    plt.yticks(ticks=np.arange(y_bottom, y_top, 2.5),
               labels=[str(x) for x in np.arange(y_bottom, y_top, 2.5)])
    plt.ylim(bottom=y_bottom, top=y_top)  # Here's the key change
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Draw red horizontal lines for avg
    plt.axhline(y=avg, color='red', linestyle='--', label='Avg')
    plt.text(950, avg * 1.05, f'Avg: {avg:.2f}', color='black')
    # Draw red vertical lines for 25% and 75% IKI
    # plt.axvline(x=iki_25, color='red', linestyle='--', label='25% IKI')
    # plt.axvline(x=iki_75, color='red', linestyle='--', label='75% IKI')

    # Text annotations
    # plt.text(iki_25, max(ac_intervals_df['AC']), f'{int(iki_25)}', color='red', horizontalalignment='right')
    # plt.text(iki_75, max(ac_intervals_df['AC']), f'{int(iki_75)}', color='red', horizontalalignment='right')
    # # Add legend
    # plt.legend()
    # Annotations for avg, total participants, and total test sections
    annotation_text = f'Avg AC: {avg:.2f}\nTotal Participants: {total_participants}\nTotal Test Sections: {total_test_sections}'
    plt.text(0.95, 0.95, annotation_text, transform=plt.gca().transAxes, horizontalalignment='right',
             verticalalignment='top')

    # save the plot
    save_dir = osp.join(DEFAULT_FIGS_DIR, 'ac_vs_iki')
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, save_file_name)
    plt.savefig(save_path)

    # Show the plot
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()


def plot_num_vs_iki(num_intervals_df, save_file_name=None, origin_df=None, interval_size=10, label_extra_info=''):
    # Convert interval to string and get the midpoint for the label
    iki_25 = origin_df['IKI'].quantile(0.25)
    iki_75 = origin_df['IKI'].quantile(0.75)
    iki_95 = origin_df['IKI'].quantile(0.95)
    iki_99 = origin_df['IKI'].quantile(0.99)

    total_num = len(origin_df)
    total_participants = len(origin_df['PARTICIPANT_ID'].unique())
    total_test_sections = len(origin_df['TEST_SECTION_ID'].unique())

    plt.figure(figsize=(12, 6))
    midpoints = num_intervals_df['IKI_interval'].apply(lambda x: (x.left + x.right) / 2).astype(int)

    # Plot the WMR vs IKI intervals
    plt.bar(midpoints, num_intervals_df['NUM'], width=interval_size, edgecolor='black')

    # Set the title and labels
    plt.title('NUM vs. Typing Interval' + label_extra_info)
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

    plt.text(iki_25, max(num_intervals_df['NUM']) * 0.9, "25%", color='black', horizontalalignment='right')
    plt.text(iki_75, max(num_intervals_df['NUM']) * 0.9, "75%", color='black', horizontalalignment='right')
    plt.text(iki_95, max(num_intervals_df['NUM']) * 0.9, "95%", color='black', horizontalalignment='right')
    plt.text(iki_99, max(num_intervals_df['NUM']) * 0.9, "99%", color='black', horizontalalignment='right')

    # # Add legend
    # plt.legend()
    # Annotations for avg, total participants, and total test sections
    annotation_text = f'total NUM: {total_num}\nTotal Participants: {total_participants}\nTotal Test Sections: {total_test_sections}'
    plt.text(0.95, 0.95, annotation_text, transform=plt.gca().transAxes, horizontalalignment='right',
             verticalalignment='top')

    # save the plot
    save_dir = osp.join(DEFAULT_FIGS_DIR, 'num_vs_iki')
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, save_file_name)
    plt.savefig(save_path)

    # Show the plot
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()


def plot_edit_distance_vs_iki(edit_distance_intervals_df, save_file_name=None, origin_df=None, interval_size=10,
                              label_extra_info=''):
    # Convert interval to string and get the midpoint for the label
    iki_25 = origin_df['IKI'].quantile(0.25)
    iki_75 = origin_df['IKI'].quantile(0.75)
    iki_95 = origin_df['IKI'].quantile(0.95)
    iki_99 = origin_df['IKI'].quantile(0.99)

    avg = origin_df['EDIT_DISTANCE'].mean()
    total_participants = len(origin_df['PARTICIPANT_ID'].unique())
    total_test_sections = len(origin_df['TEST_SECTION_ID'].unique())

    plt.figure(figsize=(12, 6))
    midpoints = edit_distance_intervals_df['IKI_interval'].apply(lambda x: (x.left + x.right) / 2).astype(int)

    # Plot the WMR vs IKI intervals
    plt.bar(midpoints, edit_distance_intervals_df['EDIT_DISTANCE'], width=interval_size, edgecolor='black')

    # Set the title and labels
    plt.title('Edit Distance vs. Typing Interval' + label_extra_info)
    plt.xlabel('Typing Interval (ms)')
    plt.ylabel('Edit Distance')

    # Set x-ticks to be the midpoints of intervals, but only label every 50ms
    plt.xticks(ticks=midpoints, labels=['' if x % (500 / interval_size) != 0 else str(x) for x in midpoints])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    y_bottom = 0.0
    y_top = 4.0
    plt.yticks(ticks=np.arange(y_bottom, y_top, 0.25),
               labels=[str(x) for x in np.arange(y_bottom, y_top, 0.25)])
    plt.ylim(bottom=y_bottom, top=y_top)  # Here's the key change

    # Draw red horizontal lines for avg
    plt.axhline(y=avg, color='red', linestyle='--', label='Avg')
    plt.text(100, avg * 1.05, f'Avg: {avg:.2f}', color='black')
    # # Draw red vertical lines for 25% and 75% IKI
    # plt.axvline(x=iki_25, color='red', linestyle='--', label='25% IKI')
    # plt.axvline(x=iki_75, color='red', linestyle='--', label='75% IKI')
    # plt.axvline(x=iki_95, color='red', linestyle='--', label='95% IKI')
    # plt.axvline(x=iki_99, color='red', linestyle='--', label='99% IKI')
    #
    # # Text annotations
    # plt.text(iki_25, max(edit_distance_intervals_df['EDIT_DISTANCE']), f'{int(iki_25)}', color='red',
    #          horizontalalignment='right')
    # plt.text(iki_75, max(edit_distance_intervals_df['EDIT_DISTANCE']), f'{int(iki_75)}', color='red',
    #          horizontalalignment='right')
    # plt.text(iki_95, max(edit_distance_intervals_df['EDIT_DISTANCE']), f'{int(iki_95)}', color='red',
    #          horizontalalignment='right')
    # plt.text(iki_99, max(edit_distance_intervals_df['EDIT_DISTANCE']), f'{int(iki_99)}', color='red',
    #          horizontalalignment='right')
    #
    # plt.text(iki_25, max(edit_distance_intervals_df['EDIT_DISTANCE']) * 0.9, "25%", color='black',
    #          horizontalalignment='right')
    # plt.text(iki_75, max(edit_distance_intervals_df['EDIT_DISTANCE']) * 0.9, "75%", color='black',
    #          horizontalalignment='right')
    # plt.text(iki_95, max(edit_distance_intervals_df['EDIT_DISTANCE']) * 0.9, "95%", color='black',
    #          horizontalalignment='right')
    # plt.text(iki_99, max(edit_distance_intervals_df['EDIT_DISTANCE']) * 0.9, "99%", color='black',
    #          horizontalalignment='right')

    # # Add legend
    # plt.legend()
    annotation_text = f'Avg EDIT_DISTANCE: {avg:.2f}\nTotal Participants: {total_participants}\nTotal Test Sections: {total_test_sections}'
    plt.text(0.95, 0.95, annotation_text, transform=plt.gca().transAxes, horizontalalignment='right',
             verticalalignment='top')

    # save the plot
    save_dir = osp.join(DEFAULT_FIGS_DIR, 'edit_distance_vs_iki')
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, save_file_name)
    plt.savefig(save_path)

    # Show the plot
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()


def plot_num_vs_wmr(save_file_name=None, origin_df=None, interval_size=0.05, label_extra_info=''):
    # use wmr_intervals_df['MODIFIED_WORD_COUNT'] / wmr_intervals_df['WORD_COUNT'] to get the WMR
    origin_df['WMR'] = origin_df['MODIFIED_WORD_COUNT'] / origin_df['WORD_COUNT']

    wmr_participant_level = origin_df.groupby('PARTICIPANT_ID').apply(
        lambda x: x['MODIFIED_WORD_COUNT'].sum() / x['WORD_COUNT'].sum())
    wmr_participant_level = wmr_participant_level.reset_index(name='WMR')
    intervals = np.arange(0.01, 0.6, interval_size)

    avg = origin_df['WMR'].mean()
    total_participants = len(origin_df['PARTICIPANT_ID'].unique())
    total_test_sections = len(origin_df['TEST_SECTION_ID'].unique())
    # plot the num vs wmr, bar is the number of data
    plt.figure(figsize=(12, 6))

    WMR_dict = {}
    for index, row in wmr_participant_level.iterrows():
        # make all the WMR in to the interval
        wmr = row['WMR']
        for i in range(len(intervals) - 1):
            if intervals[i] <= wmr < intervals[i + 1]:
                if intervals[i] in WMR_dict:
                    WMR_dict[intervals[i]] += 1
                else:
                    WMR_dict[intervals[i]] = 1
                break

    plt.bar(WMR_dict.keys(), WMR_dict.values(), width=interval_size, edgecolor='black')

    plt.title('Number of participants vs. WMR' + label_extra_info)
    plt.xlabel('WMR (%)')
    plt.ylabel('Number of participants')
    # set x-tics to be 10, 20, 30, 40, 50, 60
    plt.xticks(ticks=np.arange(0.05, 0.6, 0.05), labels=[str(int(x * 100)) for x in np.arange(0.05, 0.6, 0.05)])

    annotation_text = f'Avg WMR: {avg:.2f}\nTotal Participants: {total_participants}\nTotal Test Sections: {total_test_sections}'
    plt.text(0.95, 0.95, annotation_text, transform=plt.gca().transAxes, horizontalalignment='right',
             verticalalignment='top')

    # save the plot
    save_dir = osp.join(DEFAULT_FIGS_DIR, 'num_vs_wmr')
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, save_file_name)
    plt.savefig(save_path)

    # Show the plot
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()


def plot_sentence_info(df):
    # calculate the character count and word count for each sentence and store in ['CHAR_COUNT', 'WORD_COUNT']
    df['CHAR_COUNT'] = df['SENTENCE'].apply(lambda x: len(x))
    df['WORD_COUNT'] = df['SENTENCE'].apply(lambda x: len(x.split()))

    # plot character count vs the number of sentences having that character count
    plt.figure(figsize=(12, 6))

    number_dict = {}
    for index, row in df.iterrows():
        length = row['CHAR_COUNT']
        if length in number_dict:
            number_dict[length] += 1
        else:
            number_dict[length] = 1

    plt.bar(number_dict.keys(), number_dict.values(), width=1, edgecolor='black')

    plt.title('Character count vs. Number of sentences in Typing 37K')
    plt.xlabel('Character count')
    plt.ylabel('Number of sentences')
    plt.show()

    # plot word count vs the number of sentences having that word count
    plt.figure(figsize=(12, 6))
    number_dict = {}
    for index, row in df.iterrows():
        length = row['WORD_COUNT']
        if length in number_dict:
            number_dict[length] += 1
        else:
            number_dict[length] = 1

    plt.bar(number_dict.keys(), number_dict.values(), width=1, edgecolor='black')

    plt.title('Word count vs. Number of sentences in Typing 37K')

    plt.xlabel('Word count')
    plt.ylabel('Number of sentences')
    plt.show()

    wmr = 0
    # iterate through the number_dict
    for key, value in number_dict.items():
        wmr += value * 1 / key

    wmr /= len(df)

    print("Average WMR: ", wmr)

    # print average character count and word count
    print("Average character count: ", df['CHAR_COUNT'].mean())
    print("Average word count: ", df['WORD_COUNT'].mean())


def show_plot_info(df, save_file_name, y_label='WMR'):
    print("Total participants: ", len(df['PARTICIPANT_ID'].unique()))
    print("Total test sections: ", len(df['TEST_SECTION_ID'].unique()))
    # add all the AC world count and word count then calculate the ratio
    if y_label == 'WMR':
        print("Plotting Word Modified Ratio (WMR) vs. Typing Interval for file: ", save_file_name)
        # print("Word Modified Ratio (WMR): ", df['MODIFIED_WORD_COUNT'].sum() / df['WORD_COUNT'].sum())
    elif y_label == 'AC':
        print("Plotting Auto-corrected ratio vs. Typing Interval for file: ", save_file_name)
        # print("Auto-corrected ratio: ", df['AC_WORD_COUNT'].sum() / df['WORD_COUNT'].sum())
    elif y_label == 'MODIFICATION':
        print("Plotting Modification ratio vs. Typing Interval for file: ", save_file_name)
        # print("Modification ratio: ", df['MODIFICATION_COUNT'].sum() / df['CHAR_COUNT'].sum())
    elif y_label == 'AGE':
        print("Plotting Age vs. Typing Interval for file: ", save_file_name)
        # print("Average age: ", df['AGE'].mean())
    elif y_label == 'NUM':
        print("Plotting Number vs. Typing Interval for file: ", save_file_name)
        # print("Total number: ", len(df))
    elif y_label == 'EDIT_DISTANCE':
        print("Plotting Edit Distance vs. Typing Interval for file: ", save_file_name)
        # print("Average edit distance: ", df['EDIT_DISTANCE'].mean())

# from tools.data_loading import get_sentences_df
#
# if __name__ == "__main__":
# df = get_sentences_df()
# plot_sentence_info(df)
