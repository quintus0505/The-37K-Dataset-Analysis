from abc import ABC, abstractmethod
from data_loading import clean_participants_data, get_logdata_df, get_test_section_df
import pandas as pd
from config import logdata_columns, DEFAULT_DATASETS_DIR, DEFAULT_VISUALIZATION_DIR
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
WMR_VISUALIZATION_COLUMNS = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'WORD_COUNT', 'MODIFIED_WORK_COUNT', 'IKI', 'WPM']


class Parse(ABC):
    def __init__(self):
        self.participants_dataframe = None
        self.logdata_dataframe = None
        self.test_sections_dataframe = None

        self.word_count = 0
        self.modified_word_count = 0
        self.auto_corrected_word_count = 0

        self.test_section_count = 0

        self.iki_wmr_visualization_df = pd.DataFrame(columns=WMR_VISUALIZATION_COLUMNS)

    def save_iki_wmr_visualization(self, path):
        self.iki_wmr_visualization_df.to_csv(path, index=False)

    def load_participants(self, ite=None, keyboard='Gboard'):
        self.participants_dataframe = clean_participants_data(ite=ite, keyboard=keyboard)

    def load_logdata(self, full_log_data):
        self.logdata_dataframe = get_logdata_df(full_log_data)

    def load_test_sections(self):
        self.test_sections_dataframe = get_test_section_df()

    def compute_iki_wpm(self, test_section_df):
        if len(test_section_df) == 0:
            return np.nan

        time_list = test_section_df['TIMESTAMP'].values
        intervals = (test_section_df['TIMESTAMP'] - test_section_df['TIMESTAMP'].shift()).fillna(0)
        word_num = len(test_section_df.iloc[-1]['INPUT'].split())
        mean_interval_ms = intervals[1:].mean()
        total_time = time_list[-1] - time_list[0]
        if total_time == 0:
            return mean_interval_ms, np.nan
        wpm = word_num / (total_time / 1000 / 60)
        return mean_interval_ms, wpm

    def compute_wmr(self, full_log_data, ite=None, keyboard=None, custom_logdata_path=None):
        """
        Compute Word Modified Ratio (WMR): the ratio of words being modified during typing or after committed
        :param full_log_data: use the full log data or not
        :return:
        """
        if not custom_logdata_path:
            self.load_participants(ite=ite, keyboard=keyboard)
            self.load_logdata(full_log_data)
            self.load_test_sections()

            # get those logdata with test sections id belonging to the selected participants
            participant_ids = self.participants_dataframe['PARTICIPANT_ID'].values

            # remove those rows where PARTICIPANT_ID is not in the selected participants
            self.test_sections_dataframe = self.test_sections_dataframe[
                self.test_sections_dataframe['PARTICIPANT_ID'].isin(participant_ids)]

            # remove those rows where test sections id is not in the selected test sections
            self.logdata_dataframe = self.logdata_dataframe[
                self.logdata_dataframe['TEST_SECTION_ID'].isin(self.test_sections_dataframe['TEST_SECTION_ID'])]
        else:
            self.load_test_sections()
            self.logdata_dataframe = pd.read_csv(custom_logdata_path, names=logdata_columns, encoding='ISO-8859-1')
        # calculation based on the test sections, since the test section id are sorted, we can iterate through the
        # dataframe and calculate the word modified ratio

        # get the data with the same test section id
        test_section_ids = self.logdata_dataframe['TEST_SECTION_ID'].unique()

        print("Total test sections: ", len(test_section_ids))

        # get participant ids based on the test section ids
        participant_ids = self.test_sections_dataframe[
            self.test_sections_dataframe['TEST_SECTION_ID'].isin(test_section_ids)]['PARTICIPANT_ID'].unique()
        print("Total participants: ", len(participant_ids))


        for test_section_id in test_section_ids:
            try:
                # if test_section_id == 3098:
                #     print("test_section_id: ", test_section_id)
                # get the data with the same test section id
                test_section_df = self.logdata_dataframe[self.logdata_dataframe['TEST_SECTION_ID'] == test_section_id]

                # committed sentence from the last row of the test section
                committed_sentence = test_section_df.iloc[-1]['INPUT']
                if len(committed_sentence) < 4:
                    for i in range(2, 5):
                        if len(committed_sentence) < 3:
                            committed_sentence = test_section_df.iloc[-i]['INPUT']
                            test_section_df = test_section_df.iloc[:-1]

                # word modified flag with false value as a list of words in the committed sentence
                word_modified_flag = [False] * len(committed_sentence.split())
                if len(committed_sentence.split()) < 3:
                    continue
                for index, row in test_section_df.iterrows():
                    # if index == 20760:
                    #     print("test_section_id: ", test_section_id)

                    if row['INPUT'] != row['INPUT'] or row['DATA'] != row['DATA'] or row['INPUT'] == ' ' or row[
                        'DATA'] == ' ':
                        continue
                    if row['INPUT'] != committed_sentence[:len(row['INPUT'])]:
                        # compare each character of the input sentence with the committed sentence
                        committed_words = committed_sentence.split()
                        input_words = row['INPUT'].split()
                        typed_word_count = len(input_words)
                        last_word_index = 0
                        if typed_word_count > 1:
                            for i in range(min(typed_word_count, len(committed_words)) - 1):
                                if input_words[i] != committed_words[i]:
                                    if input_words[i] in committed_words:
                                        continue
                                    else:
                                        word_modified_flag[i] = True
                                last_word_index = i + 1

                        if len(input_words[last_word_index]) > len(committed_words[last_word_index]):
                            word_modified_flag[last_word_index] = True
                        else:
                            for j in range(len(input_words[last_word_index])):
                                if input_words[last_word_index][j] != committed_words[last_word_index][j]:
                                    word_modified_flag[last_word_index] = True
                iki, wpm = self.compute_iki_wpm(test_section_df)

                self.test_section_count += 1
                test_section_word_count = len(committed_sentence.split())
                test_section_modified_word_count = sum(word_modified_flag)

                self.word_count += test_section_word_count
                self.modified_word_count += test_section_modified_word_count

                participant_id = self.test_sections_dataframe[
                    self.test_sections_dataframe['TEST_SECTION_ID'] == test_section_id]['PARTICIPANT_ID'].values[0]
                # if test_section_modified_word_count / test_section_word_count >= 0.8:
                #     print("test_section_id: ", test_section_id)

                self.iki_wmr_visualization_df = self.iki_wmr_visualization_df.append(
                    pd.DataFrame([[participant_id, test_section_id, test_section_word_count,
                                   test_section_modified_word_count, iki, wpm]], columns=WMR_VISUALIZATION_COLUMNS))
                if self.test_section_count % 1000 == 0:
                    print('IKI: ', iki)
                    print('WPM: ', wpm)
                    print("Word Modified Ratio (WMR): ", self.modified_word_count / self.word_count)
                    print("test_section_count: ", self.test_section_count)
                    print("test_section_id: ", test_section_id)

            except:
                pass

        print("Word Modified Ratio (WMR): ", self.modified_word_count / self.word_count)


def calculate_wmr_intervals(df, interval_size=10):
    # Define the intervals for IKI
    intervals = np.arange(145, 1045, interval_size)
    df['IKI_interval'] = pd.cut(df['IKI'], bins=intervals, right=False)

    # Group by the IKI_interval and calculate the WMR for each group
    def calculate_wmr(x):
        word_count = x['WORD_COUNT'].sum()
        if word_count == 0:
            return np.nan  # Return NaN if the word count is zero to avoid division by zero
        else:
            return x['MODIFIED_WORK_COUNT'].sum() / word_count

    grouped = df.groupby('IKI_interval').apply(calculate_wmr)

    return grouped.reset_index(name='WMR')


# Plotting function for WMR vs IKI
def plot_wmr_vs_iki(wmr_intervals_df):
    # Assuming IKI_interval is a column with numerical values representing the midpoints of the intervals.
    # Convert interval to string and get the midpoint for the label
    plt.figure(figsize=(12, 6))
    midpoints = wmr_intervals_df['IKI_interval'].apply(lambda x: (x.left + x.right)/2).astype(int)

    # Plot the WMR vs IKI intervals
    plt.bar(midpoints, wmr_intervals_df['WMR'] * 100, width=10, edgecolor='black')

    # Set the title and labels
    plt.title('WMR vs. Typing Interval')
    plt.xlabel('Typing Interval (ms)')
    plt.ylabel('WMR (%)')

    # Set x-ticks to be the midpoints of intervals, but only label every 50ms
    plt.xticks(ticks=midpoints, labels=['' if x % 50 != 0 else str(x) for x in midpoints])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Show the plot
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.show()


if __name__ == "__main__":
    parser = Parse()
    custom_logdata_path = osp.join(DEFAULT_DATASETS_DIR, 'all_keyboard_logdata.csv')
    parser.compute_wmr(full_log_data=True, ite=None, keyboard='Gboard', custom_logdata_path=custom_logdata_path)

    wmr_intervals_df = calculate_wmr_intervals(parser.iki_wmr_visualization_df)
    plot_wmr_vs_iki(wmr_intervals_df)

    parser.save_iki_wmr_visualization(osp.join(DEFAULT_VISUALIZATION_DIR, 'all_keyboard_logdata_iki_wmr_visualization.csv'))
    # parser.compute_wmr(full_log_data=True)
