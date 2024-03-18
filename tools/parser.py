from abc import ABC, abstractmethod
from tools.data_loading import clean_participants_data, get_logdata_df, get_test_section_df
import pandas as pd
from config import logdata_columns, DEFAULT_DATASETS_DIR, DEFAULT_VISUALIZATION_DIR, DEFAULT_CLEANED_DATASETS_DIR, \
    DEFAULT_FIGS_DIR
import os.path as osp
import numpy as np

WMR_VISUALIZATION_COLUMNS = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'WORD_COUNT', 'MODIFIED_WORD_COUNT', 'IKI', 'WPM']
AC_VISUALIZATION_COLUMNS = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'WORD_COUNT', 'AC_WORD_COUNT', 'IKI', 'AC']
MODIFICATION_VISUALIZATION_COLUMNS = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'CHAR_COUNT', 'MODIFICATION_COUNT', 'IKI',
                                      'MODIFICATION']
AGE_VISUALIZATION_COLUMNS = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'IKI', 'AGE']


class Parse(ABC):
    def __init__(self):
        self.participants_dataframe = None
        self.logdata_dataframe = None
        self.test_sections_dataframe = None

        self.word_count = 0
        self.char_count = 0
        self.modified_word_count = 0  # word being modified during typing before committed
        self.auto_corrected_word_count = 0  # word being auto-corrected during typing before committed
        self.modification_count = 0  # modification count on char level

        self.test_section_count = 0

        self.iki_wmr_visualization_df = pd.DataFrame(columns=WMR_VISUALIZATION_COLUMNS)
        self.ac_visualization_df = pd.DataFrame(columns=AC_VISUALIZATION_COLUMNS)
        self.modification_visualization_df = pd.DataFrame(columns=MODIFICATION_VISUALIZATION_COLUMNS)
        self.age_visualization_df = pd.DataFrame(columns=AGE_VISUALIZATION_COLUMNS)

        self.test_section_ids = None

    def save_iki_wmr_visualization(self, path):
        self.iki_wmr_visualization_df.to_csv(path, index=False, header=False)

    def save_iki_ac_visualization(self, path):
        self.ac_visualization_df.to_csv(path, index=False, header=False)

    def save_modification_visualization(self, path):
        self.modification_visualization_df.to_csv(path, index=False, header=False)

    def save_iki_age_visualization(self, path):
        self.age_visualization_df.to_csv(path, index=False, header=False)

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

    def load_data(self, ite=None, keyboard=None, full_log_data=True, custom_logdata_path=None):
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
            if not osp.exists(DEFAULT_CLEANED_DATASETS_DIR):
                raise FileNotFoundError("File not found: ", DEFAULT_CLEANED_DATASETS_DIR)
            self.logdata_dataframe = pd.read_csv(custom_logdata_path, names=logdata_columns, encoding='ISO-8859-1')
        # calculation based on the test sections, since the test section id are sorted, we can iterate through the
        # dataframe and calculate the word modified ratio

        # get the data with the same test section id
        self.test_section_ids = self.logdata_dataframe['TEST_SECTION_ID'].unique()

        print("Total test sections: ", len(self.test_section_ids))

        # get participant ids based on the test section ids
        participant_ids = self.test_sections_dataframe[
            self.test_sections_dataframe['TEST_SECTION_ID'].isin(self.test_section_ids)]['PARTICIPANT_ID'].unique()
        print("Total participants: ", len(participant_ids))

    def get_test_section_df(self, test_section_id):
        test_section_df = self.logdata_dataframe[self.logdata_dataframe['TEST_SECTION_ID'] == test_section_id]

        # committed sentence from the last row of the test section
        committed_sentence = test_section_df.iloc[-1]['INPUT']
        if len(committed_sentence) < 4 or committed_sentence != committed_sentence:
            for i in range(2, 5):
                if len(committed_sentence) < 3:
                    committed_sentence = test_section_df.iloc[-i]['INPUT']
                    test_section_df = test_section_df.iloc[:-1]

        return test_section_df, committed_sentence

    def get_age(self, full_log_data, ite=None, keyboard=None, custom_logdata_path=None):
        """
        get age vs iki based on the test section id
        :return:
        """
        self.test_section_count = 0
        self.load_data(ite=ite, keyboard=keyboard, full_log_data=full_log_data, custom_logdata_path=custom_logdata_path)
        self.load_participants(ite=ite, keyboard=keyboard)
        for test_section_id in self.test_section_ids:
            try:
                test_section_df, committed_sentence = self.get_test_section_df(test_section_id)

                age = self.participants_dataframe[self.participants_dataframe['PARTICIPANT_ID'] ==
                                                  self.test_sections_dataframe[
                                                      self.test_sections_dataframe[
                                                          'TEST_SECTION_ID'] == test_section_id][
                                                      'PARTICIPANT_ID'].values[0]]['AGE'].values[0]

                iki, wpm = self.compute_iki_wpm(test_section_df)
                participant_id = self.test_sections_dataframe[
                    self.test_sections_dataframe['TEST_SECTION_ID'] == test_section_id]['PARTICIPANT_ID'].values[0]
                self.test_section_count += 1

                self.age_visualization_df = self.age_visualization_df.append(
                    pd.DataFrame([[participant_id, test_section_id, iki, age]], columns=AGE_VISUALIZATION_COLUMNS))
                if self.test_section_count % 1000 == 0:
                    print('IKI: ', iki)
                    print('AGE: ', age)
                    print("test_section_count: ", self.test_section_count)
                    print("test_section_id: ", test_section_id)
            except:
                pass

    def compute_modification(self, full_log_data, ite=None, keyboard=None, custom_logdata_path=None):
        """
        Compute the modification ratio on char level
        :return:
        """
        self.test_section_count = 0
        self.load_data(ite=ite, keyboard=keyboard, full_log_data=full_log_data, custom_logdata_path=custom_logdata_path)
        for test_section_id in self.test_section_ids:
            try:
                test_section_df, committed_sentence = self.get_test_section_df(test_section_id)

                # word modified flag with false value as a list of words in the committed sentence
                if len(committed_sentence.split()) < 3:
                    continue
                # count how many characters in the committed sentence
                committed_char_count = len(committed_sentence)
                modification_count = 0
                pre_input = ''
                for index, row in test_section_df.iterrows():
                    if row['INPUT'] != row['INPUT'] or row['DATA'] != row['DATA'] or row['INPUT'] == ' ' or row[
                        'DATA'] == ' ':
                        continue
                    current_input = row['INPUT']
                    if row['ITE_AUTO']:
                        # auto-corrected word count, we do not consider
                        pass
                    elif len(current_input) - len(pre_input) == 1 and current_input[:-1] == pre_input:

                        pass
                    elif len(current_input) < len(pre_input):
                        # using backspace to delete the last character
                        modification_count += 1
                    else:
                        # move the cursor to the middle of the sentence and start typing
                        # compare each char betwen the pre_input and current_input to find the modification
                        for i in range(len(pre_input)):
                            if pre_input[i] != current_input[i]:
                                modification_count += 1
                                break
                    pre_input = current_input
                iki, wpm = self.compute_iki_wpm(test_section_df)
                self.test_section_count += 1
                participant_id = self.test_sections_dataframe[
                    self.test_sections_dataframe['TEST_SECTION_ID'] == test_section_id]['PARTICIPANT_ID'].values[0]

                self.char_count += committed_char_count
                self.modification_count += modification_count

                self.modification_visualization_df = self.modification_visualization_df.append(
                    pd.DataFrame(
                        [[participant_id, test_section_id, committed_char_count, modification_count, iki, wpm]],
                        columns=MODIFICATION_VISUALIZATION_COLUMNS))

                if self.test_section_count % 1000 == 0:
                    print('IKI: ', iki)
                    print('WPM: ', wpm)
                    print("Modification ratio: ", self.modification_count / self.char_count)
                    print("test_section_count: ", self.test_section_count)
                    print("test_section_id: ", test_section_id)

            except:
                pass
        print("Modification ratio: ", self.modification_count / self.char_count)

    def compute_ac(self, full_log_data, ite=None, keyboard=None, custom_logdata_path=None):
        """
        Compute auto-corrected ratio on word level
        :return:
        """
        self.test_section_count = 0
        self.load_data(ite=ite, keyboard=keyboard, full_log_data=full_log_data, custom_logdata_path=custom_logdata_path)
        for test_section_id in self.test_section_ids:
            try:
                test_section_df, committed_sentence = self.get_test_section_df(test_section_id)

                # word modified flag with false value as a list of words in the committed sentence
                if len(committed_sentence.split()) < 3:
                    continue
                # auto corrected word count equals to the number of rows that ITE_AUTO is True
                auto_corrected_word_count = test_section_df[test_section_df['ITE_AUTO'] == True].shape[0]
                # if auto_corrected_word_count == 0:
                #     continue
                self.auto_corrected_word_count += auto_corrected_word_count

                world_count = len(committed_sentence.split())
                self.word_count += world_count

                iki, wpm = self.compute_iki_wpm(test_section_df)

                self.test_section_count += 1

                participant_id = self.test_sections_dataframe[
                    self.test_sections_dataframe['TEST_SECTION_ID'] == test_section_id]['PARTICIPANT_ID'].values[0]

                self.ac_visualization_df = self.ac_visualization_df.append(
                    pd.DataFrame([[participant_id, test_section_id, world_count, auto_corrected_word_count, iki, wpm]],
                                 columns=AC_VISUALIZATION_COLUMNS))
                if self.test_section_count % 1000 == 0:
                    print('IKI: ', iki)
                    print('WPM: ', wpm)
                    print("Auto-corrected ratio: ", self.auto_corrected_word_count / self.word_count)
                    print("test_section_count: ", self.test_section_count)
                    print("test_section_id: ", test_section_id)

            except:
                pass
        print("Auto-corrected ratio: ", self.auto_corrected_word_count / self.word_count)

    def compute_wmr(self, full_log_data, ite=None, keyboard=None, custom_logdata_path=None):
        """
        Compute Word Modified Ratio (WMR): the ratio of words being modified during typing or after committed
        :param full_log_data: use the full log data or not
        :return:
        """
        self.test_section_count = 0
        self.load_data(ite=ite, keyboard=keyboard, full_log_data=full_log_data, custom_logdata_path=custom_logdata_path)
        for test_section_id in self.test_section_ids:
            try:
                test_section_df, committed_sentence = self.get_test_section_df(test_section_id)

                # word modified flag with false value as a list of words in the committed sentence
                word_modified_flag = [False] * len(committed_sentence.split())
                if len(committed_sentence.split()) < 3:
                    continue
                for index, row in test_section_df.iterrows():
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
                auto_corrected_word_count = test_section_df[test_section_df['ITE_AUTO'] == True].shape[0]
                self.word_count += test_section_word_count
                self.modified_word_count += test_section_modified_word_count
                self.auto_corrected_word_count += auto_corrected_word_count

                participant_id = self.test_sections_dataframe[
                    self.test_sections_dataframe['TEST_SECTION_ID'] == test_section_id]['PARTICIPANT_ID'].values[0]

                self.iki_wmr_visualization_df = self.iki_wmr_visualization_df.append(
                    pd.DataFrame([[participant_id, test_section_id, test_section_word_count,
                                   test_section_modified_word_count, iki, wpm]], columns=WMR_VISUALIZATION_COLUMNS))
                if self.test_section_count % 1000 == 0:
                    print('IKI: ', iki)
                    print('WPM: ', wpm)
                    print("Word Modified Ratio (WMR): ", self.modified_word_count / self.word_count)
                    print("Auto-corrected ratio: ", self.auto_corrected_word_count / self.word_count)
                    print("test_section_count: ", self.test_section_count)
                    print("test_section_id: ", test_section_id)

            except:
                pass

        print("Word Modified Ratio (WMR): ", self.modified_word_count / self.word_count)

    def filter(self, ite=None, keyboard=None, filter_ratio=0.5, save_file_name=None, load_file_name=None):
        self.test_section_count = 0
        self.load_data(ite=ite, keyboard=keyboard, custom_logdata_path=load_file_name)
        temp_df = pd.DataFrame(columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'IKI'])
        for test_section_id in self.test_section_ids:
            try:
                test_section_df, committed_sentence = self.get_test_section_df(test_section_id)

                # word modified flag with false value as a list of words in the committed sentence
                iki, wpm = self.compute_iki_wpm(test_section_df)

                self.test_section_count += 1

                participant_id = self.test_sections_dataframe[
                    self.test_sections_dataframe['TEST_SECTION_ID'] == test_section_id]['PARTICIPANT_ID'].values[0]

                temp_df = temp_df.append(
                    pd.DataFrame([[participant_id, test_section_id, iki]],
                                 columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'IKI']))
            except:
                pass
        iki_filter = temp_df['IKI'].quantile(filter_ratio)

        # get those test section id with iki higher that iki_filter
        temp_df = temp_df[temp_df['IKI'] > iki_filter]

        # remain those rows in self.logdata_dataframe with the same test section id
        self.logdata_dataframe = self.logdata_dataframe[
            self.logdata_dataframe['TEST_SECTION_ID'].isin(temp_df['TEST_SECTION_ID'])]

        # add a new column for self.logdata_dataframe storing iki between each time step, if the test section id is the first new one, then the iki is 0
        # get the unique test section id
        test_section_ids = self.logdata_dataframe['TEST_SECTION_ID'].unique()
        for test_section_id in test_section_ids:
            test_section_df = self.logdata_dataframe[self.logdata_dataframe['TEST_SECTION_ID'] == test_section_id]
            iki_list = [0]
            for i in range(1, len(test_section_df)):
                iki_list.append(test_section_df.iloc[i]['TIMESTAMP'] - test_section_df.iloc[i - 1]['TIMESTAMP'])
            self.logdata_dataframe.loc[self.logdata_dataframe['TEST_SECTION_ID'] == test_section_id, 'IKI'] = iki_list

        # print the number of test sections  and unique participants num after filtering
        print("Number of test sections after filtering: ", len(temp_df['TEST_SECTION_ID'].unique()))
        print("Number of unique participants after filtering: ", len(temp_df['PARTICIPANT_ID'].unique()))

        self.logdata_dataframe.to_csv(osp.join(DEFAULT_CLEANED_DATASETS_DIR, save_file_name), index=False, header=False)


def compare_sentences(sentence1, sentence2):
    # Split the sentences into words
    words1 = sentence1.split()
    words2 = sentence2.split()

    # Ensure both sentences have the same number of words
    if len(words1) != len(words2):
        return False

    # Compare each pair of words
    for word1, word2 in zip(words1, words2):
        # Check the length of the words first
        if len(word1) != len(word2):
            return True  # Different lengths mean more than one char differs

        # Count the number of differing characters
        diff_count = sum(1 for c1, c2 in zip(word1, word2) if c1 != c2)

        # If more than one char differs, return True
        if diff_count > 1:
            return True

    # If no word has more than one char difference, return False
    return False


if __name__ == "__main__":
    parser = Parse()
    # logdata_path = osp.join(DEFAULT_CLEANED_DATASETS_DIR, 'ac_logdata.csv')
    # parser.compute_wmr(full_log_data=True, ite=None, keyboard='Gboard', custom_logdata_path=logdata_path)
    #
    # wmr_intervals_df = calculate_iki_intervals(parser.iki_wmr_visualization_df)
    # plot_wmr_vs_iki(wmr_intervals_df)
    #
    # parser.save_iki_wmr_visualization(
    #     osp.join(DEFAULT_VISUALIZATION_DIR, 'all_keyboard_logdata_iki_wmr_visualization.csv'))
    # parser.compute_wmr(full_log_data=True, ite=None, keyboard='Gboard', custom_logdata_path=logdata_path)
    # parser.compute_ac(full_log_data=True, ite=None, keyboard='Gboard', custom_logdata_path=logdata_path)

    # sentence1 = "Hello world"
    # sentence2 = "Hella warld"
    # print(compare_sentences(sentence1, sentence2))  # This should return True
