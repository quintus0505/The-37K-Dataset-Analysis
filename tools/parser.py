from abc import ABC, abstractmethod
from tools.data_loading import clean_participants_data, get_logdata_df, get_test_section_df, get_sentences_df
import pandas as pd
from config import logdata_columns, DEFAULT_DATASETS_DIR, DEFAULT_VISUALIZATION_DIR, DEFAULT_CLEANED_DATASETS_DIR, \
    DEFAULT_FIGS_DIR, test_sections_columns
import os.path as osp
import numpy as np
import Levenshtein
from tools.string_analysis import *
import Levenshtein as lev
from tools.string_functions import *

WMR_VISUALIZATION_COLUMNS = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'WORD_COUNT', 'MODIFIED_WORD_COUNT', 'IKI', 'WPM']
AC_VISUALIZATION_COLUMNS = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'WORD_COUNT', 'AC_WORD_COUNT', 'IKI', 'AC']
MODIFICATION_VISUALIZATION_COLUMNS = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'CHAR_COUNT', 'MODIFICATION_COUNT', 'IKI',
                                      'MODIFICATION']
AGE_VISUALIZATION_COLUMNS = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'IKI', 'AGE']
EDIT_DISTANCE_VISUALIZATION_COLUMNS = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'EDIT_DISTANCE', 'IKI', 'ERROR_RATE']


class Parse(ABC):
    def __init__(self):
        self.participants_dataframe = None
        self.logdata_dataframe = None
        self.test_sections_dataframe = None
        self.sentences_dataframe = None

        self.word_count = 0
        self.char_count = 0
        self.modified_word_count = 0  # word being modified during typing before committed
        self.auto_corrected_word_count = 0  # word being auto-corrected during typing before committed
        self.modification_count = 0  # modification count on char level

        self.uncorrected_error_rates = []
        self.corrected_error_rates = []
        self.immediate_error_correction_rates = []
        self.delayed_error_correction_rates = []

        self.auto_capitalization_count = 0

        self.test_section_count = 0

        self.iki_wmr_visualization_df = pd.DataFrame(columns=WMR_VISUALIZATION_COLUMNS)
        self.ac_visualization_df = pd.DataFrame(columns=AC_VISUALIZATION_COLUMNS)
        self.modification_visualization_df = pd.DataFrame(columns=MODIFICATION_VISUALIZATION_COLUMNS)
        self.age_visualization_df = pd.DataFrame(columns=AGE_VISUALIZATION_COLUMNS)
        self.edit_distance_visualization_df = pd.DataFrame(columns=EDIT_DISTANCE_VISUALIZATION_COLUMNS)

        self.test_section_ids = None

    def save_iki_wmr_visualization(self, path):
        self.iki_wmr_visualization_df.to_csv(path, index=False, header=False)

    def save_iki_ac_visualization(self, path):
        self.ac_visualization_df.to_csv(path, index=False, header=False)

    def save_modification_visualization(self, path):
        self.modification_visualization_df.to_csv(path, index=False, header=False)

    def save_edit_distance_visualization(self, path):
        self.edit_distance_visualization_df.to_csv(path, index=False, header=False)

    def save_iki_age_visualization(self, path):
        self.age_visualization_df.to_csv(path, index=False, header=False)

    def load_participants(self, ite=None, keyboard='Gboard'):
        self.participants_dataframe = clean_participants_data(ite=ite, keyboard=keyboard)

    def load_logdata(self, full_log_data):
        self.logdata_dataframe = get_logdata_df(full_log_data)

    def load_test_sections(self):
        self.test_sections_dataframe = get_test_section_df()

    def load_sentences(self):
        self.sentences_dataframe = get_sentences_df()

    def compute_iki_wpm(self, test_section_df):
        if len(test_section_df) == 0:
            return np.nan, np.nan

        # Copy the DataFrame to avoid SettingWithCopyWarning.
        df = test_section_df.copy()

        # replace the nan in INPUT as empty string

        df['INPUT'] = df['INPUT'].fillna('')
        # Add a new column for the previous timestamp.
        df['PREV_TIMESTAMP'] = df['TIMESTAMP'].shift()
        df['PREV_INPUT'] = df['INPUT'].shift()
        df.loc[df.index[0], 'PREV_INPUT'] = ''
        # Calculate IKI for all rows.
        df['IKI'] = df['TIMESTAMP'] - df['PREV_TIMESTAMP']

        # Set the IKI for the first row to NaN.
        df.loc[df.index[0], 'IKI'] = np.nan

        # Skip rows where DATA is a space when computing the IKI mean.
        is_space = (df['DATA'] == ' ')
        df.loc[is_space, 'IKI'] = np.nan

        # Skip rows where not len(df['INPUT']) - len(df['PREV_INPUT']) == 1 and df['PREV_INPUT'] != df['INPUT'][:-1]
        # when computing the IKI mean, also do not consider the first row, start from the second row
        for idx in df.index[1:]:
            current_input = df.at[idx, 'INPUT']
            prev_input = df.at[idx, 'PREV_INPUT']
            # Check if the current input is a direct continuation of the previous input
            if not (prev_input == current_input[:-1] or prev_input.startswith(current_input)):
                df.at[idx, 'IKI'] = np.nan

        # Calculate the mean IKI, excluding NaN values.
        mean_iki = df.loc[~is_space, 'IKI'].mean()

        # Calculate the number of words, considering only the last entry and ignoring spaces.
        word_num = len(test_section_df.iloc[-1]['INPUT'].split())

        # Calculate the total typing duration.
        total_time_ms = df['TIMESTAMP'].iloc[-1] - df['TIMESTAMP'].iloc[0]

        if total_time_ms == 0:
            return mean_iki, np.nan

        # Calculate words per minute.
        wpm = word_num / (total_time_ms / 1000 / 60)

        return mean_iki, wpm

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
        if len(committed_sentence) < 4 or committed_sentence != committed_sentence or committed_sentence[-1] == ' ':
            for i in range(2, 5):
                if len(committed_sentence) < 3 or committed_sentence[-1] == ' ':
                    test_section_df = test_section_df.iloc[:-1]
                    committed_sentence = test_section_df.iloc[-1]['INPUT']
        # while committed_sentence[-1] == ' ':
        #     committed_sentence = committed_sentence[:-1]
        #     test_section_df.iloc[-1]['INPUT'] = committed_sentence

        return test_section_df, committed_sentence

    def get_age(self, full_log_data, ite=None, keyboard=None, custom_logdata_path=None):
        """
        get age vs iki based on the test section id
        :return:
        """
        iter_count = 0
        self.test_section_count = 0
        self.load_data(ite=ite, keyboard=keyboard, full_log_data=full_log_data, custom_logdata_path=custom_logdata_path)
        self.load_participants(ite=ite, keyboard=keyboard)
        for test_section_id in self.test_section_ids:
            iter_count += 1
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
                if iter_count % 1000 == 0:
                    print('IKI: ', iki)
                    print('AGE: ', age)
                    print("test_section_count: ", self.test_section_count)
                    print("test_section_id: ", test_section_id)
            except:
                pass

    def compute_edit_distance(self, full_log_data, ite=None, keyboard=None, custom_logdata_path=None):
        """
        Compute the edit distance between the committed sentence and the input sentence
        :return:
        """
        iter_count = 0
        self.test_section_count = 0
        self.load_data(ite=ite, keyboard=keyboard, full_log_data=full_log_data, custom_logdata_path=custom_logdata_path)
        self.load_test_sections()
        self.load_sentences()
        for test_section_id in self.test_section_ids:
            iter_count += 1
            try:
                test_section_df, committed_sentence = self.get_test_section_df(test_section_id)
                if committed_sentence != committed_sentence:
                    continue

                iki, wpm = self.compute_iki_wpm(test_section_df)
                self.test_section_count += 1
                participant_id = self.test_sections_dataframe[
                    self.test_sections_dataframe['TEST_SECTION_ID'] == test_section_id]['PARTICIPANT_ID'].values[0]
                # edit_distance = self.test_sections_dataframe[
                #     self.test_sections_dataframe['TEST_SECTION_ID'] == test_section_id]['EDIT_DISTANCE'].values[0]
                sentence_id = self.test_sections_dataframe[
                    self.test_sections_dataframe['TEST_SECTION_ID'] == test_section_id]['SENTENCE_ID'].values[0]
                target_sentence = \
                    self.sentences_dataframe[self.sentences_dataframe['SENTENCE_ID'] == sentence_id]['SENTENCE'].values[
                        0]
                edit_distance = Levenshtein.distance(committed_sentence, target_sentence)
                error_rate = self.test_sections_dataframe[
                    self.test_sections_dataframe['TEST_SECTION_ID'] == test_section_id]['ERROR_RATE'].values[0]
                self.edit_distance_visualization_df = self.edit_distance_visualization_df.append(
                    pd.DataFrame(
                        [[participant_id, test_section_id, edit_distance,
                          iki, error_rate]],
                        columns=EDIT_DISTANCE_VISUALIZATION_COLUMNS))

                if iter_count % 1000 == 0:
                    print('IKI: ', iki)
                    print("Edit distance: ", edit_distance)
                    print("test_section_count: ", self.test_section_count)
                    print("test_section_id: ", test_section_id)

            except:
                pass

    @staticmethod
    def compute_if_c_count_for_auto_correction(str1, str2):
        """
        :param str1: original string (word)
        :param str2: auto-corrected string (word)
        :return:
        """
        # Create a matrix to store the distances and matches
        matrix = [[[0, 0] for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]

        # Initialize the first row and column of the matrix
        for i in range(len(str1) + 1):
            matrix[i][0] = [i, 0]  # Distance, Matches
        for j in range(len(str2) + 1):
            matrix[0][j] = [j, 0]  # Distance, Matches

        # Populate the matrix
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if str1[i - 1] == str2[j - 1]:
                    cost = 0
                    matches = matrix[i - 1][j - 1][1] + 1  # Increase matches
                else:
                    cost = 1
                    matches = max(matrix[i - 1][j][1], matrix[i][j - 1][1], matrix[i - 1][j - 1][1])

                # Calculate distances and update matches
                dist_del = matrix[i - 1][j][0] + 1
                dist_ins = matrix[i][j - 1][0] + 1
                dist_sub = matrix[i - 1][j - 1][0] + cost

                min_dist = min(dist_del, dist_ins, dist_sub)
                matrix[i][j] = [min_dist, matches]

                # Ensure the match count does not decrease
                if min_dist == dist_sub:
                    matrix[i][j][1] = max(matrix[i][j][1], matrix[i - 1][j - 1][1])

        # The last element of the matrix contains the distance and the matches
        distance, matches = matrix[-1][-1]
        char_count = len(str2)
        return distance, matches, char_count

    def reformat_input(self, test_section_df):
        reformatted_input = ""
        pre_input = ""
        reformat_if_count = 0
        reformat_c_count = 0
        reformat_f_count = 0
        bsp_count = 0
        bsp_index_list = []
        auto_corrected_word_count = 0

        auto_correct_flag = False

        immediate_error_correction_count = 0
        delayed_error_correction_count = 0

        for index, row in test_section_df.iterrows():
            if row['INPUT'] != row['INPUT']:
                continue
            current_input = row['INPUT']
            if current_input != current_input:
                current_input = ''
            if current_input == pre_input:
                continue
            if len(current_input) > len(pre_input):
                if current_input[:len(pre_input)] == pre_input:
                    # normal typing after last input
                    reformatted_input += current_input[len(pre_input):]
                elif current_input[:len(pre_input)].lower() == pre_input.lower():
                    # auto capitalization
                    self.auto_capitalization_count += 1
                    auto_correct_flag = True
                    # first find the low or upper case different and correct it
                    for i in range(len(pre_input)):
                        if pre_input[i] != current_input[i]:
                            # change the reformatted_input[i] to current_input[i]
                            # count how many bsp before i
                            bsp_count_before = 0
                            if bsp_index_list:
                                for bsp_index in bsp_index_list:
                                    if bsp_index < i:
                                        bsp_count_before += 1
                            reformatted_input = reformatted_input[:i + 2 * bsp_count_before] + current_input[
                                i] + reformatted_input[i + 2 * bsp_count_before + 1:]
                            break
                    # then add the rest of the input
                    reformat_if_count += 1
                    reformat_f_count += 1
                    reformatted_input += current_input[len(pre_input):]

                    delayed_error_correction_count += 1

                else:
                    # move the cursor to the middle of the sentence and start typing,
                    # we just assume that every keystroke result in only one change
                    for i in range(len(pre_input)):
                        if pre_input[i] != current_input[i]:
                            # insert current_input[i] to the reformatted_input at the same position
                            # TODO: Let's assume that all this action result in correct typing
                            bsp_count_before = 0
                            if bsp_index_list:
                                for bsp_index in bsp_index_list:
                                    if bsp_index < i:
                                        bsp_count_before += 1
                            reformatted_input = reformatted_input[:i + 2 * bsp_count_before] + current_input[
                                i] + reformatted_input[i + 2 * bsp_count_before:]
                            reformat_if_count += 1
                            break

                    delayed_error_correction_count += 1
            elif row['ITE_AUTO'] or len(current_input) == len(pre_input):
                #  use auto correction
                if not row['ITE_AUTO'] and current_input.lower() == pre_input.lower():
                    if len(current_input) > 1:
                        self.auto_capitalization_count += 1
                    for i in range(len(pre_input)):
                        if pre_input[i] != current_input[i]:
                            bsp_count_before = 0
                            if bsp_index_list:
                                for bsp_index in bsp_index_list:
                                    if bsp_index < i:
                                        bsp_count_before += 1
                            reformatted_input = reformatted_input[:i + 2 * bsp_count_before] + current_input[
                                i] + reformatted_input[i + 1 + 2 * bsp_count_before:]
                            break
                    reformat_if_count += 1
                    if len(current_input) == 1:
                        auto_correct_flag = False
                    else:
                        auto_correct_flag = True

                    pre_input = current_input

                    delayed_error_correction_count += 1
                    continue

                elif not row['ITE_AUTO'] and current_input[:-1] == pre_input[:-1]:
                    # not detected as autocorrected but the log looks like that, maybe multiple input in one keystroke
                    bsp_count_before = 0
                    if bsp_index_list:
                        for bsp_index in bsp_index_list:
                            if bsp_index < len(current_input) - 1:
                                bsp_count_before += 1
                    # replace the last character in the reformatted_input with the last character in the current_input
                    reformatted_input = reformatted_input[:len(current_input) - 2 + 2 * bsp_count_before] + \
                                        current_input[-1] + \
                                        reformatted_input[len(current_input) - 1 + 2 * bsp_count_before:]
                    reformat_if_count += 1
                    if len(current_input) == 1 or (current_input[-1] == '.' and pre_input[-1] == ' '):
                        auto_correct_flag = False
                    else:
                        auto_correct_flag = True

                    pre_input = current_input

                    immediate_error_correction_count += 1

                    continue
                elif len(reformatted_input.split()) > 1:
                    # replace the last word in the reformatted_input
                    # with the last word in the current_input
                    # if more than one words in the current_input, only consider the last word
                    reformatted_input = reformatted_input.rsplit(' ', 1)[0] + ' ' + current_input.rsplit(' ', 1)[1]
                    word_before_modification = pre_input.rsplit(' ', 1)[1]
                    word_after_modification = current_input.rsplit(' ', 1)[1]

                    delayed_error_correction_count += 1
                    auto_correct_flag = True
                else:
                    # if only one word is typed
                    reformatted_input = current_input
                    word_before_modification = pre_input
                    word_after_modification = current_input

                    delayed_error_correction_count += 1
                    auto_correct_flag = True

                if_count, c_count, word_count = self.compute_if_c_count_for_auto_correction(word_before_modification,
                                                                                            word_after_modification)
                reformat_if_count += if_count
                reformat_c_count += c_count
                reformat_f_count += 1
                auto_corrected_word_count += word_count
            else:
                # using backspace to delete
                if len(pre_input) - len(current_input) == 1 and pre_input[:-1] == current_input:
                    # if the backspace is used to delete the last character
                    reformatted_input += '<'
                    bsp_count += 1
                    bsp_index_list.append(len(current_input) - 1)

                    immediate_error_correction_count += 1

                elif lev.distance(pre_input, current_input) == 1:  # let us assume no miss detected autocorrection
                    # or multiple input in one keystroke
                    # find if the last word in the reformatted_input is not the same
                    # as the last word in the current_input
                    # move the cursor to the middle of the sentence and use backspace to delete
                    for i in range(len(pre_input)):
                        if pre_input[i] != current_input[i]:
                            bsp_count_before = 0
                            if bsp_index_list:
                                for bsp_index in bsp_index_list:
                                    if bsp_index < i - 1:
                                        bsp_count_before += 1
                            reformatted_input = reformatted_input[:i + 2 * bsp_count_before] + '<' + reformatted_input[
                                                                                                     i + 2 * bsp_count_before:]
                            reformat_c_count += 1
                            bsp_count += 1
                            bsp_index_list.append(i)
                            break
                    delayed_error_correction_count += 1
                elif pre_input.rsplit(' ', 1)[1] != current_input.rsplit(' ', 1)[1]:
                    # miss detected autocorrection
                    # replace the last word in the reformatted_input with the last word in the current_input
                    auto_correct_flag = True
                    reformatted_input = reformatted_input.rsplit(' ', 1)[0] + ' ' + current_input.rsplit(' ', 1)[1]
                    word_before_modification = pre_input.rsplit(' ', 1)[1]
                    word_after_modification = current_input.rsplit(' ', 1)[1]
                    if_count, c_count, word_count = self.compute_if_c_count_for_auto_correction(
                        word_before_modification,
                        word_after_modification)
                    reformat_if_count += if_count
                    reformat_c_count += c_count
                    reformat_f_count += 1
                    auto_corrected_word_count += word_count

                    delayed_error_correction_count += 1
                else:
                    # move the cursor to the middle of the sentence and use backspace to delete
                    for i in range(len(pre_input)):
                        if pre_input[i] != current_input[i]:
                            bsp_count_before = 0
                            if bsp_index_list:
                                for bsp_index in bsp_index_list:
                                    if bsp_index < i - 1:
                                        bsp_count_before += 1
                            reformatted_input = reformatted_input[:i + 2 * bsp_count_before] + '<' + reformatted_input[
                                                                                                     i + 2 * bsp_count_before:]
                            reformat_c_count += 1
                            bsp_count += 1
                            bsp_index_list.append(i)
                            break
                    delayed_error_correction_count += 1
            pre_input = current_input

        return reformatted_input, reformat_if_count, reformat_c_count, auto_corrected_word_count, \
               reformat_f_count, auto_correct_flag, immediate_error_correction_count, delayed_error_correction_count

    def compute_error_rate_correction(self, full_log_data, ite=None, keyboard=None, custom_logdata_path=None):
        """
        Compute the error rate between the committed sentence and the input sentence
        return corrected error rate, uncorrected error rate, immediate error correction rate and delayed error correction rate
        :return:
        """
        total_correct_count, total_inf_count, total_if_count, total_fix_count = 0, 0, 0, 0
        self.test_section_count = 0
        iter_count = 0
        self.load_data(ite=ite, keyboard=keyboard, full_log_data=full_log_data, custom_logdata_path=custom_logdata_path)
        self.load_test_sections()
        self.load_sentences()

        auto_corrected_test_section_count = 0
        abandoned_test_section_df = pd.DataFrame(columns=logdata_columns)

        detected_autocorrected_test_section_ids = []

        for test_section_id in self.test_section_ids:
            iter_count += 1
            # if test_section_id == 2855:
            #     print("test_section_id: ", test_section_id)
            try:
                test_section_df, committed_sentence = self.get_test_section_df(test_section_id)
                if committed_sentence != committed_sentence:
                    continue
                sentence_id = self.test_sections_dataframe[
                    self.test_sections_dataframe['TEST_SECTION_ID'] == test_section_id]['SENTENCE_ID'].values[0]
                target_sentence = self.sentences_dataframe[self.sentences_dataframe['SENTENCE_ID'] == sentence_id][
                    'SENTENCE'].values[0]

                reformatted_input, auto_corrected_if_count, auto_corrected_c_count, \
                auto_corrected_word_count, auto_correct_count, auto_correct_flag, \
                immediate_error_correction_count, delayed_error_correction_count = self.reformat_input(test_section_df)

                if auto_correct_flag:
                    auto_corrected_test_section_count += 1
                    if ite is None:
                        detected_autocorrected_test_section_ids.append(test_section_id)
                    continue

                target_sentence += 'eof'
                committed_sentence += 'eof'
                reformatted_input += 'eof'
                flagged_IS = flag_input_stream(reformatted_input)

                _, MSD = min_string_distance(target_sentence, committed_sentence)

                alignments = []

                align(target_sentence, committed_sentence, MSD, len(target_sentence), len(committed_sentence), "", "",
                      alignments)

                all_triplets = stream_align(flagged_IS, alignments)
                all_edited_triplets = assign_position_values(all_triplets)
                all_error_lists = error_detection(all_edited_triplets)
                lev_distance = lev.distance(target_sentence, committed_sentence)
                for error_list in all_error_lists:
                    # remove the "eof"
                    error_list = error_list[:-3]
                    inf_count, if_count, correct_count, fix_count = count_component(error_list, verbose=False)
                    if inf_count == lev_distance:
                        break
                # correct_count, inf_count, if_count, fix_count = track_typing_errors(target_sentence,
                #                                                                     reformatted_input)
                correct_count += auto_corrected_c_count - auto_corrected_word_count
                if_count += auto_corrected_if_count
                fix_count += auto_correct_count
                self.test_section_count += 1

                uncorrected_error_rate = inf_count / (correct_count + inf_count + if_count)
                corrected_error_rate = if_count / (correct_count + inf_count + if_count)
                immediate_error_correction_rate = immediate_error_correction_count / len(committed_sentence)
                delayed_error_correction_rate = delayed_error_correction_count / len(committed_sentence)

                self.corrected_error_rates.append(corrected_error_rate)
                self.uncorrected_error_rates.append(uncorrected_error_rate)
                self.immediate_error_correction_rates.append(immediate_error_correction_rate)
                self.delayed_error_correction_rates.append(delayed_error_correction_rate)
                # if uncorrected_error_rate > 0.25 or corrected_error_rate > 0.25:
                #     print("test_section_count: ", self.test_section_count)
                #     print("test_section_id: ", test_section_id)
                #     print("Corrected error rate: ", corrected_error_rate)
                #     print("Uncorrected error rate: ", uncorrected_error_rate)
                if iter_count % 1000 == 0:
                    print("Total test sections count", iter_count)
                    print("Selected test sections count: ", self.test_section_count)
                    print("Detected auto corrected test section count: ", auto_corrected_test_section_count)
                    print("Auto capitalization count: ", self.auto_capitalization_count)
                    print("test_section_id: ", test_section_id)
                    print("Corrected error rate mean: ", np.mean(self.corrected_error_rates))
                    print("Corrected error rate std: ", np.std(self.corrected_error_rates))
                    print("Uncorrected error rate mean: ", np.mean(self.uncorrected_error_rates))
                    print("Uncorrected error rate std: ", np.std(self.uncorrected_error_rates))
                    print("Immediate error correction rate mean: ", np.mean(self.immediate_error_correction_rates))
                    print("Immediate error correction rate std: ", np.std(self.immediate_error_correction_rates))
                    print("Delayed error correction rate mean: ", np.mean(self.delayed_error_correction_rates))
                    print("Delayed error correction rate std: ", np.std(self.delayed_error_correction_rates))

            except:
                # try:
                #     auto_corrected_word_count, auto_correct_count, auto_correct_flag = self.reformat_input(test_section_df)
                # except:
                # try:
                #     auto_corrected_word_count, auto_correct_count, auto_correct_flag = self.reformat_input(test_section_df)
                # except:
                pass
                # add current test section to abandoned_test_section_df
            abandoned_test_section_df = abandoned_test_section_df.append(test_section_df)

        print("Total test sections count", iter_count)
        print("Selected test sections count: ", self.test_section_count)
        print("Detected auto corrected test section count: ", auto_corrected_test_section_count)
        print("Auto capitalization count: ", self.auto_capitalization_count)
        print("Corrected error rate mean: ", np.mean(self.corrected_error_rates))
        print("Corrected error rate std: ", np.std(self.corrected_error_rates))
        print("Uncorrected error rate mean: ", np.mean(self.uncorrected_error_rates))
        print("Uncorrected error rate std: ", np.std(self.uncorrected_error_rates))
        print("Immediate error correction rate mean: ", np.mean(self.immediate_error_correction_rates))
        print("Immediate error correction rate std: ", np.std(self.immediate_error_correction_rates))
        print("Delayed error correction rate mean: ", np.mean(self.delayed_error_correction_rates))
        print("Delayed error correction rate std: ", np.std(self.delayed_error_correction_rates))

        # save the abandoned test sections to a csv file
        abandoned_test_section_df.to_csv(osp.join(DEFAULT_CLEANED_DATASETS_DIR, 'abandoned_test_sections.csv'),
                                         index=False)
        if ite is None:
            detected_autocorrected_test_section_df = self.test_sections_dataframe[
                self.test_sections_dataframe['TEST_SECTION_ID'].isin(detected_autocorrected_test_section_ids)]
            detected_autocorrected_test_section_df.to_csv(
                osp.join(DEFAULT_CLEANED_DATASETS_DIR, 'detected_autocorrected_test_sections.csv'), index=False)

    def compute_modification(self, full_log_data, ite=None, keyboard=None, custom_logdata_path=None):
        """
        Compute the modification ratio on char level
        :return:
        """
        iter_count = 0
        self.test_section_count = 0
        self.load_data(ite=ite, keyboard=keyboard, full_log_data=full_log_data, custom_logdata_path=custom_logdata_path)
        for test_section_id in self.test_section_ids:
            iter_count += 1
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
                try:
                    iki, wpm = self.compute_iki_wpm(test_section_df)
                except:
                    iki, wpm = np.nan, np.nan
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

                if iter_count % 1000 == 0:
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
        iter_count = 0
        self.load_data(ite=ite, keyboard=keyboard, full_log_data=full_log_data, custom_logdata_path=custom_logdata_path)
        for test_section_id in self.test_section_ids:
            iter_count += 1
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
                if iter_count % 1000 == 0:
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
        iter_count = 0
        self.load_data(ite=ite, keyboard=keyboard, full_log_data=full_log_data, custom_logdata_path=custom_logdata_path)
        for test_section_id in self.test_section_ids:
            iter_count += 1
            try:
                test_section_df, committed_sentence = self.get_test_section_df(test_section_id)

                # word modified flag with false value as a list of words in the committed sentence
                word_modified_flag = [False] * len(committed_sentence.split())
                if len(committed_sentence.split()) < 3:
                    continue
                committed_words = committed_sentence.split()
                for index, row in test_section_df.iterrows():
                    if row['INPUT'] != row['INPUT'] or row['DATA'] != row['DATA'] or row['INPUT'] == ' ' or row[
                        'DATA'] == ' ':
                        continue
                    if row['INPUT'] != committed_sentence[:len(row['INPUT'])]:
                        # compare each character of the input sentence with the committed sentence
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
                try:
                    iki, wpm = self.compute_iki_wpm(test_section_df)
                except:
                    iki, wpm = 0, 0
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
                if iter_count % 1000 == 0:
                    print('IKI: ', iki)
                    print('WPM: ', wpm)
                    print("Word Modified Ratio (WMR): ", self.modified_word_count / self.word_count)
                    print("Auto-corrected ratio: ", self.auto_corrected_word_count / self.word_count)
                    print("test_section_count: ", self.test_section_count)
                    print("test_section_id: ", test_section_id)

            except:
                pass

        print("Word Modified Ratio (WMR): ", self.modified_word_count / self.word_count)

    def filter_percentage(self, ite=None, keyboard=None, filter_ratio=0.5, save_file_name=None, load_file_name=None):
        iter_count = 0
        self.test_section_count = 0
        self.load_data(ite=ite, keyboard=keyboard, custom_logdata_path=load_file_name)
        temp_df = pd.DataFrame(columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'IKI'])
        for test_section_id in self.test_section_ids:
            iter_count += 1
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

    def filter_iki(self, ite=None, keyboard=None, filter_iki=200, save_file_name=None, load_file_name=None,
                   wmr_file_name=None, threshold=1.0, wmr_df=None):
        iter_count = 0
        self.test_section_count = 0
        self.load_data(ite=ite, keyboard=keyboard, custom_logdata_path=load_file_name)
        # load the wmr visualization data
        if wmr_df is None:
            wmr_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, wmr_file_name),
                                 names=WMR_VISUALIZATION_COLUMNS,
                                 encoding='ISO-8859-1')
        # get the test section id with iki around the filter_iki, wmr_df iki should > filter_iki - 1 and < filter_iki + 1
        test_section_ids = wmr_df[(wmr_df['IKI'] > filter_iki - threshold) & (wmr_df['IKI'] < filter_iki + threshold)][
            'TEST_SECTION_ID']
        # remain those rows in self.logdata_dataframe with the same test section id

        # filter the logdata dataframe with the test section ids and save the filtered data
        self.logdata_dataframe = self.logdata_dataframe[
            self.logdata_dataframe['TEST_SECTION_ID'].isin(test_section_ids)]
        self.logdata_dataframe.to_csv(osp.join(DEFAULT_CLEANED_DATASETS_DIR, save_file_name), index=False, header=False)
        print("Number of unique test sections after filtering: ",
              len(self.logdata_dataframe['TEST_SECTION_ID'].unique()))

    def rebuild_typing(self, wmr_df=None, rebuild_save_file_name=None, filtered_logdata_save_file_name=None):
        rebuild_columns = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'REFERENCE_SENTENCE', 'TYPED_SENTENCE',
                           'COMMITTED_SENTENCE', 'WMR']
        rebuild_df = pd.DataFrame(columns=rebuild_columns)
        self.load_test_sections()
        self.load_sentences()
        test_section_ids = self.logdata_dataframe['TEST_SECTION_ID'].unique()
        for test_section_id in test_section_ids:
            try:
                participant_id = self.test_sections_dataframe[
                    self.test_sections_dataframe['TEST_SECTION_ID'] == test_section_id]['PARTICIPANT_ID'].values[0]
                test_section_df, committed_sentence = self.get_test_section_df(test_section_id)
                if committed_sentence != committed_sentence:
                    continue
                sentence_id = self.test_sections_dataframe[
                    self.test_sections_dataframe['TEST_SECTION_ID'] == test_section_id]['SENTENCE_ID'].values[0]
                reference_sentence = self.sentences_dataframe[
                    self.sentences_dataframe['SENTENCE_ID'] == sentence_id]['SENTENCE'].values[0]
                reformatted_input, auto_corrected_if_count, auto_corrected_c_count, \
                auto_corrected_word_count, auto_correct_count, auto_correct_flag, \
                immediate_error_correction_count, delayed_error_correction_count = self.reformat_input(test_section_df)
                wmr = wmr_df[wmr_df['TEST_SECTION_ID'] == test_section_id]['MODIFIED_WORD_COUNT'].values[0] / \
                      wmr_df[wmr_df['TEST_SECTION_ID'] == test_section_id]['WORD_COUNT'].values[0]
                rebuild_df = rebuild_df.append(
                    pd.DataFrame([[participant_id, test_section_id, reference_sentence,
                                   reformatted_input, committed_sentence, wmr]], columns=rebuild_columns))
            except:
                print("Error in test section id: ", test_section_id)
                # remove the test section id from the filtered logdata
                self.logdata_dataframe = self.logdata_dataframe[
                    self.logdata_dataframe['TEST_SECTION_ID'] != test_section_id]

        rebuild_df.to_csv(osp.join(DEFAULT_CLEANED_DATASETS_DIR, rebuild_save_file_name), index=False, header=True)
        self.logdata_dataframe.to_csv(osp.join(DEFAULT_CLEANED_DATASETS_DIR, filtered_logdata_save_file_name),
                                      index=False,
                                      header=False)
        print("Unique test sections after rebuilding: ", len(rebuild_df['TEST_SECTION_ID'].unique()))
        print("Mean WMR: ", rebuild_df['WMR'].mean())
        print("Number of sentences with no modification: ", rebuild_df[rebuild_df['WMR'] == 0].shape[0])


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
    reference = "the quick brown fox"
    typed = "th quix<ck brpown"
    typed_2 = 'thhe<<e quic<<ckk<<< browwn<<<n foxx<<x'
    result = 'the qu bron fox'
    correct_count, inf_count, if_count, fix_count = parser.track_typing_errors(reference, typed_2)
    print("Correct count: ", correct_count)
    print("Incorrect-not-fix count: ", inf_count)
    print("Incorrect fix count: ", if_count)
    print("fix count: ", fix_count)
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
