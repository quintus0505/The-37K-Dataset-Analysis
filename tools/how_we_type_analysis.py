import os.path as osp
import pandas as pd
import os
from config import DEFAULT_ROOT_DIR
from tools.parser import Parse
from tools.string_analysis import *
HOW_WE_TYPE_DIR = osp.join(DEFAULT_ROOT_DIR, 'original_data', 'How_we_type_mobile_dataset_typing_log')
LOG_DIR = osp.join(HOW_WE_TYPE_DIR, 'Typing_log')

original_sentences_columns = ['sentence_n', 'sentence']
sentences_columns = ['SENTENCE_ID', 'SENTENCE']
# systime	id	block	sentence_n	trialtime	event	layout	message	touchx	touchy
original_log_columns = ['systime', 'id', 'block', 'SENTENCE_ID', 'trialtime', 'DATA', 'layout', 'INPUT', 'touchx',
                        'touchy']
used_log_columns = [ 'id', 'block', 'SENTENCE_ID', 'DATA', 'INPUT']


def load_sentences_df():
    sentences_path = osp.join(HOW_WE_TYPE_DIR, 'Sentences.csv')
    sentences_df = pd.read_csv(sentences_path, usecols=original_sentences_columns)
    # rename columns
    sentences_df.columns = sentences_columns
    return sentences_df


def load_log_df():
    # all csv file stored in LOG_DIR, load them all and concat into one dataframe, only use 'SENTENCE_ID', 'DATA' and 'INPUT'
    log_df = None
    selected_columns_id = [original_log_columns.index(col) for col in used_log_columns if col in original_log_columns]
    for file in os.listdir(LOG_DIR):
        if file.endswith("2.csv"):
            file_path = osp.join(LOG_DIR, file)
            # only use 'SENTENCE_ID', 'DATA' and 'INPUT'
            df = pd.read_csv(file_path, names=original_log_columns, usecols=selected_columns_id)
            # remove the first row
            df = df.iloc[1:]
            if log_df is None:
                log_df = df
            else:
                log_df = pd.concat([log_df, df], ignore_index=True)

    # rename
    log_df.columns = used_log_columns
    log_df['TEST_SECTION_ID'] = (log_df['SENTENCE_ID'] != log_df['SENTENCE_ID'].shift()).cumsum()
    # add a column 'ITE_AUTO' with all 0
    log_df['ITE_AUTO'] = 0
    return log_df


def get_test_sections_df(test_section_id):
    # get the test section dataframe by test_section_id
    test_section_df = log_df[log_df['TEST_SECTION_ID'] == test_section_id]
    committed_sentence = test_section_df['INPUT'].iloc[-1]
    return test_section_df, committed_sentence


if __name__ == '__main__':
    sentences_df = load_sentences_df()
    log_df = load_log_df()

    parser = Parse()
    test_section_ids = log_df['TEST_SECTION_ID'].unique()
    total_correct_count, total_inf_count, total_if_count, total_fix_count = 0, 0, 0, 0
    test_section_count = 0
    for test_section_id in test_section_ids:
        # if test_section_id == 141:
        #     print("test_section_id: ", test_section_id)
        try:
            test_section_df, committed_sentence = get_test_sections_df(test_section_id)
            sentence_id = int(test_section_df['SENTENCE_ID'].iloc[0])
            target_sentence = sentences_df[sentences_df['SENTENCE_ID'] == sentence_id]['SENTENCE'].iloc[0]
            while committed_sentence[-1] == ' ':
                committed_sentence = committed_sentence[:-1]
            reformatted_input, auto_corrected_if_count, auto_corrected_c_count, \
            auto_corrected_word_count, auto_correct_count = parser.reformat_input(test_section_df)
            correct_count, inf_count, if_count, fix_count = track_typing_errors(target_sentence,
                                                                                reformatted_input)
            total_correct_count += correct_count + auto_corrected_c_count - auto_corrected_word_count
            total_inf_count += inf_count
            total_if_count += if_count + auto_corrected_if_count
            total_fix_count += fix_count + auto_correct_count
            test_section_count += 1
            uncorrected_error_rate = inf_count / (correct_count + inf_count + if_count)
            corrected_error_rate = if_count / (correct_count + inf_count + if_count)
            # if uncorrected_error_rate > 0.25 or corrected_error_rate > 0.25:
            #     print("test_section_count: ", self.test_section_count)
            #     print("test_section_id: ", test_section_id)
            #     print("Corrected error rate: ", corrected_error_rate)
            #     print("Uncorrected error rate: ", uncorrected_error_rate)
            if test_section_count % 1000 == 0:
                print("test_section_count: ", test_section_count)
                print("test_section_id: ", test_section_id)
                uncorrected_error_rate = total_inf_count / (total_correct_count + total_inf_count + total_if_count)
                corrected_error_rate = total_if_count / (total_correct_count + total_inf_count + total_if_count)
                print("Corrected error rate: ", corrected_error_rate)
                print("Uncorrected error rate: ", uncorrected_error_rate)

        except:
            pass
    uncorrected_error_rate = total_inf_count / (total_correct_count + total_inf_count + total_if_count)
    corrected_error_rate = total_if_count / (total_correct_count + total_inf_count + total_if_count)
    print("Corrected error rate: ", corrected_error_rate)
    print("Uncorrected error rate: ", uncorrected_error_rate)
