import os.path as osp
import numpy as np
import pandas as pd
import re
from io import StringIO
import scipy.stats.mstats as mstats
from config import *


def clean_participants_data(ite=None, keyboard=None):
    df = pd.DataFrame(columns=participant_columns)

    data_path = OPEN_PARTICIPANTS_PATH

    # Read the entire file content
    with open(data_path, 'r', encoding='ISO-8859-1') as file:
        file_content = file.read()

    # Perform the replacements (solving problems in the CSV format of USING_FEATURES)
    file_content = file_content.replace('\\"', '\\').replace(']"', '"').replace('[', '')

    # Use StringIO to simulate a file object for pd.read_csv
    data_io = StringIO(file_content)

    print("loading data from {}".format(data_path))

    df = pd.read_csv(data_io, header=0, names=df.columns, usecols=range(len(participant_columns)))
    print("loaded unique participants: ", len(df['PARTICIPANT_ID'].unique()))
    # remove those rows where LAYOUT is not qwerty
    df = df[df['LAYOUT'] == 'qwerty']

    # remove those rows where DEVICE is not mobile
    df = df[df['KEYBOARD_TYPE'] == 'mobile']
    if not ite:
        df = df[df['USING_FEATURES'] == '\\no\\']
    elif 'autocorrection' in ite:
        df = df[df['USING_FEATURES'].str.contains('autocorrection', na=False)]
    elif 'prediction' in ite:
        df = df[df['USING_FEATURES'].str.contains('prediction', na=False)]
    elif 'swype' in ite:
        df = df[df['USING_FEATURES'].str.contains('swype', na=False)]

    if keyboard:
        if keyboard == 'Gboard':
            df = df[df['USING_APP'] == 'Gboard']
        elif keyboard == 'SwiftKey':
            df = df[df['USING_APP'] == 'SwiftKey']

    # clean the duplicate rows of USING_APP
    df["USING_APP"] = df["USING_APP"].astype(str).apply(lambda x: re.sub(r'(?i)^Kika.*$', 'Kika', x))
    df["USING_APP"] = df["USING_APP"].astype(str).apply(lambda x: re.sub(r'(?i)^Go.*$', 'Go', x))
    df["USING_APP"] = df["USING_APP"].astype(str).apply(lambda x: re.sub(r'^Gramm.*$', 'Grammarly', x))
    df["USING_APP"] = df["USING_APP"].astype(str).apply(lambda x: re.sub(r'^Cheetah.*$', 'Cheetah', x))
    df["USING_APP"] = df["USING_APP"].astype(str).apply(lambda x: re.sub(r'^Fle.*$', 'Flesky', x))
    df["USING_APP"] = df["USING_APP"].astype(str).apply(lambda x: re.sub(r'^Facemoji.*$', 'Facemoji', x))
    df["USING_APP"] = df["USING_APP"].astype(str).apply(lambda x: re.sub(r'(?i)^SwiftKey.*$', 'SwiftKey', x))

    # # drop those with "USING_FEATURES" containing "other" and "notsure"
    # df = df[~df['USING_FEATURES'].str.contains('other|notsure', na=False)]
    print("cleaned unique participants: ", len(df['PARTICIPANT_ID'].unique()))
    return df


def get_logdata_df(full_log_data=False, ite=None, keyboard=None, data_path=None):
    if full_log_data:
        data_path = OPEN_INPUT_LOGDATA_FULL_PATH
        print("loading data from {}".format(data_path))

        # Set chunksize to read the CSV file in chunks
        chunksize = 5000

        # get the selected columns id from the logdata_columns and input_logdata_columns
        selected_columns_id = [input_logdata_columns.index(col) for col in logdata_columns if
                               col in input_logdata_columns]
        # Use a generator expression to read the CSV file in chunks
        chunks = (chunk for chunk in pd.read_csv(data_path, names=logdata_columns,
                                                 usecols=selected_columns_id, encoding='ISO-8859-1',
                                                 chunksize=chunksize))

        # Concatenate all chunks into a single DataFrame
        df = pd.concat(chunks, ignore_index=True)
        # find those test sections id which has 'ITE_AUTO' value as 1
        if not ite:
            remove_test_section_id = df[df['ITE_AUTO'] == 1]['TEST_SECTION_ID'].unique()
            # remove those test sections id from the dataframe
            df = df[~df['TEST_SECTION_ID'].isin(remove_test_section_id)]

            remove_test_section_id = df[df['ITE_PRED'] == 1]['TEST_SECTION_ID'].unique()
            # remove those test sections id from the dataframe
            df = df[~df['TEST_SECTION_ID'].isin(remove_test_section_id)]

            # remove those test sections id which has more than 1 char in ''DATA'
            remove_test_section_id = df[df['DATA'].str.len() > 1]['TEST_SECTION_ID'].unique()
            # remove those test sections id from the dataframe
            df = df[~df['TEST_SECTION_ID'].isin(remove_test_section_id)]

            remove_test_section_id = df[df['ITE_SWYP'] == 1]['TEST_SECTION_ID'].unique()
            # remove those test sections id from the dataframe
            df = df[~df['TEST_SECTION_ID'].isin(remove_test_section_id)]


        # group the dataframe by 'TEST_SECTION_ID', the same test section id sort by timestamp
        df = df.sort_values(by=['TEST_SECTION_ID', 'TIMESTAMP'])

    else:
        if not data_path:
            raise FileNotFoundError("No data path provided for the logdata. Please provide the data path")  # noqa
        elif not osp.exists(data_path):
            raise FileNotFoundError("File not found: {}, perhaps you should first generate it".format(data_path))
        print("loading data from {}".format(data_path))
        df = pd.read_csv(data_path, names=logdata_columns, usecols=range(len(logdata_columns)),
                         encoding='ISO-8859-1')

        remove_test_section_id = df[df['DATA'].str.len() > 1]['TEST_SECTION_ID'].unique()
        # remove those test sections id from the dataframe
        df = df[~df['TEST_SECTION_ID'].isin(remove_test_section_id)]

    # get unique test section id
    print("loaded unique test sections: ", len(df['TEST_SECTION_ID'].unique()))
    # get those participants id in test_sections_dataframe with the test section id in target_df
    test_section_ids = df['TEST_SECTION_ID'].unique()
    participant_ids = get_test_section_df()[get_test_section_df()['TEST_SECTION_ID'].isin(test_section_ids)][
        'PARTICIPANT_ID'].unique()
    print("Total participants: ", len(participant_ids))
    return df


def get_test_section_df():
    df = pd.DataFrame(columns=test_sections_columns)

    data_path = OPEN_TEST_SECTIONS_PATH

    print("loading data from {}".format(data_path))
    df = pd.read_csv(data_path, names=df.columns, usecols=range(len(test_sections_columns)),
                     encoding='ISO-8859-1')
    print("loaded unique setences: ", len(df['SENTENCE_ID'].unique()))
    return df


def build_open_input_logdata_test(test_section_num=1000):
    # build open_input_logdata_test.csv with the first test_section_num test section
    df = get_logdata_df(full_log_data=True)
    # get first test_section_num test section id
    test_section_ids = df['TEST_SECTION_ID'].unique()[:test_section_num]
    new_df = df[df['TEST_SECTION_ID'].isin(test_section_ids)]
    # save without header and index
    new_df.to_csv(osp.join(DEFAULT_DATASETS_DIR, 'open_input_logdata_test.csv'), index=False, header=False)
    print("done")


def build_custom_logdata(ite=None, keyboard=None, data_path=None, file_name='custom_logdata.csv'):
    participants_dataframe = clean_participants_data(ite=ite, keyboard=keyboard)
    if not data_path:
        logdata_dataframe = get_logdata_df(full_log_data=True, ite=ite, keyboard=keyboard)
    else:
        logdata_dataframe = get_logdata_df(full_log_data=False, ite=ite, keyboard=keyboard, data_path=data_path)
    test_sections_dataframe = get_test_section_df()

    # get those logdata with test sections id belonging to the selected participants
    participant_ids = participants_dataframe['PARTICIPANT_ID'].values

    # remove those rows where PARTICIPANT_ID is not in the selected participants
    test_sections_dataframe = test_sections_dataframe[
        test_sections_dataframe['PARTICIPANT_ID'].isin(participant_ids)]

    # remove those rows where test sections id is not in the selected test sections
    logdata_dataframe = logdata_dataframe[
        logdata_dataframe['TEST_SECTION_ID'].isin(test_sections_dataframe['TEST_SECTION_ID'])]

    logdata_dataframe.to_csv(osp.join(DEFAULT_DATASETS_DIR, file_name), index=False, header=False)


def get_sheet_info(sheet_name):
    path = osp.join(DEFAULT_DATASETS_DIR, sheet_name)
    test_sections_dataframe = get_test_section_df()
    target_df = pd.read_csv(path, names=logdata_columns, usecols=range(len(logdata_columns)),
                            encoding='ISO-8859-1')

    # get those participants id in test_sections_dataframe with the test section id in target_df
    participant_ids = test_sections_dataframe[test_sections_dataframe['TEST_SECTION_ID'].isin(target_df['TEST_SECTION_ID'])][
        'PARTICIPANT_ID'].unique()

    print("Total participants: ", len(participant_ids))
    print("Total test sections: ", len(target_df['TEST_SECTION_ID'].unique()))


if __name__ == "__main__":
    # df = clean_participants_data()
    # df = get_logdata_df()
    # print(df.head())
    # df = get_test_section_df()
    # build_open_input_logdata_test(test_section_num=1000)
    data_path = osp.join(DEFAULT_DATASETS_DIR, 'all_keyboard_logdata.csv')
    # build_custom_logdata(ite=None, keyboard='Gboard', file_name='gboard_logdata.csv')
    get_sheet_info('all_keyboard_logdata.csv')
