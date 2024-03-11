import os.path as osp
import numpy as np
import pandas as pd
import re
from io import StringIO
import scipy.stats.mstats as mstats
from config import *


def clean_participants_data():
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


def get_logdata_df(full_log_data=False):
    if full_log_data:
        data_path = OPEN_INPUT_LOGDATA_FULL_PATH
        print("loading data from {}".format(data_path))

        # Set chunksize to read the CSV file in chunks
        chunksize = 1000

        # get the selected columns id from the logdata_columns and input_logdata_columns
        selected_columns_id = [input_logdata_columns.index(col) for col in logdata_columns if col in input_logdata_columns]
        # Use a generator expression to read the CSV file in chunks
        chunks = (chunk for chunk in pd.read_csv(data_path, names=logdata_columns,
                                                 usecols=selected_columns_id, encoding='ISO-8859-1',
                                                 chunksize=chunksize))

        # Concatenate all chunks into a single DataFrame
        df = pd.concat(chunks, ignore_index=True)
        # group the dataframe by 'TEST_SECTION_ID', the same test section id sort by timestamp
        df = df.sort_values(by=['TEST_SECTION_ID', 'TIMESTAMP'])

    else:
        data_path = OPEN_INPUT_LOGDATA_TEST_PATH
        if not osp.exists(data_path):
            raise FileNotFoundError("File not found: {}, perhaps you should first generate it".format(data_path))
        print("loading data from {}".format(data_path))
        df = pd.read_csv(data_path, names=logdata_columns, usecols=range(len(logdata_columns)),
                         encoding='ISO-8859-1')

    print("loaded unique test sections: ", len(df['TEST_SECTION_ID'].unique()))
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


if __name__ == "__main__":
    # df = clean_participants_data()
    # df = get_logdata_df()
    # print(df.head())
    # df = get_test_section_df()
    build_open_input_logdata_test(10000)
