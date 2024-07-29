import os.path as osp
import numpy as np
import pandas as pd
import re
from io import StringIO
import scipy.stats.mstats as mstats
from config import *
import os
from tqdm import tqdm

# set pd warnings to ignore
pd.options.mode.chained_assignment = None  # default='warn'


def clean_participants_data(ite=None, keyboard=None, os=None):
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

    # remove those rows where OS is not mobile
    df = df[df['KEYBOARD_TYPE'] == 'mobile']
    if not ite:
        df = df[df['USING_FEATURES'] == '\\no\\']
    else:
        # remove rows with notsure and other
        df = df[~df['USING_FEATURES'].str.contains('notsure|other', na=False)]

        if 'autocorrection' in ite and 'prediction' not in ite and 'swipe' not in ite:
            df = df[df['USING_FEATURES'].str.contains('autocorrection', na=False)
                    & ~df['USING_FEATURES'].str.contains('prediction', na=False)
                    & ~df['USING_FEATURES'].str.contains('swipe', na=False)]
        elif 'autocorrection' not in ite and 'prediction' in ite and 'swipe' not in ite:
            df = df[df['USING_FEATURES'].str.contains('prediction', na=False)
                    & ~df['USING_FEATURES'].str.contains('autocorrection', na=False)
                    & ~df['USING_FEATURES'].str.contains('swipe', na=False)]
        elif 'autocorrection' not in ite and 'prediction' not in ite and 'swipe' in ite:
            df = df[df['USING_FEATURES'].str.contains('swipe', na=False)
                    & ~df['USING_FEATURES'].str.contains('autocorrection', na=False)
                    & ~df['USING_FEATURES'].str.contains('prediction', na=False)]
        elif 'autocorrection' in ite and 'prediction' in ite and 'swipe' not in ite:
            df = df[df['USING_FEATURES'].str.contains('autocorrection', na=False)
                    & df['USING_FEATURES'].str.contains('prediction', na=False)
                    & ~df['USING_FEATURES'].str.contains('swipe', na=False)]
        elif 'autocorrection' in ite and 'prediction' not in ite and 'swipe' in ite:
            df = df[df['USING_FEATURES'].str.contains('autocorrection', na=False)
                    & ~df['USING_FEATURES'].str.contains('prediction', na=False)
                    & df['USING_FEATURES'].str.contains('swipe', na=False)]
        elif 'autocorrection' not in ite and 'prediction' in ite and 'swipe' in ite:
            df = df[df['USING_FEATURES'].str.contains('prediction', na=False)
                    & ~df['USING_FEATURES'].str.contains('autocorrection', na=False)
                    & df['USING_FEATURES'].str.contains('swipe', na=False)]
        elif 'autocorrection' in ite and 'prediction' in ite and 'swipe' in ite:
            df = df[df['USING_FEATURES'].str.contains('autocorrection', na=False)
                    & df['USING_FEATURES'].str.contains('prediction', na=False)
                    & df['USING_FEATURES'].str.contains('swipe', na=False)]

    if keyboard:
        if keyboard == 'Gboard':
            df = df[df['USING_APP'] == 'Gboard']
        elif keyboard == 'SwiftKey':
            df = df[df['USING_APP'] == 'SwiftKey']

    if os:
        if os == 'Android':
            # getting those rows where BROWSER contain 'Android'
            df = df[df['BROWSER'].str.contains('Android', na=False)]
        elif os == 'iOS':
            # getting those rows where BROWSER do not contain 'iOS'
            df = df[df['BROWSER'].str.contains('iPhone', na=False)]

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
        chunksize = 2000

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
            df = remove_auto_corrected_test_sections(df)
            df = remove_predicted_test_sections(df)
            df = remove_swipe_test_sections(df)
            abandoned_test_sections_df = pd.read_csv(
                osp.join(DEFAULT_CLEANED_DATASETS_DIR, 'abandoned_test_sections.csv'))
            # remove those rows where TEST_SECTION_ID is in the abandoned_test_sections_df
            df = df[~df['TEST_SECTION_ID'].isin(abandoned_test_sections_df['TEST_SECTION_ID'])]
            # remove those rows where TEST_SECTION_ID is in the abandoned_test_sections_df
        elif 'autocorrection' in ite and 'prediction' not in ite and 'swipe' not in ite:
            df = remove_predicted_test_sections(df)
            df = remove_swipe_test_sections(df)
        elif 'autocorrection' not in ite and 'prediction' in ite and 'swipe' not in ite:
            df = remove_auto_corrected_test_sections(df)
            df = remove_swipe_test_sections(df)
        elif 'autocorrection' not in ite and 'prediction' not in ite and 'swipe' in ite:
            df = remove_auto_corrected_test_sections(df)
            df = remove_predicted_test_sections(df)
        elif 'autocorrection' in ite and 'prediction' in ite and 'swipe' not in ite:
            df = remove_swipe_test_sections(df)
        elif 'autocorrection' in ite and 'prediction' not in ite and 'swipe' in ite:
            df = remove_predicted_test_sections(df)
        elif 'autocorrection' not in ite and 'prediction' in ite and 'swipe' in ite:
            df = remove_auto_corrected_test_sections(df)

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
    # test_section_ids = df['TEST_SECTION_ID'].unique()
    # participant_ids = get_test_section_df()[get_test_section_df()['TEST_SECTION_ID'].isin(test_section_ids)][
    #     'PARTICIPANT_ID'].unique()
    # print("Total participants: ", len(participant_ids))
    # get one new column 'TRAILTIME' which is the time difference between the current row and the first row. The first
    # row of each test section will be 0
    # df = df.groupby('TEST_SECTION_ID').apply(lambda x: x.assign(TRAILTIME=x['TIMESTAMP'] - x['TIMESTAMP'].iloc[0]))
    # print the columns name
    print(df.columns)
    return df


# TODO: Check if the removeing strategy works for prediction and swipe
def remove_auto_corrected_test_sections(df):
    # find those test sections id which has 'ITE_AUTO' value as 1
    remove_test_section_id = df[df['ITE_AUTO'] == 1]['TEST_SECTION_ID'].unique()
    # remove those test sections id from the dataframe
    df = df[~df['TEST_SECTION_ID'].isin(remove_test_section_id)]

    # # remove those test sections id which has more than 1 char in ''DATA'
    # remove_test_section_id = df[df['DATA'].str.len() > 1]['TEST_SECTION_ID'].unique()
    # # remove those test sections id from the dataframe
    # df = df[~df['TEST_SECTION_ID'].isin(remove_test_section_id)]
    return df


def remove_predicted_test_sections(df):
    remove_test_section_id = df[df['ITE_PRED'] == 1]['TEST_SECTION_ID'].unique()
    # remove those test sections id from the dataframe
    df = df[~df['TEST_SECTION_ID'].isin(remove_test_section_id)]
    return df


def remove_swipe_test_sections(df):
    remove_test_section_id = df[df['ITE_SWYP'] == 1]['TEST_SECTION_ID'].unique()
    # remove those test sections id from the dataframe
    df = df[~df['TEST_SECTION_ID'].isin(remove_test_section_id)]
    return df


def get_test_section_df():
    df = pd.DataFrame(columns=test_sections_columns)

    data_path = OPEN_TEST_SECTIONS_PATH

    print("loading data from {}".format(data_path))
    df = pd.read_csv(data_path, names=df.columns, usecols=range(len(test_sections_columns)),
                     encoding='ISO-8859-1')
    return df


def get_sentences_df():
    df = pd.DataFrame(columns=sentences_columns)

    data_path = OPEN_SENTENCES_PATH

    print("loading data from {}".format(data_path))
    df = pd.read_csv(data_path, names=df.columns, usecols=range(len(sentences_columns)),
                     encoding='ISO-8859-1')

    return df


def build_open_input_logdata_test(test_section_num=1000):
    # build open_input_logdata_test.csv with the first test_section_num test section
    df = get_logdata_df(full_log_data=True)
    # get first test_section_num test section id
    test_section_ids = df['TEST_SECTION_ID'].unique()[:test_section_num]
    new_df = df[df['TEST_SECTION_ID'].isin(test_section_ids)]
    # save without header and index
    new_df.to_csv(osp.join(DEFAULT_CLEANED_DATASETS_DIR, 'open_input_logdata_test.csv'), index=False, header=False)
    print("done")


def build_custom_logdata(ite=None, keyboard=None, data_path=None, os=None, file_name='custom_logdata.csv'):
    if not osp.exists(DEFAULT_DATASETS_DIR):
        raise FileNotFoundError("No data path provided for the logdata. check the original data")

    participants_dataframe = clean_participants_data(ite=ite, keyboard=keyboard, os=os)

    if not data_path:
        logdata_dataframe = get_logdata_df(full_log_data=True, ite=ite, keyboard=keyboard)
    else:
        logdata_dataframe = get_logdata_df(full_log_data=False, ite=ite, keyboard=keyboard, data_path=data_path)

    test_sections_dataframe = get_test_section_df()

    # Filter logdata with test sections id belonging to the selected participants
    participant_ids = participants_dataframe['PARTICIPANT_ID'].values

    test_sections_dataframe = test_sections_dataframe[
        test_sections_dataframe['PARTICIPANT_ID'].isin(participant_ids)]

    logdata_dataframe = logdata_dataframe[
        logdata_dataframe['TEST_SECTION_ID'].isin(test_sections_dataframe['TEST_SECTION_ID'])]
    # ADD PARTICIPANT_ID to logdata_dataframe
    logdata_dataframe = pd.merge(logdata_dataframe, test_sections_dataframe[['TEST_SECTION_ID', 'PARTICIPANT_ID']],
                                 on='TEST_SECTION_ID', how='left')
    test_section_ids = logdata_dataframe['TEST_SECTION_ID'].unique()
    print("Total test sections: ", len(test_section_ids))
    if not data_path:
        if not osp.exists(DEFAULT_CLEANED_DATASETS_DIR):
            os.makedirs(DEFAULT_CLEANED_DATASETS_DIR)
        print("Saving data to {}".format(osp.join(DEFAULT_CLEANED_DATASETS_DIR, file_name)))

        # Save the DataFrame in chunks of 2000 rows
        # chunk_size = 100000
        # num_chunks = len(logdata_dataframe) // chunk_size + (1 if len(logdata_dataframe) % chunk_size > 0 else 0)
        #
        # with open(osp.join(DEFAULT_CLEANED_DATASETS_DIR, file_name), 'w') as f:
        #     for i in tqdm(range(num_chunks), desc="Writing to CSV"):
        #         start = i * chunk_size
        #         end = start + chunk_size
        #         chunk = logdata_dataframe.iloc[start:end]
        #         chunk.to_csv(f, index=False, header=(i == 0))

        logdata_dataframe.to_csv(osp.join(DEFAULT_CLEANED_DATASETS_DIR, file_name), index=False, header=False)
    return logdata_dataframe, participants_dataframe, test_sections_dataframe


def get_sheet_info(sheet_name, test_sections_dataframe=None):
    path = osp.join(DEFAULT_CLEANED_DATASETS_DIR, sheet_name)
    if test_sections_dataframe is None:
        test_sections_dataframe = get_test_section_df()
    # target_df = pd.read_csv(path, names=logdata_columns, usecols=range(len(logdata_columns)),
    #                         encoding='ISO-8859-1')

    target_df = pd.read_csv(path)

    # get those participants id in test_sections_dataframe with the test section id in target_df
    try:
        participant_ids = target_df['PARTICIPANT_ID'].unique()
    except:
        participant_ids = \
            test_sections_dataframe[test_sections_dataframe['TEST_SECTION_ID'].isin(target_df['TEST_SECTION_ID'])][
                'PARTICIPANT_ID'].unique()
    print("sheet name: ", sheet_name)
    print("Total participants: ", len(participant_ids))
    print("Total test sections: ", len(target_df['TEST_SECTION_ID'].unique()))


if __name__ == "__main__":
    # dataframe = clean_participants_data()
    # dataframe = get_logdata_df()
    # print(dataframe.head())
    # dataframe = get_test_section_df()
    # build_open_input_logdata_test(test_section_num=1000)
    data_path = osp.join(DEFAULT_CLEANED_DATASETS_DIR, 'all_keyboard_logdata.csv')
    # build_custom_logdata(ite=None, keyboard='Gboard', file_name='gboard_logdata.csv')
    # get_sheet_info('gboard_no_ite_logdata (remove auto-correct flag).csv')
    get_sheet_info("one_finger_gboard_no_ite_logdata.csv")
    get_sheet_info("one_finger_gboard_ac_logdata.csv")
    get_sheet_info("one_finger_swiftkey_no_ite_logdata.csv")
    get_sheet_info("one_finger_swiftkey_ac_logdata.csv")

    get_sheet_info("two_fingers_gboard_no_ite_logdata.csv")
    get_sheet_info("two_fingers_gboard_ac_logdata.csv")
    get_sheet_info("two_fingers_swiftkey_no_ite_logdata.csv")
    get_sheet_info("two_fingers_swiftkey_ac_logdata.csv")
