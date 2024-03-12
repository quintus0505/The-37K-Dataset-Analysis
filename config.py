import os.path as osp

DEFAULT_ROOT_DIR = osp.join((osp.abspath(osp.dirname(__file__))))
DEFAULT_DATASETS_DIR = osp.join(DEFAULT_ROOT_DIR, 'data')
DEFAULT_VISUALIZATION_DIR = osp.join(DEFAULT_ROOT_DIR, 'visualization')

OPEN_INPUT_LOGDATA_FULL_PATH = osp.join(DEFAULT_DATASETS_DIR, 'open_input_logdata_full.csv')
# OPEN_INPUT_LOGDATA_TEST_PATH = osp.join(DEFAULT_DATASETS_DIR, 'open_input_logdata_test.csv')
OPEN_INPUT_LOGDATA_TEST_PATH = osp.join(DEFAULT_DATASETS_DIR, 'open_input_logdata_no_ite_test.csv')

OPEN_PARTICIPANTS_PATH = osp.join(DEFAULT_DATASETS_DIR, 'open_participants.csv')
OPEN_TEST_SECTIONS_PATH = osp.join(DEFAULT_DATASETS_DIR, 'open_test_sections.csv')

participant_columns = ['PARTICIPANT_ID', 'BROWSER', 'BROWSER_LANGUAGE', 'DEVICE', 'SCREEN_W', 'SCREEN_H', 'AGE',
                       'GENDER',
                       'HAS_TAKEN_TYPING_COURSE', 'LAYOUT', 'WPM', 'ERROR_RATE', 'NATIVE_LANGUAGE', 'KEYBOARD_TYPE',
                       'USING_APP',
                       'USING_FEATURES', 'FINGERS', 'TIME_SPENT_TYPING', 'TYPE_ENGLISH', 'P_KPD', 'P_IKI', 'P_BSP',
                       'P_ECPC',
                       'P_UILEN', 'P_KSPC', 'ITE_SWYPE', 'ITE_PREDICT', 'ITE_AUTOCORR']

input_logdata_columns = ['TIMESTAMP', 'LOG_DATA_ID', 'TEST_SECTION_ID', 'TYPE', 'KEY', 'CODE', 'DATA', 'INPUT',
                         'PRESSED', 'DEVICE_ORIENTATION_1', 'DEVICE_ORIENTATION_2', 'DEVICE_ORIENTATION_3',
                         'SCREEN_ORIENTATION', 'INPUT_LEN', 'INPUT_PRLEN',
                         'INPUT_LDIST', 'INPUT_PRLDIST', 'INPUT_LC', 'INPUT_PRLC', 'ITE_SWYP', 'ITE_PRED', 'ITE_AUTO']

logdata_columns = ['TIMESTAMP', 'TEST_SECTION_ID', 'DATA', 'INPUT',
                   'ITE_SWYP', 'ITE_PRED', 'ITE_AUTO']  # columns to keep for computation

test_sections_columns = ['TEST_SECTION_ID', 'SENTENCE_ID', 'PARTICIPANT_ID', 'USER_INPUT', 'INPUT_TIME',
                         'EDIT_DISTANCE', 'ERROR_RATE', 'WPM', 'INPUT_LENGTH', 'ERROR_LEN', 'POTENTIAL_WPM',
                         'POTENTIAL_LENGTH', 'DEVICE', 'TS_IKI', 'TS_KPD', 'TS_BSP', 'TS_ECPC', 'TS_UILEN', 'TS_KS',
                         'TS_KSPC', 'N_SWYP', 'N_PRED', 'N_AUTO', 'PR_SWYP', 'PR_PRED', 'PR_AUTO', 'TS_NUM_WORDS']
