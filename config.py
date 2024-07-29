import os.path as osp

DEFAULT_ROOT_DIR = osp.join((osp.abspath(osp.dirname(__file__))))
DEFAULT_DATASETS_DIR = osp.join(DEFAULT_ROOT_DIR, 'original_data')
DEFAULT_CLEANED_DATASETS_DIR = osp.join(DEFAULT_ROOT_DIR, 'cleaned_data')
DEFAULT_VISUALIZATION_DIR = osp.join(DEFAULT_ROOT_DIR, 'visualization')
DEFAULT_FIGS_DIR = osp.join(DEFAULT_ROOT_DIR, 'figs')

OPEN_INPUT_LOGDATA_FULL_PATH = osp.join(DEFAULT_DATASETS_DIR, 'open_input_logdata_full.csv')
OPEN_PARTICIPANTS_PATH = osp.join(DEFAULT_DATASETS_DIR, 'open_participants.csv')
OPEN_TEST_SECTIONS_PATH = osp.join(DEFAULT_DATASETS_DIR, 'open_test_sections.csv')
OPEN_SENTENCES_PATH = osp.join(DEFAULT_DATASETS_DIR, 'sentences.csv')

HOW_WE_TYPE_DATA_DIR = osp.join(DEFAULT_ROOT_DIR, 'How_we_type_data')
HOW_WE_TYPE_FINGER_DATA_DIR = osp.join(HOW_WE_TYPE_DATA_DIR, 'How_we_type_mobile_dataset_finger_motion_capture')
HOW_WE_TYPE_GAZE_DATA_DIR = osp.join(HOW_WE_TYPE_DATA_DIR, 'How_we_type_mobile_dataset_gaze')
HOW_WE_TYPE_TYPING_LOG_DATA_DIR = osp.join(HOW_WE_TYPE_DATA_DIR, 'How_we_type_mobile_dataset_typing_log')

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

sentences_columns = ['SENTENCE_ID', 'SENTENCE', 'SOURCE']

# keys,q,w,e,r,t,y,u,i,o,p,å,a,s,d,f,g,h,j,k,l,ö,ä,z,x,c,v,b,n,m,<,,enter
# x,566,697,828,959,1090,1221,1352,1483,1614,1745,1876,566,697,828,959,1090,1221,1352,1483,1614,1745,1876,828,959,1090,1221,1352,1483,1614,1810,1193,1737
# y,1955,1955,1955,1955,1955,1955,1955,1955,1955,1955,1955,2185,2185,2185,2185,2185,2185,2185,2185,2185,2185,2185,2415,2415,2415,2415,2415,2415,2415,2415,2645,2645

key_height = 230
key_width = 131
half_key_height = 115
half_key_width = 65.5


# x y presented above is the center of the key
how_we_type_key_coordinate = {'q': [501.5, 1840, 632.5, 2070],
                              'w': [632.5, 1840, 763.5, 2070],
                              'e': [763.5, 1840, 894.5, 2070],
                              'r': [894.5, 1840, 1025.5, 2070],
                              't': [1025.5, 1840, 1156.5, 2070],
                              'y': [1156.5, 1840, 1287.5, 2070],
                              'u': [1287.5, 1840, 1418.5, 2070],
                              'i': [1418.5, 1840, 1549.5, 2070],
                              'o': [1549.5, 1840, 1680.5, 2070],
                              'p': [1680.5, 1840, 1811.5, 2070],
                              'å': [1811.5, 1840, 1942.5, 2070],
                              'a': [501.5, 2070, 632.5, 2300],
                              's': [632.5, 2070, 763.5, 2300],
                              'd': [763.5, 2070, 894.5, 2300],
                              'f': [894.5, 2070, 1025.5, 2300],
                              'g': [1025.5, 2070, 1156.5, 2300],
                              'h': [1156.5, 2070, 1287.5, 2300],
                              'j': [1287.5, 2070, 1418.5, 2300],
                              'k': [1418.5, 2070, 1549.5, 2300],
                              'l': [1549.5, 2070, 1680.5, 2300],
                              'ö': [1680.5, 2070, 1811.5, 2300],
                              'ä': [1811.5, 2070, 1942.5, 2300],
                              'z': [763.5, 2300, 894.5, 2530],
                              'x': [894.5, 2300, 1025.5, 2530],
                              'c': [1025.5, 2300, 1156.5, 2530],
                              'v': [1156.5, 2300, 1287.5, 2530],
                              'b': [1287.5, 2300, 1418.5, 2530],
                              'n': [1418.5, 2300, 1549.5, 2530],
                              'm': [1549.5, 2300, 1680.5, 2530],
                              '<': [1680.5, 2300, 1942.5, 2530],
                              ' ': [1193 - 5 * half_key_width, 2530, 1193 + 5 * half_key_width, 2760],
                              'shift': [501.5, 2300, 763.5, 2530],
                              'symbol': [501.5, 2530, 1193 - 5 * half_key_width, 2760],
                              'enter': [1193 + 5 * half_key_width, 2530, 1942.5, 2760]}
