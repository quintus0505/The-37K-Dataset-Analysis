import pandas as pd
from tqdm import tqdm
from config import *
from tools.data_loading import *
from tools.parser import *
from tools.visualization import *
import argparse

parser = argparse.ArgumentParser()

# for data cleaning
parser.add_argument("--data-cleaning", action="store_true", default=False, help="generate logdata file")

# options for data cleaning and analysis
parser.add_argument("--auto-correct", action="store_true", default=False, help='if considering auto-correct')
parser.add_argument("--predict", action="store_true", default=False, help='if considering prediction')
parser.add_argument("--swipe", action="store_true", default=False, help='if considering swipe')
parser.add_argument("--keyboard", type=str, default=None, help='keyboard type')
parser.add_argument("--os", type=str, default=None, help='os type')
# parser.add_argument("--finger", type=str, choices=['one', 'two'], default=None, help='finger num')

# for both visualization and analysis
parser.add_argument("--analyze", action="store_true", default=False, help="anylyze the data")
parser.add_argument("--visualize", action="store_true", default=False, help="visualize the data")
parser.add_argument("--wmr", action="store_true", default=False, help="wmr")
parser.add_argument("--ac", action="store_true", default=False, help="ac")
parser.add_argument("--modification", action="store_true", default=False, help="modification")
parser.add_argument("--age", action="store_true", default=False, help="age")
parser.add_argument("--num", action="store_true", default=False, help="num")
parser.add_argument("--edit-distance", action="store_true", default=False, help="edit distance")

parser.add_argument("--visualize-by-edit-distance", type=int, default=0,
                    help='split the dataset and visualize by edit distance')
parser.add_argument("--visualize-by-sentence-length", type=int, default=0,
                    help='split the dataset and visualize by sentence length')

# visualization options
parser.add_argument("--avg", action="store_true", default=False, help="visualize the horizontal line of average")

# for filter
parser.add_argument("--filter", action="store_true", default=False, help='filter the data')
parser.add_argument("--rebuild", action="store_true", default=False, help='rebuild the typing data')
parser.add_argument("--percentage", type=float, default=0,
                    help='filter those test section with iki below this percentage')
parser.add_argument("--iki", type=float, default=200, help='filter those log data with iki around this value')

# for data analysis
parser.add_argument('--error-rate', action="store_true", default=False,
                    help="compute corrected and uncorrected error rate")

args = parser.parse_args()


def split_by_finger_num(df, participants_dataframe=None, test_sections_dataframe=None):
    # Function to clean participants data should be defined elsewhere
    if participants_dataframe is None:
        participants_dataframe = clean_participants_data()

    if test_sections_dataframe is None:
        test_sections_dataframe = get_test_section_df()

    # Initialize empty lists to collect data frames
    one_finger_list = []
    two_fingers_list = []

    unique_test_sections = df['TEST_SECTION_ID'].unique()
    total_unique_test_sections = len(unique_test_sections)
    print("Total unique test sections:", total_unique_test_sections)

    for test_section_id in tqdm(unique_test_sections, desc="Processing test sections"):
        test_section_df = df[df['TEST_SECTION_ID'] == test_section_id]
        # get the participant ID from the test_sections_dataframe
        participant_data = test_sections_dataframe[
            test_sections_dataframe['TEST_SECTION_ID'] == test_section_id
            ]

        if participant_data.empty:
            continue

        participant_id = participant_data['PARTICIPANT_ID'].values[0]

        # Extract the finger usage associated with the participant ID
        finger_data = participants_dataframe[
            participants_dataframe['PARTICIPANT_ID'] == participant_id
            ]

        if finger_data.empty:
            continue

        finger_use_value = finger_data['FINGERS'].values[0]

        # Determine the finger use based on the extracted value
        if 'both_hands' in finger_use_value:
            finger_use = 'two_fingers'
        elif 'right_hand' in finger_use_value or 'left_hand' in finger_use_value:
            finger_use = 'one_finger'
        elif 'thumbs' in finger_use_value:
            finger_use = 'thumbs'
        else:
            finger_use = 'unknown'

        if finger_use == 'one_finger':
            one_finger_list.append(test_section_df)
        elif finger_use == 'two_fingers':
            two_fingers_list.append(test_section_df)

    # Concatenate the collected data frames into single data frames
    one_finger_df = pd.concat(one_finger_list, ignore_index=True) if one_finger_list else pd.DataFrame(
        columns=df.columns)
    two_fingers_df = pd.concat(two_fingers_list, ignore_index=True) if two_fingers_list else pd.DataFrame(
        columns=df.columns)

    return one_finger_df, two_fingers_df


def get_name_str():
    if not (args.auto_correct or args.predict or args.swipe):
        ite = None
        ite_str = 'no_ite_'
    else:
        ite = []
        ite_str = ''
        if args.auto_correct:
            ite.append('autocorrection')
            ite_str += 'ac_'
        if args.predict:
            ite.append('prediction')
            ite_str += 'predict_'
        if args.swipe:
            ite.append('swipe')
            ite_str += 'swipe_'
    print("ite: ", ite)
    if args.keyboard:
        keyboard = args.keyboard
        keyboard_str = keyboard.lower() + '_'
    else:
        keyboard = None
        keyboard_str = ''
    print("keyboard: ", keyboard)
    if args.os:
        typing_os = args.os
        os_str = typing_os.lower() + '_'
    else:
        typing_os = None
        os_str = ''
    name_str = os_str + keyboard_str + ite_str

    return name_str, ite, keyboard, typing_os


if __name__ == "__main__":

    if args.data_cleaning:
        print("cleaning data")
        name_info, ite_list, kbd, os = get_name_str()
        file_name = name_info + 'logdata.csv'
        print("Generating logdata file: ", file_name)
        logdata_dataframe, participants_dataframe, test_sections_dataframe = build_custom_logdata(ite=ite_list,
                                                                                                  keyboard=kbd,
                                                                                                  file_name=file_name,
                                                                                                  os=os,
                                                                                                  # data_path=osp.join(
                                                                                                  #     DEFAULT_CLEANED_DATASETS_DIR,
                                                                                                  #     file_name)
                                                                                                  )
        # get_sheet_info(sheet_name=file_name, test_sections_dataframe=test_sections_dataframe)

        print("Logdata file generated successfully")
        print("splitting logdata file with finger number")
        one_finger_df, two_fingers_df = split_by_finger_num(logdata_dataframe, participants_dataframe,
                                                            test_sections_dataframe)
        one_finger_file_name = 'one_finger_' + file_name
        two_fingers_file_name = 'two_fingers_' + file_name
        one_finger_df.to_csv(osp.join(DEFAULT_CLEANED_DATASETS_DIR, one_finger_file_name), index=False, header=True)
        two_fingers_df.to_csv(osp.join(DEFAULT_CLEANED_DATASETS_DIR, two_fingers_file_name), index=False, header=True)

    if args.analyze:
        print("analyzing data")
        parser = Parse()
        name_info, ite_list, kbd, os = get_name_str()
        file_name = name_info + 'logdata.csv'
        logdata_path = osp.join(DEFAULT_CLEANED_DATASETS_DIR, file_name)
        if not osp.exists(logdata_path):
            raise FileNotFoundError("File not found: ", file_name)
        print("Analyzing logdata file: ", file_name)
        if args.modification:
            parser.compute_modification(full_log_data=True, ite=ite_list, keyboard=args.keyboard,
                                        custom_logdata_path=logdata_path)
        if args.wmr:
            parser.compute_wmr(full_log_data=True, ite=ite_list, keyboard=args.keyboard,
                               custom_logdata_path=logdata_path)
        if args.ac:
            parser.compute_ac(full_log_data=True, ite=ite_list, keyboard=args.keyboard,
                              custom_logdata_path=logdata_path)
        if args.age:
            parser.get_age(full_log_data=True, ite=ite_list, keyboard=args.keyboard,
                           custom_logdata_path=logdata_path)
        save_file_name = name_info + 'logdata_visualization.csv'
        if not osp.exists(DEFAULT_VISUALIZATION_DIR):
            os.makedirs(DEFAULT_VISUALIZATION_DIR)
        if args.modification:
            parser.save_modification_visualization(
                osp.join(DEFAULT_VISUALIZATION_DIR, 'modification_' + save_file_name))
        if args.wmr:
            parser.save_iki_wmr_visualization(osp.join(DEFAULT_VISUALIZATION_DIR, 'wmr_' + save_file_name))

        if args.auto_correct and args.ac:
            parser.save_iki_ac_visualization(osp.join(DEFAULT_VISUALIZATION_DIR, 'ac_' + save_file_name))
        if args.age:
            parser.save_iki_age_visualization(osp.join(DEFAULT_VISUALIZATION_DIR, 'age_' + save_file_name))
        if args.edit_distance:
            parser.compute_edit_distance(full_log_data=True, ite=ite_list, keyboard=args.keyboard,
                                         custom_logdata_path=logdata_path)
            parser.save_edit_distance_visualization(
                osp.join(DEFAULT_VISUALIZATION_DIR, 'edit_distance_' + save_file_name))

    if args.filter:
        parser = Parse()
        if args.percentage:
            filter_ratio = args.percentage
            print("filtering data below {} % perscent".format(args.percentage))
            name_info, ite_list, kbd, os = get_name_str()
            save_file_name = 'filtered_percentage_' + str(args.percentage) + "_" + name_info + 'logdata.csv'
            file_name = name_info + 'logdata.csv'
            logdata_path = osp.join(DEFAULT_CLEANED_DATASETS_DIR, file_name)
            parser.filter_percentage(filter_ratio=filter_ratio, ite=ite_list, keyboard=kbd,
                                     save_file_name=save_file_name,
                                     load_file_name=logdata_path)
        if args.iki:
            filter_iki = args.iki
            print("filtering data around {} ms".format(args.iki))
            name_info, ite_list, kbd, os = get_name_str()
            save_file_name = 'filtered_iki_' + str(args.iki) + "_" + name_info + 'logdata.csv'
            file_name = name_info + 'logdata.csv'
            logdata_path = osp.join(DEFAULT_CLEANED_DATASETS_DIR, file_name)
            wmr_file_name = 'wmr_' + name_info + 'logdata_visualization.csv'
            wmr_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, wmr_file_name),
                                 names=WMR_VISUALIZATION_COLUMNS,
                                 encoding='ISO-8859-1')
            parser.filter_iki(filter_iki=filter_iki, ite=ite_list, keyboard=kbd, save_file_name=save_file_name,
                              load_file_name=logdata_path, wmr_file_name=wmr_file_name, wmr_df=wmr_df)
            if args.rebuild:
                rebuild_save_file_name = 'filtered_iki_' + str(args.iki) + "_" + name_info + 'rebuild.csv'
                parser.rebuild_typing(wmr_df=wmr_df,
                                      rebuild_save_file_name=rebuild_save_file_name,
                                      filtered_logdata_save_file_name=save_file_name)

    if args.error_rate:
        print("computing error rate")
        name_info, ite_list, kbd, os = get_name_str()
        file_name = name_info + 'logdata.csv'
        logdata_path = osp.join(DEFAULT_CLEANED_DATASETS_DIR, file_name)
        parser = Parse()
        parser.compute_error_rate_correction(full_log_data=True, ite=ite_list, keyboard=kbd,
                                             custom_logdata_path=logdata_path)

    if args.visualize:
        visualize_mean = args.avg
        interval_size = 10
        print("visualizing data")
        name_info, ite_list, kbd, os = get_name_str()
        ac_file_name = ''
        wmr_file_name = ''
        if args.auto_correct:
            ac_file_name = 'ac_' + name_info + 'logdata_visualization.csv'

        wmr_file_name = 'wmr_' + name_info + 'logdata_visualization.csv'
        modification_file_name = 'modification_' + name_info + 'logdata_visualization.csv'

        if args.modification:
            modification_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, modification_file_name),
                                                        names=MODIFICATION_VISUALIZATION_COLUMNS,
                                                        encoding='ISO-8859-1')
            show_plot_info(modification_visualization_df, save_file_name=modification_file_name.split('.')[0],
                           y_label='MODIFICATION')
            iki_interval_df = calculate_iki_intervals(modification_visualization_df,
                                                      y_label='MODIFICATION', interval_size=interval_size)
            plot_modification_vs_iki(iki_interval_df, save_file_name=modification_file_name.split('.')[0],
                                     interval_size=interval_size, origin_df=modification_visualization_df,
                                     visualize_mean=visualize_mean)
            # avg = origin_df['MODIFICATION_COUNT'].sum() / origin_df['CHAR_COUNT'].sum() * 100
            modification_visualization_df['AVG'] = modification_visualization_df['MODIFICATION_COUNT'] / \
                                                   modification_visualization_df['CHAR_COUNT'] * 100
            iki_99 = modification_visualization_df['IKI'].quantile(0.99)
            print("99% of IKI: ", iki_99)
            modification_visualization_df = modification_visualization_df[modification_visualization_df['IKI'] < iki_99]
            one_finger_modification_df, two_fingers_modification_df = split_by_finger_num(modification_visualization_df)
            # print the mean, std of MODIFICATION of one finger
            print("One finger MODIFICATION mean and std")
            print("MODIFICATION: ", one_finger_modification_df['AVG'].mean(),
                  one_finger_modification_df['AVG'].std())

            # print the mean, std of MODIFICATION of two fingers
            print("Two fingers MODIFICATION mean and std")
            print("MODIFICATION: ", two_fingers_modification_df['AVG'].mean(),
                  two_fingers_modification_df['AVG'].std())

            print("Total MODIFICATION mean and std")
            print("MODIFICATION: ", modification_visualization_df['AVG'].mean(),
                  modification_visualization_df['AVG'].std())

        if args.wmr:
            wmr_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, wmr_file_name),
                                               names=WMR_VISUALIZATION_COLUMNS,
                                               encoding='ISO-8859-1')
            show_plot_info(wmr_visualization_df, save_file_name=wmr_file_name.split('.')[0], y_label='WMR')
            iki_interval_df = calculate_iki_intervals(wmr_visualization_df, y_label='WMR',
                                                      interval_size=interval_size)
            plot_wmr_vs_iki(iki_interval_df, save_file_name=wmr_file_name.split('.')[0],
                            interval_size=interval_size, origin_df=wmr_visualization_df, visualize_mean=visualize_mean)

            plot_num_vs_wmr(save_file_name='num_vs_wmr_' + name_info + 'logdata_visualization',
                            interval_size=0.005, origin_df=wmr_visualization_df, visualize_mean=visualize_mean)
            # filter out those IKI out of 99%
            # compute the 99% of IKI
            iki_99 = wmr_visualization_df['IKI'].quantile(0.99)
            print("99% of IKI: ", iki_99)
            wmr_visualization_df = wmr_visualization_df[wmr_visualization_df['IKI'] < iki_99]
            one_finger_wmr_df, two_fingers_wmr_df = split_by_finger_num(wmr_visualization_df)
            # print the mean, std of WMR, WPM, IKI of one finger
            print("One finger WMR mean and std")
            print("WMR: ", one_finger_wmr_df['WMR'].mean(), one_finger_wmr_df['WMR'].std())
            print("WPM: ", one_finger_wmr_df['WPM'].mean(), one_finger_wmr_df['WPM'].std())
            print("IKI: ", one_finger_wmr_df['IKI'].mean(), one_finger_wmr_df['IKI'].std())

            # print the mean, std of WMR, WPM, IKI of two fingers
            print("Two fingers WMR mean and std")
            print("WMR: ", two_fingers_wmr_df['WMR'].mean(), two_fingers_wmr_df['WMR'].std())
            print("WPM: ", two_fingers_wmr_df['WPM'].mean(), two_fingers_wmr_df['WPM'].std())
            print("IKI: ", two_fingers_wmr_df['IKI'].mean(), two_fingers_wmr_df['IKI'].std())

            print("Total WMR mean and std")
            print("WMR: ", wmr_visualization_df['WMR'].mean(), wmr_visualization_df['WMR'].std())
            print("WPM: ", wmr_visualization_df['WPM'].mean(), wmr_visualization_df['WPM'].std())
            print("IKI: ", wmr_visualization_df['IKI'].mean(), wmr_visualization_df['IKI'].std())

        if ac_file_name and args.ac:
            ac_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, ac_file_name),
                                              names=AC_VISUALIZATION_COLUMNS,
                                              encoding='ISO-8859-1')
            show_plot_info(ac_visualization_df, save_file_name=ac_file_name.split('.')[0], y_label='AC')
            iki_interval_df = calculate_iki_intervals(ac_visualization_df, y_label='AC',
                                                      interval_size=interval_size)
            plot_ac_vs_iki(iki_interval_df, save_file_name=ac_file_name.split('.')[0], interval_size=interval_size,
                           origin_df=ac_visualization_df, visualize_mean=visualize_mean)
        if args.edit_distance:
            edit_distance_visualization_df = pd.read_csv(
                osp.join(DEFAULT_VISUALIZATION_DIR, 'edit_distance_' + name_info + 'logdata_visualization.csv'),
                names=EDIT_DISTANCE_VISUALIZATION_COLUMNS,
                encoding='ISO-8859-1')
            show_plot_info(edit_distance_visualization_df,
                           save_file_name='edit_distance_' + name_info + 'logdata_visualization',
                           y_label='EDIT_DISTANCE')
            iki_interval_df = calculate_iki_intervals(edit_distance_visualization_df, y_label='EDIT_DISTANCE',
                                                      interval_size=interval_size)
            plot_edit_distance_vs_iki(iki_interval_df,
                                      save_file_name='edit_distance_' + name_info + 'logdata_visualization',
                                      interval_size=interval_size, origin_df=edit_distance_visualization_df,
                                      visualize_mean=visualize_mean)
        if args.age:
            age_visualization_df = pd.read_csv(
                osp.join(DEFAULT_VISUALIZATION_DIR, 'age_' + name_info + 'logdata_visualization.csv'),
                names=AGE_VISUALIZATION_COLUMNS,
                encoding='ISO-8859-1')
            show_plot_info(age_visualization_df, save_file_name='age_' + name_info + 'logdata_visualization',
                           y_label='AGE')
            iki_interval_df = calculate_iki_intervals(age_visualization_df, y_label='AGE',
                                                      interval_size=interval_size)
            plot_age_vs_iki(iki_interval_df, save_file_name='age_' + name_info + 'logdata_visualization',
                            interval_size=interval_size, origin_df=age_visualization_df)
            # remove the one with extreme IKI > 2000
            age_visualization_df = age_visualization_df[age_visualization_df['IKI'] < 20000]
            # print the IKI WPM mean and std for one finger and two fingers
        if args.num:
            # visualizing how many test section in each iki interval, based on wmr_logdata
            num_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, wmr_file_name),
                                               names=WMR_VISUALIZATION_COLUMNS,
                                               encoding='ISO-8859-1')
            show_plot_info(num_visualization_df, save_file_name=wmr_file_name.split('.')[0], y_label='NUM')
            iki_interval_df = calculate_iki_intervals(num_visualization_df, y_label='NUM',
                                                      interval_size=interval_size)
            plot_num_vs_iki(iki_interval_df, save_file_name='num_' + name_info + 'logdata_visualization',
                            interval_size=interval_size, origin_df=num_visualization_df)
            # print mean and std of IKI and WPM
            print("mean and std of 'IKI' and 'WPM'")
            # filter out the one with extreme IKI
            # print those IKI > 2000
            print(num_visualization_df[num_visualization_df['IKI'] > 2000])
            num_visualization_df = num_visualization_df[num_visualization_df['IKI'] < 20000]

            print(num_visualization_df[['IKI', 'WPM']].mean())
            print(num_visualization_df[['IKI', 'WPM']].std())

    if args.visualize_by_edit_distance:
        visualize_mean = args.avg
        interval_size = 10
        print("visualizing data")
        name_info, ite_list, kbd, os = get_name_str()
        ac_file_name = ''
        wmr_file_name = ''
        if args.auto_correct:
            ac_file_name = 'ac_' + name_info + 'logdata_visualization.csv'

        wmr_file_name = 'wmr_' + name_info + 'logdata_visualization.csv'
        modification_file_name = 'modification_' + name_info + 'logdata_visualization.csv'
        parser = Parse()
        parser.edit_distance_visualization_df = pd.read_csv(
            osp.join(DEFAULT_VISUALIZATION_DIR, 'edit_distance_' + name_info + 'logdata_visualization.csv'),
            names=EDIT_DISTANCE_VISUALIZATION_COLUMNS,
            encoding='ISO-8859-1')
        edit_distance_gte_ids = parser.edit_distance_visualization_df[
            parser.edit_distance_visualization_df['EDIT_DISTANCE'] >= args.visualize_by_edit_distance][
            'TEST_SECTION_ID'].tolist()

        edit_distance_lt_ids = parser.edit_distance_visualization_df[
            parser.edit_distance_visualization_df['EDIT_DISTANCE'] < args.visualize_by_edit_distance][
            'TEST_SECTION_ID'].tolist()

        gte_label_extra_info = ' (with edit distance gte ' + str(args.visualize_by_edit_distance) + ' )'
        lt_label_extra_info = ' (with edit distance lt ' + str(args.visualize_by_edit_distance) + ' )'
        if args.wmr:
            wmr_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, wmr_file_name),
                                               names=WMR_VISUALIZATION_COLUMNS,
                                               encoding='ISO-8859-1')
            gte_wmr_visualization_df = wmr_visualization_df[
                wmr_visualization_df['TEST_SECTION_ID'].isin(edit_distance_gte_ids)]

            lt_wmr_visualization_df = wmr_visualization_df[
                wmr_visualization_df['TEST_SECTION_ID'].isin(edit_distance_lt_ids)]

            gte_wmr_file_name = 'edit_distance_gte_' + str(args.visualize_by_edit_distance) + '_' + wmr_file_name
            lt_wmr_file_name = 'edit_distance_lt_' + str(args.visualize_by_edit_distance) + '_' + wmr_file_name

            gte_num_vs_wmr_save_file_name = 'edit_distance_gte_' + str(
                args.visualize_by_edit_distance) + '_' + 'num_vs_wmr_' + name_info + 'logdata_visualization'

            lt_num_vs_wmr_save_file_name = 'edit_distance_lt_' + str(
                args.visualize_by_edit_distance) + '_' + 'num_vs_wmr_' + name_info + 'logdata_visualization'

            show_plot_info(gte_wmr_visualization_df, save_file_name=gte_wmr_file_name.split('.')[0], y_label='WMR')
            iki_interval_df = calculate_iki_intervals(gte_wmr_visualization_df, y_label='WMR',
                                                      interval_size=interval_size)
            plot_wmr_vs_iki(iki_interval_df, save_file_name=gte_wmr_file_name.split('.')[0],
                            interval_size=interval_size, origin_df=gte_wmr_visualization_df,
                            label_extra_info=gte_label_extra_info, visualize_mean=visualize_mean)

            plot_num_vs_wmr(save_file_name=gte_num_vs_wmr_save_file_name,
                            interval_size=0.005, origin_df=gte_wmr_visualization_df,
                            label_extra_info=gte_label_extra_info, visualize_mean=visualize_mean)

            show_plot_info(lt_wmr_visualization_df, save_file_name=lt_wmr_file_name.split('.')[0], y_label='WMR')
            iki_interval_df = calculate_iki_intervals(lt_wmr_visualization_df, y_label='WMR',
                                                      interval_size=interval_size)
            plot_wmr_vs_iki(iki_interval_df, save_file_name=lt_wmr_file_name.split('.')[0],
                            interval_size=interval_size, origin_df=lt_wmr_visualization_df,
                            label_extra_info=lt_label_extra_info, visualize_mean=visualize_mean)

            plot_num_vs_wmr(save_file_name=lt_num_vs_wmr_save_file_name,
                            interval_size=0.005, origin_df=lt_wmr_visualization_df,
                            label_extra_info=lt_label_extra_info, visualize_mean=visualize_mean)

        if args.modification:
            modification_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, modification_file_name),
                                                        names=MODIFICATION_VISUALIZATION_COLUMNS,
                                                        encoding='ISO-8859-1')

            gte_modification_visualization_df = modification_visualization_df[
                modification_visualization_df['TEST_SECTION_ID'].isin(edit_distance_gte_ids)]

            lt_modification_visualization_df = modification_visualization_df[
                modification_visualization_df['TEST_SECTION_ID'].isin(edit_distance_lt_ids)]

            gte_modification_file_name = 'edit_distance_gte_' + str(
                args.visualize_by_edit_distance) + '_' + modification_file_name
            lt_modification_file_name = 'edit_distance_lt_' + str(
                args.visualize_by_edit_distance) + '_' + modification_file_name

            show_plot_info(gte_modification_visualization_df, save_file_name=gte_modification_file_name.split('.')[0],
                           y_label='MODIFICATION')
            iki_interval_df = calculate_iki_intervals(gte_modification_visualization_df,
                                                      y_label='MODIFICATION', interval_size=interval_size)
            plot_modification_vs_iki(iki_interval_df, save_file_name=gte_modification_file_name.split('.')[0],
                                     interval_size=interval_size, origin_df=gte_modification_visualization_df,
                                     label_extra_info=gte_label_extra_info, visualize_mean=visualize_mean)

            show_plot_info(lt_modification_visualization_df, save_file_name=lt_modification_file_name.split('.')[0],
                           y_label='MODIFICATION')
            iki_interval_df = calculate_iki_intervals(lt_modification_visualization_df,
                                                      y_label='MODIFICATION', interval_size=interval_size)
            plot_modification_vs_iki(iki_interval_df, save_file_name=lt_modification_file_name.split('.')[0],
                                     interval_size=interval_size, origin_df=lt_modification_visualization_df,
                                     label_extra_info=lt_label_extra_info, visualize_mean=visualize_mean)

        if ac_file_name and args.ac:
            ac_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, ac_file_name),
                                              names=AC_VISUALIZATION_COLUMNS,
                                              encoding='ISO-8859-1')

            gte_ac_visualization_df = ac_visualization_df[
                ac_visualization_df['TEST_SECTION_ID'].isin(edit_distance_gte_ids)]

            lt_ac_visualization_df = ac_visualization_df[
                ac_visualization_df['TEST_SECTION_ID'].isin(edit_distance_lt_ids)]

            gte_ac_file_name = 'edit_distance_gte_' + str(
                args.visualize_by_edit_distance) + '_' + ac_file_name
            lt_ac_file_name = 'edit_distance_lt_' + str(
                args.visualize_by_edit_distance) + '_' + ac_file_name

            show_plot_info(gte_ac_visualization_df, save_file_name=gte_ac_file_name.split('.')[0], y_label='AC')
            iki_interval_df = calculate_iki_intervals(gte_ac_visualization_df, y_label='AC',
                                                      interval_size=interval_size)
            plot_ac_vs_iki(iki_interval_df, save_file_name=gte_ac_file_name.split('.')[0], interval_size=interval_size,
                           origin_df=gte_ac_visualization_df, label_extra_info=gte_label_extra_info,
                           visualize_mean=visualize_mean)

            show_plot_info(lt_ac_visualization_df, save_file_name=lt_ac_file_name.split('.')[0], y_label='AC')
            iki_interval_df = calculate_iki_intervals(lt_ac_visualization_df, y_label='AC',
                                                      interval_size=interval_size)
            plot_ac_vs_iki(iki_interval_df, save_file_name=lt_ac_file_name.split('.')[0], interval_size=interval_size,
                           origin_df=lt_ac_visualization_df, label_extra_info=lt_label_extra_info,
                           visualize_mean=visualize_mean)

    if args.visualize_by_sentence_length:
        visualize_mean = args.avg
        interval_size = 10
        print("visualizing data")
        name_info, ite_list, kbd, os = get_name_str()
        ac_file_name = ''
        wmr_file_name = ''
        if args.auto_correct:
            ac_file_name = 'ac_' + name_info + 'logdata_visualization.csv'

        wmr_file_name = 'wmr_' + name_info + 'logdata_visualization.csv'
        modification_file_name = 'modification_' + name_info + 'logdata_visualization.csv'
        parser = Parse()
        parser.load_sentences()
        parser.load_test_sections()
        parser.sentences_dataframe['SENTENCE_LENGTH'] = parser.sentences_dataframe['SENTENCE'].apply(
            lambda x: len(x.split()))
        parser.test_sections_dataframe = parser.test_sections_dataframe.merge(
            parser.sentences_dataframe[['SENTENCE_ID', 'SENTENCE_LENGTH']],
            on='SENTENCE_ID',
            how='left'
        )

        sentence_length_gte_ids = parser.test_sections_dataframe[
            parser.test_sections_dataframe['SENTENCE_LENGTH'] >= args.visualize_by_sentence_length][
            'TEST_SECTION_ID'].tolist()

        sentence_length_lt_ids = parser.test_sections_dataframe[
            parser.test_sections_dataframe['SENTENCE_LENGTH'] < args.visualize_by_sentence_length][
            'TEST_SECTION_ID'].tolist()

        gte_label_extra_info = ' (with sentence length gte ' + str(args.visualize_by_sentence_length) + ' )'
        lt_label_extra_info = ' (with sentence length lt ' + str(args.visualize_by_sentence_length) + ' )'

        if args.wmr:
            wmr_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, wmr_file_name),
                                               names=WMR_VISUALIZATION_COLUMNS,
                                               encoding='ISO-8859-1')
            gte_wmr_visualization_df = wmr_visualization_df[
                wmr_visualization_df['TEST_SECTION_ID'].isin(sentence_length_gte_ids)]

            lt_wmr_visualization_df = wmr_visualization_df[
                wmr_visualization_df['TEST_SECTION_ID'].isin(sentence_length_lt_ids)]

            gte_wmr_file_name = 'sentence_length_gte_' + str(args.visualize_by_sentence_length) + '_' + wmr_file_name
            lt_wmr_file_name = 'sentence_length_lt_' + str(args.visualize_by_sentence_length) + '_' + wmr_file_name

            gte_num_vs_wmr_save_file_name = 'sentence_length_gte_' + str(
                args.visualize_by_sentence_length) + '_' + 'num_vs_wmr_' + name_info + 'logdata_visualization'

            lt_num_vs_wmr_save_file_name = 'sentence_length_lt_' + str(
                args.visualize_by_sentence_length) + '_' + 'num_vs_wmr_' + name_info + 'logdata_visualization'

            show_plot_info(gte_wmr_visualization_df, save_file_name=gte_wmr_file_name.split('.')[0], y_label='WMR')
            iki_interval_df = calculate_iki_intervals(gte_wmr_visualization_df, y_label='WMR',
                                                      interval_size=interval_size)
            plot_wmr_vs_iki(iki_interval_df, save_file_name=gte_wmr_file_name.split('.')[0],
                            interval_size=interval_size, origin_df=gte_wmr_visualization_df,
                            label_extra_info=gte_label_extra_info, visualize_mean=visualize_mean)

            plot_num_vs_wmr(save_file_name=gte_num_vs_wmr_save_file_name,
                            interval_size=0.005, origin_df=gte_wmr_visualization_df,
                            label_extra_info=gte_label_extra_info, visualize_mean=visualize_mean)

            show_plot_info(lt_wmr_visualization_df, save_file_name=lt_wmr_file_name.split('.')[0], y_label='WMR')
            iki_interval_df = calculate_iki_intervals(lt_wmr_visualization_df, y_label='WMR',
                                                      interval_size=interval_size)
            plot_wmr_vs_iki(iki_interval_df, save_file_name=lt_wmr_file_name.split('.')[0],
                            interval_size=interval_size, origin_df=lt_wmr_visualization_df,
                            label_extra_info=lt_label_extra_info, visualize_mean=visualize_mean)

            plot_num_vs_wmr(save_file_name=lt_num_vs_wmr_save_file_name,
                            interval_size=0.005, origin_df=lt_wmr_visualization_df,
                            label_extra_info=lt_label_extra_info, visualize_mean=visualize_mean)

        if args.modification:
            modification_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, modification_file_name),
                                                        names=MODIFICATION_VISUALIZATION_COLUMNS,
                                                        encoding='ISO-8859-1')

            gte_modification_visualization_df = modification_visualization_df[
                modification_visualization_df['TEST_SECTION_ID'].isin(sentence_length_gte_ids)]

            lt_modification_visualization_df = modification_visualization_df[
                modification_visualization_df['TEST_SECTION_ID'].isin(sentence_length_lt_ids)]

            gte_modification_file_name = 'sentence_length_gte_' + str(
                args.visualize_by_sentence_length) + '_' + modification_file_name
            lt_modification_file_name = 'sentence_length_lt_' + str(
                args.visualize_by_sentence_length) + '_' + modification_file_name

            show_plot_info(gte_modification_visualization_df, save_file_name=gte_modification_file_name.split('.')[0],
                           y_label='MODIFICATION')
            iki_interval_df = calculate_iki_intervals(gte_modification_visualization_df,
                                                      y_label='MODIFICATION', interval_size=interval_size)
            plot_modification_vs_iki(iki_interval_df, save_file_name=gte_modification_file_name.split('.')[0],
                                     interval_size=interval_size, origin_df=gte_modification_visualization_df,
                                     label_extra_info=gte_label_extra_info, visualize_mean=visualize_mean)

            show_plot_info(lt_modification_visualization_df, save_file_name=lt_modification_file_name.split('.')[0],
                           y_label='MODIFICATION')
            iki_interval_df = calculate_iki_intervals(lt_modification_visualization_df,
                                                      y_label='MODIFICATION', interval_size=interval_size)
            plot_modification_vs_iki(iki_interval_df, save_file_name=lt_modification_file_name.split('.')[0],
                                     interval_size=interval_size, origin_df=lt_modification_visualization_df,
                                     label_extra_info=lt_label_extra_info, visualize_mean=visualize_mean)

        if ac_file_name and args.ac:
            ac_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, ac_file_name),
                                              names=AC_VISUALIZATION_COLUMNS,
                                              encoding='ISO-8859-1')

            gte_ac_visualization_df = ac_visualization_df[
                ac_visualization_df['TEST_SECTION_ID'].isin(sentence_length_gte_ids)]

            lt_ac_visualization_df = ac_visualization_df[
                ac_visualization_df['TEST_SECTION_ID'].isin(sentence_length_lt_ids)]

            gte_ac_file_name = 'sentence_length_gte_' + str(
                args.visualize_by_sentence_length) + '_' + ac_file_name
            lt_ac_file_name = 'sentence_length_lt_' + str(
                args.visualize_by_sentence_length) + '_' + ac_file_name

            show_plot_info(gte_ac_visualization_df, save_file_name=gte_ac_file_name.split('.')[0], y_label='AC')
            iki_interval_df = calculate_iki_intervals(gte_ac_visualization_df, y_label='AC',
                                                      interval_size=interval_size)
            plot_ac_vs_iki(iki_interval_df, save_file_name=gte_ac_file_name.split('.')[0], interval_size=interval_size,
                           origin_df=gte_ac_visualization_df, label_extra_info=gte_label_extra_info,
                           visualize_mean=visualize_mean)

            show_plot_info(lt_ac_visualization_df, save_file_name=lt_ac_file_name.split('.')[0], y_label='AC')
            iki_interval_df = calculate_iki_intervals(lt_ac_visualization_df, y_label='AC',
                                                      interval_size=interval_size)
            plot_ac_vs_iki(iki_interval_df, save_file_name=lt_ac_file_name.split('.')[0], interval_size=interval_size,
                           origin_df=lt_ac_visualization_df, label_extra_info=lt_label_extra_info,
                           visualize_mean=visualize_mean)
