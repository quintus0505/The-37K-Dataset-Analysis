from config import *
from tools.data_loading import *
from tools.parser import *
from tools.visualization import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data-cleaning", action="store_true", default=False, help="generate logdata file")
parser.add_argument("--auto-correct", action="store_true", default=False, help='if considering auto-correct')
parser.add_argument("--predict", action="store_true", default=False, help='if considering prediction')
parser.add_argument("--swipe", action="store_true", default=False, help='if considering swipe')
parser.add_argument("--keyboard", type=str, default=None, help='keyboard type')
parser.add_argument("--os", type=str, default=None, help='os type')
parser.add_argument("--analyze", action="store_true", default=False, help="anylyze the data")
parser.add_argument("--visualize", action="store_true", default=False, help="visualize the data")
parser.add_argument("--filter", type=float, default=0, help='filter those test section with iki below this percentage')
parser.add_argument('--error-rate', action="store_true", default=False, help="compute corrected and uncorrected error rate")

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

args = parser.parse_args()


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
        build_custom_logdata(ite=ite_list, keyboard=kbd, file_name=file_name, os=os)
        get_sheet_info(sheet_name=file_name)

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
        filter_ratio = args.filter
        print("filtering data below {} % perscent".format(args.filter))
        name_info, ite_list, kbd, os = get_name_str()
        save_file_name = 'filtered_' + str(args.filter) + "_" + name_info + 'logdata.csv'
        file_name = name_info + 'logdata.csv'
        logdata_path = osp.join(DEFAULT_CLEANED_DATASETS_DIR, file_name)
        parser.filter(filter_ratio=filter_ratio, ite=ite_list, keyboard=kbd, save_file_name=save_file_name,
                      load_file_name=logdata_path)

    if args.error_rate:
        print("computing error rate")
        name_info, ite_list, kbd, os = get_name_str()
        file_name = name_info + 'logdata.csv'
        logdata_path = osp.join(DEFAULT_CLEANED_DATASETS_DIR, file_name)
        parser = Parse()
        parser.compute_error_rate_correction(full_log_data=True, ite=ite_list, keyboard=kbd, custom_logdata_path=logdata_path)

    if args.visualize:
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
                                     interval_size=interval_size, origin_df=modification_visualization_df)

        if args.wmr:
            wmr_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, wmr_file_name),
                                               names=WMR_VISUALIZATION_COLUMNS,
                                               encoding='ISO-8859-1')
            show_plot_info(wmr_visualization_df, save_file_name=wmr_file_name.split('.')[0], y_label='WMR')
            iki_interval_df = calculate_iki_intervals(wmr_visualization_df, y_label='WMR',
                                                      interval_size=interval_size)
            plot_wmr_vs_iki(iki_interval_df, save_file_name=wmr_file_name.split('.')[0],
                            interval_size=interval_size, origin_df=wmr_visualization_df)

            plot_num_vs_wmr(save_file_name='num_vs_wmr_' + name_info + 'logdata_visualization',
                            interval_size=0.005, origin_df=wmr_visualization_df)

        if ac_file_name and args.ac:
            ac_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, ac_file_name),
                                              names=AC_VISUALIZATION_COLUMNS,
                                              encoding='ISO-8859-1')
            show_plot_info(ac_visualization_df, save_file_name=ac_file_name.split('.')[0], y_label='AC')
            iki_interval_df = calculate_iki_intervals(ac_visualization_df, y_label='AC',
                                                      interval_size=interval_size)
            plot_ac_vs_iki(iki_interval_df, save_file_name=ac_file_name.split('.')[0], interval_size=interval_size,
                           origin_df=ac_visualization_df)
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
                                      interval_size=interval_size, origin_df=edit_distance_visualization_df)
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

    if args.visualize_by_edit_distance:
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
                            label_extra_info=gte_label_extra_info)

            plot_num_vs_wmr(save_file_name=gte_num_vs_wmr_save_file_name,
                            interval_size=0.005, origin_df=gte_wmr_visualization_df,
                            label_extra_info=gte_label_extra_info)

            show_plot_info(lt_wmr_visualization_df, save_file_name=lt_wmr_file_name.split('.')[0], y_label='WMR')
            iki_interval_df = calculate_iki_intervals(lt_wmr_visualization_df, y_label='WMR',
                                                      interval_size=interval_size)
            plot_wmr_vs_iki(iki_interval_df, save_file_name=lt_wmr_file_name.split('.')[0],
                            interval_size=interval_size, origin_df=lt_wmr_visualization_df,
                            label_extra_info=lt_label_extra_info)

            plot_num_vs_wmr(save_file_name=lt_num_vs_wmr_save_file_name,
                            interval_size=0.005, origin_df=lt_wmr_visualization_df,
                            label_extra_info=lt_label_extra_info)

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
                                     label_extra_info=gte_label_extra_info)

            show_plot_info(lt_modification_visualization_df, save_file_name=lt_modification_file_name.split('.')[0],
                           y_label='MODIFICATION')
            iki_interval_df = calculate_iki_intervals(lt_modification_visualization_df,
                                                      y_label='MODIFICATION', interval_size=interval_size)
            plot_modification_vs_iki(iki_interval_df, save_file_name=lt_modification_file_name.split('.')[0],
                                     interval_size=interval_size, origin_df=lt_modification_visualization_df,
                                     label_extra_info=lt_label_extra_info)

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
                           origin_df=gte_ac_visualization_df, label_extra_info=gte_label_extra_info)

            show_plot_info(lt_ac_visualization_df, save_file_name=lt_ac_file_name.split('.')[0], y_label='AC')
            iki_interval_df = calculate_iki_intervals(lt_ac_visualization_df, y_label='AC',
                                                      interval_size=interval_size)
            plot_ac_vs_iki(iki_interval_df, save_file_name=lt_ac_file_name.split('.')[0], interval_size=interval_size,
                           origin_df=lt_ac_visualization_df, label_extra_info=lt_label_extra_info)

    if args.visualize_by_sentence_length:
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
                            label_extra_info=gte_label_extra_info)

            plot_num_vs_wmr(save_file_name=gte_num_vs_wmr_save_file_name,
                            interval_size=0.005, origin_df=gte_wmr_visualization_df,
                            label_extra_info=gte_label_extra_info)

            show_plot_info(lt_wmr_visualization_df, save_file_name=lt_wmr_file_name.split('.')[0], y_label='WMR')
            iki_interval_df = calculate_iki_intervals(lt_wmr_visualization_df, y_label='WMR',
                                                      interval_size=interval_size)
            plot_wmr_vs_iki(iki_interval_df, save_file_name=lt_wmr_file_name.split('.')[0],
                            interval_size=interval_size, origin_df=lt_wmr_visualization_df,
                            label_extra_info=lt_label_extra_info)

            plot_num_vs_wmr(save_file_name=lt_num_vs_wmr_save_file_name,
                            interval_size=0.005, origin_df=lt_wmr_visualization_df,
                            label_extra_info=lt_label_extra_info)

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
                                     label_extra_info=gte_label_extra_info)

            show_plot_info(lt_modification_visualization_df, save_file_name=lt_modification_file_name.split('.')[0],
                           y_label='MODIFICATION')
            iki_interval_df = calculate_iki_intervals(lt_modification_visualization_df,
                                                      y_label='MODIFICATION', interval_size=interval_size)
            plot_modification_vs_iki(iki_interval_df, save_file_name=lt_modification_file_name.split('.')[0],
                                     interval_size=interval_size, origin_df=lt_modification_visualization_df,
                                     label_extra_info=lt_label_extra_info)

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
                           origin_df=gte_ac_visualization_df, label_extra_info=gte_label_extra_info)

            show_plot_info(lt_ac_visualization_df, save_file_name=lt_ac_file_name.split('.')[0], y_label='AC')
            iki_interval_df = calculate_iki_intervals(lt_ac_visualization_df, y_label='AC',
                                                      interval_size=interval_size)
            plot_ac_vs_iki(iki_interval_df, save_file_name=lt_ac_file_name.split('.')[0], interval_size=interval_size,
                           origin_df=lt_ac_visualization_df, label_extra_info=lt_label_extra_info)






