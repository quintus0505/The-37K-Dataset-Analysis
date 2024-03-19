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
parser.add_argument("--analyze", action="store_true", default=False, help="anylyze the data")
parser.add_argument("--visualize", action="store_true", default=False, help="visualize the data")
parser.add_argument("--filter", type=float, default=0, help='filter those test section with iki below this percentage')

parser.add_argument("--wmr", action="store_true", default=False, help="wmr")
parser.add_argument("--ac", action="store_true", default=False, help="ac")
parser.add_argument("--modification", action="store_true", default=False, help="modification")
parser.add_argument("--age", action="store_true", default=False, help="age")
parser.add_argument("--num", action="store_true", default=False, help="num")
parser.add_argument("--edit-distance", action="store_true", default=False, help="edit distance")

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
    name_str = keyboard_str + ite_str

    return name_str, ite, keyboard


if __name__ == "__main__":

    if args.data_cleaning:
        print("cleaning data")
        name_info, ite_list, kbd = get_name_str()
        file_name = name_info + 'logdata.csv'
        print("Generating logdata file: ", file_name)
        build_custom_logdata(ite=ite_list, keyboard=kbd, file_name=file_name)
        get_sheet_info(sheet_name=file_name)

    if args.analyze:
        print("analyzing data")
        parser = Parse()
        name_info, ite_list, kbd = get_name_str()
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
        name_info, ite_list, kbd = get_name_str()
        save_file_name = 'filtered_' + str(args.filter) + "_" + name_info + 'logdata.csv'
        file_name = name_info + 'logdata.csv'
        logdata_path = osp.join(DEFAULT_CLEANED_DATASETS_DIR, file_name)
        parser.filter(filter_ratio=filter_ratio, ite=ite_list, keyboard=kbd, save_file_name=save_file_name,
                      load_file_name=logdata_path)

    if args.visualize:
        interval_size = 10
        print("visualizing data")
        name_info, ite_list, kbd = get_name_str()
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
            plot_edit_distance_vs_iki(iki_interval_df, save_file_name='edit_distance_' + name_info + 'logdata_visualization',
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
