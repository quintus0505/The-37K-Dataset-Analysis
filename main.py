from config import *
from tools.data_loading import *
from tools.parser import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data-cleaning", action="store_true", default=False, help="generate logdata file")
parser.add_argument("--auto-correct", action="store_true", default=False, help='if considering auto-correct')
parser.add_argument("--predict", action="store_true", default=False, help='if considering prediction')
parser.add_argument("--swipe", action="store_true", default=False, help='if considering swipe')
parser.add_argument("--keyboard", type=str, default=None, help='keyboard type')
parser.add_argument("--analyze", action="store_true", default=False, help="anylyze the data")
parser.add_argument("--visualize", action="store_true", default=False, help="visualize the data")

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
        parser.compute_wmr(full_log_data=True, ite=None, keyboard='Gboard', custom_logdata_path=logdata_path)
        parser.compute_ac(full_log_data=True, ite=None, keyboard='Gboard', custom_logdata_path=logdata_path)

        save_file_name = name_info + 'logdata_visualization.csv'
        parser.save_iki_wmr_visualization(osp.join(DEFAULT_VISUALIZATION_DIR, 'wmr_' + save_file_name))
        if args.auto_correct:
            parser.save_iki_ac_visualization(osp.join(DEFAULT_VISUALIZATION_DIR, 'ac_' + save_file_name))

    if args.visualize:
        print("visualizing data")
        name_info, ite_list, kbd = get_name_str()
        ac_file_name = ''
        wmr_file_name = ''
        if args.auto_correct:
            ac_file_name = 'ac_' + name_info + 'logdata_visualization.csv'
            wmr_file_name = 'wmr_' + name_info + 'logdata_visualization.csv'
        else:
            wmr_file_name = 'wmr_' + name_info + 'logdata_visualization.csv'

        if ac_file_name:
            ac_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, ac_file_name),
                                              names=AC_VISUALIZATION_COLUMNS,
                                              encoding='ISO-8859-1')
            show_plot_info(ac_visualization_df, save_file_name=ac_file_name.split('.')[0])
            ac_visualization_df = calculate_iki_intervals(ac_visualization_df)
            plot_ac_vs_iki(ac_visualization_df, save_file_name=ac_file_name.split('.')[0])
        wmr_visualization_df = pd.read_csv(osp.join(DEFAULT_VISUALIZATION_DIR, wmr_file_name), names=WMR_VISUALIZATION_COLUMNS,
                        encoding='ISO-8859-1')
        show_plot_info(wmr_visualization_df, save_file_name=wmr_file_name.split('.')[0])
        wmr_visualization_df = calculate_iki_intervals(wmr_visualization_df)
        plot_wmr_vs_iki(wmr_visualization_df, save_file_name=wmr_file_name.split('.')[0])
