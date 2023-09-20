import os
import pandas as pd
import argparse
from io_utils import dump_json_object, load_json_object, dump_pickle_object, load_pickle_object
from preprocessing import Image_Split, Image_Convert, Image_Merge
from filtering import Cell_Detecting_and_Counting, Filtering

plate_map = { 'PLATE_I':'p1','PLATE_II':'p2','PLATE_III':'p3','PLATE_IV':'p4',
               'PLATE_V':'p5', 'PLATE_VI':'p6', 'PLATE_VII':'p7', 'PLATE_VIII':'p8'}

# 이미지 각 경로
base_path = r"D:\\Dataset\\Mitochondria_new\\"
original_path = base_path + r"original\\"
split4x4_path = base_path + r"split8x8\\"
splitRGB_path = base_path + r"splitRGB\\"
convert_path = base_path + r"convert\\"
filter_path = base_path + r"filter\\"
detect_save_path = base_path + r"detect\\"
merge_path = base_path + r"merge\\"

def arg_parse():
    desc = "Pytorch"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-d_c', '--cell_detection_counting', type=str, required=False, default='f', help='If this flag is t then Directory name to save the model')
    parser.add_argument('-dc_path', '--detect_img_path', type=str, required=False, default=convert_path, help='Image path to be detected')

    parser.add_argument('-nu_sy_f', '--nucleus_sytox_filtering', type=str, required=False, default='f', help='If this flag is t then Directory name to save the model')

    parser.add_argument('-tm_f', '--tmrm_filtering', type=str, required=False, default='f', help='If this flag is t then Directory name to save the model')
    parser.add_argument('-tm_th', '--tmrm_bad_filter_threshold', type=float, required=False, default=20, help='Sytox convert criteria threshold')

    return parser.parse_args()


def run_of_img_filtering(args):

    cell_detection = args.cell_detection_counting
    detect_img_path = args.detect_img_path

    nu_sy_filter = args.nucleus_sytox_filtering
    tm_filter = args.tmrm_filtering
    tm_bad_thr = args.tmrm_bad_filter_threshold

    rgb_dic = {'TMRM': [], 'Sytox': [], 'Nucleus': []}
    splitRGB_list = load_json_object('splitRGB_list.json', compress=False)

    # Cell detection & counting on the transformed images
    if cell_detection == 't':
        cell_det = Cell_Detecting_and_Counting(base_path, detect_save_path)
        cell_cnt_result = cell_det.detection_counting(detect_img_path)
    else :
        cell_cnt_result = pd.read_excel(base_path + 'number_of_cells.xlsx', index_col = 0)

    # Split 'crop_num' piece images into rgb
    if nu_sy_filter == 't':
        img_filter = Filtering(base_path, convert_path, filter_path)
        detect_bad_list, detect_filter_result = img_filter.nucleus_and_sytox_filtering(cell_cnt_result, rgb_dic, splitRGB_list)
        dump_json_object(detect_bad_list, 'detect_bad_list.json')
        dump_json_object(detect_filter_result, 'detect_filter_result.json')
    else:
        detect_filter_result = load_json_object('detect_filter_result.json', compress=False)

    if tm_filter == 't':
        img_filter = Filtering(base_path, convert_path, filter_path)
        areaRatio_bad_list, areaRatio_filter_result = img_filter.tmrm_filtering(detect_filter_result, rgb_dic, tmrm_bad_thr=tm_bad_thr)
        dump_json_object(areaRatio_bad_list, 'areaRatio_bad_list.json')
        dump_json_object(areaRatio_filter_result, 'areaRatio_filter_result.json')

if __name__ == '__main__':
    args = arg_parse()
    print(vars(args))

    # image preprocessing
    run_of_img_filtering(args)

