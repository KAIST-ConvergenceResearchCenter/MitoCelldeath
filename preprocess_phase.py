import os
import argparse
from io_utils import dump_json_object, load_json_object, dump_pickle_object, load_pickle_object
from preprocessing import Image_Split, Image_Convert, Image_Merge


plate_map = { 'PLATE_I':'p1','PLATE_II':'p2','PLATE_III':'p3','PLATE_IV':'p4',
               'PLATE_V':'p5', 'PLATE_VI':'p6', 'PLATE_VII':'p7', 'PLATE_VIII':'p8'}

upper_folder_list = ['split8x8', 'splitRGB', 'convert', 'filter', 'detect', 'merge']
lower_folder_list = [[],['tmrm', 'sytox', 'nucleus'], ['tmrm', 'tmrmBinary', 'sytox', 'nucleus'], ['tmrm ', 'sytox', 'nucleus'], ['sytox', 'nucleus'], []]

# 이미지 각 경로
base_path = r"D:\\Dataset\\Mitochondria_new\\"
original_path = base_path + r"original\\"
split4x4_path = base_path + r"split8x8\\"
splitRGB_path = base_path + r"splitRGB\\"
convert_path = base_path + r"convert\\"
filter_path = base_path + r"filter\\"
detect_path = base_path + r"detect\\"
merge_path = base_path + r"merge\\"

def arg_parse():
    desc = "Pytorch"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-org_sp', '--org_split', type=str, required=False, default='f', help='If this flag is t then Original data 4x4split')
    parser.add_argument('-cp_n', '--crop_num', type=int, required=False, default=8, help='')

    parser.add_argument('-rgb_sp', '--rgb_split', type=str, required=False, default='f', help='If this flag is t then Directory name to save the model')

    parser.add_argument('-rgb_c', '--rgb_convert', type=str, required=False, default='f', help='If this flag is t then Directory name to save the model')
    parser.add_argument('-sy_th', '--sytox_convert_threshold', type=float, required=False, default=30, help='Sytox convert criteria threshold')
    parser.add_argument('-sy_rm_th', '--sytox_remove_threshold', type=float, required=False, default=220, help='Sytox convert criteria threshold')
    parser.add_argument('-nu_th', '--nucleus_convert_threshold', type=float, required=False, default=5, help='Nucleus convert criteria threshold')

    return parser.parse_args()

# 이미지 저장을 위한 폴더 생성
def create_folder(base_path):
    for i in range(len(upper_folder_list)) :
        upper_folder_name = upper_folder_list[i]
        lower_folder_name = lower_folder_list[i]
        if not os.path.exists(base_path + upper_folder_name):
            os.makedirs(base_path + upper_folder_name)
            if lower_folder_name:
                for f in lower_folder_name:
                    os.makedirs(base_path + f'{upper_folder_name}\\' + f)


def run_of_img_preprocessing(args):

    org_split = args.org_split
    rgb_split = args.rgb_split
    rgb_convert = args.rgb_convert

    original_list = []
    split_list = []
    rgb_dic = {'TMRM': [], 'Sytox': [], 'Nucleus': []}

    # Split original data into 4x4
    if org_split == 't':
        img_split = Image_Split(original_path, split4x4_path, splitRGB_path, convert_path)
        original_list, split_list = img_split.original_split(original_list, split_list, plate_map, crop_num=args.crop_num)
        dump_json_object(original_list, 'original_list.json')
        dump_json_object(split_list, 'split_list.json')
    else:
        split_list = load_json_object('split_list.json', compress=False)

    # Split 'crop_num' piece images into rgb
    if rgb_split == 't':
        img_split = Image_Split(original_path, split4x4_path, splitRGB_path, convert_path)
        splitRGB_list = img_split.piece_img_rgb_split(split_list, rgb_dic)
        #dump_json_object(splitRGB_list, 'splitRGB_list.json')
    else:
        splitRGB_list = load_json_object('splitRGB_list.json', compress=False)

    # Convert rgb(tmrm, sytox, nucleus) images
    if rgb_convert == 't':
        img_convert = Image_Convert(original_path, split4x4_path, splitRGB_path, convert_path)
        img_convert.rgb_img_convert(splitRGB_list, sy_thr=args.sytox_convert_threshold, sy_rm_thr=args.sytox_remove_threshold, nu_thr=args.nucleus_convert_threshold)

if __name__ == '__main__':
    args = arg_parse()
    print(vars(args))

    # image preprocessing
    create_folder(base_path)
    run_of_img_preprocessing(args)

