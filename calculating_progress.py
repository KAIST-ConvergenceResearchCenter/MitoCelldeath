import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from io_utils import load_json_object, dump_json_object

def create_save_folder(save_dir, fname):
    if os.path.exists(save_dir + fname):
        shutil.rmtree(save_dir + fname)
        os.makedirs(save_dir + fname)
    else:
        os.makedirs(save_dir + fname)

class Input_Image_Split(object):

    def __init__(self, input_img_dir, output_save_dir):
        self.input_dir = input_img_dir
        self.output_save_dir = output_save_dir
        self.crop_num = 8

    def peices_img_save(self, img, fname, grid_w, grid_h, range_w, range_h):

        i = 0
        peices_img_list = []

        for w in range(range_w):
            for h in range(range_h):
                img_peice_num = "{}.png".format("_{0:03d}".format(i))
                bbox = (h * grid_h, w * grid_w, (h + 1) * (grid_h), (w + 1) * (grid_w))
                img_crop = img.crop(bbox)  # 가로 세로 시작, 가로 세로 끝

                crop_img_name = fname + img_peice_num
                peices_img_list.append(crop_img_name)

                # if not os.path.isdir(file_path + crop_img_name):
                img_crop.save(self.output_save_dir + f'{fname}\\' + crop_img_name)

                i += 1

        return peices_img_list

    def img_split(self, file_list):
        analysis_img = {}
        for file in file_list:
            fname = str(file.split('.')[0])
            create_save_folder(self.output_save_dir, fname)

            load_img = Image.open(self.input_dir + file)
            (img_h, img_w) = load_img.size

            grid_w = img_w / self.crop_num  # crop width
            grid_h = img_h / self.crop_num  # crop height
            range_w = int(img_w / grid_w)
            range_h = int(img_h / grid_h)

            peices_img_list = self.peices_img_save(load_img, fname, grid_w, grid_h, range_w, range_h)
            if fname not in analysis_img:
                analysis_img.update({fname: peices_img_list})

        print(' -- Total number of original images           :', len(file_list))
        print(' -- Total number of 8x8 split original images :', len(list(analysis_img.values())))
        return analysis_img


class Filtering(object):
    def __init__(self, saved_dir):
        self.saved_dir = saved_dir

    def get_area_ratio(self, img, area_sum):
        img_h, img_w = img.shape
        rectangle_area = img_h * img_w
        ratio = area_sum / rectangle_area
        return 100 * ratio

    def ratio_calculate(self, img, tmrbBinary_thr=20):
        _, thresh = cv2.threshold(img, tmrbBinary_thr, 255, cv2.THRESH_BINARY)
        white_area_sum = np.sum(thresh == 255)
        result_ratio = self.get_area_ratio(img, white_area_sum)
        return result_ratio

    def area_ratio_to_excel(self, ratio_dict, img_path, save_name):
        sorted_dict = dict(sorted(ratio_dict.items(), key=lambda x: x[1], reverse=True))
        df = pd.DataFrame({'image_id': list(sorted_dict.keys()),
                           'ratio': list(sorted_dict.values())})
        df.to_excel(img_path + save_name + ".xlsx")

    def img_delete_for_filtering_result(self, img_path, bad_img_list):
        for i in tqdm(range(len(bad_img_list))):
            #print(get_path + img, save_path)
            img = bad_img_list[i]
            os.remove(img_path + img)

    def tmrm_filtering(self, split_list, tmrm_bad_thr):

        all_bad_img_list = {}
        all_filtering_result = {}

        for img_folder, fname in split_list.items():
            print(img_folder)

            tmrm_ratio_result = {}
            bad_img_list = []
            filtering_result = []

            img_path = self.saved_dir + f'{img_folder}\\'

            #TMRM 면적 비율 기준으로 TMRM 이미지 drop
            for i in tqdm(range(len(fname))):
                img_name = fname[i]
                img = cv2.imread(img_path + img_name, cv2.IMREAD_GRAYSCALE)
                ratio = self.ratio_calculate(img)

                if img_name not in tmrm_ratio_result.keys():
                    tmrm_ratio_result.update({img_name: ratio})

                if ratio < tmrm_bad_thr:
                    bad_img_list.append(img_name)

            self.area_ratio_to_excel(tmrm_ratio_result, img_path, f'{img_folder}_ratio')
            print('--- Find Bad images in TMRM images :', len(bad_img_list))

            self.img_delete_for_filtering_result(img_path, bad_img_list)
            all_bad_img_list.update({img_folder: bad_img_list})

            tmrm_filtering_result = [x for x in fname if x not in bad_img_list]
            all_filtering_result.update({img_folder : tmrm_filtering_result})

            print('--- TMRM filtering result based on tmrm area ratio    :', len(tmrm_filtering_result))

        return all_bad_img_list, all_filtering_result