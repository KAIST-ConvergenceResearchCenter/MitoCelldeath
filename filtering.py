import os
import cv2
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
from yolo_detect.detect import detection

class Cell_Detecting_and_Counting(object):

    def __init__(self, base_path, detect_save_path):
        self.base_path = base_path

        self.detect_save_path = detect_save_path
        self.save_sytox_path = self.detect_save_path + r'sytox\\'
        self.save_nucleus_path = self.detect_save_path + r'nucleus\\'

    def cell_cnt_result_to_excel(self, cell_dict, save_name):
        sorted_dict = dict(sorted(cell_dict.items()))
        df = pd.DataFrame({'image_id': list(sorted_dict.keys()),
                           'number': list(sorted_dict.values())})
        df.to_excel("D:\\Dataset\\Mitochondria_new\\" + save_name + ".xlsx")

    def cell_detect(self, detect_source, img_type):

        cell_cnt = {}
        if img_type == 'sytox':
            get_model = 'sytox_trained_model.pt'
            get_yaml = 'sytox_data.yaml'
            save_detect = self.save_sytox_path
        else:
            get_model = 'nucleus_trained_model.pt'
            get_yaml = 'nucleus_data.yaml'
            save_detect = self.save_nucleus_path

        cell_num = detection(cell_cnt, get_model, detect_source, get_yaml, save_detect)
        self.cell_cnt_result_to_excel(cell_num, f'number_of_{img_type}_cells')

        return cell_num

    def all_cell_counting(self, sytox_cell_num, nucleus_cell_num):
        sytox_id = [img_id.partition('_S')[0] for img_id in sytox_cell_num.keys()]
        nucleus_id = [img_id.partition('_N')[0] for img_id in nucleus_cell_num.keys()]

        sy_df = pd.DataFrame()
        sy_df['image_id'] = sytox_id
        sy_df['sytox_cell'] = sytox_cell_num.values()
        sy_df = sy_df.set_index('image_id')

        nu_df = pd.DataFrame()
        nu_df['image_id'] = nucleus_id
        nu_df['nucleus_cell'] = nucleus_cell_num.values()
        nu_df = nu_df.set_index('image_id')

        result = pd.merge(sy_df, nu_df, how='outer', on='image_id')

        s = result['sytox_cell']
        n = result['nucleus_cell']
        ratio_of_cell_death = (s/(s+n)) * 100
        result['cell_ratio'] = ratio_of_cell_death
        result = result.fillna(0)
        print(result)

        return result

    def detection_counting(self, detect_img_path):
        print('Detect sytox and nucleus cells')

        get_sytox_img = detect_img_path + r'sytox\\'
        get_nucleus_img = detect_img_path + r'nucleus\\'


        sytox_cell_num = self.cell_detect(get_sytox_img, 'sytox')
        nucleus_cell_num = self.cell_detect(get_nucleus_img, 'nucleus')

        cell_cnt_result = self.all_cell_counting(sytox_cell_num, nucleus_cell_num)
        cell_cnt_result.to_excel(self.base_path + 'number_of_cells.xlsx')

        print(' -- Total number of sytox images detected   :', len(sytox_cell_num))
        print(' -- Total number of nucleus images detected :', len(nucleus_cell_num))


        return cell_cnt_result


class Filtering(object):
    def __init__(self, base_path, convert_path, filter_path):
        self.base_path = base_path

        self.convert_path = convert_path
        self.filter_path = filter_path

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

    def area_ratio_to_excel(self, ratio_dict, save_name):
        sorted_dict = dict(sorted(ratio_dict.items(), key=lambda x: x[1], reverse=True))
        df = pd.DataFrame({'image_id': list(sorted_dict.keys()),
                           'ratio': list(sorted_dict.values())})
        ratio_show(df['image_id'], df['ratio'])
        df.to_excel(self.base_path + save_name + ".xlsx")

    def img_copy_for_filtering_result(self, img_list, rgb_type):

        if rgb_type == 'TMRM':
            get_path = self.convert_path + r'tmrm\\'
            save_path = self.filter_path + r'tmrm\\'
        elif rgb_type == 'Sytox':
            get_path = self.convert_path + r'sytox\\'
            save_path = self.filter_path + r'sytox\\'
        elif rgb_type == 'Nucleus':
            get_path = self.convert_path + r'nucleus\\'
            save_path = self.filter_path + r'nucleus\\'


        for i in tqdm(range(len(img_list))):
            #print(get_path + img, save_path)
            img = img_list[i]
            shutil.copy(get_path + img, save_path)

    def img_delete_for_filtering_result(self, bad_img_list, rgb_type):

        if rgb_type == 'TMRM':
            filter_path = self.filter_path + r'tmrm\\'
        elif rgb_type == 'Sytox':
            filter_path = self.filter_path + r'sytox\\'
        elif rgb_type == 'Nucleus':
            filter_path = self.filter_path + r'nucleus\\'

        for i in tqdm(range(len(bad_img_list))):
            #print(get_path + img, save_path)
            img = bad_img_list[i]
            os.remove(filter_path + img)

    def nucleus_and_sytox_filtering(self, cell_cnt_result, bad_img_dic, img_list):
        filtering_result = {'TMRM': [], 'Sytox': [], 'Nucleus': []}

        bad_img_list = cell_cnt_result[
            (cell_cnt_result['nucleus_cell'] == 0) | (cell_cnt_result['sytox_cell'] > cell_cnt_result['nucleus_cell'])]
            #| (cell_cnt_result['nucleus_cell'] < 10)]
        bad_img_list.to_excel(self.base_path + 'cell_bad_list.xlsx')

        print('--- Find Bad images :', len(bad_img_list))

        bad_img_id = bad_img_list.index.values
        bad_img_dic['TMRM'] = [id + '_TMRM.png' for id in bad_img_id]
        bad_img_dic['Sytox'] = [id + '_Sytox.png' for id in bad_img_id]
        bad_img_dic['Nucleus'] = [id + '_Nucleus.png' for id in bad_img_id]

        tmrm_filtering_result = [x for x in img_list['TMRM'] if x not in bad_img_dic['TMRM']]
        sytox_filtering_result = [x for x in img_list['Sytox'] if x not in bad_img_dic['Sytox']]
        nucleus_filtering_result = [x for x in img_list['Nucleus'] if x not in bad_img_dic['Nucleus']]

        filtering_result.update({'TMRM': tmrm_filtering_result})
        filtering_result.update({'Sytox': sytox_filtering_result})
        filtering_result.update({'Nucleus': nucleus_filtering_result})

        print('--- TMRM filtering result based on detect    :', len(tmrm_filtering_result))
        print('--- Sytox filtering result based on detect   :', len(sytox_filtering_result))
        print('--- Nucleus filtering result based on detect :', len(nucleus_filtering_result))

        self.img_copy_for_filtering_result(tmrm_filtering_result, 'TMRM')
        self.img_copy_for_filtering_result(sytox_filtering_result, 'Sytox')
        self.img_copy_for_filtering_result(nucleus_filtering_result, 'Nucleus')

        return bad_img_dic, filtering_result

    def tmrm_filtering(self, img_list, bad_img_dic, tmrm_bad_thr):
        tmrm_ratio_result = {}
        filtering_result = {'TMRM': [], 'Sytox': [], 'Nucleus': []}

        # TMRM 면적 비율 기준으로 TMRM, Sytox, Nucleus 이미지 drop
        for i in tqdm(range(len(img_list['TMRM']))):
            img_name = img_list['TMRM'][i]
            img = cv2.imread(str(self.convert_path + r'tmrm\\' + img_name), cv2.IMREAD_GRAYSCALE)
            ratio = self.ratio_calculate(img)

            if img_name not in tmrm_ratio_result.keys():
                tmrm_ratio_result.update({img_name: ratio})

            if ratio < tmrm_bad_thr:
                bad_img_dic['TMRM'].append(img_name)
                bad_img_dic['Sytox'].append(img_name.replace("TMRM", "Sytox"))
                bad_img_dic['Nucleus'].append(img_name.replace("TMRM", "Nucleus"))

        self.area_ratio_to_excel(tmrm_ratio_result, 'tmrm_ratio')

        print('--- Find Bad images in TMRM images :', len(bad_img_dic['TMRM']))
        print('--- Sytox and Nucleus have the same number of bad images')

        self.img_delete_for_filtering_result(bad_img_dic['TMRM'], 'TMRM')
        self.img_delete_for_filtering_result(bad_img_dic['Sytox'], 'Sytox')
        self.img_delete_for_filtering_result(bad_img_dic['Nucleus'], 'Nucleus')

        tmrm_filtering_result = [x for x in img_list['TMRM'] if x not in bad_img_dic['TMRM']]
        sytox_filtering_result = [x for x in img_list['Sytox'] if x not in bad_img_dic['Sytox']]
        nucleus_filtering_result = [x for x in img_list['Nucleus'] if x not in bad_img_dic['Nucleus']]

        filtering_result.update({'TMRM': tmrm_filtering_result})
        filtering_result.update({'Sytox': sytox_filtering_result})
        filtering_result.update({'Nucleus': nucleus_filtering_result})

        print('--- TMRM filtering result based on tmrm area ratio    :', len(tmrm_filtering_result))
        print('--- Sytox filtering result based on tmrm area ratio   :', len(sytox_filtering_result))
        print('--- Nucleus filtering result based on tmrm area ratio :', len(nucleus_filtering_result))

        return bad_img_dic, filtering_result

def ratio_show(img_id, ratio):
    x_len = np.arange(len(img_id))

    plt.figure(figsize=(12, 5))
    # plt.axis([0, x_len, 0, 100])     # X, Y축의 범위: [xmin, xmax, ymin, ymax]
    #plt.xticks(range(0, len(img_id), 10000))

    plt.plot(x_len, ratio)
    # plt.bar(x_len, rate,  width=W_)
    plt.xlabel('TMRM Image ID (Order By)', fontsize=14)
    plt.ylabel('Area Ratio', fontsize=14)
    plt.title('Mitochondrial network width ratio for each slice image (network area width / slice image width)')
    plt.tight_layout()
    plt.savefig("D:\\Dataset\\Mitochondria_new\\areaRatio_tmrm.png")
    #plt.show()

    bin = math.floor(round(ratio[0]) / 10 + 1) * 10
    ratio.plot(kind='hist', figsize=(12, 5), bins=bin, color='steelblue', edgecolor='black')
    #plt.yticks(range(0, 20000, 2000))
    plt.title('Number of image for mitochondrial network area ratio')
    plt.xlabel('TMRM Area Ratio')
    plt.ylabel('Number of TMRM Images')
    plt.savefig("D:\\Dataset\\Mitochondria_new\\areaRatio_tmrm_hist.png")
    #plt.show()
