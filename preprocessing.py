import os
import cv2
from PIL import Image
import numpy as np
import skimage.morphology as mp
import scipy.ndimage.morphology as sm
from tqdm import tqdm


class Image_Split(object):

    def __init__(self, original_path, split8x8_path, splitRGB_path, convert_path):
        self.original_path = original_path
        self.split8x8_path = split8x8_path
        self.splitRGB_path = splitRGB_path
        self.convert_path = convert_path

        self.org_tmrm_path = self.splitRGB_path + r'tmrm\\'
        self.org_sytox_path = self.splitRGB_path + r'sytox\\'
        self.org_nucleus_path = self.splitRGB_path + r'nucleus\\'

        self.convert_tmrm_path = self.convert_path + r'tmrm\\'
        self.convert_tmrmBinary_path = self.convert_path + r'tmrmBinary\\'
        self.convert_sytox_path = self.convert_path + r'sytox\\'
        self.convert_nucleus_path = self.convert_path + r'nucleus\\'

        print('{}'.format('*') * 30)
        print(' Start image preprocessing')
        print('{}'.format('*') * 30)

    def get_split_size(self, img_w, img_h, crop_num):
        # crop 할 사이즈 : grid_w, grid_h
        grid_w = img_w / crop_num  # crop width
        grid_h = img_h / crop_num  # crop height
        range_w = int(img_w / grid_w)
        range_h = int(img_h / grid_h)

        return grid_w, grid_h, range_w, range_h


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
                img_crop.save(self.split8x8_path + crop_img_name)

                i += 1

        return peices_img_list

    def original_split(self, original_list, split_list, plate_map, crop_num):
        img_check = 0
        print(f'Split all original images into {crop_num}x{crop_num}')

        for (root, directories, files) in os.walk(self.original_path):
            img_check += len(files)
            for file in files:
                plate_name = str(root.split('\\')[-1])
                org_id = str(file.split('.')[0])
                org_path = os.path.join(root, file)

                if file:
                    fname = plate_map[plate_name] + '_' + org_id
                    original_list.append(fname)

                    load_img = Image.open(org_path)
                    (img_h, img_w) = load_img.size

                    grid_w, grid_h, range_w, range_h = self.get_split_size(img_h, img_w, crop_num)
                    peices_img_list = self.peices_img_save(load_img, fname, grid_w, grid_h, range_w, range_h)
                    split_list.extend(peices_img_list)

        print(' -- Checked images :', img_check)
        print(' -- Total number of original images           :', len(original_list))
        print(' -- Total number of 8x8 split original images :', len(split_list))

        return original_list, split_list

    def piece_img_rgb_split(self, img_list, rgb_dic):
        print('Split into RGB images')

        for i in tqdm(range(len(img_list))):
            img = img_list[i]
            fname = img.split('.')[0]
            img_split = cv2.imread(self.split8x8_path + img, cv2.IMREAD_COLOR)
            b, g, r = cv2.split(img_split)

            zeros = np.zeros((img_split.shape[0], img_split.shape[1]), dtype="uint8")
            tmrm = cv2.merge([zeros, zeros, r])
            sytox = cv2.merge([zeros, g, zeros])
            nucleus = cv2.merge([b, zeros, zeros])

            tmrm_save_fname = fname + '_TMRM.png'
            sytox_save_fname = fname + '_Sytox.png'
            nucleus_save_fname = fname + '_Nucleus.png'

            if tmrm_save_fname:
                if tmrm_save_fname not in rgb_dic.get('TMRM'):
                    rgb_dic['TMRM'].append(tmrm_save_fname)
            if sytox_save_fname:
                if sytox_save_fname not in rgb_dic.get('Sytox'):
                    rgb_dic['Sytox'].append(sytox_save_fname)
            if nucleus_save_fname:
                if nucleus_save_fname not in rgb_dic.get('Nucleus'):
                    rgb_dic['Nucleus'].append(nucleus_save_fname)

            cv2.imwrite(self.org_tmrm_path + tmrm_save_fname, tmrm)
            cv2.imwrite(self.org_sytox_path + sytox_save_fname, sytox)
            cv2.imwrite(self.org_nucleus_path + nucleus_save_fname, nucleus)

        print(' -- Total number of original tmrm    :', len(rgb_dic['TMRM']))
        print(' -- Total number of original sytox   :', len(rgb_dic['Sytox']))
        print(' -- Total number of original nucleus :', len(rgb_dic['Nucleus']))

        return rgb_dic

class Image_Convert(Image_Split):
    def __init__(self, original_path, split8x8_path, splitRGB_path, convert_path):
            Image_Split.__init__(self, original_path, split8x8_path, splitRGB_path, convert_path)

    def sytox_and_nucleus_converter(self, img_list, rgb_type, convert_threshold):

        converted_list = []

        if rgb_type == 'sytox':
            read_img_path = self.org_sytox_path
            convert_save_path = self.convert_sytox_path
        else :
            read_img_path = self.org_nucleus_path
            convert_save_path = self.convert_nucleus_path

        for i in tqdm(range(len(img_list))):
            img = img_list[i]
            read_img = cv2.imread(str(read_img_path + img))
            img_grayscale = cv2.cvtColor(read_img, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(img_grayscale, convert_threshold, 255, cv2.THRESH_BINARY)

            ## binary fill
            binary_fill_img = sm.binary_fill_holes(thresh)

            ## bording
            disk2 = mp.disk(2)
            bording_img = mp.binary_dilation(binary_fill_img, disk2)
            bording_img = mp.binary_erosion(bording_img, disk2)
            bording_img = mp.binary_opening(bording_img, disk2)
            bording_img = mp.binary_closing(bording_img, disk2)
            bording_img = sm.binary_fill_holes(bording_img)

            bording_img_to_cv = (bording_img * 1).astype('uint8')  # cv2의 기본 데이터타입은 'uint8'이므로 바꿔준다.
            masked_bording_img = cv2.bitwise_or(read_img, read_img, mask=bording_img_to_cv)

            cv2.imwrite(convert_save_path + img, masked_bording_img)

            converted_list.append(masked_bording_img)

        return converted_list

    def tmrm_converter(self, org_tmrm, org_sytox, sy_rm_thr, tmrm_binary_thr=20, create_binary=True):

        converted_list = []

        for i in tqdm(range(len(org_tmrm))):
            org_tmrm_img = org_tmrm[i]
            org_sytox_img = org_sytox[i]

            read_tmrm = cv2.imread(self.org_tmrm_path + org_tmrm_img, cv2.IMREAD_COLOR)
            tm_b, tm_g, tm_r = cv2.split(read_tmrm)

            read_sytox = cv2.imread(self.org_sytox_path + org_sytox_img, cv2.IMREAD_COLOR)
            sy_b, sy_g, sy_r = cv2.split(read_sytox)

            ret, thresh = cv2.threshold(sy_g, sy_rm_thr, 255, cv2.THRESH_BINARY)

            zeros = np.zeros((tm_g.shape[0], tm_g.shape[1]), dtype="uint8")

            ## binary fill
            binary_fill_img = sm.binary_fill_holes(thresh)

            # bording
            disk2 = mp.disk(2)  # 디스크 생성
            bording_img = mp.binary_dilation(binary_fill_img, disk2)
            bording_img = mp.binary_erosion(bording_img, disk2)
            bording_img = mp.binary_opening(bording_img, disk2)
            bording_img = mp.binary_closing(bording_img, disk2)
            bording_img = sm.binary_fill_holes(bording_img)
            bording_img_to_cv = (bording_img * 1).astype('uint8')  # cv2의 기본 데이터타입은 'uint8'이므로 바꿔준다.
            masked_bording_img = cv2.bitwise_or(sy_g, sy_g, mask=bording_img_to_cv)

            converted_tmrm_gray = cv2.subtract(tm_r, masked_bording_img)
            converted_tmrm = cv2.merge([zeros, zeros, converted_tmrm_gray])
            cv2.imwrite(self.convert_tmrm_path + org_tmrm_img, converted_tmrm)
            converted_list.append(converted_tmrm)

            if create_binary:
                _, converted_tmrmBinary = cv2.threshold(converted_tmrm_gray, tmrm_binary_thr, 255, cv2.THRESH_BINARY)
                cv2.imwrite(self.convert_tmrmBinary_path + org_tmrm_img, converted_tmrmBinary)

        return converted_list

    def rgb_img_convert(self, rgb_list, sy_thr, sy_rm_thr, nu_thr):
        print('Convert RGB images')

        org_tmrm_list = rgb_list['TMRM']
        org_sytox_list = rgb_list['Sytox']
        org_nucleus_list = rgb_list['Nucleus']

        converted_tmrm = self.tmrm_converter(org_tmrm_list, org_sytox_list, sy_rm_thr)
        converted_sytox = self.sytox_and_nucleus_converter(org_sytox_list, 'sytox', sy_thr)
        converted_nucleus = self.sytox_and_nucleus_converter(org_nucleus_list, 'nucleus', nu_thr)

        print(' -- Total number of converted tmrm    : ', len(converted_tmrm))
        print(' -- Total number of converted sytox   : ', len(converted_sytox))
        print(' -- Total number of converted nucleus : ', len(converted_nucleus))

        #return converted_tmrm, converted_sytox, converted_nucleus

class Image_Merge(object):

    def __init__(self, filter_path, merge_path):
        self.filter_path = filter_path
        self.merge_path = merge_path

        self.filter_tmrm_path = self.filter_path + r'tmrm\\'
        self.filter_sytox_path = self.filter_path + r'sytox\\'
        self.filter_nucleus_path = self.filter_path + r'nucleus\\'

        print('{}'.format('*') * 30)
        print(' Start image merging')
        print('{}'.format('*') * 30)

    def rbg_merge(self, img_list):
        print('Merge tmrm and sytox images')

        merged_list = []

        for i in tqdm(range(len(img_list['TMRM']))):
            img_id = img_list['TMRM'][i].partition('_T')[0]
            tmrm = img_list['TMRM'][i]
            sytox = [sy_img for sy_img in img_list['Sytox'] if img_id in sy_img][0]
            nucleus = [nu_img for nu_img in img_list['Nucleus'] if img_id in nu_img][0]

            tm_img = cv2.imread(self.filter_tmrm_path + tmrm, cv2.IMREAD_COLOR)
            tm_b, tm_g, tm_r = cv2.split(tm_img)

            sy_img = cv2.imread(self.filter_sytox_path + sytox, cv2.IMREAD_COLOR)
            sy_b, sy_g, sy_r = cv2.split(sy_img)

            nu_img = cv2.imread(self.filter_nucleus_path + nucleus, cv2.IMREAD_COLOR)
            nu_b, nu_g, nu_r = cv2.split(nu_img)

            # merged = cv2.merge([nu_b, sy_g, tm_r])

            zeros = np.zeros((tm_img.shape[0], tm_img.shape[1]), dtype="uint8")
            merged = cv2.merge([zeros, sy_g, tm_r])

            cv2.imwrite(self.merge_path + img_id + '.png', merged)

            merged_list.append(img_id)

        print(' -- Total number of merged images : ', len(merged_list))

        return merged_list






