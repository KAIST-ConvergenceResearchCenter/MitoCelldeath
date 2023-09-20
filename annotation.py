import os
import json
import cv2
import random
import math
import shutil

class Create_Annotation(object):

    def __init__(self, base_path, filter_path, clustering_map):
        self.base_path = base_path
        self.filter_path = filter_path
        self.clustering_map = clustering_map

        print('{}'.format('*') * 40)
        print(' Create annotations for model training')
        print('{}'.format('*') * 40)

    def get_name(self, image, rgb_type):
        # id = image.split('.')[0]
        # fname = id + f'_{rgb_type}.png'
        # original_fname = id + f'_{rgb_type}'

        img_split = image.split('_')
        id = img_split[0] + '_' + img_split[1] + '_' + img_split[2]
        fname = image.replace('Sytox','TMRM')
        original_fname = fname.split('.')[0]

        return id, fname, original_fname

    def image_size(self, rgb_type, image):
        img = cv2.imread(self.filter_path + f'{rgb_type}\\' + image)  # 이미지 변수 할당
        img_h, img_w, img_d = img.shape
        return img_h, img_w

    def annotate(self, rgb_type):

        annotation_dic = {}

        with open(self.base_path + 'syB_clustering_result.json', "r") as classified_json:
            clustering_set = json.load(classified_json)

        for key, val in clustering_set.items():
            # print('- ',key, len(val))
            if key == 'cluster1' or key == 'cluster2':
                for image in val:
                    id, fname, original_fname = self.get_name(image, rgb_type)
                    img_h, img_w = self.image_size(rgb_type, fname)

                    annotation_dic.update({original_fname: []})

                    image_form = {
                        "fname": str(fname),
                        "ori_fname": original_fname,
                        "id": id,
                        "type": str(rgb_type),
                        "progress": str(self.clustering_map[key]),
                        "height": img_h,
                        "width": img_w,
                    }
                    annotation_dic[original_fname].append(image_form)

        print(f'--- Completion of {rgb_type} annotations')

        return annotation_dic

class Split_Annotation(object):
    def __init__(self, filter_path, dataset_path):
        self.filter_path = filter_path
        self.filter_tmrm_path = self.filter_path + r'tmrm\\'
        self.filter_sytox_path = self.filter_path + r'sytox\\'
        self.filter_nucleus_path = self.filter_path + r'nucleus\\'

        self.dataset_path = dataset_path + r'dataset\\'

        set_folder = ['train', 'val', 'test']

        os.makedirs(self.dataset_path, exist_ok=True)
        for i in range(len(set_folder)):
            if os.path.exists(self.dataset_path + set_folder[i]):
                shutil.rmtree(self.dataset_path + set_folder[i])
                os.makedirs(self.dataset_path + set_folder[i])
            else:
                os.makedirs(self.dataset_path + set_folder[i])

    def sort_by_progress(self, dataset, image_ids, progress_dic):
        for x in image_ids:
            progress = dataset[x][0]['progress']
            if progress == 'mild':
                progress_dic['mild'].append(x)
            elif progress == 'medium':
                progress_dic['medium'].append(x)

        print(progress_dic.keys())
        #print(len(progress_dic['healthy']), len(progress_dic['severe']))
        print(len(progress_dic['mild']), len(progress_dic['medium']))

        return progress_dic

    def split_dataset(self, get_progress, data_dic, val_ratio, test_ratio):

        for progress, image_ids in get_progress.items():
            #if progress == 'healthy' or progress == 'severe':
            if progress == 'mild' or progress == 'medium':
                random.shuffle(image_ids)
                random.shuffle(image_ids)

                num_train = math.ceil(len(image_ids) * (1 - val_ratio))
                num_val = math.ceil((len(image_ids) - num_train) * (1 - test_ratio))

                image_ids_train, image_ids_val, image_ids_test = image_ids[:num_train], image_ids[
                                                                                    num_train:num_val + num_train], image_ids[
                                                                                                                    num_val + num_train:]
                for train_img in image_ids_train:
                    data_dic['train'].append(train_img)
                    shutil.copy(self.filter_tmrm_path + f'{train_img}.png', self.dataset_path + r'train\\')
                for val_img in image_ids_val:
                    data_dic['val'].append(val_img)
                    shutil.copy(self.filter_tmrm_path + f'{val_img}.png', self.dataset_path + r'val\\')
                for test_img in image_ids_test:
                    data_dic['test'].append(test_img)
                    shutil.copy(self.filter_tmrm_path + f'{test_img}.png', self.dataset_path + r'test\\')

        return data_dic

    def trainValTest_split(self, annFile, rgb_type):

        print('Split annotations into train, val, test')

        #progress_dic = {'healthy': [], 'severe': []}
        progress_dic = {'mild': [], 'medium': []}
        data_dic = {'train': [], 'val': [], 'test': []}

        train_data = {}
        val_data = {}
        test_data = {}

        image_ids = [x for x in annFile.keys()]
        get_progress = self.sort_by_progress(annFile, image_ids, progress_dic)
        data_dic = self.split_dataset(get_progress, data_dic, val_ratio=0.6, test_ratio=0.5)

        for tr in data_dic['train']:
            value = annFile[tr]
            train_data[tr] = value

        output_train_json = os.path.join(self.dataset_path, f'train_annotation_{rgb_type}.json')
        print(f'write {output_train_json}')
        with open(output_train_json, 'w') as train_writer:
            json.dump(train_data, train_writer, indent=4)

        for tr in data_dic['val']:
            value = annFile[tr]
            val_data[tr] = value

        output_val_json = os.path.join(self.dataset_path, f'val_annotation_{rgb_type}.json')
        print(f'write {output_val_json}')
        with open(output_val_json, 'w') as val_writer:
            json.dump(val_data, val_writer, indent=4)

        for tr in data_dic['test']:
            value = annFile[tr]
            test_data[tr] = value

        output_test_json = os.path.join(self.dataset_path, f'test_annotation_{rgb_type}.json')
        print(f'write {output_test_json}')
        with open(output_test_json, 'w') as test_writer:
            json.dump(test_data, test_writer, indent=4)

        print(' -- Total number of train set      :', len(data_dic['train']), data_dic['train'][:5])
        print(' -- Total number of validation set :', len(data_dic['val']), data_dic['val'][:5])
        print(' -- Total number of test set       :', len(data_dic['test']), data_dic['test'][:5])