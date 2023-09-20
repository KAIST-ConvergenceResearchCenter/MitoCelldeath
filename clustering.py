import os
import cv2
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 2 to use

class Result_Image_Show(object):

    def __init__(self, base_path, filter_path, detect_path):
        self.base_path = base_path
        self.filter_path = filter_path
        self.detect_path = detect_path

        self.tmrm_filter_path = self.filter_path + r'tmrm\\'
        self.sytox_filter_path = self.filter_path + r'sytox\\'
        self.nucleus_filter_path = self.filter_path + r'nucleus\\'

        self.tmrmGray_filter_path = self.base_path + r'filter_new\\tmrmGray\\'

        self.sytox_detect_path = self.detect_path + r'sytox\\'
        self.nucleus_detect_path = self.detect_path + r'nucleus\\'

    def get_rgb_list(self, image_ids, cluster):

        tmrm_img_list = [self.tmrm_filter_path + image.replace('Sytox', 'TMRM') for image in image_ids]
        tmrmGray_img_list = [self.tmrmGray_filter_path + image.replace('Sytox', 'TMRM') for image in image_ids]
        sytox_img_list = [self.sytox_filter_path + image for image in image_ids]
        nucleus_img_list = [self.nucleus_filter_path + image.replace('Sytox', 'Nucleus') for image in image_ids]

        sytox_detect_img_list = [self.sytox_detect_path + image for image in image_ids]
        nucleus_detect_img_list = [self.nucleus_detect_path + image.replace('Sytox', 'Nucleus') for image in image_ids]

        # tmrm_img_list = [self.tmrm_filter_path + image.replace('Nucleus', 'TMRM') for image in image_ids]
        # sytox_img_list = [self.sytox_filter_path + image.replace('Nucleus', 'Sytox') for image in image_ids]
        # nucleus_img_list = [self.nucleus_filter_path + image for image in image_ids]
        #
        # sytox_detect_img_list = [self.sytox_detect_path + image.replace('Nucleus', 'Sytox') for image in image_ids]
        # nucleus_detect_img_list = [self.nucleus_detect_path + image for image in image_ids]

        # tmrm_img_list = [self.tmrm_filter_path + image for image in image_ids]
        # sytox_img_list = [self.sytox_filter_path + image.replace('TMRM', 'Sytox') for image in image_ids]
        # nucleus_img_list = [self.nucleus_filter_path + image.replace('TMRM', 'Nucleus') for image in image_ids]
        #
        # sytox_detect_img_list = [self.sytox_detect_path + image.replace('TMRM', 'Sytox') for image in image_ids]
        # nucleus_detect_img_list = [self.nucleus_detect_path + image.replace('TMRM', 'Nucleus') for image in image_ids]

        print(f' -- Get {len(tmrm_img_list)} tmrm images from {cluster}')
        print(f' -- Get {len(sytox_img_list)} sytox images from {cluster}')
        print(f' -- Get {len(nucleus_img_list)} nucleus images from {cluster}')

        return tmrm_img_list, tmrmGray_img_list, sytox_img_list, nucleus_img_list, sytox_detect_img_list, nucleus_detect_img_list

    def draw_sample_data(self, files, clu):
        plt.figure(figsize=(25, 25))

        # plot each image in the cluster
        for index, file in enumerate(files):
            plt.subplot(10, 10, index + 1)
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        if r'filter\\tmrm' in files[0]:
            plt.savefig(self.base_path + f'{clu}_tmrm.png')
        elif r'filter\\sytox' in files[0]:
            plt.savefig(self.base_path + f'{clu}_sytox.png')
        elif r'filter\\nucleus' in files[0]:
            plt.savefig(self.base_path + f'{clu}_nucleus.png')
        elif r'detect\\sytox' in files[0]:
            plt.savefig(self.base_path + f'{clu}_detect_sytox.png')
        elif r'detect\\nucleus' in files[0]:
            plt.savefig(self.base_path + f'{clu}_detect_nucleus.png')
        elif r'filter_new\\tmrmGray' in files[0]:
            plt.savefig(self.base_path + f'{clu}_tmrmGray.png')

        #plt.show()
        #print(clustering_map[clu])



def img_clustering(predict_data, number_clusters, pca_component):

    print(f'Clustering with {number_clusters} progress') # (healthy, mild, medium, severe)

    # get a list of the filenames
    filenames = np.array(list(predict_data.keys()))
    #print(filenames, len(filenames))

    feat = [val[0].detach().cpu().numpy() for val in predict_data.values()]
    feat = np.array(list(feat))
    feat = feat.reshape(-1, 4096)  # reshape so that there are N samples of 4096 vectors

    kmeans = KMeans(n_clusters=number_clusters, n_init=1)  # random_state=42 -> 항상 랜덤값이 같게 하기위해 쓴다.
    # kmeans.fit(x)
    kmeans.fit(feat)
    print('Clustering Result')
    print(kmeans.labels_, len(kmeans.labels_))

    clustering_result = {}

    for file, cluster in zip(filenames, kmeans.labels_):
        if f'cluster{cluster}' not in clustering_result.keys():
            clustering_result[f'cluster{cluster}'] = []
            clustering_result[f'cluster{cluster}'].append(file[0])
        else:
            clustering_result[f'cluster{cluster}'].append(file[0])

    for k in clustering_result.keys():
        print(f' -- Number of {k} images :', len(clustering_result[k]))

    return clustering_result


def clustering_result_show(base_path, filter_path, detect_path, img_show_num):
    print('Clustering results show')
    with open(base_path + 'syB_clustering_result.json', "r") as classified_json:
        clustering_set = json.load(classified_json)

    img_show = Result_Image_Show(base_path, filter_path, detect_path)

    for clt in clustering_set.keys():
        cluster = clustering_set[clt]
        random.shuffle(cluster)
        #tmrm_img_list, sytox_img_list, nucleus_img_list, sytox_detect_img_list, nucleus_detect_img_list = img_show.get_rgb_list(cluster[:img_show_num], str(clt))
        tmrm_img_list, tmrmGray_img_list, sytox_img_list, nucleus_img_list, sytox_detect_img_list, nucleus_detect_img_list = img_show.get_rgb_list(cluster[:img_show_num], str(clt))
        img_show.draw_sample_data(tmrm_img_list, clt)
        img_show.draw_sample_data(sytox_img_list, clt)
        img_show.draw_sample_data(nucleus_img_list, clt)
        img_show.draw_sample_data(sytox_detect_img_list, clt)
        img_show.draw_sample_data(nucleus_detect_img_list, clt)

        img_show.draw_sample_data(tmrmGray_img_list, clt)
