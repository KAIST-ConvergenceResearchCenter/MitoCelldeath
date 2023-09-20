import os
import pandas as pd
import argparse
from io_utils import dump_json_object, load_json_object, dump_pickle_object, load_pickle_object
from feature_extraction import feature_extract
from clustering import img_clustering, clustering_result_show
from annotation import Create_Annotation, Split_Annotation

#clustering_map = {'cluster0': 'medium', 'cluster1': 'mild', 'cluster2': 'healthy', 'cluster3': 'severe'}
#clustering_map = {'cluster2': 'healthy', 'cluster3': 'severe'}
clustering_map = {'cluster1': 'mild', 'cluster2': 'medium'}

# 이미지 각 경로
base_path = r"D:\\Dataset\\Mitochondria_new\\"
original_path = base_path + r"original\\"
split4x4_path = base_path + r"split8x8\\"
splitRGB_path = base_path + r"splitRGB\\"
convert_path = base_path + r"convert\\"
filter_path = base_path + r"filter\\"
detect_path = base_path + r"detect\\"

dataset_path = os.getcwd() + r'\\efficientNet\\'


def arg_parse():
    desc = "Pytorch"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-fe_e', '--feature_extract', type=str, required=False, default='f', help='If this flag is t then Directory name to save the model')

    parser.add_argument('-cl', '--clustering', type=str, required=False, default='f', help='If this flag is t then Directory name to save the model')
    parser.add_argument('-cls_n', '--clustering_class_num', type=int, required=False, default=4, help='')
    parser.add_argument('-vi', '--visualize', type=str, required=False, default='f', help='If this flag is t then Directory name to save the model')
    parser.add_argument('-vi_n', '--visualize_img_num', type=int, required=False, default=100, help='')

    parser.add_argument('-an', '--creat_annotation', type=str, required=False, default='f', help='If this flag is t then Directory name to save the model')

    return parser.parse_args()



def run_of_img_clustering(args):

    feat_extract = args.feature_extract

    clustering = args.clustering
    cls_n = args.clustering_class_num
    visualize = args.visualize
    visualize_n = args.visualize_img_num

    # image feature extracting and clustering
    if feat_extract == 't':
        #img_path = filter_path + r'nucleusBinary\\'
        #img_path = 'D:\\Dataset\\Mitochondria_new\\filter_3filt_10%\\sytoxBinary\\'
        img_path = 'D:\\Dataset\\Mitochondria_new\\filter_new\\sytoxNo\\'
        #img_path = 'D:\\Dataset\\Mitochondria_new\\filter_new\\sytoxBinary\\'
        feat_result = feature_extract(img_path)
        dump_pickle_object(feat_result, 'feature_extract_result_np.pkl', compress=False)
    else:
        feat_result = load_pickle_object('feature_extract_result_np.pkl', compress=False)

    if clustering == 't':
        clustering_result = img_clustering(feat_result, number_clusters=cls_n, pca_component=10)
        dump_json_object(clustering_result, 'syB_clustering_result.json')

    if visualize == 't':
        clustering_result_show(base_path, filter_path, detect_path, img_show_num=visualize_n)

if __name__ == '__main__':
    args = arg_parse()
    print(vars(args))

    # image clustering
    run_of_img_clustering(args)

    # create train, val, test annotation
    if args.creat_annotation == 't':
        create_anno = Create_Annotation(base_path, filter_path, clustering_map)
        tmrm_annotation = create_anno.annotate('TMRM')
        dump_json_object(tmrm_annotation, 'annotation_tmrm.json')

        #tmrm_annotation = load_json_object('annotation_tmrm.json', compress=False)
        anno_split = Split_Annotation(filter_path, dataset_path)
        anno_split.trainValTest_split(tmrm_annotation, 'tmrm')

        # sytox_annotation = create_anno.annotate('Sytox')
        # dump_json_object(sytox_annotation, 'annotation_sytox.json')
        # # anno_split.trainValTest_split(sytox_annotation, 'sytox')
        #
        # nucleus_annotation = create_anno.annotate('Nucleus')
        # dump_json_object(nucleus_annotation, 'annotation_nucleus.json')
        # # anno_split.trainValTest_split(nucleus_annotation, 'nucleus')
