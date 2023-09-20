import os
import numpy as np
import pandas as pd
from io_utils import load_json_object, dump_json_object, dump_pickle_object, load_pickle_object
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from feature_extraction import feature_extract
from calculating_progress import Input_Image_Split, Filtering
import matplotlib.pyplot as plt
from PIL import Image
from tabulate import tabulate

def arg_parse():
    desc = "Pytorch"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-img_fname', '--image_file_name', type=str, required=False, default='', help='If this flag is t then Directory name to save the model')
    parser.add_argument('-img_dir', '--image_directory', type=str, required=True, default='', help='If this flag is t then Directory name to save the model')
    parser.add_argument('-save_dir', '--save_image_directory', type=str, required=False, default=r'D:\\Dataset\\Mitochondria_new\\cell_analysis\\',
                        help='If this flag is t then Directory name to save the model')
    parser.add_argument('-tm_th', '--tmrm_bad_filter_threshold', type=float, required=False, default=20, help='Sytox convert criteria threshold')

    return parser.parse_args()

def similarity_calculate(healthy_centroid, severe_centroid, file_feat):

    # Cosine similarity
    healthy_similarity = cosine_similarity(healthy_centroid, file_feat)[0][0]
    severe_similarity = cosine_similarity(severe_centroid, file_feat)[0][0]

    # Jaccard similarity coefficient score
    # healthy_similarity = jaccard_score(healthy_centroid, file_feat, average="micro")
    # severe_similarity = jaccard_score(severe_centroid, file_feat, average="micro")

   #print(healthy_similarity, severe_similarity)

    if healthy_similarity > severe_similarity:
        progress_result = 'healthy'
        similarity_score = healthy_similarity
    else:
        progress_result = 'severe'
        similarity_score = severe_similarity

    return progress_result, similarity_score, healthy_similarity, severe_similarity


if __name__ == '__main__':
    args = arg_parse()
    print(vars(args))

    input_img_fname = args.image_file_name
    input_img_dir = args.image_directory + '\\'
    output_save_dir = args.save_image_directory
    tm_bad_thr = args.tmrm_bad_filter_threshold

    clustering_centroid = load_json_object('center_val_result.json', compress=False)
    healthy_centroid = np.array([clustering_centroid['cluster0']])
    severe_centroid = np.array([clustering_centroid['cluster1']])

    input_img_split = Input_Image_Split(input_img_dir, output_save_dir)
    split_img_filtering = Filtering(output_save_dir)

    # if input_img_fname:
    #     file_list = [input_img_fname]
    #     split_list = input_img_split.img_split(file_list)
    #     all_bad_img_list, all_filtering_result = split_img_filtering.tmrm_filtering(split_list, tmrm_bad_thr=tm_bad_thr)
    # else:
    #     file_list = os.listdir(input_img_dir)
    #     file_list = [file for file in file_list if file.endswith(".png")]
    #     split_list = input_img_split.img_split(file_list)
    #     all_bad_img_list, all_filtering_result = split_img_filtering.tmrm_filtering(split_list, tmrm_bad_thr=tm_bad_thr)
    #
    # dump_json_object(all_bad_img_list, 'cell_analysis\\bad_image_list.json')
    # dump_json_object(all_filtering_result, 'cell_analysis\\analysis_image_list.json')

    all_filtering_result = load_json_object('cell_analysis\\analysis_image_list.json', compress=False)

    # Feature Extract for the image list to be analyzed
    for img_folder in all_filtering_result.keys():
        print(img_folder)
        feat_result = feature_extract(output_save_dir + f'{img_folder}\\' )
        dump_pickle_object(feat_result, f'cell_analysis\\{img_folder}\\{img_folder}_featResult_np.pkl', compress=False)

        #feat_result = load_pickle_object(f'cell_analysis\\{img_folder}\\{img_folder}_featResult_np.pkl', compress=False)
        similarity_score_matrix = np.zeros([8 * 8])
        progress_matrix = np.asarray(np.zeros([8 * 8]), dtype = str)

        df = pd.DataFrame(columns=['img_fname', 'healthy_sim', 'severe_sim'])  ## 1. 데이터 초기화 - 열 이름 지정
        i = 0
        for file_name in list(feat_result.keys()):
            file_num = file_name.split('.')[0][-3:]
            file_feat = np.array([feat_result[file_name][0].detach().cpu().numpy()])

            progress_result, similarity_score, healthy_similarity, severe_similarity = similarity_calculate(healthy_centroid, severe_centroid, file_feat)

            df.loc[i] = [file_name, healthy_similarity, severe_similarity]
            i += 1

            similarity_score_matrix[int(file_num)] = round(similarity_score, 4)
            progress_matrix[int(file_num)] = progress_result
        print(similarity_score_matrix)
        print(progress_matrix)
        print(tabulate(df, headers='keys', tablefmt='pretty'))
        #print(similarity_score_matrix.reshape(8, 8))
        #print(progress_matrix.reshape(8, 8))