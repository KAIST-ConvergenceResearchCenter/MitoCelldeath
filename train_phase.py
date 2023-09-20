from __future__ import print_function, division
import os
import pandas as pd
import argparse
import numpy as np
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from torchvision import transforms
from torch.utils.data import DataLoader
from efficientNet.dataloader import TMRM_Dataset, build_model
from efficientNet.data_check import dataset_check
from efficientNet.train_test import train_model
from efficientNet.test_visualize import test_and_visualize_model
from efficientNet.visualize import visual

plate_map = { 'PLATE_I':'p1','PLATE_II':'p2','PLATE_III':'p3','PLATE_IV':'p4',
               'PLATE_V':'p5', 'PLATE_VI':'p6', 'PLATE_VII':'p7', 'PLATE_VIII':'p8'}

# 이미지 각 경로
base_path = r"D:\\Dataset\\Mitochondria_new\\"
original_path = base_path + r"original\\"
split4x4_path = base_path + r"split8x8\\"
splitRGB_path = base_path + r"splitRGB\\"
convert_path = base_path + r"convert\\"
filter_path = base_path + r"filter\\"
detect_path = base_path + r"detect\\"

dataset_path = os.getcwd() + r'\\efficientNet\\'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu

def arg_parse():
    desc = "Pytorch"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-train', type=str, required=False, default='f', help='If this flag is t then Directory name to save the model')
    parser.add_argument('-pre', '--pretrained', type=str, required=False, default='t', help='The number of epochs')
    parser.add_argument('-fin', '--fine_tune', type=str, required=False, default='t', help='The number of epochs')

    parser.add_argument('-test', type=str, required=False, default='f', help='If this flag is t then Directory name to save the model')

    parser.add_argument('-ep', '--epoch', type=int, default=50, help='The number of epochs')
    parser.add_argument('-ba', '--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='The size of batch')
    parser.add_argument('-sa_fn', '--save_model_folder_name', type=str, required=False, default='exp_train', help='Nucleus convert criteria threshold')
    parser.add_argument('-sa_dir', '--save_model_path', type=str, required=False, default=dataset_path + r'dataset\\save_model\\', help='Nucleus convert criteria threshold')

    parser.add_argument('-vi_test', type=str, required=False, default='f', help='If this flag is t then Directory name to save the model')
    return parser.parse_args()

## Seed 고정
## 시드를 고정해 모델의 추론 값이 일정하게 유지 될 수 있도록함. 시드를 고정하지 않으면 커널을 다시 시작하여 모델을 학습 할 때 마다 결과값이 다를 수 있음
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _init_fn(seed):
    np.random.seed(int(seed))


def run_of_model_training(args):
    seed = 42
    num_epoch = args.epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    save_model = args.save_model_path
    model_fname = args.save_model_folder_name
    save_path = save_model + f'{model_fname}\\'

    training = args.train
    testing = args.test
    visualized = args.vi_test

    seed_everything(int(seed))

    if not os.path.exists(dataset_path + 'dataset\\save_model'):
        os.makedirs(dataset_path + 'dataset\\save_model')

    all_data_dir = dataset_path + 'dataset\\'

    annotation_train = all_data_dir + 'train_annotation_tmrm.json'
    image_dir_train = all_data_dir + r'train\\'

    annotation_val = all_data_dir + 'val_annotation_tmrm.json'
    image_dir_val = all_data_dir + r'val\\'

    annotation_test = all_data_dir + 'test_annotation_tmrm.json'
    image_dir_test = all_data_dir + r'test\\'

    tmrm_train = TMRM_Dataset(annotation_train, image_dir_train,
                              transform=transforms.Compose(
                                  [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    tmrm_val = TMRM_Dataset(annotation_val, image_dir_val,
                            transform=transforms.Compose(
                                [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    tmrm_test = TMRM_Dataset(annotation_test, image_dir_test,
                             transform=transforms.Compose(
                                 [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    # batch_size : 모델을 한 번 학습시킬 때 몇 개의 데이터를 넣을지 정함. 1 배치가 끝날때마다 파라미터를 조정함
    # shuffle : 데이터를 섞을지 정함
    # num_workers : 몇개의 subprocesses를 가동시킬건지 정함
    # drop_last : 배치별로 묶고 남은 데이터를 버릴지 (True) 여부를 정함
    dataloader_train = DataLoader(tmrm_train, batch_size=batch_size, shuffle=True, worker_init_fn=_init_fn(seed))
    dataloader_val = DataLoader(tmrm_val, batch_size=batch_size, shuffle=True, worker_init_fn=_init_fn(seed))
    dataloader_test = DataLoader(tmrm_test, batch_size=batch_size, shuffle=True, worker_init_fn=_init_fn(seed))

    dataloader = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}

    # Data Check
    # dataset_check(dataloader)

    ### Loading Model ###
    model = build_model(pretrained=args.pretrained, fine_tune=args.fine_tune)    # True or False
    # print(model.classifier)
    summary(model, input_size=(3, 224, 224), device='cpu')

    model = model.to(device)

    # 분류 문제이므로 손실 함수는 CrossEntropyLoss 사용
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=4)

    if training == 't':
        train_model(dataloader, model, criterion, optimizer_ft, scheduler, save_model, num_epochs=num_epoch,
                    folder_name=model_fname)

    if testing == 't':
        checkpoint = torch.load(f=save_path + 'bestcheckpoint.pth.tar')
        print('Best Model of {}'.format(model_fname))
        print('Best Epoch: {}, Best ACC: {}'.format(checkpoint['epoch'], checkpoint['best_acc']))
        ## TEST!
        test_and_visualize_model(dataloader, criterion, model, save_path, checkpoint, phase='test')

    if visualized == 't':
        checkpoint = torch.load(f=save_path + 'bestcheckpoint.pth.tar')
        ## TEST!
        visual(dataloader, save_path, model, checkpoint, phase='test')

if __name__ == '__main__':
    args = arg_parse()
    print(vars(args))

    # image clustering
    run_of_model_training(args)

