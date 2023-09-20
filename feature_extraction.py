import torch
import os
import numpy as np

from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 2 to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Load_Dataset(Dataset):
    def __init__(self, image_list, root_dir, transform=None):
        self.image_list = image_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_id = self.image_list[index]

        img_name = os.path.join(self.root_dir, img_id)
        image = Image.open(img_name).convert('RGB')
        image = np.array(image)
        #print(image)
        y_iname = img_id

        # img_2 = Image.fromarray(image[:, :50, 2])  # NumPy array to PIL image
        # img_2.show()


        #print(y_iname)
        if self.transform:
            image = self.transform(image)

        return (image, y_iname)

def custom_imshow(img):
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

def feature_extract(image_dir):

    images = []

    print('Feature extraction from filtered images')

    # creates a ScandirIterator aliased as files
    with os.scandir(image_dir) as files:
        for file in files:
            if file.name.endswith('.png'):
                images.append(file.name)

    print(len(images), images[:5])

    # 기본적으로 PyTorch는 이미지 데이터셋을 [Batch Size, Channel, Width, Height] 순서대로 저장함
    dataset = Load_Dataset(images, image_dir,
                              transform=transforms.Compose(
                                  [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    dataloader = DataLoader(dataset, shuffle=False)

    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
    model.eval()
    model = model.to(device)

    # transformed images show
    # for batch_idx, (inputs, fname) in enumerate(dataloader):
    #     custom_imshow(inputs[0])


    data = {}
    with torch.no_grad():
        for i, (inputs, fname) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device)
            fname = fname[0]
            # inputs_0 = inputs[0].detach().cpu().numpy()
            # print(inputs_0, inputs[0].shape)

            feat = model(inputs)
            data[fname] = feat

    return data




