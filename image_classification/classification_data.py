import os

import torch
from PIL import Image
from torch.utils.data import Dataset,random_split
import torchvision.transforms as T

from classification_config import *
from common.utils import *
import pandas as pd


class ImageLabelDataset(Dataset):
    #初始化
    def __init__(self, image_dir, label_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(image_dir))
        label_df = pd.read_csv(label_path)
        self.labels = label_df['target'].tolist()

    #获取数据集长度
    def __len__(self):
        return len(self.image_names)

    #根据索引号获取元素:(input,target)=(image,label)
    def __getitem__(self, idx):
        #1.构建图片的网站访问路径
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        #2.打开图片
        image = Image.open(image_path).convert("RGB")
        #3.图像转换,(调整大小,转换Tensor)
        if self.transform is not None:
            img_tensor = self.transform(image)
        else:  #(抛出异常)
            raise ValueError("transform 参数不能为None!")
        #4.找到图片对应的标签
        img_label = self.labels[idx]

        return img_tensor, img_label

#创建数据集并划分
def create_datasets():
    transform=T.Compose([
        T.Resize((IMG_H,IMG_W)),
        T.ToTensor()
    ])
    #创建数据集
    dataset=ImageLabelDataset(IMG_PATH,LABELS_PATH,transform=transform)
    #划分数据集
    train_dataset,val_dataset,test_dataset=random_split(dataset,[TRAIN_RATIO,VAL_RATIO,TEST_RATIO])
    return train_dataset,val_dataset,test_dataset

if __name__ == '__main__':
    # dataset=NoiseImageDataset(IMG_PATH)
    # print(dataset.image_names)
    train_dataset, val_dataset, test_dataset=create_datasets()
    print(len(train_dataset))
    print(len(val_dataset))

    print(len(test_dataset))



