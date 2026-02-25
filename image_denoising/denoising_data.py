import os

import torch
from PIL import Image
from torch.utils.data import Dataset,random_split
import torchvision.transforms as T

from denoising_config import *
from common.utils import *


class NoiseImageDataset(Dataset):
    # 初始化
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(image_dir))

    # 获取数据集长度
    def __len__(self):
        return len(self.image_names)

    # 根据索引号获取元素:(input,target)=(noise_image,image)
    def __getitem__(self, idx):
        # 1.构建图片的网站访问路径
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        # 2.打开图片
        image = Image.open(image_path).convert("RGB")
        # 3.图像转换,(调整大小,转换Tensor)
        if self.transform is not None:
            img_tensor = self.transform(image)
        else:  # (抛出异常)
            raise ValueError("transform 参数不能为None!")
        # 4.加入噪声,得到模型输入的input
        image_noise_tensor = img_tensor + NOISE_FACTOR * torch.randn_like(img_tensor)
        # 将图片数据范围限制在(0,1)
        image_noise_tensor = torch.clamp(image_noise_tensor, 0, 1)
        return image_noise_tensor, img_tensor

#创建数据集并划分
def create_datasets():
    transform=T.Compose([
        T.Resize((IMG_H,IMG_W)),
        T.ToTensor()
    ])
    #创建数据集
    dataset=NoiseImageDataset(IMG_PATH,transform=transform)
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



