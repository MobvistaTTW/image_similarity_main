from torch import nn
import torch


class Classifier(nn.Module):
    #初始化,传入分类数
    def __init__(self,n_classes=5):
        super(Classifier,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,8,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(8,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),

            nn.Linear(16*16*16,n_classes)
        )

    #定义前向传播
    def forward(self,x):
        return self.model(x)

if __name__ == '__main__':
    input=torch.randn(32,3,64,64)
    print(Classifier(input))
