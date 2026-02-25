import torch.nn as nn
import torch

#自定义神经网络类:基于CNN的去噪器
#自定义神经网络类:基于CNN的去噪器
class ConvDeNoiser(nn.Module):
    def __init__(self):
        super(ConvDeNoiser,self).__init__()
        #编码器
            #卷积层
        self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,stride=1,padding=1)
        #通用池化层
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        #解码器
        #转置卷积成
        self.conv_t1=nn.ConvTranspose2d(in_channels=8,out_channels=8,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.conv_t2=nn.ConvTranspose2d(in_channels=8,out_channels=16,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.conv_t3=nn.ConvTranspose2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1,output_padding=1)
        #输出普通卷积层
        self.conv_out=nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        #编码
        #第一次卷积,激活,池化
        x=torch.relu(self.conv1(x))
        x=self.pool(x)

        #第二次卷积,激活,池化
        x=torch.relu(self.conv2(x))
        x=self.pool(x)

        #第三次卷积,激活,池化
        x=torch.relu(self.conv3(x))
        x=self.pool(x)

        #解码
        #第一次转置卷积
        x=torch.relu(self.conv_t1(x))

        #第二次转置卷积
        x=torch.relu(self.conv_t2(x))

        #第三次转置卷积
        x=torch.relu(self.conv_t3(x))

        #最终输出:sigmoid限制在0-1
        y=torch.sigmoid(self.conv_out(x))

        return y

if __name__ == '__main__':
    input=torch.randn(12,3,64,64)
    model=ConvDeNoiser()
    output=model(input)
    print(output.shape)