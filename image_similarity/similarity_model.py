import torch.nn as nn
import torch


#分别定义编码器类和解码器类
class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder,self).__init__()
        #编码器
        #卷积层
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.conv5=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv6=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        #通用池化层
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)

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

        #第四次卷积,激活,池化
        x = torch.relu(self.conv4(x))
        x = self.pool(x)

        #第五次卷积,激活,池化
        x = torch.relu(self.conv5(x))
        x = self.pool(x)

        #第六次卷积,激活,池化
        x = torch.relu(self.conv6(x))
        x = self.pool(x)

        #压缩为向量形式(N，512)返回
        x=x.squeeze(-1).squeeze(-1)

        return x

class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder,self).__init__()
        # 解码器
        # 转置卷积成
        self.conv_t1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.conv_t2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.conv_t3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.conv_t4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.conv_t5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.conv_t6 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)


    def forward(self,x):
        #回复4维张量形状(N,512,1,1)
        x=x.unsqueeze(-1).unsqueeze(-1)
        # 解码
        # 第一次转置卷积
        x = torch.relu(self.conv_t1(x))

        # 第二次转置卷积
        x = torch.relu(self.conv_t2(x))

        # 第三次转置卷积
        x = torch.relu(self.conv_t3(x))

        # 第四次转置卷积
        x = torch.relu(self.conv_t4(x))

        # 第五次转置卷积
        x = torch.relu(self.conv_t5(x))

        # # 第六次转置卷积
        # x = torch.relu(self.conv_t6(x))

        # 最终输出:sigmoid限制在0-1
        x = torch.sigmoid(self.conv_t6(x))

        return x



if __name__ == '__main__':
    input=torch.randn(10,3,64,64)
    encoder=ConvEncoder()
    decoder=ConvDecoder()
    #前向传播
    embeddings=encoder(input)
    print("嵌入向量形状:",embeddings.shape)
    output=decoder(embeddings)
    print("重构图像形状:",output.shape)