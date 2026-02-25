import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import  DataLoader
from tqdm import tqdm

from common.utils import *
from denoising_config import *
from denoising_data import  create_datasets
from denoising_model import ConvDeNoiser
from denoising_engine import evaluate

def test_new_data(model,test_loader,device):
    #1.取一个批次的测试图像
    data_iter=iter(test_loader)
    noise_imgs,imgs=next(data_iter)

    #2.模型预测
    with torch.no_grad():
        inputs=noise_imgs.to(device)
        outputs=model(inputs)

    #3.数据转换
    images_numpy=imgs.permute(0,2,3,1).numpy()
    noise_imgs_numpy=noise_imgs.permute(0,2,3,1).numpy()
    outputs_numpy=outputs.permute(0,2,3,1).cpu().numpy()

    #4.画图
    fig,axes=plt.subplots(3,10,figsize=(25,4),sharex=True,sharey=True)

    for ax_row,imgs in zip(axes,[images_numpy,noise_imgs_numpy,outputs_numpy]):
        for ax,img in zip(ax_row,imgs):
            ax.imshow(img)
            ax.set_axis_off()
    plt.show()

if __name__ == '__main__':
    seed_everything(SEED)
    # 1.定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2.创建数据集
    _, _, test_dataset = create_datasets()
    print("=============测试数据集创建完成===========")
    # 3.创建加载器
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE)
    print("=============测试数据加载器创建完成===========")
    # 4.定义模型,加载参数
    model = ConvDeNoiser().to(device)
    model.load_state_dict(torch.load(DENOISER_MODEL_NAME))
    print("========模型加载完成=========")
    # 5.测试
    test_new_data(model,test_loader, device)
    test_loss=evaluate(model,test_loader,nn.MSELoss(),device)
    print(f"测试误差为{test_loss:.6f}")



