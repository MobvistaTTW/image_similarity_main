import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import  DataLoader
from tqdm import tqdm

from common.utils import *
from classification_config import *
from classification_data import  create_datasets
from classification_model import Classifier
from classification_engine import evaluate

def predict(model,test_loader,device):
    #1.取一个批次的测试图像
    data_iter=iter(test_loader)
    imgs,labels=next(data_iter)

    #2.模型预测
    with torch.no_grad():
        input=imgs.to(device)
        outputs=model(input)

    #获取预测标签
    pred_labels=outputs.cpu().argmax(dim=-1).numpy()

    #3.数据转换
    images=imgs.permute(0,2,3,1).numpy()

    #4.画图
    fig,axes=plt.subplots(1,10,figsize=(25,4),sharex=True,sharey=True)

    for i in range(10):
        axes[i].imshow(images[i])
        axes[i].set_axis_off()
        pred_class=classification_names[pred_labels[i]]
        print(f"{i+1} 的 预测标签为 {pred_labels[i]},他的真实标签为{labels[i]},他的预测分类为{pred_class}")
        print()

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
    model = Classifier().to(device)
    model.load_state_dict(torch.load(CLASSIFIER_MODEL_NAME))
    print("========模型加载完成=========")
    # 5.测试
    predict(model,test_loader, device)
    test_loss=evaluate(model,test_loader,device)
    print(f"测试误差为{test_loss:.6f}")



