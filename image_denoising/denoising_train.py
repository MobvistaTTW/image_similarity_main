import torch
from chromadb.api import DataLoader
from torch import nn,optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from denoising_config import *
from denoising_data import create_datasets
from denoising_model import ConvDeNoiser
from denoising_engine import *
from common.utils import seed_everything
from tqdm import tqdm

if __name__ == '__main__':
    seed_everything(SEED)
    #1.定义设备
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #2.创建数据集
    train_dataset, val_dataset, _=create_datasets()
    print("=============数据集创建完成===========")
    #3.创建加载器
    train_loader=DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True,drop_last=True)
    val_loader=DataLoader(val_dataset,batch_size=VAL_BATCH_SIZE)
    print("=============数据加载器创建完成===========")
    #4.定义模型
    model=ConvDeNoiser().to(device)
    #5.损失函数
    loss_fn=nn.MSELoss()
    #6.优化器
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)
    #7.训练核心流程
    print("========训练开始=========")
    min_val_loss=float('inf')
    for epoch in tqdm(range(EPOCHS)):
        #训练
        train_loss=train_one_epoch(model=model,trian_loader=train_loader,loss_fn=loss_fn,optimizer=optimizer,device=device)
        #验证
        val_loss=validate(model=model,val_loader=val_loader,loss_fn=loss_fn,device=device)

        print(f"Epoch:{epoch+1},Train Loss:{train_loss:.6f},Val Loss: {val_loss:.6f}")

        if val_loss<min_val_loss:
            print("验证损失集减少,保存模型")
            min_val_loss=val_loss
            torch.save(model.state_dict(),DENOISER_MODEL_NAME)

    print("==========训练结束==========")
    print(f"最小的验证损失值为:{min_val_loss}")

