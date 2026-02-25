import torch

#训练一个轮次
def train_one_epoch(model,trian_loader,loss_fn,optimizer,device):
    #开启训练模式
    model.train()
    #定义损失函数
    total_loss=0
    #遍历这一批训练数据
    for noise_img,image in trian_loader:
        input=noise_img.to(device)
        target=image.to(device)

        output=model(input)

        loss=loss_fn(output,target)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        total_loss+=loss.item()

    return total_loss/len(trian_loader)

# 验证
def validate(model,val_loader,loss_fn,device):
    #开启验证模式
    model.eval()
    total_loss=0
    with torch.no_grad():
        for noise_img,image in val_loader:
            input=noise_img.to(device)
            target=image.to(device)

            output=model(input)
            loss = loss_fn(output, target)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def evaluate(model, test_loader, loss_fn, device):
    # 验证
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for noise_img, image in test_loader:
            input = noise_img.to(device)
            target = image.to(device)

            output = model(input)
            loss = loss_fn(output, target)
            total_loss += loss.item()
    return total_loss / len(test_loader)