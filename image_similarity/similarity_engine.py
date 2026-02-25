import torch

#训练一个轮次
def train_one_epoch(encoder,decoder,trian_loader,loss_fn,optimizer,device):
    #开启训练模式
    encoder.train()
    decoder.train()
    #定义损失函数
    total_loss=0
    total_num=0
    #遍历这一批训练数据
    for input,target in trian_loader:
        input=input.to(device)
        target=target.to(device)

        embedding=encoder(input)
        output=decoder(embedding)

        loss=loss_fn(output,target)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        total_loss+=loss.item()*input.shape[0]
        total_num+=input.shape[0]

    return total_loss/total_num

# 验证
def validate(encoder,decoder,val_loader,loss_fn,device):
    #开启验证模式
    encoder.eval()
    decoder.eval()
    total_loss=0
    total_num=0
    with torch.no_grad():
        for input,target in val_loader:
            input=input.to(device)
            target=target.to(device)

            embedding = encoder(input)
            output = decoder(embedding)
            loss = loss_fn(output, target)
            total_loss += loss.item()*input.shape[0]
            total_num+=input.shape[0]
    return total_loss / total_num


def evaluate(encoder,decoder,test_loader,loss_fn,device):
    #开启验证模式
    encoder.eval()
    decoder.eval()
    total_loss=0
    total_num=0
    with torch.no_grad():
        for input,target in test_loader:
            input=input.to(device)
            target=target.to(device)

            embedding = encoder(input)
            output = decoder(embedding)
            loss = loss_fn(output, target)
            total_loss += loss.item()*input.shape[0]
            total_num+=input.shape[0]
    return total_loss / total_num