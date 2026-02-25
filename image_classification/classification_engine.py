import torch


# 训练一个轮次
def train_one_epoch(model, trian_loader, loss_fn, optimizer, device):
    # 开启训练模式
    model.train()
    # 定义损失函数
    total_loss = 0
    total_num = 0
    # 遍历这一批训练数据
    for input, target in trian_loader:
        input = input.to(device)
        target = target.to(device)

        output = model(input)

        loss = loss_fn(output, target)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        total_loss += loss.item() * input.shape[0]
        total_num += input.shape[0]

    return total_loss / total_num


# 验证
def validate(model, val_loader, loss_fn, device):
    # 开启验证模式
    model.eval()
    total_loss = 0
    total_num = 0
    with torch.no_grad():
        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = loss_fn(output, target)
            total_loss += loss.item() * input.shape[0]
            total_num += input.shape[0]
    return total_loss / total_num


def evaluate(model, test_loader,  device):
    # 测试
    model.eval()
    total_num = 0
    test_acc_num = 0
    with torch.no_grad():
        for input, target in test_loader:
            input = input.to(device)
            target = target.to(device)

            #前向传播
            output = model(input)

            # 得到预测分类
            pred = output.argmax(dim=-1)

            # 累加准确个数
            test_acc_num += pred.eq(target).sum().item()

            #累加测试总数
            total_num += input.shape[0]

    # 返回准确率
    return test_acc_num / total_num
