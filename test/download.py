import torch
import torchvision.models as models
import os

# 导入一个可能的下载辅助函数（如果使用torch.hub）
# from torch.hub import download_url_to_file

# --- 方案一：直接指定可用的镜像源URL下载权重文件 ---
# 这是一个常用的GitHub加速代理，可以尝试用它来下载
# 注意：这个方法需要你手动处理文件存放路径，或者修改torch.hub的下载逻辑
# 更简单的做法是让torchvision从镜像源加载，但torchvision直接加载不支持换源
# 所以推荐手动下载后加载

# 1. 定义权重的保存路径 (可以放在torch默认的缓存目录，也可以自定义)
cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints/')
filename = 'vgg16-397923af.pth'  # 这是你之前下载失败的文件名
filepath = os.path.join(cache_dir, filename)

# 2. 如果文件不存在，则手动下载
if not os.path.exists(filepath):
    print("权重文件不存在，开始下载...")
    # 使用一个代理镜像URL（比如用 ghproxy.com 代理 GitHub 原始地址）
    # 原始的官方URL是：https://download.pytorch.org/models/vgg16-397923af.pth
    mirror_url = "https://ghproxy.com/https://download.pytorch.org/models/vgg16-397923af.pth"

    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)

    # 使用 torch.hub 的下载工具 (如果可用)
    try:
        # PyTorch 1.10+ 推荐使用 torch.hub.download_url_to_file
        torch.hub.download_url_to_file(mirror_url, filepath)
        print(f"文件已下载到: {filepath}")
    except Exception as e:
        print(f"通过 torch.hub 下载失败: {e}")
        # 备选：使用 requests 库下载（如果安装了的话）
        try:
            import requests

            print("尝试使用 requests 下载...")
            response = requests.get(mirror_url, stream=True)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"文件已下载到: {filepath}")
        except ImportError:
            print("未安装 requests 库，无法使用备选方案。")
        except Exception as e_req:
            print(f"使用 requests 下载也失败了: {e_req}")

# 3. 加载本地权重文件
if os.path.exists(filepath):
    print("从本地加载权重文件...")
    # 先初始化模型结构（不加载预训练权重）
    vgg = models.vgg16(weights=None)
    # 然后加载下载好的本地权重
    state_dict = torch.load(filepath, map_location='cpu')
    vgg.load_state_dict(state_dict)
    print("VGG16 模型加载成功！")
else:
    print("错误：无法找到或下载权重文件。请检查网络或手动下载。")
    # 可以在这里抛出异常或使用其他备选方案
    vgg = models.vgg16(weights=None)  # 至少得到一个未训练的模型结构

# --- 原来的代码（会直接触发下载）可以注释掉 ---
# vgg = models.VGG16(weights=VGGL16_Weights.IMAGENET1K_V1)

# --- 后续你的 loss_net 代码可以保持不变 ---
# from torch import nn
# loss_net = nn.Sequential( * list(vgg.features.children())[:10])