# utils.py 文件

import numpy as np  # 导入 numpy 库并简写为 np
import torch  # 导入 PyTorch 库
import os  # 导入 os 库，用于操作系统的相关功能
import random  # 导入 random 库，用于生成随机数
import re

"""
    为什么做这个:
        可重复性: 在科学研究和模型训练中，可重复性非常重要。通过固定随机种子，可以确保每次运行代码时得到相同的结果，便于调试和验证。
        调试: 当代码中出现错误时，固定的随机种子可以帮助定位问题，因为每次运行的条件都是相同的。
        比较不同模型或参数: 在比较不同模型或参数的效果时，固定的随机种子可以排除随机性带来的干扰，使得比较更加公平和准确。
"""

# 实现一个统一设置随机数种子的函数，消除随机性
def seed_everything(seed):
    # 定义一个函数 seed_everything，参数为 seed（随机种子）
    random.seed(seed)
    # 为 Python 的内置 random 模块设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置 PYTHONHASHSEED 环境变量，确保 Python 的哈希种子一致
    np.random.seed(seed)
    # 为 numpy 的随机数生成器设置随机种子
    # pytorch设置
    torch.manual_seed(seed)
    # 为 PyTorch 的 CPU 随机数生成器设置随机种子
    torch.cuda.manual_seed(seed)
    # 为 PyTorch 的 CUDA 随机数生成器设置随机种子
    torch.backends.cudnn.deterministic = True
    # 设置 PyTorch 的 cudnn 模式为确定性模式，确保每次运行结果一致
    torch.backends.cudnn.benchmark = False
    # 关闭 cudnn 的自动优化功能，以保证每次运行结果一致

def sorted_alphanum(lists):
    convert=lambda str:int(str) if str.isdigit() else str.lower()
    alphanum_key=lambda file_name:[convert(x) for x in re.split(r'([0-9]+)',file_name)]
    return sorted(lists,key=alphanum_key)


if __name__ == '__main__':
    print(sorted_alphanum(['1.jpg', '10.jpg', '2.jpg']))
