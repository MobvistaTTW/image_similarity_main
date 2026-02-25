import math
import os

import chromadb
from chromadb import EmbeddingFunction
from chromadb.api.types import Images, Embeddings
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from common.utils import sorted_alphanum
from image_similarity.similarity_config import *
from image_similarity.similarity_model import *


# 自定义嵌入函数
class ImageEmbeddingFunction(EmbeddingFunction[Images]):
    # 初始化,传入自己的嵌入模型(编码模型,encoder)
    def __init__(self, model) -> None:
        self.model = model.to('cpu')
        return

    # 调用方法
    def __call__(self, input: Images) -> Embeddings:
        # 将输入图像转换为Tensor
        input_tensor = torch.tensor(np.array(input), dtype=torch.float32)
        # 前向传播,得到模型输出
        with torch.no_grad():
            output = self.model(input_tensor)
        # 将处处转换为ndarray,返回
        return output.numpy()


# 1.加载所有图片(字典{id,image})
def get_id2images(image_dir, transform):
    id2images = {}
    # 读取目录下所有图片名
    images_names = sorted_alphanum(os.listdir(image_dir))
    # 遍历文件名,打开图片转换
    with tqdm(total=len(images_names)) as pbar:
        for i, image_names in enumerate(images_names):
            # 1.构建图片的网站访问路径
            image_path = os.path.join(image_dir, image_names)
            # 2.打开图片
            image = Image.open(image_path).convert("RGB")
            # 3.图像转换,(调整大小,转换Tensor)
            img_tensor = transform(image)
            # 4.转换为ndarray
            id2images[str(i)] = img_tensor.numpy()

            #更新进度条
            pbar.update(1)

    return id2images


# 2.获取chroma的集合
def get_embedding_collection(encoder):
    # 2.1 创建客户端
    path = os.path.join('..', PACKAGE_NAME, CHROMA_BACKEND_PATH)
    client = chromadb.PersistentClient(path)
    # 2.2 创建集合
    collection = client.get_or_create_collection(
        name=IMAGE_COLLECTION_NAME,
        embedding_function=ImageEmbeddingFunction(encoder)
    )
    return collection


# 生成所以图像的嵌入向量(预处理,初始化)
def create_embeddings(encoder):
    # 1.加载所有图片(字典{})
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor()
    ])
    print("正在加载所有数据...:")
    id2images = get_id2images(IMG_PATH, transform)
    print("图片加载完毕")

    ids = list(id2images.keys())
    imgs = list(id2images.values())

    # 2.获取chroma的集合
    collection = get_embedding_collection(encoder)
    # 3.执行写入chroma的操作
    print("正在写入chroma数据:")
    batchs = math.ceil(len(ids) / CHROMA_INSERT_BATCH_SIZE)
    for i in tqdm(range(batchs),desc='写入数据'):
        start = i * CHROMA_INSERT_BATCH_SIZE
        end = min((i+1) * CHROMA_INSERT_BATCH_SIZE, len(ids))
        collection.upsert(
            ids=ids[start:end],
            images=imgs[start:end]
        )
    print("写入数据完成")


# 相似图片搜索
def search_similar_image_ids(collection, image, cnt=5):
    result = collection.query(
        query_images=image.numpy(),
        n_results=cnt,
        # where={'label':{'$eq':{3}}}
    )
    similar_ids=[int(id) for id in result['ids'][0]]
    return similar_ids

if __name__ == '__main__':
    #1.加载编码器模型
    encoder=ConvEncoder()
    state_dict=torch.load(ENCODER_MODEL_NAME)
    encoder.load_state_dict(state_dict)
    print("模型加载完成")
    #2.写入数据,生成嵌入向量
    create_embeddings(encoder)