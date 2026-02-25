import os


import torch
import matplotlib.pyplot as plt
from PIL import Image

from common.utils import seed_everything
from image_similarity.similarity_config import *
from image_similarity.similarity_data import create_datasets
from image_similarity.similarity_embeddings import get_embedding_collection, search_similar_image_ids
from image_similarity.similarity_model import ConvEncoder

if __name__ == '__main__':
    # seed_everything(SEED)
    # 1.创建数据集
    _, _, test_dataset = create_datasets()
    print("=============测试数据集创建完成===========")
    # 2.创建加载器
    image,_=test_dataset[0]
    print(image.shape)
    print("=============测试数据集加载完成===========")
    # 3.定义模型,加载参数
    model = ConvEncoder()
    state_dict=torch.load(ENCODER_MODEL_NAME)
    model.load_state_dict(state_dict)

    print("========模型加载完成=========")
    # 4.获取chroma集合
    collection=get_embedding_collection(model)
    # print(collection.peek(limit=5))

    #5.测试得到图片id
    similar_image_ids=search_similar_image_ids(collection,image,5)
    print("相似图片id:",similar_image_ids)

    #画图显示
    fig,axes,=plt.subplots(2,5,figsize=(25,4))
    #测试图片
    image=image.permute(1,2,0)
    axes[0,2].imshow(image)
    #返回结果
    for i in range(len(similar_image_ids)):
        #拼接文件名
        image_name=str(similar_image_ids[i])+'.jpg'
        #读取图片
        image_path=os.path.join(IMG_PATH,image_name)
        image=Image.open(image_path).convert('RGB')
        #画出图像
        axes[1,i].imshow(image)

    for ax in axes.flat:
        ax.axis('off')
    plt.show()

