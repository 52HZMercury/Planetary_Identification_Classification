import os

import cv2

import pandas as pd
import numpy as np

import torch

import matplotlib.pyplot as plt
from PIL._imaging import display

import torch
print(torch.__version__)  #注意是双下划线

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

from torchvision import models

# 载入预训练图像分类模型
model = models.resnet18(pretrained=True)
# model = models.resnet152(pretrained=True)

model = model.eval()
model = model.to(device)

from torchvision import transforms
# 图像预处理
# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

# img_path = 'test_img/banana1.jpg'
# img_path = 'test_img/husky1.jpeg'
img_path = 'test_img/banana1.jpg'

# img_path = 'test_img/cat_dog.jpg'
# 用 pillow 载入
from PIL import Image
img_pil = Image.open(img_path)

np.array(img_pil).shape

input_img = test_transform(img_pil) # 预处理
input_img.shape

input_img = input_img.unsqueeze(0).to(device)
input_img.shape

# 执行前向预测，得到所有类别的 logit 预测分数
pred_logits = model(input_img)
pred_logits.shape

# pred_logits
import torch.nn.functional as F
pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算
pred_softmax.shape

plt.figure(figsize=(8,4))

x = range(1000)
y = pred_softmax.cpu().detach().numpy()[0]

ax = plt.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
plt.ylim([0, 1.0]) # y轴取值范围
# plt.bar_label(ax, fmt='%.2f', fontsize=15) # 置信度数值

plt.xlabel('Class', fontsize=20)
plt.ylabel('Confidence', fontsize=20)
plt.tick_params(labelsize=16) # 坐标文字大小
plt.title(img_path, fontsize=25)

plt.show()

n = 10
top_n = torch.topk(pred_softmax, n)
#top_n

# 解析出类别
pred_ids = top_n[1].cpu().detach().numpy().squeeze()
#pred_ids

# 解析出置信度
confs = top_n[0].cpu().detach().numpy().squeeze()
#confs

df = pd.read_csv('imagenet_class_index.csv')

idx_to_labels = {}
for idx, row in df.iterrows():
    idx_to_labels[row['ID']] = [row['wordnet'], row['class']]

# 用 opencv 载入原图
img_bgr = cv2.imread(img_path)
for i in range(n):
    class_name = idx_to_labels[pred_ids[i]][1]  # 获取类别名称
    confidence = confs[i] * 100  # 获取置信度
    text = '{:<15} {:>.4f}'.format(class_name, confidence)
    print(text)

    # !图片，添加的文字，左上角坐标，字体，字号，bgr颜色，线宽
    img_bgr = cv2.putText(img_bgr, text, (25, 50 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)

# 保存图像
cv2.imwrite('output/img_pred.jpg', img_bgr)

# 载入预测结果图像
img_pred = Image.open('output/img_pred.jpg')
#img_pred

fig = plt.figure(figsize=(18,6))

# 绘制左图-预测图
ax1 = plt.subplot(1,2,1)
ax1.imshow(img_pred)
ax1.axis('off')

# 绘制右图-柱状图
ax2 = plt.subplot(1,2,2)
x = df['ID']
y = pred_softmax.cpu().detach().numpy()[0]
ax2.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)

plt.ylim([0, 1.0]) # y轴取值范围
plt.title('{} Classification'.format(img_path), fontsize=30)
plt.xlabel('Class', fontsize=20)
plt.ylabel('Confidence', fontsize=20)
ax2.tick_params(labelsize=16) # 坐标文字大小

plt.tight_layout()
fig.savefig('output/预测图+柱状图.jpg')

pred_df = pd.DataFrame() # 预测结果表格
for i in range(n):
    class_name = idx_to_labels[pred_ids[i]][1] # 获取类别名称
    label_idx = int(pred_ids[i]) # 获取类别号
    wordnet = idx_to_labels[pred_ids[i]][0] # 获取 WordNet
    confidence = confs[i] * 100 # 获取置信度
    pred_df = pred_df._append({'Class':class_name, 'Class_ID':label_idx, 'Confidence(%)':confidence, 'WordNet':wordnet}, ignore_index=True) # 预测结果表格添加一行
#display(pred_df) # 展示预测结果表格

pred_df.to_csv('output/预测结果表格.csv', index=False)