import matplotlib
matplotlib.rc("font",family='SimHei') # 中文字体

from PIL import ImageFont, ImageDraw
# 导入中文字体，指定字号
font = ImageFont.truetype('SimHei.ttf', 32)

import os
import time
import shutil
import cv2
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
import gc

import torch
import torch.nn.functional as F
from torchvision import models

import mmcv

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# 后端绘图，不显示，只保存
import matplotlib
matplotlib.use('Agg')

idx_to_labels = np.load('idx_to_labels_zh.npy', allow_pickle=True).item()

model = torch.load('checkpoint/best-0.846.pth')
model = model.eval().to(device)

from torchvision import transforms

# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])


def pred_single_frame(img, n=5):
    '''
    输入摄像头画面bgr-array，输出前n个图像分类预测结果的图像bgr-array
    '''
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
    img_pil = Image.fromarray(img_rgb)  # array 转 pil
    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

    top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别
    confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析出置信度

    # 在图像上写字
    draw = ImageDraw.Draw(img_pil)
    # 在图像上写字
    for i in range(len(confs)):
        pred_class = idx_to_labels[pred_ids[i]]
        text = '{:<15} {:>.3f}'.format(pred_class, confs[i])
        # 文字坐标，中文字符串，字体，rgba颜色
        draw.text((50, 100 + 50 * i), text, font=font, fill=(255, 0, 0, 1))

    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # RGB转BGR

    return img_bgr, pred_softmax


input_video = 'test_img/video/月.mp4'
# input_video = 'test_img/video/最真实的行星及其卫星.mp4'
output_path = 'output/video/output_pred4.mp4'


'''方案一'''
# 创建临时文件夹，存放每帧结果
temp_out_dir = time.strftime('%Y%m%d%H%M%S')
os.mkdir(temp_out_dir)
print('创建临时文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))

# 读入待预测视频
imgs = mmcv.VideoReader(input_video)

# prog_bar = mmcv.ProgressBar(len(imgs))

# 对视频逐帧处理
for frame_id, img in enumerate(imgs):
    ## 处理单帧画面
    img, pred_softmax = pred_single_frame(img, n=5)

    # 将处理后的该帧画面图像文件，保存至 /tmp 目录下
    cv2.imwrite(f'{temp_out_dir}/{frame_id:06d}.jpg', img)

    # prog_bar.update()  # 更新进度条

# 把每一帧串成视频文件
mmcv.frames2video(temp_out_dir, output_path, fps=imgs.fps, fourcc='mp4v')

shutil.rmtree(temp_out_dir)  # 删除存放每帧画面的临时文件夹
print('删除临时文件夹', temp_out_dir)
print('视频已生成', output_path)


'''方案二'''
#
# def pred_single_frame_bar(img):
#     '''
#     输入pred_single_frame函数输出的bgr-array，加柱状图，保存
#     '''
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
#     fig = plt.figure(figsize=(18, 6))
#     # 绘制左图-视频图
#     ax1 = plt.subplot(1, 2, 1)
#     ax1.imshow(img)
#     ax1.axis('off')
#     # 绘制右图-柱状图
#     ax2 = plt.subplot(1, 2, 2)
#     x = idx_to_labels.values()
#     y = pred_softmax.cpu().detach().numpy()[0] * 100
#     ax2.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
#     plt.xlabel('类别', fontsize=20)
#     plt.ylabel('置信度', fontsize=20)
#     ax2.tick_params(labelsize=16)  # 坐标文字大小
#     plt.ylim([0, 100])  # y轴取值范围
#     plt.xlabel('类别', fontsize=25)
#     plt.ylabel('置信度', fontsize=25)
#     plt.title('图像分类预测结果', fontsize=30)
#     plt.xticks(rotation=90)  # 横轴文字旋转
#
#     plt.tight_layout()
#     fig.savefig(f'{temp_out_dir}/{frame_id:06d}.jpg')
#     # 释放内存
#     fig.clf()
#     plt.close()
#     gc.collect()
#
#
# # 创建临时文件夹，存放每帧结果
# temp_out_dir = time.strftime('%Y%m%d%H%M%S')
# os.mkdir(temp_out_dir)
# print('创建临时文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))
# # 读入待预测视频
# imgs = mmcv.VideoReader(input_video)
#
# # prog_bar = mmcv.utils.progressbar.ProgressBar(len(imgs))
#
# # 对视频逐帧处理
# for frame_id, img in enumerate(imgs):
#     ## 处理单帧画面
#     img, pred_softmax = pred_single_frame(img, n=5)
#     img = pred_single_frame_bar(img)
#
#     # prog_bar.update()  # 更新进度条
#
# # 把每一帧串成视频文件
# mmcv.frames2video(temp_out_dir, output_path, fps=imgs.fps, fourcc='mp4v')
#
# shutil.rmtree(temp_out_dir)  # 删除存放每帧画面的临时文件夹
# print('删除临时文件夹', temp_out_dir)
# print('视频已生成', output_path)