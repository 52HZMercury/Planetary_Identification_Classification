import matplotlib
matplotlib.rc("font",family='SimHei') # 中文字体
from PIL import ImageFont, ImageDraw
# 导入中文字体，指定字号
font = ImageFont.truetype('SimHei.ttf', 32)
from PIL import Image, ImageFont, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 载入预训练图像分类模型
model = models.resnet18(pretrained=True)
# model = models.resnet152(pretrained=True)

model = model.eval()
model = model.to(device)

df = pd.read_csv('imagenet_class_index.csv')
idx_to_labels = {}
for idx, row in df.iterrows():
    idx_to_labels[row['ID']] = [row['wordnet'], row['Chinese']]

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
# img_path = 'test_img/watermelon1.jpg'
img_path = 'test_img/cat_dog.jpg'

img_pil = Image.open(img_path) # 用 pillow 载入

input_img = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
pred_logits = model(input_img) # 执行前向预测，得到所有类别的 logit 预测分数
pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算

plt.figure(figsize=(8,4))

x = range(1000)
y = pred_softmax.cpu().detach().numpy()[0]

ax = plt.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
plt.ylim([0, 1.0]) # y轴取值范围
# plt.bar_label(ax, fmt='%.2f', fontsize=15) # 置信度数值

plt.title(img_path, fontsize=30)
plt.xlabel('类别', fontsize=20)
plt.ylabel('置信度', fontsize=20)
plt.tick_params(labelsize=16) # 坐标文字大小

plt.show()

n = 10
top_n = torch.topk(pred_softmax, n) # 取置信度最大的 n 个结果
pred_ids = top_n[1].cpu().detach().numpy().squeeze() # 解析出类别
confs = top_n[0].cpu().detach().numpy().squeeze() # 解析出置信度

draw = ImageDraw.Draw(img_pil)
for i in range(n):
    class_name = idx_to_labels[pred_ids[i]][1]  # 获取类别名称
    confidence = confs[i] * 100  # 获取置信度
    text = '{:<15} {:>.4f}'.format(class_name, confidence)
    print(text)

    # 文字坐标，中文字符串，字体，rgba颜色
    draw.text((50, 100 + 50 * i), text, font=font, fill=(255, 0, 0, 1))

# 保存图像
img_pil.save('output/img_pred.jpg')

fig = plt.figure(figsize=(18,6))

# 绘制左图-预测图
ax1 = plt.subplot(1,2,1)
ax1.imshow(img_pil)
ax1.axis('off')

# 绘制右图-柱状图
ax2 = plt.subplot(1,2,2)
x = df['ID']
y = pred_softmax.cpu().detach().numpy()[0]
ax2.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
plt.ylim([0, 1.0]) # y轴取值范围
plt.xlabel('类别', fontsize=20)
plt.ylabel('置信度', fontsize=20)
ax2.tick_params(labelsize=16) # 坐标文字大小

plt.title('{} 图像分类预测结果'.format(img_path), fontsize=30)

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
pred_df

pred_df.to_csv('output/预测结果表格.csv', index=False)