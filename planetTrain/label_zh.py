import json

import numpy as np

# 加载npy文件
loaded_arr = np.load('idx_to_labels.npy',allow_pickle=True)
# 打印加载的数组
print(loaded_arr)

# 创建一个NumPy数组
# 给定的字典
dict_data = {0: '彗星', 1: '地球', 2: '木星', 3: '火星', 4: '水星',
              5: '月球', 6: '海王星', 7: '冥王星', 8: '土星', 9: '太阳',
              10: '天王星', 11: '金星'}


dtype = [('key', int), ('value', 'U10')]  # 'U10' 表示最大长度为 10 的 Unicode 字符串

# 将字典项转换为结构化数组的元组列表
structured_data = {k:v for k, v in dict_data.items()}

# 保存结构化数组为 .npy 文件
np.save('idx_to_labels_zh.npy', structured_data)

# 加载npy文件
loaded_arr2 = np.load('idx_to_labels_zh.npy',allow_pickle=True)
# 打印加载的数组
print(loaded_arr2)



# 路径
npy_path = "idx_to_labels_zh.npy"
json_path = "idx_to_labels_zh.json"

# 读取
file = np.load(npy_path, allow_pickle=True).item()
# print("转换前：", file.dtype)        # 查看数据类型

# 存为json
with open(json_path, "w", encoding="utf-8") as new_file:
    new_file.write(json.dumps(file, indent=2, ensure_ascii=False))