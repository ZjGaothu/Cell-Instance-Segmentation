from __future__ import absolute_import
import skimage.morphology as sm
from skimage import morphology,feature
from scipy import ndimage as ndi
from skimage import morphology,color,data
import cv2
import imageio
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt

# 读取路径
res_path= './test_data/best_modeldata2_results'
image_list = sorted([osp.join(res_path, image) for image in os.listdir(res_path)])
# 保存路径
result_path = './data/dataset2/testREStif'
if not osp.exists(result_path):
        os.mkdir(result_path)
visual_path = './data/dataset2/visualRES'
if not osp.exists(visual_path):
        os.mkdir(visual_path)
for index in range(len(image_list)):
    print(index)
    # 阈值分割
    temp_img = cv2.imread(image_list[index],cv2.IMREAD_GRAYSCALE)
    _,temp_img = cv2.threshold(temp_img,80,255,cv2.THRESH_BINARY)

    # 二值图两次膨胀 弥补孔洞
    temp_img=sm.dilation(temp_img,sm.square(4))
    temp_img=sm.dilation(temp_img,sm.square(6))

    # 连通域提取
    connectivity = 4
    _, label_img, _, _ = cv2.connectedComponentsWithStats(temp_img , connectivity , cv2.CV_32S)

    # 各标签两次膨胀 恢复细胞大小
    label_img=sm.dilation(label_img,sm.square(6))
    label_img=sm.dilation(label_img,sm.square(4))

    #可视化
    height, width = temp_img.shape[:2]
    label = np.unique(label_img)
    visual_img = np.zeros((height, width, 3))
    for lab in label:
        if lab == 0:
            continue
        color = np.random.randint(low=0, high=255, size=3)
        visual_img[label_img==lab, :] = color
    imageio.imwrite(osp.join(visual_path, 'mask{:0>3d}.png'.format(index)),visual_img.astype(np.uint8))
    imageio.imwrite(osp.join(result_path, 'mask{:0>3d}.tif'.format(index)),label_img.astype(np.uint16))