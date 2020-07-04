from __future__ import absolute_import

import cv2
import imageio
import numpy as np
import os
import os.path as osp
    
def visual(img, gt):
    img = cv2.imread(img, -1)
    gt = cv2.imread(gt, -1)
    label = np.unique(gt)
    height, width = img.shape[:2]
    visual_img = np.zeros((height, width, 3))
    for lab in label:
         if lab == 0:
             continue
         color = np.random.randint(low=10, high=255, size=3)
         visual_img[gt==lab, :] = color
    return img.astype(np.uint8), gt.astype(np.uint8)

def RotateAntiClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 0)
    return new_img
def RotateClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 1)
    return new_img
if __name__ == "__main__":
    import skimage.morphology as sm
    ''' 顺序: 不变 + 顺时针90 '''
    image_path = './data/dataset2/train_part'
    gt_path = './data/dataset2/train_GT/SEG'
    images = sorted([osp.join(image_path, img) for img in os.listdir(image_path) if img.find('.tif') != -1])
    gts = sorted([osp.join(gt_path, gt) for gt in os.listdir(gt_path) if gt.find('.tif') != -1])
    visual_img_path = './data/dataset2/trainuint8'
    visual_gt_path = './data/dataset2/train_GT/mask_uint8'
    visual_bi_path = './data/dataset2/train_GT/mask_bin_ero'
    if not osp.exists(visual_img_path):
        os.mkdir(visual_img_path)
    if not osp.exists(visual_gt_path):
        os.mkdir(visual_gt_path)
    if not osp.exists(visual_bi_path):
        os.mkdir(visual_bi_path)
    # 一种变换 数据增广
    repeat = 2
    for i in range(repeat):
        for idx, (image,gt) in enumerate(zip(images,gts)):
            print(idx)
            img, visual_img = visual(image, gt)
            mask = visual_img

            label = np.unique(mask)
            height, width = img.shape[:2]
            er_img = np.zeros((height, width))
            for lab in label:
                if lab == 0:
                    continue
                temp_img = np.zeros((height, width))
                temp_img[mask == lab] = 255
                temp_img = sm.erosion(temp_img,sm.square(20))
                er_img = er_img + temp_img
            _,mask = cv2.threshold(er_img, 0, 255, 0)
        
            if i == 1:
                img = RotateClockWise90(img)
                visual_img = RotateClockWise90(visual_img)
                mask = RotateClockWise90(mask) 
            cv2.imwrite(osp.join(visual_img_path, '{:0>3d}.png'.format(idx+len(images)*i)), img.astype(np.uint8))
            cv2.imwrite(osp.join(visual_gt_path, '{:0>3d}.png'.format(idx+len(images)*i)), visual_img.astype(np.uint8))
            cv2.imwrite(osp.join(visual_bi_path, '{:0>3d}.png'.format(idx+len(images)*i)), mask.astype(np.uint8))
    # 测试集
    image_path = './data/dataset2/test'
    visual_test_path = './data/dataset2/testuint8'
    if not osp.exists(visual_test_path):
        os.mkdir(visual_test_path)
    images = sorted([osp.join(image_path, img) for img in os.listdir(image_path) if img.find('.tif') != -1])
    for idx, (image, gt) in enumerate(zip(images, images)):
        print(idx)
        img, visual_img = visual(image, image)
        cv2.imwrite(osp.join(visual_test_path, '{:0>3d}.png'.format(idx)), img.astype(np.uint8))
