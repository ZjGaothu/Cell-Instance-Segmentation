import os
from skimage import io, transform
import torch
import torchvision
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET 
from model import U2NETP 
# 数据归一化
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)

    return dn
# 参数
def get_args():
    parser = argparse.ArgumentParser(description='Train the U2net on images and target masks')
    parser.add_argument("--img_dir", help="data directory", default="/home/gaozijing/gzj/PRseg/data/dataset2/testuint8/", type=str)
    parser.add_argument("--modelname", help="model name", default="u2netse2ero", type=str)
    parser.add_argument('--scale', type=int, default=320, help='resize shape')
    parser.add_argument("--themodel", help="model name", default="u2netse2ero_bce_itr_1800_train_1.586861_tar_0.206399", type=str)
    return parser.parse_args()

def main():
    args = get_args()
    model_name= args.modelname
    image_dir = args.img_dir
    # 预测的保存路径
    prediction_dir = './test_data/' +args.themodel + '_results/'
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    # 模型的路径
    model_dir = './saved_models/'+ model_name + '/' + args.themodel + '.pth'

    
    img_name_list = glob.glob(image_dir + '*')
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,lbl_name_list = [],transform=transforms.Compose([RescaleT(args.scale),ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)


    net = U2NET(3,1)
    net.load_state_dict(torch.load(model_dir))
#     若多卡训练，则取消下一行的注释
#     net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_dir).items()})
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    for i_test, data_test in enumerate(test_salobj_dataloader):
        print(i_test)
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # 顶层输出
        pred = d1[:,0,:,:]
        # 输出归一化
        pred = normPRED(pred)
        del d1,d2,d3,d4,d5,d6,d7
        image_name = img_name_list[i_test]
        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        # 转换为uint8的彩图格式
        im = Image.fromarray(predict_np*255).convert('RGB')
        img_name = image_name.split("/")[-1]
        # 放缩至原图大小
        imo = im.resize((500,500),resample=Image.BILINEAR)
        pb_np = np.array(imo)
        # 获取文件名
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]
        imo.save(prediction_dir+imidx+'.png')

if __name__ == "__main__":
    main()
