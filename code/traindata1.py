import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from multi_iou import multi_iou_acc

from model import U2NET
from model import U2NETP
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()))

	return loss0, loss

def get_args():
    parser = argparse.ArgumentParser(description='Train the U2net on images and target masks')
    parser.add_argument('-e', '--epochs',type=int, default=1000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batchsize',type=int, nargs='?', default=12,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate',type=float,default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument("--img_dir", help="data directory", default="./data/dataset1/trainuint8/", type=str)
    parser.add_argument("--mask_dir", help="data directory", default="./data/dataset1/train_GT/mask_bin/", type=str)
    parser.add_argument("--modelname", help="model name", default="u2netdata1", type=str)
    parser.add_argument("--ext", help="ext", default=".png", type=str)
    parser.add_argument("--savefre", help="save frequency", default=1000, type=int)
    parser.add_argument('--scale', type=int, default=320, help='resize shape')
    parser.add_argument('--crop', type=int, default=288, help='crop shape')
    return parser.parse_args()

# 归一化
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

'''------------  train part  ------------ '''
# 参数
args = get_args()
model_name = args.modelname
img_dir = args.img_dir
mask_dir = args.mask_dir
ext = args.ext
epoch_num = args.epochs
batch_size_train = args.batchsize
save_frq = args.savefre
model_dir = './saved_models/' + model_name +'/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

''' 数据列表 '''
img_list = glob.glob(img_dir + '*' + ext)
mask_list = []
for img_path in img_list:
    img_name = img_path.split("/")[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    mask_list.append(mask_dir + imidx + ext)

''' 读取数据集 '''
salobj_dataset = SalObjDataset(
    img_name_list=img_list,
    lbl_name_list=mask_list,
    transform=transforms.Compose([
        RescaleT(args.scale),
        RandomCrop(args.crop),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

''' 加在网络  '''
net = U2NET(3, 1)
net.cuda()
# 是否多卡训练
net=nn.DataParallel(net)
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
bce_loss = nn.BCELoss(size_average=True)

print("---start training---")
train_num = len(img_list)
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
ious_ = []
dice_ = []
losse_ = []
iouss = 0
dicess = 0
for epoch in range(0, epoch_num):
    net.train()
    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1
        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
        optimizer.zero_grad()

        
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
        # 计算iou 与 dice
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        
        iou = 0.0
        dice = 0.0
        for k in range(labels_v.shape[1]):
            temp_label = labels_v[k]
            temp_label = torch.squeeze(temp_label)
            cal_pred = pred[k].cpu().data.numpy()
            cal_gt = temp_label.cpu().data.numpy()
            cal_pred = (cal_pred > 0.5).astype(int)
            smooth = 1.0
            intersection = np.sum(cal_pred*cal_gt)
            merge = ((cal_pred+cal_gt)>0).astype(int)
            merge = np.sum(merge)
            dice += ((2. * intersection + smooth) / (np.sum(cal_pred) + np.sum(cal_gt) + smooth))
            iou += intersection/merge
        dice = dice / labels_v.shape[1]
        iou = iou / labels_v.shape[1]
        iouss += iou
        dicess += dice
        # 记录中间结果参数
        if ite_num % 20 == 0:
            ious_.append(iouss/ite_num4val)
            dice_.append(dicess/ite_num4val)
            losse_.append(loss2.item())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_tar_loss += loss2.item()
        # 删除暂时的变量
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f , iou: %3f , dice: %3f" % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val,iouss/ite_num4val,dicess/ite_num4val))
        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f.pth" % (ite_num, running_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            iouss = 0
            
            dicess = 0
            net.train()  # resume train
            ite_num4val = 0
ious_ = np.array(ious_)
dice_ = np.array(dice_)
losse_ = np.array(losse_)
np.savetxt('ioudata1.txt',ious_,fmt="%3f", delimiter="\t")
np.savetxt('dicedata1.txt',dice_,fmt="%3f", delimiter="\t")
np.savetxt('lossdata1.txt',losse_,fmt="%3f", delimiter="\t")

