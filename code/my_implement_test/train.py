import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from dataset import Celldataset
import numpy as np
import glob

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET
from model import U2NETP
from dataloader import loaddata
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
bce_loss = nn.BCELoss(size_average=True)
# 损失函数
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


model_name = 'u2netmyself'
# 读取数据
img_path = '/home/gaozijing/gzj/PRseg/data/dataset1/trainuint8/'
mask_path = '/home/gaozijing/gzj/PRseg/data/dataset1/train_GT/mask_bin/'
model_dir = './saved_models/' + model_name +'/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# 超参数
epochs = 10000
batchsize = 12
batchsize_val = 1

# label_ext = '.png'
tra_image_list = glob.glob(img_path + '*' + '.png')
tra_mask_list = []
for img_path in tra_image_list:
    img_name = img_path.split("/")[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    tra_mask_list.append(mask_path + imidx + '.png') 
dataset = Celldataset(tra_image_list,tra_mask_list,istraining='train',transform=None)
mydataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=1)

# data_dir = '/home/gaozijing/gzj/PRseg/data/dataset1/'
# tra_image_dir = '/trainuint8/'
# tra_label_dir = '/train_GT/mask_bin/'

# epochs = 10000

# image_ext = '.png'
# label_ext = '.png'

# model_dir = './saved_models/' + model_name +'/'
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
# epoch_num = 10000
# batch_size_train = 8
# batch_size_val = 1
# train_num = 0
# val_num = 0

# tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

# tra_lbl_name_list = []
# for img_path in tra_img_name_list:
# 	img_name = img_path.split("/")[-1]

# 	aaa = img_name.split(".")
# 	bbb = aaa[0:-1]
# 	imidx = bbb[0]
# 	for i in range(1,len(bbb)):
# 		imidx = imidx + "." + bbb[i]

# 	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

# print("---")
# print("train images: ", len(tra_img_name_list))
# print("train labels: ", len(tra_lbl_name_list))
# print("---")

# train_num = len(tra_img_name_list)
# salobj_dataset = SalObjDataset(
#     img_name_list=tra_img_name_list,
#     lbl_name_list=tra_lbl_name_list,
#     transform=transforms.Compose([
#         Rescale(480),
#         ToTensorLab(flag=0)]))
# mydataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)


net = U2NET(3,1)
if torch.cuda.is_available():
    net.cuda()
# net=nn.DataParallel(net)
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

print("---Start training---")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
temp_count = 0
save_frq = 2

for epoch in range(epochs):
    net.train()
    for i, value in enumerate(mydataloader):
        ite_num = ite_num + 1
        temp_count =temp_count + 1
        inputs, labels = value['image'], value['label']
        inputs,labels = inputs.type(torch.FloatTensor),labels.type(torch.FloatTensor)
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
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_tar_loss += loss2.item()

        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
        avg_loss = running_loss / temp_count
        avg_tar_loss = running_tar_loss /temp_count
        print("[epoch: %3d/%3d, ite: %d] train loss: %3f, tar: %3f ， iou:%3f, dice:%3f" % (epoch + 1, epochs, ite_num, avg_loss, avg_tar_loss,iou,dice))

    if epoch % 20 == 0:
        torch.save(net.state_dict(), model_dir + model_name+"_%d_train_%3f_tar_%3f.pth" % (epoch,avg_loss,avg_tar_loss))
        running_loss = 0.0
        running_tar_loss = 0.0
        net.train()
        temp_count = 0

if __name__ == "__main__":
    main()

