import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
from dataset import Celldataset
import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
# from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from dataloader import loaddata
from model import U2NET 
from model import U2NETP 
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

# def save_output(image_name,pred,d_dir):
#     predict = pred
#     predict = predict.squeeze()
#     predict_np = predict.cpu().data.numpy()

#     im = Image.fromarray(predict_np*255).convert('RGB')
#     img_name = image_name.split("/")[-1]
#     image = io.imread(image_name)
#     imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

#     pb_np = np.array(imo)

#     aaa = img_name.split(".")
#     bbb = aaa[0:-1]
#     imidx = bbb[0]
#     for i in range(1,len(bbb)):
#         imidx = imidx + "." + bbb[i]

#     imo.save(d_dir+imidx+'.png')

def main():
    model_name='u2netse2ero'#u2netp
    model_test_name = 'u2netse2ero_bce_itr_1800_train_1.586861_tar_0.206399'
    prediction_dir = './test_data/' + model_test_name + '_1results/'
    model_dir = './saved_models/'+ model_name + '/' + model_test_name + '.pth'
    img_path = '/home/gaozijing/gzj/PRseg/data/dataset2/testuint8/'
    mask_path = img_path
    
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    label_ext = '.png'
    tra_image_list = glob.glob(img_path + '*' + '.png')
    tra_mask_list = []
    for img_path in tra_image_list:
        img_name = img_path.split("/")[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        tra_mask_list.append(mask_path + imidx + '.png') 
    dataset = Celldataset(tra_image_list,tra_mask_list,'test',transform=None)
    mydataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)


    net = U2NET(3,1)
    net.load_state_dict(torch.load(model_dir))
#     net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_dir).items()})
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    for i_test, data_test in enumerate(mydataloader):
        print(i_test)

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        image_name = tra_image_list[i_test]
        d_dir = prediction_dir
        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        im = Image.fromarray(predict_np*255).convert('RGB')
        img_name = image_name.split("/")[-1]
        image = io.imread(image_name)
        imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
        pb_np = np.array(imo)
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        imo.save(d_dir+imidx+'.png')
        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
