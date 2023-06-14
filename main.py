from Networks import SRResNet
from Preprocess import create_list
from Datasets import DIV_2K
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms, utils
from torch.autograd import Variable
import random
from torch.utils.data import Dataset,DataLoader
import os
import cv2
import glob
import numpy as np

path = 'D:\\Pytorch\\Data\\'
lr = 0.00001

create_list(path)

def load(path,shape):
    img= cv2.imread(path)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= cv2.resize(img, shape)
    return img


def get_data(path):
    X=[]
    Y=[]
    for folder in glob.glob(path+ str('/*')):
        for img_path in glob.glob(folder+ str('/*')):      
            if folder == os.path.join(path, 'HR'):
                X.append(load(img_path, (384, 384)))
            elif folder == os.path.join(path, 'LR'):
                Y.append(load(img_path, (96,96)))

    X= np.array(X).astype(np.float32)
    Y= np.array(Y).astype(np.float32)
    
    return X/255.0, Y/255.0



def train(training_data, optimizer, model, Loss_type, epoch,testing_data):
    for param_group in optimizer.param_groups:
        print('param_group LR = ',param_group['lr'])
        param_group['lr']=lr
    print("Epoch = {}, lr = {}".format(epoch,optimizer.param_groups[0]["lr"]))
    model.train()

    for iter, batch in enumerate(training_data, 0):
        input,target = Variable(batch), Variable(testing_data[iter], requires_grad=False)
        # print("input:",input.shape)
        # print("target:",target.shape)
        # input = input.permute(2,0,1 )
        # target = target.permute(2,0,1)

        input = input.cuda()
        target = target.cuda()

        output = model(input)
        # print("output:",output.shape)
        loss = Loss_type(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter%10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.5}".format(epoch, iter, len(training_data), loss))

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__=='__main__':
    if not torch.cuda.is_available():
        raise Exception("No GPU")
    torch.cuda.manual_seed(random.randint(1,10000))
    cudnn.benchmark = True
    # HR_train, LR_train= get_data(path)
    # train_set = DIV_2K(path)
    # training_dataset = DataLoader(dataset=train_set,batch_size=5,shuffle=True)

    train_data = DIV_2K(txt='./training.txt',transform=transforms.ToTensor())
    test_data = DIV_2K(txt='./testing.txt',transform=transforms.ToTensor())
    training_dataload = DataLoader(dataset=train_data,batch_size=15,shuffle=True)
    # testing_dataload = DataLoader(dataset=test_data,batch_size=5,shuffle=True)

    print("Building model")
    SR = SRResNet()
    Loss_type = nn.MSELoss(size_average=False)

    print("GPU setting")
    SR = SR.cuda()
    Loss_type = Loss_type.cuda()
    optimizer = optim.Adam(SR.parameters(),lr=lr)
    Epoch = 100
    for epoch in range(Epoch):
        train(train_data, optimizer, SR, Loss_type, epoch,test_data)
        if epoch%10 == 0:
            save_checkpoint(SR, epoch)
    # train_set = 