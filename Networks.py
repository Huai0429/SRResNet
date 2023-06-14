import torch 
import torch.nn as nn
import math

class Residual_block(nn.Module):
    def __init__(self):
        super(Residual_block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64,out_channels=64,bias = False,kernel_size = 3,stride = 1,padding = 1)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,bias = False,kernel_size = 3,stride = 1,padding = 1)
    def forward(self,x):
        temp = x
        x = self.conv1(x)
        x = self.relu(self.in1(x))
        x = self.conv2(x)
        x = self.in1(x)
        x = torch.add(x,temp)
        return x
        
class SRResNet(nn.Module):
    def __init__(self):
        super(SRResNet,self).__init__()
        self.residual = self.makelayer(Residual_block,16)
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels=64,kernel_size=9,stride=1,padding=4,bias=False)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(in_channels = 64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        self.IN = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(in_channels = 64,out_channels=64,kernel_size=9,stride=1,padding=4,bias=False)
        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1)
        )
        self.conv_final = nn.Conv2d(in_channels = 64,out_channels=3,kernel_size=9,stride=1,padding=4,bias=False)
        # 參數初始化
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self,input):
        x = self.conv1(input)
        temp1 = self.relu(x)
        temp2 = torch.add(temp1,self.residual(temp1))
        temp3 = torch.add(temp2,self.residual(temp2))
        temp4 = torch.add(temp3,self.residual(temp3))
        temp5 = torch.add(temp4,self.residual(temp4))
        temp6 = torch.add(temp5,self.residual(temp5))
        x = self.IN(self.conv2(temp6))
        x = torch.add(x,temp1)
        x = self.upscale2x(x)
        x = self.upscale2x(x)
        x = self.conv_final(x)
        return x
    def makelayer(self,block,num):
        layers = []
        for i in range(num):
            layers.append(block())
        return nn.Sequential(*layers)
        
