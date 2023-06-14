import torch
from PIL import Image
import numpy as np
import cv2

path = './checkpoint/model_epoch_90.pth'
SRModel = torch.load(path)["model"]
SRModel = SRModel.cuda()

img = cv2.imread('./testset/mandrill.png')
img = cv2.resize(img,(96,96))
cv2.imwrite('./testset/mandrill_LR.png',img)

img1 = np.array(img,dtype=np.float32)
im_input = np.array(img,dtype=np.float32).transpose(2,0,1)
im_input = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
im_input = torch.from_numpy(im_input/255.).float()

im_input = im_input.cuda()
result = SRModel(im_input)

result = result.permute(2,3,1,0)
result = result.cpu()
result = result.detach().numpy()
result = (result*255).astype(np.uint8)

result = np.squeeze(result, 3)
print(result.shape)
cv2.imshow('',result)
cv2.waitKey(0)



