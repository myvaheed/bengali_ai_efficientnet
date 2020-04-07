# %%
from IPython import get_ipython;   
get_ipython().magic('reset -sf')

# %%
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sklearn
import csv
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
# %%
# CHECKPOINT_FNAME = 'checkpoint_val.chk'
CHECKPOINT_FNAME = 'checkpoint_10'
BACKBONE = 'efficientnet-b3'

HEIGHT = 137
WIDTH = 236
SIZE = 128

TEST = ['./kaggle/bengaliai-cv19/test_image_data_0.parquet',
         './kaggle/bengaliai-cv19/test_image_data_1.parquet',
         './kaggle/bengaliai-cv19/test_image_data_2.parquet',
        './kaggle/bengaliai-cv19/test_image_data_3.parquet'
        ]

TEST_LABELS = './kaggle/bengaliai-cv19/test.csv'

# %%
# https://www.kaggle.com/iafoss/image-preprocessing-128x128

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

# %%
class BengaliDatasetTest(Dataset):

    def __init__(self, fname):
        print(fname)
        self.df = pd.read_parquet(fname)
        self.data = 255 - self.df.iloc[:, 1:].values.reshape(-1,HEIGHT, WIDTH).astype(np.uint8)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\n'
        return string

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.iloc[idx,0]
        #normalize each image by its max val
        img = (self.data[idx]*(255.0/self.data[idx].max())).astype(np.uint8)
        img = crop_resize(img)
        # img = img.astype(np.float32)/255.0
        return img.reshape(SIZE, SIZE), name

# %%

labels = pd.read_csv(TEST_LABELS)

# %%
print("labels shape = ", labels.shape)


# %%

from efficientnet_pytorch.utils import (
    round_filters,
    get_same_padding_conv2d
)


# %%
# https://arxiv.org/pdf/1811.00202.pdf
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = p
        self.eps = eps
    
    def forward(self, x):
        return (torch.mean(torch.abs(x**self.p), dim=(2,3)) + self.eps)**(1.0/self.p)


class EffNetTuned(nn.Module):
    def __init__(self):
        super(EffNetTuned, self).__init__()
        self.eff = EfficientNet.from_name(BACKBONE)
        # 1 channel
        Conv2d = get_same_padding_conv2d(image_size = self.eff._global_params.image_size)
        out_channels = round_filters(32, self.eff._global_params)
        self.eff._conv_stem = Conv2d(1, out_channels, kernel_size=3, stride=2, bias=False)

        self.eff._avg_pooling = GeM()
        self.eff._dropout = nn.Dropout(p=0.2, inplace=False)
        self.fc = nn.Sequential(
            nn.Linear(in_features=1000, out_features=512),
            nn.BatchNorm1d(512)
        ) 

    def forward(self, x):
        x = self.eff(x)
        x = self.fc(x)
        return x

class EffMultioutputNet(nn.Module):
    def __init__(self):
        super(EffMultioutputNet, self).__init__()
        self.backbone = EffNetTuned()
        self.cosRootClassifier = nn.Linear(in_features=512, out_features=168)
        self.cosVowelClassifier = nn.Linear(in_features=512, out_features=11)
        self.cosConsonantClassifier = nn.Linear(in_features=512, out_features=7)

    def forward(self, x):
        root = self.backbone(x)
        root = self.cosRootClassifier(root)

        vowel = self.backbone(x)
        vowel = self.cosVowelClassifier(vowel)

        consonant = self.backbone(x)
        consonant = self.cosConsonantClassifier(consonant)

        return root, vowel, consonant

# %%

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    return model



# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = EffMultioutputNet()
model = load_checkpoint(CHECKPOINT_FNAME)
# checkpoint = torch.load(CHECKPOINT_FNAME, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])    
model = model.to(device)

# %%
row_id,target = [],[]
for fname in TEST:
    test_image = BengaliDatasetTest(fname)
    dl = torch.utils.data.DataLoader(test_image,batch_size=128,num_workers=4,shuffle=False)
    with torch.no_grad():
        for x,y in tqdm(dl):
            x = x.unsqueeze(1).float().cuda()
            p1,p2,p3 = model(x)
            p1 = p1.argmax(1).view(-1).cpu()
            p2 = p2.argmax(1).view(-1).cpu()
            p3 = p3.argmax(1).view(-1).cpu()
            for idx,name in enumerate(y):
                row_id += [f'{name}_vowel_diacritic',f'{name}_grapheme_root',
                           f'{name}_consonant_diacritic']
                target += [p1[idx].item(),p2[idx].item(),p3[idx].item()]
                
sub_df = pd.DataFrame({'row_id': row_id, 'target': target})
sub_df.to_csv('submission.csv', index=False)
sub_df.head(20)

# %%
