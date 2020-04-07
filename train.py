# %%
from IPython import get_ipython;   
get_ipython().magic('reset -sf')

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils.data import Dataset
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    
    RandomGamma,
    ShiftScaleRotate ,
    GaussNoise,
    Blur,
    MotionBlur,   
    GaussianBlur,
)

import torch.nn as nn
from torch.functional import F
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import math
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish
import sklearn
import csv
from tqdm import tqdm
import gc
import os
import sys
from torch import optim
# %%
CHECKPOINT_FNAME = 'checkpoint_val.chk'
LOG_FNAME = 'train_logs.csv'
BACKBONE = 'efficientnet-b3'
N_EPOCHS = 15 
BATCH_SIZE = 16

RESUME = False
if os.path.isfile(CHECKPOINT_FNAME):
    RESUME = True

LITE = True

HEIGHT = 137
WIDTH = 236
SIZE = 128

TRAIN = ['./kaggle/bengaliai-cv19/train_image_data_0.parquet',
         './kaggle/bengaliai-cv19/train_image_data_1.parquet',
         './kaggle/bengaliai-cv19/train_image_data_2.parquet',
        './kaggle/bengaliai-cv19/train_image_data_3.parquet']
TRAIN = ['./kaggle/bengaliai-cv19/train_data_0.feather',
        './kaggle/bengaliai-cv19/train_data_1.feather',
        './kaggle/bengaliai-cv19/train_data_2.feather',
        './kaggle/bengaliai-cv19/train_data_3.feather']
TRAIN_LABELS = './kaggle/bengaliai-cv19/train.csv'

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

def get_image(df, idx): 
    img0 = 255 - df.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype(np.uint8)
    img = (img0*(255.0/img0.max())).astype(np.uint8)
    return img

def Resize(df,size=128):
    resized = {} 
    df = df.set_index('image_id')
    for i in tqdm(range(df.shape[0])):
       # image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        image0 = 255 - df.loc[df.index[i]].values.reshape(HEIGHT, WIDTH).astype(np.uint8)
    #normalize each image by its max val
        img = (image0*(255.0/image0.max())).astype(np.uint8)
        image = crop_resize(img)
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized

# %%

# for i in range(2,4):
#     data = pd.read_parquet(TRAIN[i])
#     data = Resize(data)
#     data.to_feather(f'train_data_{i}.feather')
#     del data
#     gc.collect()

# %%
class BengaliDataset(Dataset):

    def __init__(self, df, labels, transform=True):
        self.grapheme_roots = labels['grapheme_root'].values.astype(np.uint8)
        self.vowel_diacritics = labels['vowel_diacritic'].values.astype(np.uint8)
        self.consonant_diacritics = labels['consonant_diacritic'].values.astype(np.uint8)
        self.image_ids = df.iloc[:, 0].values
        self.data = df.iloc[:, 1:].values
        self.transform = transform

        self.aug = Compose([ 
            ShiftScaleRotate(p=1,border_mode=cv2.BORDER_CONSTANT,value =1),
            OneOf([
                ElasticTransform(p=0.1, alpha=1, sigma=50, alpha_affine=50,border_mode=cv2.BORDER_CONSTANT,value =1),
                GridDistortion(distort_limit =0.05 ,border_mode=cv2.BORDER_CONSTANT,value =1, p=0.1),
                OpticalDistortion(p=0.1, distort_limit= 0.05, shift_limit=0.2,border_mode=cv2.BORDER_CONSTANT,value =1)                  
                ], p=0.3),
            OneOf([
                GaussNoise(var_limit=1.0),
                Blur(),
                GaussianBlur(blur_limit=3)
                ], p=0.4),    
            RandomGamma(p=0.8)])

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\n'
        return string

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        grapheme_root = self.grapheme_roots[index]
        vowel_diacritic = self.vowel_diacritics[index]
        consonant_diacritic = self.consonant_diacritics[index]

        # image_id = image_id + '.png'

        # img0 = 255 - self.data[index, :].reshape(HEIGHT, WIDTH).astype(np.uint8)
        # img = (img0*(255.0/img0.max())).astype(np.uint8)
        # img = crop_resize(img)
        img = self.data[index, :].reshape(SIZE, SIZE)
        if self.transform:
            img = self.aug(image=img)['image']

        infor = {'index' : index, 
                 'image_id' : image_id} 

        return img, grapheme_root, vowel_diacritic, consonant_diacritic, infor

# %%

if not LITE:
    try:
        del df
        gc.collect()
    except NameError:
        pass
    labels_all = pd.read_csv(TRAIN_LABELS)
    df = pd.read_feather(TRAIN[0])
    for i in range(1, 4):
        read = pd.read_feather(TRAIN[i])
        gc.collect()
        df = pd.concat([df, read], ignore_index=True)
        gc.collect()
    labels = labels_all[:len(df)]
    gc.collect()
else:
    try:
        del df
        gc.collect()
    except NameError:
        pass

    labels_all = pd.read_csv(TRAIN_LABELS)
    labels = labels_all[labels_all.grapheme_root.isin([59,60,61,62,63,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95])]

    df_all = pd.read_feather(TRAIN[0])
    df_all = df_all.loc[df_all.image_id.isin(labels.image_id.values)]
    gc.collect()
    
    for i in range(1, 4):
        read = pd.read_feather(TRAIN[i])
        read = read.loc[read.image_id.isin(labels.image_id.values)]
        gc.collect()
        df_all = pd.concat([df_all, read], ignore_index=True)
        gc.collect()

    df = df_all.loc[df_all.image_id.isin(labels.image_id.values)]
    del df_all, labels_all
    gc.collect()

# %%
print("labels shape = ", labels.shape)
print("df shape = ", df.shape)
# %%
def visualize(original_image,aug_image):
    fontsize = 18
    f, ax = plt.subplots(1, 2, figsize=(8, 8))
    ax[0].imshow(original_image)
    ax[0].set_title('Original image', fontsize=fontsize)
    ax[1].imshow(aug_image)
    ax[1].set_title('Augmented image', fontsize=fontsize)

# %%
# https://github.com/hysts/pytorch_image_classification/blob/master/optim.py
class LARSOptimizer(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr=0.02,
                 momentum=0,
                 weight_decay=0,
                 eps=1e-9,
                 thresh=1e-2):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
            thresh=thresh)
        super(LARSOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            eps = group['eps']
            thresh = group['thresh']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)
                local_lr = weight_norm / (
                    eps + grad_norm + weight_decay * weight_norm)
                local_lr = torch.where(weight_norm < thresh,
                                       torch.ones_like(local_lr), local_lr)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state[
                            'momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(lr * local_lr, d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(lr * local_lr, d_p)
                p.data.add_(-1.0, buf)

        return loss

# %%
# https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py
class CosFace(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    @staticmethod
    def cosine_sim(x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1,w2).clamp(min=eps)

    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        cosine = CosFace.cosine_sim(input, self.weight)
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        return output

    def __repr__(self):
        return self.CosFace.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class OneHot(nn.Module):
    def __init__(self, depth):
        super(OneHot,self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).to(device)
    def forward(self, X_in):
        X_in = X_in.long()
        # ones = torch.sparse.torch.eye(self.depth).to(X_in.device)
        return self.ones.index_select(0, X_in)
        # return Variable(selfones.index_select(0,batch).ones.index_select(0,X_in.data))
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

class CosFace2(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(CosFace2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.one_hot = OneHot(out_features)
        self.weight = Parameter(torch.Tensor(in_features, out_features)).to(device)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        y = self.one_hot(label)
        # y = torch.zeros_like(label)
        # y.scatter_(0, label.view(-1, 1), 1.0)

        x = F.normalize(x, dim=1)
        W = F.normalize(self.weight, dim=0)
        logits = torch.mm(x, W)
        target_logits = logits - self.m
        logits = logits * (1 - y) + target_logits * y
        logits *= self.s
        # return F.softmax(logits, dim=1)
        return logits

    def __repr__(self):
        return self.CosFace.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

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
        # self.eff = EfficientNet.from_name('efficientnet-b0', override_params={'num_classes': 1000}, in_channels=1)
        self.eff = EfficientNet.from_pretrained(BACKBONE, num_classes=1000, in_channels=1)

        self.eff._avg_pooling = GeM()
        self.eff._dropout = nn.Dropout(p=0.2, inplace=False)
        self.fc = nn.Sequential(
            nn.Linear(in_features=1000, out_features=512),
            nn.BatchNorm1d(512)
        ) 

    def forward(self, x):
        x = self.eff(x)
        x = self.fc(x)
        # return F.normalize(x, dim=1)
        return x

class EffMultioutputNet(nn.Module):
    def __init__(self):
        super(EffMultioutputNet, self).__init__()
        self.backbone = EffNetTuned()
        # self.cosRootClassifier = nn.Sequential(
        #     nn.Linear(in_features=1000, out_features=512),
        #     nn.Linear(in_features=512, out_features=168)
        # ) 
        # self.cosVowelClassifier = nn.Sequential(
        #     nn.Linear(in_features=1000, out_features=512),
        #     nn.Linear(in_features=512, out_features=11)
        # ) 
        # self.cosConsonantClassifier = nn.Sequential(
        #     nn.Linear(in_features=1000, out_features=512),
        #     nn.Linear(in_features=512, out_features=7)
        # ) 
        self.cosRootClassifier = nn.Linear(in_features=512, out_features=168)
        self.cosVowelClassifier = nn.Linear(in_features=512, out_features=11)
        self.cosConsonantClassifier = nn.Linear(in_features=512, out_features=7)
        # self.cosRootClassifier = CosFace(in_features=512, out_features=168)
        # self.cosVowelClassifier = CosFace(in_features=512, out_features=11)
        # self.cosConsonantClassifier = CosFace(in_features=512, out_features=7)
        # self.cosRootClassifier = CosFace2(in_features=512, out_features=168)
        # self.cosVowelClassifier = CosFace2(in_features=512, out_features=11)
        # self.cosConsonantClassifier = CosFace2(in_features=512, out_features=7)

    def forward(self, x, root_label, vowel_label, consonant_label):
        root = self.backbone(x)
        root = self.cosRootClassifier(root)
        # root = self.cosRootClassifier(root, root_label)
        # root = F.softmax(root, dim=1)

        vowel = self.backbone(x)
        vowel = self.cosVowelClassifier(vowel)
        # vowel = self.cosVowelClassifier(vowel, vowel_label)
        # vowel = F.softmax(vowel, dim=1)

        consonant = self.backbone(x)
        consonant = self.cosConsonantClassifier(consonant)
        # consonant = self.cosConsonantClassifier(consonant, consonant_label)
        # consonant = F.softmax(consonant, dim=1)

        return root, vowel, consonant

# %%

def macro_recall_multi(pred_graphemes, true_graphemes,pred_vowels,true_vowels,pred_consonants,true_consonants, n_grapheme=20, n_vowel=11, n_consonant=4):
    #pred_y = torch.split(pred_y, [n_grapheme], dim=1)
    pred_label_graphemes = torch.argmax(pred_graphemes, dim=1).cpu().numpy()
    true_label_graphemes = true_graphemes.cpu().numpy()
    pred_label_vowels = torch.argmax(pred_vowels, dim=1).cpu().numpy()
    true_label_vowels = true_vowels.cpu().numpy()
    pred_label_consonants = torch.argmax(pred_consonants, dim=1).cpu().numpy()
    true_label_consonants = true_consonants.cpu().numpy()   

    recall_grapheme = sklearn.metrics.recall_score(pred_label_graphemes, true_label_graphemes, average='macro')
    recall_vowel = sklearn.metrics.recall_score(pred_label_vowels, true_label_vowels, average='macro')
    recall_consonant = sklearn.metrics.recall_score(pred_label_consonants, true_label_consonants, average='macro')
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2, 1, 1])
    #print(f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}, '
    #       f'total {final_score}')
    return final_score

# %%
from sklearn.model_selection import train_test_split
train_labels , valid_labels = train_test_split(labels, test_size=0.20, shuffle=False) ## Split Labels
train_df, valid_df = train_test_split(df, test_size=0.20, shuffle=False) ## split data
del df, labels
gc.collect()

# %%

train_dataset = BengaliDataset(train_df ,train_labels,transform = True) 
valid_dataset = BengaliDataset(valid_df ,valid_labels,transform = False) 
torch.cuda.empty_cache()
gc.collect()

# %%

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=True)
del train_df, valid_df, train_labels, valid_labels 
torch.cuda.empty_cache()
gc.collect()

# %%
torch.cuda.empty_cache()
gc.collect()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%

epoch = 0
criterion = nn.CrossEntropyLoss()
# %%
model = EffMultioutputNet()
# optimizer =  optim.Adam(model.parameters(), lr=1e-5)
optimizer = LARSOptimizer(model.parameters(), lr=0.0002, weight_decay=1e-3) 
# optimizer = optim.RMSprop([
#     {'params': model.backbone.parameters(), 'lr': 1e-3},
#     {'params': model.cosRootClassifier.parameters(), 'lr': 1e-5},
#     {'params': model.cosVowelClassifier.parameters(), 'lr': 1e-5},
#     {'params': model.cosConsonantClassifier.parameters(), 'lr': 1e-5},
# ])

if RESUME:
    checkpoint = torch.load(CHECKPOINT_FNAME, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    history = pd.read_csv(LOG_FNAME)
else:
    history = pd.DataFrame()
    
model = model.to(device)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-5, epochs=N_EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.0)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=N_EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.0)

# %%
def train(epoch, history):
    model.train()
    loss = 0.0
    acc= 0.0
    total = 0.0
    running_loss = 0.0
    running_acc = 0.0
    running_recall = 0.0
    for idx, (inputs,labels1,labels2,labels3, inform) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs = inputs.float().to(device)
        labels1 = labels1.long().to(device)
        labels2 = labels2.long().to(device)
        labels3 = labels3.long().to(device)
        total += len(inputs)
        optimizer.zero_grad()
        outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1), labels1, labels2, labels3)
        loss1 = 2*criterion(outputs1,labels1)
        loss2 = 1* criterion(outputs2,labels2)
        loss3 = 1*criterion(outputs3,labels3)
        running_loss += loss1.item()+loss2.item()+loss3.item()
        running_recall+= macro_recall_multi(outputs1,labels1,outputs2,labels2,outputs3,labels3)
        running_acc += (outputs1.argmax(1)==labels1).float().mean()
        running_acc += (outputs2.argmax(1)==labels2).float().mean()
        running_acc += (outputs3.argmax(1)==labels3).float().mean()
        (loss1+loss2+loss3).backward()
        scheduler.step()
        optimizer.step()
        
    print(' running_recall : ', running_recall)
    print(' running_loss : ', running_loss)
    print(' running_acc : ', running_acc)
    loss = running_loss/len(train_loader)
    acc = running_acc/(len(train_loader)*3)
    print(' train epoch : {}\tacc : {:.2f}%'.format(epoch,running_acc/(len(train_loader)*3)))
    print('loss : {:.4f}'.format(running_loss/len(train_loader)))
    print('recall: {:.4f}'.format(running_recall/len(train_loader)))
    total_train_recall = running_recall/len(train_loader)
    torch.cuda.empty_cache()
    gc.collect()
    history.loc[epoch, 'train_loss'] = loss
    history.loc[epoch,'train_acc'] = acc.cpu().numpy()
    history.loc[epoch,'train_recall'] = total_train_recall
    return  total_train_recall

def evaluate(epoch,history):
    model.eval()
    loss = 0.0
    acc = 0.0
    total = 0.0
    running_loss = 0.0
    running_acc = 0.0
    running_recall = 0.0
    with torch.no_grad():
        for idx, (inputs,labels1,labels2,labels3, inform) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            inputs = inputs.float().to(device)
            labels1 = labels1.long().to(device)
            labels2 = labels2.long().to(device)
            labels3 = labels3.long().to(device)
            total += len(inputs)
            outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1), labels1, labels2, labels3)
            loss1 = 2*criterion(outputs1,labels1)
            loss2 = criterion(outputs2,labels2)
            loss3 = criterion(outputs3,labels3)
            running_loss += loss1.item()+loss2.item()+loss3.item()
            running_recall+= macro_recall_multi(outputs1,labels1,outputs2,labels2,outputs3,labels3)
            running_acc += (outputs1.argmax(1)==labels1).float().mean()
            running_acc += (outputs2.argmax(1)==labels2).float().mean()
            running_acc += (outputs3.argmax(1)==labels3).float().mean()
                
    loss = running_loss/len(valid_loader)
    acc = running_acc/(len(valid_loader)*3)
    total_recall = running_recall/len(valid_loader) 
    print('val epoch: {} \tval acc : {:.2f}%'.format(epoch,running_acc/(len(valid_loader)*3)))
    print('loss : {:.4f}'.format(running_loss/len(valid_loader)))
    print('recall: {:.4f}'.format(running_recall/len(valid_loader)))
    history.loc[epoch, 'valid_loss'] = loss
    history.loc[epoch, 'valid_acc'] = acc.cpu().numpy()
    history.loc[epoch, 'valid_recall'] = total_recall
    return  total_recall

# %%
history = pd.DataFrame()
valid_recall = 0.0
best_valid_recall = 0.0
for epoch in range(last_epoch+1, N_EPOCHS):
    torch.cuda.empty_cache()
    gc.collect()
    train_recall = train(epoch, history)
    valid_recall = evaluate(epoch, history)
    if valid_recall > best_valid_recall:
        print(f'Validation recall has increased from:  {best_valid_recall:.4f} to: {valid_recall:.4f}. Saving checkpoint')
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, CHECKPOINT_FNAME) ## Saving model weights based on best validation accuracy.
        history.to_csv(LOG_FNAME)
        best_valid_recall = valid_recall ## Set the new validation Recall score to compare with next epoch

# %%
history.to_csv(LOG_FNAME)
history.head()


# %%
