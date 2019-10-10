import os
from skimage.io import imread
import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from .myio import load_one_image
from itertools import product
from scipy.ndimage import zoom
from tqdm import tqdm_notebook
from multiprocessing import Pool

def narrow(arr,axis,start,end):
    if axis<0:
        axis = arr.ndim-axis
    return arr[(slice(None),)*axis+(slice(start,end,1),)+(slice(None),)*(arr.ndim-axis-1)]

def _shift(arr, shift, axis):
    if shift == 0:
        return arr

    if axis < 0:
        axis += arr.ndim

    dim_size = arr.shape[axis]
    after_start = dim_size - shift
    slice_shape=list(arr.shape)
    slice_shape[axis]=abs(shift)
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)
        before = np.ones(slice_shape)*arr.min()
        after = narrow(arr,axis, after_start, shift)
    else:
        before = narrow(arr,axis, 0, dim_size - shift)
        after = np.ones(slice_shape)*arr.min()
    return np.concatenate([after, before], axis)

class MyTransform():
    
    def __init__(self,flip=False,mean_change=0,std_change=0,crop=None,seed=None,zoom=0.0,rotate=0,shift=0,out_size=None):
        self.do_flip = flip
        self.rotate_angle=rotate
        self.mean_change = mean_change
        self.std_change = std_change
        np.random.seed(seed)
        self.zoom_factor=zoom

        if isinstance (crop,tuple):
            self.cropx=crop[0]
            self.cropy=crop[1]
        else:
            self.cropx=crop
            self.cropy=crop
        if isinstance (shift,tuple):
            self.shiftx=shift[0]
            self.shifty=shift[1]
        else:
            self.shiftx=shift
            self.shifty=shift
        if isinstance (out_size,tuple):
            self.out_sizex=out_size[0]
            self.out_sizey=out_size[1]
        else:
            self.out_sizex=out_size
            self.out_sizey=out_size
        
    def random(self,imgs):
        sqz=False
        imgs=imgs.copy()
        if len(imgs.shape)==2:
            imgs=np.expand_dims(imgs, axis=0)
            sqz=True
        cropx,cropy = imgs.shape[1:3] if self.cropx is None else (self.cropx,self.cropy)
        out_sizex,out_sizey = imgs.shape[1:3] if self.out_sizex is None else (self.out_sizex,self.out_sizey)
        imgs=imgs.transpose(1,2,0)
        if (self.std_change>0) or (self.mean_change>0):
            for i,ix in enumerate(self.channels):
                imgs[i]=imgs[i]*np.random.normal(loc=1,scale=self.std_change)+np.random.normal(loc=0,scale=self.mean_change)
        if self.do_flip:
            if (np.random.randint(2)>0):
                imgs = self.flip(imgs)
        if self.rotate_angle>0:
            angle=np.random.randint(-self.rotate_angle,self.rotate_angle)
            imgs=self.rotate(imgs,angle)       
        if self.shiftx>0:
            imgs=self.img_shift(imgs,np.random.randint(-self.shiftx,self.shiftx),np.random.randint(-self.shifty,self.shifty))
        if self.zoom!=0:
            factor=np.random.randn(1)[0]*self.zoom_factor
            imgs=self.zoom(imgs,factor)
        x0=max(imgs.shape[1]//2-cropx//2,0)
        y0=max(imgs.shape[0]//2-cropy//2,0)
        imgs=self.crop(imgs,x0,y0,cropx,cropy)
        if (imgs.shape[0]!=out_sizey) or (imgs.shape[1]!=out_sizex):
            imgs=self.resize(imgs,out_sizex, out_sizey)
        imgs=imgs.transpose(2,0,1)        
        if sqz:
            imgs=imgs.squeeze(0)
        return imgs
    
    def flip(self,img,axis=1):
        return np.flip(img,axis=axis)
    
    def img_shift(self,img,x,y):
        return _shift(_shift(img,x,1),y,0)
    
    def crop(self,img,x,y,width,hight):
        if width>img.shape[1]:
            img=np.concatenate([np.ones((img.shape[0],(width-img.shape[1])//2+1,img.shape[-1]))*img.min(),
                                img,
                                np.ones((img.shape[0],(width-img.shape[1])//2+1,img.shape[-1]))*img.min()],1)
        if hight>img.shape[0]:
            img=np.concatenate([np.ones(((hight-img.shape[0])//2+1,img.shape[1],img.shape[-1]))*img.min(),
                                img,
                                np.ones(((hight-img.shape[0])//2+1,img.shape[1],img.shape[-1]))*img.min()],0)
        
        return img[x:x+width,y:y+hight,:]
    
    def change_mean_std(self,img,mean,std):
        if (isinstance(mean,list)):
            for i,(m,s) in zip(mean,std):
                img[...,i] = img[...,i]*s+m
        else:
            img = img*std + mean
        return img
    
    def resize(self,img,width,hight):
        return transform.resize(img,(hight,width),anti_aliasing=True)
    
    def zoom(self,img,factor):
        timg=transform.rescale(img,1.0+factor,multichannel=True,mode='constant',cval=float(img.min()))
        return transform.rescale(img,1.0+factor,multichannel=True,mode='constant',cval=float(img.min()))
    
    def rotate(self,img,angle,resize=True):
        return transform.rotate(img, angle, resize=resize, center=None, order=1, 
                                mode='constant', cval=img.min(), clip=True, preserve_range=False)   
    

    
    
class sampler():
    
    def __init__(self,arr,norm_ratio,sampled_ratios,unique_col=None):
        self.arr=arr
        self.norm_ratio = norm_ratio
        self.sampled_ratios = sampled_ratios
        self.unique_col=unique_col
        
    def do_unique(self,indxes):
        if self.unique_col is not None:
            u,ind = np.unique(self.unique_col[indxes],return_index=True)
            return indxes[ind]
        else:
            return indexs
        
    def __call__(self,index_arr=None):
        if index_arr is None:
            index_arr = Ellipsis
        sampled = []
        indxes=np.argwhere(~self.arr[index_arr].any(axis=1)>0).squeeze()
        np.random.shuffle(indxes)
        indxes = self.do_unique(indxes)    
        sampled.append(indxes[:int(indxes.shape[0]*(self.norm_ratio-np.floor(self.norm_ratio)))])
        for i in range(int(self.norm_ratio)):
            indxes=np.argwhere(~self.arr[index_arr].any(axis=1)>0).squeeze()
            np.random.shuffle(indxes)  
            sampled.append(self.do_unique(indxes))
           
        for i,s in enumerate(self.sampled_ratios):
            s_=s
            if s_>1:
                s_=np.floor(s_)
                for j in range(int(s_)):
                    indxes=np.argwhere(self.arr[index_arr][...,i]>0).squeeze()
                    np.random.shuffle(indxes) 
                    sampled.append(self.do_unique(indxes))
                s_=s-s_
            if s_>0:
                    indxes=np.argwhere(self.arr[index_arr][...,i]>0).squeeze()
                    np.random.shuffle(indxes)
                    indxes = self.do_unique(indxes)
                    sampled.append(indxes[:int(indxes.shape[0]*s_)])
        return np.concatenate(sampled)

    
class simple_sampler():
    
    def __init__(self,arr,ratio):
        self.arr=arr
        self.ratio = ratio
        
    def __call__(self):
        indxes=np.arange(self.arr.shape[0])
        np.random.shuffle(indxes)
        return indxes[:int(self.arr.shape[0]*self.ratio)]

class Mixup():
    def __init__(self,alpha=0.4,device='gpu'):
        self.alpha=alpha
        self.device=device
        
    def __call__(self,images,targets):
        lambd = np.random.beta(self.alpha, self.alpha, targets.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        shuffle = torch.randperm(targets.size(0)).to(self.device)
        lambd=torch.tensor(lambd,dtype=torch.float).to(self.device)
        out_images = (lambd*images.transpose(0,-1)+(1-lambd)*images[shuffle].transpose(0,-1)).transpose(0,-1)
        out_targets = (lambd*targets.transpose(0,-1)+(1-lambd)*targets[shuffle].transpose(0,-1)).transpose(0,-1)
        return out_images, out_targets

class ImageDataset(Dataset):
    """Image dataset."""

    def __init__(self, df, base_path, transform=None,out_shape=None,window_eq=False,equalize=True,rescale=False):
        """
        Args:
            df - dataframe 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(ImageDataset, self).__init__()
        self.df = df
        self.pids = df.PatientID.values
        self.transform = transform
        self.base_path = base_path
        self.out_shape=out_shape
        self.window_eq=window_eq
        self.equalize = equalize
        self.rescale=rescale

    def __len__(self):
        return self.pids.shape[0] 

    def __getitem__(self, idx):
        sample=load_one_image(self.pids[idx],equalize=self.equalize,base_path=self.base_path,file_path='',
                              window_eq=self.window_eq,rescale=self.rescale)
        sample = torch.tensor(sample,dtype=torch.float) \
            if self.transform is None else torch.tensor(self.transform(sample),dtype=torch.float)
        if len(sample.shape)==2:
            sample = sample.unsqueeze(0)
        if self.out_shape is not None:
            if sample.shape != self.out_shape:
                print ("Error in idx {}".format(idx))
                print (sample.shape,sample)
                sample = torch.randn(self.out_shape)*1e-5
        return sample

class FeatursDataset(Dataset):
    """Image dataset."""

    def __init__(self, df, features,num_neighbors, ref_column,order_column,target_columns=None):
        """
        Args:
            Todo
        """
        super(FeatursDataset, self).__init__()
        self.df = df.sort_values([ref_column,order_column])
        self.num_neighbors = num_neighbors
        self.ref_column = ref_column
        self.target_columns=target_columns
        self.target_tensor=None if target_columns is None else torch.tensor(df[self.target_columns].values,dtype=torch.float)
        self.features=features
        self.ref_arr=np.zeros((self.df.shape[0],1+2*self.num_neighbors),dtype=np.long)
        for i in range(-self.num_neighbors,self.num_neighbors+1):
            self.ref_arr[:,i+self.num_neighbors]=np.where(self.df[ref_column]==self.df[ref_column].shift(i),
                                                          np.roll(self.df.index.values,i),
                                                          self.df.index.values)
        self.ref_arr=torch.tensor(self.ref_arr[np.argsort(self.ref_arr[:,self.num_neighbors])])
                
                              

    def __len__(self):
        return self.ref_arr.shape[0] 

    def __getitem__(self, idx):
        sample=self.features[self.ref_arr[idx]]
        return sample if self.target_tensor is None else (sample, self.target_tensor[idx])


class FeatursDatasetCor(Dataset):
    """Image dataset."""

    def __init__(self, df, features,num_neighbors, ref_column,target_columns=None):
        """
        Args:
            Todo
        """
        super(FeatursDatasetCor, self).__init__()
        self.num_neighbors = num_neighbors
        self.ref_column = ref_column
        self.target_columns=target_columns
        self.target_tensor=None if target_columns is None else torch.tensor(df[self.target_columns].values,dtype=torch.float)
        self.features=features
        self.ref_arr=np.zeros((df.shape[0],1+2*self.num_neighbors),dtype=np.long)
        unq,si=np.unique(df[self.ref_column].values,return_inverse=True)
        for i in tqdm_notebook(range(unq.shape[0]), leave=False):
            sinx = np.where(si==i)[0]
            r=np.corrcoef(self.features[sinx].numpy())
            self.ref_arr[sinx]=sinx[np.argsort(-r)][:,:1+2*self.num_neighbors]

    def __len__(self):
        return self.ref_arr.shape[0] 

    def __getitem__(self, idx):
        sample=self.features[self.ref_arr[idx]]
        return sample if self.target_tensor is None else (sample, self.target_tensor[idx])


class DatasetCat(Dataset):
    '''Concatenate Datasets'''
    
    def __init__(self,datasets):
        '''
        Args: datasets - an iterable containing the datasets
        '''
        super(DatasetCat, self).__init__()
        self.datasets=datasets
        assert len(self.datasets)>0
        for dataset in datasets:
            assert len(self.datasets[0])==len(dataset),"Datasets length should be equal"
            
    def __len__(self):
        return len(self.datasets[0])
    
    def __getitem__(self, idx):
        outputs = tuple(dataset.__getitem__(idx) for i in self.datasets for dataset in (i if isinstance(i, tuple) else (i,)))
        return tuple(output for i in outputs for output in (i if isinstance(i, tuple) else (i,)))
