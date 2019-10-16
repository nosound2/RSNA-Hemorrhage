'''
Data loaders and manipulation
BY: Yuval
'''

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
    # A numpy implementation to torch.narrow
    if axis<0:
        axis = arr.ndim-axis
    return arr[(slice(None),)*axis+(slice(start,end,1),)+(slice(None),)*(arr.ndim-axis-1)]

def _shift(arr, shift, axis, fill_value=None):
    ''' Shits an image. Fill the empty space
        Arges:
            arr  : numpy array with the image to shift
            shift: shift step
            axis : axis to shift
            fill_value: a value to fill empty spaces - default - None => Use min value in array
        Return:
            new shifted array
        Update: Yuval 12/10/2019
    '''
            
    if shift == 0:
        return arr
    if fill_value is None:
        fill_value = arr.min()
    if axis < 0:
        axis += arr.ndim

    dim_size = arr.shape[axis]
    after_start = dim_size - shift
    slice_shape=list(arr.shape)
    slice_shape[axis]=abs(shift)
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)
        before = np.ones(slice_shape)*fill_value
        after = narrow(arr,axis, after_start, shift)
    else:
        before = narrow(arr,axis, 0, dim_size - shift)
        after = np.ones(slice_shape)*fill_value
    return np.concatenate([after, before], axis)

class MyTransform():
    '''
    My implementation for image transformation class
    Args:
     flip        : do a right/left mirroring.  default - False
     mean_change : mean change.                default - 0
     std_change  : std change.                 default - 0
     crop=None   : crop image to size. If tuple - x,y. the crop position will chabge randomly.  
                   default -None, keep image. the crop position will
     seed        : seed to use for random.     default - None
     zoom=0.0    : Zoom images, by             default - 0 (stay the same)
     rotate=0    : Rotation Angle, Deg         default - 0 
     shift=0     : Shift image                 default - 0
     out_size    : Output image size, int (x=y) or tuple 
                                               default - None => keep input                                           
   Methods:
       random: random transform using the init parameters
       Args:
           imags - numpy array with one or more images, for multiple images the first dim should be the channle
       Returns 
           numpy array with randomly transformed images of size out_size
           
   Updated: Yuval 12/10/19
   '''
    
    def __init__(self,
                 flip=False,
                 mean_change=0,
                 std_change=0,
                 crop=None,
                 seed=None,
                 zoom=0.0,
                 rotate=0,
                 shift=0,
                 out_size=None):
        
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
#            for i,ix in enumerate(self.channels):
#                imgs[i]=imgs[i]*np.random.normal(loc=1,scale=self.std_change)+np.random.normal(loc=0,scale=self.mean_change)
            imgs=self.change_mean_std(imgs,np.random.randn(1)[0]*self.mean_change,1+np.random.randn(1)[0]*self.std_change)
        if self.do_flip:
            if (np.random.randint(2)>0):
                imgs = self.flip(imgs)
        if self.rotate_angle>0:
            angle=np.random.randint(-self.rotate_angle,self.rotate_angle)
            imgs=self.rotate(imgs,angle)       
        if self.shiftx>0:
            imgs=self.img_shift(imgs,np.random.randint(-self.shiftx,self.shiftx),np.random.randint(-self.shifty,self.shifty))
        if self.zoom_factor!=0:
            if isinstance(self.zoom_factor,tuple):
                factor_x=np.random.randn(1)[0]*self.zoom_factor[0]
                factor_y=np.random.randn(1)[0]*self.zoom_factor[1]
                factor=(1+factor_x,1+factor_y)
            else:
                factor=1+np.random.randn(1)[0]*self.zoom_factor
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
#        timg=transform.rescale(img,1.0+factor,multichannel=True,mode='constant',cval=float(img.min()))
        return transform.rescale(img,factor,multichannel=True,mode='constant',cval=float(img.min()))
    
    def rotate(self,img,angle,resize=True):
        return transform.rotate(img, angle, resize=resize, center=None, order=1, 
                                mode='constant', cval=img.min(), clip=True, preserve_range=False)   
    

    
    
class sampler():
    '''
    sampler class for RSNA 2019. sample the images according to the tagets vector
    
    Args:
        arr:            numpy array with the target vectors
        norm_ratio:     float - the ratio of sampling for all zero target vector
        sampled ratios: a numpy vector,len: arr.shape[-1], sampling ratio by target value.
        unique_col:     numpy vector length arr.shape[0], 
                        with values which will be uniqued (no 2 samples would have the same value in this column
                        default: None (don't use)
    Methods:
        __call__:
        Args:
            index_arr: index vector, sample only from this index. default: None
            
    Update: Yuval 12/10/19
    '''
    
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
            return indxes
        
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
    '''
    Simple sampler, will shuffle and sample a part of an array (dim - 0)
    Args:
        arr   :numpy array
        ratio :float, the sampling ratio, if>1 the same as =1
    Methods:
        __call__:
            Return numpy vector, type long, with sampled indexes
    Update: Yuval 12/10/19
    '''
    
    def __init__(self,arr,ratio):
        self.arr=arr
        self.ratio = ratio
        
    def __call__(self):
        indxes=np.arange(self.arr.shape[0])
        np.random.shuffle(indxes)
        return indxes[:int(self.arr.shape[0]*self.ratio)]

class Mixup():
    '''
    Method for mixup augmentation - TODO doc
    '''
    def __init__(self,alpha=0.4,device='gpu'):
        self.alpha=alpha
        self.device=device
        
    def __call__(self,images,targets):
        lambd = np.random.beta(self.alpha, self.alpha, targets.size(0))
        lambd = np.abs(lambd-0.5)+0.5 #np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        shuffle = torch.randperm(targets.size(0)).to(self.device)
        lambd=torch.tensor(lambd,dtype=torch.float).to(self.device)
        out_images = (lambd*images.transpose(0,-1)+(1-lambd)*images[shuffle].transpose(0,-1)).transpose(0,-1)
        out_targets = torch.cat([targets.unsqueeze(-1),
                                 targets[shuffle].unsqueeze(-1),
                                 lambd.unsqueeze(-1).repeat(1,targets.shape[-1]).unsqueeze(-1)],-1)
#        out_targets = (lambd*targets.transpose(0,-1)+(1-lambd)*targets[shuffle].transpose(0,-1)).transpose(0,-1)
        return out_images, out_targets

class ImageDataset(Dataset):
    '''
        RSNA 2019 Image (DICOM) dataset to use in Pytorch dataloader.
        Base class: Dataset
        Args:
            df              : Data frame with the image ids
            base_path       : File path for the images 
            transform=None  : Transfor method. to perform after the images are loaded. default: None - no transform
            out_shape=None  : Expected output shape - used only for sanity check.      default: None - no check
            window_eq=False : Do window equaliztion: (for backward competability, don't use it anymore use WSO)
                              False - No equalization
                              True  - Do equalization with window = [(40,80),(80,200),(600,2800)]
                              tuple/list shaped as above 
            equalize         : Equalize - return (image-mean)/std
            rescale=False    : Use DICOM parameters for rescale, done automaticaly if windows_eq!=False
       Update:Yuval 12/10/19       
    '''

    def __init__(self, df, base_path, transform=None,out_shape=None,window_eq=False,equalize=True,rescale=False):
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
    '''
        RSNA 2019 features dataset to use in Pytorch dataloader.
        Base class: Dataset
        Args:
            df              : Data frame
            features        : pytorch tensor with features.
                              Shape:
                                  option1 - (df.shape[0],num_of_features) - normal mode
                                  option2 - Not Implemented here yet
                                            (df.shape[0],N,num_of_features) - TTA mode, 
                                            will select random feature vector from same raw. 
                                      
            num_neighbors   : int, Number of neighbor to return, output will be shape (1+2*num_neighbors,features.shape[-1])
            ref_column      : string, The name of the column in df with the series id
            order_column    : string, The name of the column in df with the data which will determine the neighbors
            target_columns  : list of strings/None, names of column in df with the target data, 
                              default None - no target data will be returned
        Methods:
            __calls__:
                return: sample - tensor size (1+2*num_neighbors,features.shape[-1])
                        if target column is defined, return tuple with the 2nd valiable: 
                            targets - tensor size (1+2*num_neighbors,len(target_columns)), dtype=torch.float
        Update:Yuval 12/10/19       
    '''
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
    """Not Used, like FeatursDataset but determine neighbors according to feature distance"""

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

class FullHeadImageDataset(Dataset):
    '''
        RSNA 2019  full head dataset to use in Pytorch dataloader.
        return all the slices from a scan in the right order.
        Base class: Dataset
        Args:
            df              : Data frame
            base_path       : File path for the images 
            SeriesIDs       : numpy array with scan series ids, each call the mathod will return one full series
            transform       : Transfor method. to perform after the images are loaded. 
                              The same transformation is done for all images in a series
                              default: None - no transform
            out_shape       : Expected output shape - used only for sanity check.
                              default: None - no check
            window_eq       : Do window equaliztion: (for backward competability, don't use it anymore use WSO)
                              False - No equalization [default]
                              True  - Do equalization with window = [(40,80),(80,200),(600,2800)]
                              tuple/list shaped as above 
            equalize        : Equalize - return (image-mean)/std [default - False]
            rescale         : Use DICOM parameters for rescale, done automaticaly if windows_eq!=False
                               default - True
            ref_column      : string, The name of the column in df with the series id
            order_column    : string, The name of the column in df with the data which will determine the neighbors
            target_columns  : list of strings/None, names of column in df with the target data, 
                              default None - no target data will be returned
        Methods:
            __calls__:
                return: sample - tensor size (# of images in series,image shape)
                        if target column is defined, return tuple with the 2nd valiable: 
                            targets - tensor size (# of images in series,len(target_columns)), dtype=torch.float
        Update:Yuval 12/10/19       
    '''
    
    def __init__(self, df,
                 base_path,
                 SeriesIDs,
                 ref_column,
                 order_column,
                 transform=None,
                 window_eq=False,
                 equalize=False,
                 rescale=True, 
                 target_columns=None):
        super(FullHeadImageDataset, self).__init__()
        self.df = df
        self.SeriesIDs=SeriesIDs
        self.ref_column=ref_column
        self.order_column=order_column
        self.target_columns=target_columns
        self.pids = df.PatientID.values
        self.transform = transform
        self.base_path = base_path
        self.window_eq=window_eq
        self.equalize = equalize
        self.rescale=rescale
        self.ref_arr=df[ref_column].values
        self.order_arr=df[order_column].values
        self.target_tensor=None if target_columns is None else torch.tensor(df[self.target_columns].values,dtype=torch.float)


    def __len__(self):
        return self.SeriesIDs.shape[0]

    def __getitem__(self, idx):
        head_idx=np.where(self.ref_arr==self.SeriesIDs[idx])[0]
        sorted_head_idx=head_idx[np.argsort(self.order_arr[head_idx])]
        samples=[]
        for i in sorted_head_idx:
            sample=load_one_image(self.pids[i],equalize=self.equalize,base_path=self.base_path,file_path='',
                                  window_eq=self.window_eq,rescale=self.rescale)[None]
            samples.append(sample)
        headimages=np.concatenate(samples,0)
        headimages = torch.tensor(headimages,dtype=torch.float) \
                if self.transform is None else torch.tensor(self.transform(headimages),dtype=torch.float)
        headimages=headimages[:,None]  # lat's make a batch out of it.
        if self.target_tensor is not None:
            targets=self.target_tensor[sorted_head_idx]

        return headimages if self.target_tensor is None else (headimages, targets)

class FullHeadDataset(Dataset):
    '''
        RSNA 2019 full head scan features dataset to use in Pytorch dataloader.
        Base class: Dataset
        Args:
            df              : Data frame
            SeriesIDs       : numpy array with scan series ids, each call the mathod will return one full series
            features        : pytorch tensor with features.
                              Shape:
                                  option1 - (df.shape[0],num_of_features) - normal mode
                                  option2 - (df.shape[0],N,num_of_features) - TTA mode, 
                                            will select random feature vector from same raw. 
                                      
            ref_column      : string, The name of the column in df with the series id
            order_column    : string, The name of the column in df with the data which will determine the neighbors
            target_columns  : list of strings/None, names of column in df with the target data, 
                              default None - no target data will be returned
        Methods:
            __calls__:
                return: sample - tensor size (# of images in series,features.shape[-1])
                        if target column is defined, return tuple with the 2nd valiable: 
                            targets - tensor size (# of images in series,len(target_columns)), dtype=torch.float
        Update:Yuval 12/10/19       
    '''
    def __init__(self, df, SeriesIDs,features, ref_column,order_column,target_columns=None,max_len=60):
        """
        Args:
            Todo
        """
        super(FullHeadDataset, self).__init__()
        self.ref_column = ref_column
        self.target_columns=target_columns
        self.target_tensor=None if target_columns is None else torch.tensor(df[self.target_columns].values,dtype=torch.float)
        self.features=features
        self.ref_arr=df[ref_column].values
        self.order_arr=df[order_column].values
        self.max_len=max_len
        self.SeriesIDs=SeriesIDs
                
                              

    def __len__(self):
        return self.SeriesIDs.shape[0] 

    def __getitem__(self, idx):
        sample = torch.zeros((self.max_len,self.features.shape[-1]),dtype=torch.float)
        head_idx=np.where(self.ref_arr==self.SeriesIDs[idx])[0]
        sorted_head_idx=head_idx[np.argsort(self.order_arr[head_idx])]
        if self.features.dim()==3:
            tta_idx=np.random.randint(0,self.features.shape[1],size=head_idx.shape[0])
            sample[:head_idx.shape[0]]=self.features[sorted_head_idx,tta_idx]
        else:
            sample[:head_idx.shape[0]]=self.features[sorted_head_idx]
        if self.target_tensor is not None:
            targets = -1*torch.ones((self.max_len,self.target_tensor.shape[-1]),dtype=torch.float)
            targets[:head_idx.shape[0]]=self.target_tensor[sorted_head_idx]
        return sample if self.target_tensor is None else (sample, targets)



class DatasetCat(Dataset):
    '''
    Concatenate datasets for Pytorch dataloader
    The normal pytorch implementation does it only for raws. this is a "column" implementation
    Arges:
        datasets: list of datasets, of the same length
    Updated: Yuval 12/10/2019
    '''
    
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
