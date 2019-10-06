import os
from skimage.io import imread
import numpy as np
import torch
import pydicom

DEFAULT_IMAGES_BASE_PATH = '/media/nvme/data/RSNA/'
train_path='stage_1_train_images/'
test_path='stage_1_test_images/'
file_format='ID_{}.dcm'

def load_one_image(pid,
                   equalize=False,
                   base_path=DEFAULT_IMAGES_BASE_PATH,
                   file_path=train_path,
                   file_format=file_format,
                   window_eq=False,rescale=False):
    dcm_data = pydicom.dcmread(base_path+ file_path+ file_format.format(pid))
    try:
        pixels = dcm_data.pixel_array
    except:
        print('got some error with pid {}.format(pid)')
        pixels = np.zeros((512,512))
    if pixels.shape[0]==pixels.shape[1]:
        m = max(pixels.shape)
        p = np.ones((m,m))*pixels[0,0]
        p[(m-pixels.shape[0])//2:(m-pixels.shape[0])//2+pixels.shape[0],
          (m-pixels.shape[1])//2:(m-pixels.shape[1])//2+pixels.shape[1]]=pixels
        pixels = p
    if window_eq:
        if window_eq==3:
            pixels = dcm_window3(dcm_data,pixels)
            if equalize:
                for i in range(3):
                    pixels[i] = (pixels[i]-pixels[i].mean())/(pixels[i].std()+1e-6)
        else:
            pixels = dcm_window(dcm_data,pixels)
            if rescale:
                b = dcm.RescaleIntercept
                m = dcm.RescaleSlope
                pixel_array = m * pixel_array + b       
            if equalize:
                pixels = (pixels-pixels.mean())/(pixels.std()+1e-6)
    
    return pixels

def dcm_window(dcm,pixel_array,window = None):
    ymin=0
    ymax=255
    if window is None:
        c = int(dcm.WindowCenter[0] if type(dcm.WindowCenter) == pydicom.multival.MultiValue else dcm.WindowCenter)
        w = int(dcm.WindowWidth[0] if type(dcm.WindowWidth) == pydicom.multival.MultiValue else dcm.WindowWidth)
    else:
        c , w = window
    b = dcm.RescaleIntercept
    m = dcm.RescaleSlope
    x = m * pixel_array + b
    # windowing C.11.2.1.2.1 Default LINEAR Function
    y = np.zeros_like(x)
    y[x <= (c - 0.5 - (w - 1) / 2)] = ymin
#        ymin + np.tanh(((x[x <= (c - 0.5 - (w - 1) / 2)] - (c - 0.5)) / (w - 1) + 0.5))* (ymax - ymin)  
    y[x > (c - 0.5 + (w - 1) / 2)] = ymax
#        ymax + np.tanh(((x[x > (c - 0.5 + (w - 1) / 2)] - (c - 0.5)) / (w - 1) + 0.5))* (ymax - ymin)   
    y[(x > (c - 0.5 - (w - 1) / 2)) & (x <= (c - 0.5 + (w - 1) / 2))] = \
      ((x[(x > (c - 0.5 - (w - 1) / 2)) & (x <= (c - 0.5 + (w - 1) / 2))] - (c - 0.5)) / (w - 1) + 0.5) * (ymax - ymin) + ymin
#    y = ((x - (c - 0.5)) / (w - 1) + 0.5) * (ymax - ymin) + ymin
    return y/255.0

def dcm_window3(dcm,pixel_array,windows = None):
    y = np.zeros((3,)+pixel_array.shape,dtype=np.float)
    if windows is None:
        windows=[(40,80),(80,200),(600,2800)]
    for i in range(3):
        y[i]=dcm_window(dcm,pixel_array,window = windows[i])
    return y
                 