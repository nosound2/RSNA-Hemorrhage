import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
import logging
import abc
import sys
from tqdm import tqdm_notebook
import torch.utils.data as D
import torch.nn.functional as F
from apex import amp 
#from .mymodels import out_to_predict,out_to_predict_in,out_to_predict_test,out_to_predict_test_simple
import copy
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from .mymodels import mean_model
def get_model_device(model):
    p = next(model.parameters())
    if p.is_cuda:
        device = torch.device("cuda:{}".format(p.get_device()))
    else:
        device = torch.device('cpu')
    return device
    
def model_train(model,optimizer,train_dataset,batch_size,num_epochs,loss_func,
                weights=None,accumulation_steps=1,
                weights_func=None,do_apex=True,validate_dataset=None,
                validate_loss=None,metric=None,param_schedualer=None,
                weights_data=False,history=None,return_model=False,model_apexed=False,
                num_workers=7,sampler=None,graph=None,k_lossf=0.01,pre_process=None,
                call_progress=None,use_batchs=True,best_average=1):
    
    if history is None:
        history = []
    num_average_models=min(num_epochs,best_average)
    best_models=np.empty(num_average_models+1,dtype=object)
    best_val_loss=1e6*np.ones(num_average_models+1,dtype=np.float)
    device = get_model_device(model)
    if do_apex and not model_apexed and (device.type=='cuda'):
        model_apexed=True
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    model.zero_grad()
    tq_epoch=tqdm_notebook(range(num_epochs))
    lossf=None
    for epoch in tq_epoch:
        best_models[1:]=best_models[:num_average_models]
        best_val_loss[1:]=best_val_loss[:num_average_models]
        torch.cuda.empty_cache()
        model.do_grad()
        if param_schedualer:
            param_schedualer(epoch)
        _=model.train()
        batch_size_= batch_size if use_batchs else None
        if sampler:
            data_loader=D.DataLoader(D.Subset(train_dataset,sampler()),num_workers=num_workers,
                                     batch_size=batch_size if use_batchs else None,
                                     shuffle=use_batchs)
        else:
            data_loader=D.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
        sum_loss = 0.
        if metric:
            metric.zero()
        tq_batch = tqdm_notebook(data_loader,leave=True)
        model.zero_grad()
        for i,(batchs) in enumerate(tq_batch):
            x_batch=batchs[0].to(device) if len(batchs)==2 else [x.to(device) for x in batchs[:-1]]  
            y_batch=batchs[-1].to(device)
            if pre_process is not None:
                x_batch,y_batch = pre_process(x_batch,y_batch)
            if weights_data:
                weights=x_batch[-1]
                x_batch=x_batch[:-1]
            if weights_func:
                weights=weights_func(weights,epoch,i)
            y_preds = model(x_batch) if not isinstance(x_batch,list) else model(*x_batch)  

            if weights is not None:
                loss = loss_func(y_preds,y_batch,weights=weights)/accumulation_steps
            else:
                loss = loss_func(y_preds,y_batch)/accumulation_steps

            if model_apexed:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                model.zero_grad()
            
            if lossf:
                lossf = (1-k_lossf)*lossf+k_lossf*loss.detach().item()*accumulation_steps
            else:
                lossf = loss.detach().item()*accumulation_steps
            if graph is not None:
                graph(lossf)
            batch_postfix={'loss':lossf}
            if metric:
                if isinstance(y_preds,tuple):
                    yp = tuple(y_preds[0].detach().cpu())
                else:
                    yp =y_preds.detach().cpu()
                batch_postfix.update(metric.calc(yp,y_batch.cpu().detach()))
            tq_batch.set_postfix(**batch_postfix)

            sum_loss=sum_loss+loss.detach().item()*accumulation_steps
        
        epoch_postfix={'loss':sum_loss/len(data_loader)}
        if metric:
            epoch_postfix.update(metric.calc_sums())
        tq_epoch.set_postfix(**batch_postfix)
        history.append(batch_postfix)

        if validate_dataset:
            if validate_loss is None:
                vloss = loss_func
                val_weights = weights
            else:
                vloss = validate_loss
                val_weights =None
            res=model_evaluate(model,
                               validate_dataset,
                               batch_size = batch_size if use_batchs else None ,
                               loss_func=vloss,
                               weights=val_weights,
                               metric=metric,
                               do_apex=False,
                               num_workers=num_workers)
                                     
            history[-1].update(res[1])
            best_val_loss[0] = res[0]
            best_models[0] = copy.deepcopy(model).to('cpu')
            best_models[0].no_grad()
            best_models=best_models[np.argsort(best_val_loss)]
            best_val_loss=best_val_loss[np.argsort(best_val_loss)]
#            if res[0]<best_val_loss:
#                best_val_loss=res[0]
#                best_model=copy.deepcopy(model)
#                best_model.no_grad()
            tq_epoch.set_postfix(res[1])
        print(history[-1])
        if call_progress is not None:
            call_progress(history)
    if num_average_models>1:
        best_model=mean_model(best_models[:num_average_models])
        model=model.to('cpu')
        best_model=best_model.to(device)
        res=model_evaluate(best_model,
                               validate_dataset,
                               batch_size = batch_size if use_batchs else None ,
                               loss_func=vloss,
                               weights=val_weights,
                               metric=metric,
                               do_apex=False,
                               num_workers=num_workers)
        best_model=best_model.to('cpu')
        model=model.to(device)
        if res[0]>best_val_loss[0]:
            best_model=best_models[0]
            print (best_val_loss[0])
        else:
            print (res)
    else:
        best_model=best_models[0]
        print (best_val_loss)
    return (history,best_model) if return_model else history




def model_run(model,dataset,do_apex=True,batch_size=32,num_workers=6):
    _=model.eval()
    model.no_grad()
    device = get_model_device(model)
    if do_apex and (device.type=='cuda'):
        model = amp.initialize(model, opt_level="O1",verbosity=0)
    res_list=[]
    data_loader=D.DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    for batchs in tqdm_notebook(data_loader):
        y_preds=model(*[x.to(device) for x in batchs]) if isinstance(batchs,tuple) else model(batchs.to(device))
        res_list.append(tuple([y.cpu() for y in y_preds]) if isinstance(y_preds,tuple) else y_preds.cpu())
    return tuple([torch.cat(tens) for tens in map(list, zip(*res_list))]) if isinstance(res_list[0],tuple) else torch.cat(res_list)

def models_run(models,dataset,do_apex=True,batch_size=32,num_workers=6):
    islist = isinstance(models,list)    
    if islist:
        models_=models
    else:
        models_=[models]
    for model in models_:
        _=model.eval()
        model.no_grad()
    device = get_model_device(models_[0])
     
    if do_apex and (device.type=='cuda'):
        for model in models_:
            model = amp.initialize(model, opt_level="O1",verbosity=0)
    res_list=[]
    for model in models_:        
        res_list.append([])
    data_loader=D.DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    for batchs in tqdm_notebook(data_loader):
        for i,model in enumerate(models_):
            y_preds=model(*[x.to(device) for x in batchs]) if isinstance(batchs,tuple) else model(batchs.to(device))
            res_list[i].append(tuple([y.cpu().detach() for y in y_preds]) if isinstance(y_preds,tuple) else y_preds.cpu().detach())
    res=[]
    for i in range(len(models_)):
        res.append(tuple([torch.cat(tens) for tens in map(list, 
                                                   zip(*res_list[i]))]) if isinstance(res_list[i][0],tuple) else torch.cat(res_list[i]))

    return tuple(zip(*res)) 



def model_evaluate(model,
                   validate_dataset,
                   batch_size,
                   loss_func,
                   weights=None,
                   metric=None,
                   do_apex=False,
                   num_workers=6):
    _=model.eval()
    model.no_grad()
    device = get_model_device(model)
    if do_apex and (device.type=='cuda'):
        model = amp.initialize(model, opt_level="O1",verbosity=0)
    data_loader=D.DataLoader(validate_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    sum_loss = 0.
    lossf=None
    if metric:
        metric.zero()
    tq_batch = tqdm_notebook(data_loader,leave=True)
    for i,(batchs) in enumerate(tq_batch):
        x_batch=batchs[0].to(device) if len(batchs)==2 else [x.to(device) for x in batchs[:-1]]  
        y_batch=batchs[-1]
        y_preds = model(x_batch) if not isinstance(x_batch,list) else model(*x_batch)
        if weights is None:
            loss = loss_func(y_preds,y_batch.to(device))
        else:
            loss = loss_func(y_preds,y_batch.to(device),weights)
        sum_loss=sum_loss+loss.detach().item()
        if lossf:
            lossf = 0.98*lossf+0.02*loss.detach().item()
        else:
            lossf = loss.item()
        batch_postfix={'val_loss':lossf}
        if metric:
            if isinstance(y_preds,tuple):
                yp = tuple(y_preds[0].detach().cpu())
            else:
                yp =y_preds.detach().cpu()
            batch_postfix.update(metric.calc(yp,y_batch.cpu().detach(),prefix='val_'))

        tq_batch.set_postfix(**batch_postfix)
    epoch_postfix={'val_loss':sum_loss/len(data_loader)}
    if metric:
        epoch_postfix.update(metric.calc_sums('val_'))
                                     
    return sum_loss/len(data_loader), epoch_postfix

class loss_graph():
    def __init__(self,fig,ax,num_epoch=1,batch2epoch=100,limits=None):
        self.num_epoch=num_epoch
        self.batch2epoch=batch2epoch
        self.loss_arr=np.zeros(num_epoch*batch2epoch,dtype=np.float)
        self.arr_size=num_epoch*batch2epoch
        self.num_points=0
        self.fig=fig
        self.ax = ax
        self.limits=limits if limits is not None else (-1000,1000)
        self.ticks = (np.arange(0, num_epoch*batch2epoch+1, step=batch2epoch),np.arange(0, num_epoch+1, step=1))
    def __call__(self,loss):
        if self.num_points==self.arr_size:
            new_arr=np.zeros(self.arr_size+self.batch2epoch,dtype=np.float)
            new_arr[:self.arr_size]=self.loss_arr
            self.loss_arr=new_arr
        self.loss_arr[self.num_points]=max(self.limits[0],min(self.limits[1],loss))
        self.num_points=self.num_points+1
        _=self.ax.clear()
        _=self.ax.plot(self.loss_arr[0:self.num_points])
        _=self.ax.set_xlabel('batch')
        _=self.ax.set_ylabel('loss')
        _=plt.xticks(self.ticks[0],self.ticks[1])
        _=self.fig.canvas.draw()

   