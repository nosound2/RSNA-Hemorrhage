import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
import math
import copy
import collections
from pytorchcv.model_provider import get_model as ptcv_get_model
from pytorchcv.models.common import conv3x3_block
import pretrainedmodels
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output 

class Window(nn.Module):
    def forward(self, x):
        return torch.clamp(x,0,1)

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features,weights=None):
        super(ArcMarginProduct, self).__init__()
        if weights is None:
            self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
            self.reset_parameters()
        else:
            self.weight = nn.Parameter(weights)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
#        self.k.data=torch.ones(1,dtype=torch.float)

    def forward(self, features):
        cosine = F.linear(l2_norm(features), l2_norm(self.weight))
        return cosine

class ArcClassifier(nn.Module):
    def __init__(self,in_features, out_features,weights=None):
        super(ArcClassifier, self).__init__()
        self.classifier = ArcMarginProduct(in_features, out_features,weights=weights)
        self.dropout1=nn.Dropout(p=0.5, inplace=True)
        
    def forward(self, x,eq):
        out = self.dropout1(x-eq)
        out = self.classifier(out)
        return out

    def no_grad(self):
        for param in self.parameters():
            param.requires_grad=False

    def do_grad(self):
        for param in self.parameters():
            param.requires_grad=True


class MyDenseNet(nn.Module):
    def __init__(self,model,
                 num_classes,
                 num_channels=1,
                 strategy='copy',
                 add_noise=0.,
                 drop_out=0.5,
                 arcface=False,
                 return_features=False,
                 norm=False,
                 intermediate=0,
                 extra_pool=1,
                 pool_type='avg',
                 wso=None,
                 dont_do_grad=['wso'],
                 do_bn=False):
        super(MyDenseNet, self).__init__()
        self.features= torch.nn.Sequential()
        self.num_channels=num_channels
        self.dont_do_grad=dont_do_grad
        self.pool_type=pool_type
        self.norm=norm
        self.return_features=return_features
        self.num_classes=num_classes
        self.extra_pool=extra_pool
        if wso is not None:
            conv_ = nn.Conv2d(1,self.num_channels, kernel_size=(1, 1))
            if hasattr(wso, '__iter__'):
                conv_.weight.data.copy_(torch.tensor([[[[1./wso[0][1]]]],[[[1./wso[1][1]]]],[[[1./wso[2][1]]]]]))
                conv_.bias.data.copy_(torch.tensor([0.5 - wso[0][0]/wso[0][1],
                                                    0.5 - wso[1][0]/wso[1][1],
                                                    0.5 -wso[2][0]/wso[2][1]]))

            self.features.add_module('wso_conv',conv_)
            self.features.add_module('wso_window',nn.Sigmoid())
            if do_bn:
                self.features.add_module('wso_norm',nn.BatchNorm2d(self.num_channels))
            else:
                self.features.add_module('wso_norm',nn.InstanceNorm2d(self.num_channels))
        if (strategy == 'copy') or (num_channels!=3):
            base = list(list(model.children())[0].named_children())[1:]
            conv0 = model.state_dict()['features.conv0.weight']
            new_conv=nn.Conv2d(self.num_channels, conv0.shape[0], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            a=(np.arange(3*(self.num_channels//3+1),dtype=np.int)%3)
            np.random.shuffle(a)
            for i in range(self.num_channels):
                new_conv.state_dict()['weight'][:,i,:,:]=conv0.clone()[:,a[i],:,:]*(1.0+torch.randn_like(conv0[:,a[i],:,:])*add_noise)
            self.features.add_module('conv0',new_conv)
        else:
            base = list(list(model.children())[0].named_children())
        for (n,l) in base:
            self.features.add_module(n,l)
        if intermediate==0:
            self.num_features=list(model.children())[-1].in_features
            self.intermediate=None
        else:
            self.num_features=intermediate
            self.intermediate=nn.Linear(list(model.children())[-1].in_features, self.num_features)
        self.dropout1=nn.Dropout(p=drop_out, inplace=True)
        if arcface:
            self.classifier=ArcMarginProduct(self.num_features, num_classes)
        else:
            self.classifier = nn.Linear(self.num_features//self.extra_pool, self.num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        if self.pool_type=='avg':
            x = F.avg_pool3d(x.unsqueeze(1), kernel_size=(self.extra_pool,)+x.size()[2:]).view(x.size(0), -1)
        else:
            x = F.max_pool3d(x.unsqueeze(1), kernel_size=(self.extra_pool,)+x.size()[2:]).view(x.size(0), -1)
#        x = F.max_pool1d(x.view(x.unsqueeze(1),self.extra_pool).squeeze()
        x = self.dropout1(x)
        if self.intermediate is not None:
            x = self.intermediate(x)
            x = F.relu(x)
        features = x
        if self.norm:
            features = l2_norm(features,axis=1)
        out = self.classifier(features)
        return out if not self.return_features else (out,features)
    
    def parameter_scheduler(self,epoch):
        do_first=['classifier','wso']
        if epoch>0:
            for n,p in self.named_parameters():
                p.requires_grad=True
        else:
            for n,p in self.named_parameters():
                p.requires_grad= any(nd in n for nd in do_first)
                
    def no_grad(self):
        for param in self.parameters():
            param.requires_grad=False

    def do_grad(self):
        for n,p in self.named_parameters():
            p.requires_grad=  not any(nd in n for nd in self.dont_do_grad)

class MySENet(nn.Module):
    def __init__(self,model,
                 num_classes,
                 num_channels=3,
                 dropout=0.2,
                 return_features=False,
                 wso=None,
                 full_copy=False,
                 dont_do_grad=['wso'],
                 extra_pool=1):
        super(MySENet, self).__init__()
        self.num_classes=num_classes
        self.return_features=return_features
        self.num_channels = num_channels
        self.features= torch.nn.Sequential()
        self.extra_pool=extra_pool
        self.dont_do_grad=dont_do_grad
        if full_copy:
            for (n,l) in list(list(model.children())[0].named_children()):
                self.features.add_module(n,l)
            if wso is not None:
                self.dont_do_grad=model.dont_do_grad
        else:
            if wso is not None:
                conv_ = nn.Conv2d(1,self.num_channels, kernel_size=(1, 1))
                if hasattr(wso, '__iter__'):
                    conv_.weight.data.copy_(torch.tensor([[[[1./wso[0][1]]]],[[[1./wso[1][1]]]],[[[1./wso[2][1]]]]]))
                    conv_.bias.data.copy_(torch.tensor([0.5 - wso[0][0]/wso[0][1],
                                                        0.5 - wso[1][0]/wso[1][1],
                                                        0.5 -wso[2][0]/wso[2][1]]))

                self.features.add_module('wso_conv',conv_)
                self.features.add_module('wso_relu',nn.Sigmoid())
                self.features.add_module('wso_norm',nn.InstanceNorm2d(self.num_channels))

#            layer0= torch.nn.Sequential()
#            layer0.add_module('conv1',model.conv1)
#            layer0.add_module('bn1',model.bn1)                        
            se_layers={'layer0':model.layer0,
                       'layer1':model.layer1,
                       'layer2':model.layer2,
                       'layer3':model.layer3,
                       'layer4':model.layer4}
            for key in se_layers:
                self.features.add_module(key,se_layers[key])
        self.dropout = dropout if dropout is None else nn.Dropout(p=dropout, inplace=True)
        self.classifier=nn.Linear(model.last_linear.in_features//self.extra_pool, self.num_classes)
        
        
    def forward(self, x):
        x = self.features(x)
        x = F.max_pool3d(x.unsqueeze(1), kernel_size=(self.extra_pool,)+x.size()[2:]).view(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x) 
        features = x
        out = self.classifier(features)
        return out if not self.return_features else (out,features) 
    
    def parameter_scheduler(self,epoch):
        do_first=['classifier']
        if epoch>0:
            for n,p in self.named_parameters():
                p.requires_grad=True
        else:
            for n,p in self.named_parameters():
                p.requires_grad= any(nd in n for nd in do_first)
                
    def no_grad(self):
        for param in self.parameters():
            param.requires_grad=False


    def do_grad(self):
        for n,p in self.named_parameters():
            p.requires_grad=  not any(nd in n for nd in self.dont_do_grad)

class MyEfficientNet(nn.Module):
    def __init__(self,model,num_classes,num_channels=3,dropout=0.5,return_features=False,wso=True,
                 full_copy=False,
                 dont_do_grad=['wso'],
                 extra_pool=1,
                 num_features=None):
        super(MyEfficientNet, self).__init__()
        self.num_classes=num_classes
        self.return_features=return_features
        self.num_channels = num_channels
        self.features= torch.nn.Sequential()
        self.extra_pool=extra_pool
        self.dont_do_grad=dont_do_grad
        if full_copy:
            for (n,l) in list(list(model.children())[0].named_children()):
                self.features.add_module(n,l)
            if wso is not None:
                self.dont_do_grad=model.dont_do_grad

        else:
            if wso is not None:
                conv_ = nn.Conv2d(1,self.num_channels, kernel_size=(1, 1))
                if hasattr(wso, '__iter__'):
                    conv_.weight.data.copy_(torch.tensor([[[[1./wso[0][1]]]],[[[1./wso[1][1]]]],[[[1./wso[2][1]]]]]))
                    conv_.bias.data.copy_(torch.tensor([0.5 - wso[0][0]/wso[0][1],
                                                        0.5 - wso[1][0]/wso[1][1],
                                                        0.5 -wso[2][0]/wso[2][1]]))

                self.features.add_module('wso_conv',conv_)
                self.features.add_module('wso_relu',nn.Sigmoid())
                self.features.add_module('wso_norm',nn.InstanceNorm2d(self.num_channels))
            for (n,l) in list(list(model.children())[0].named_children()):
                self.features.add_module(n,l)
        self.dropout = dropout if dropout is None else nn.Dropout(p=dropout, inplace=True)
        if num_features is None:
            self.classifier=nn.Linear(model.output.fc.in_features//self.extra_pool, self.num_classes)
        else:
            self.classifier=nn.Linear(num_features, self.num_classes)
        
        
    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, kernel_size=x.size(-1)).view(x.size(0), -1)
        if self.extra_pool>1:
            x = x.view(x.shape[0],x.shape[1]//self.extra_pool,self.extra_pool).mean(-1)
        if self.dropout is not None:
            x = self.dropout(x)
        features = x
        out = self.classifier(features)
        return out if not self.return_features else (out,features) 
    
    def parameter_scheduler(self,epoch):
        do_first=['classifier']
        if epoch>0:
            for n,p in self.named_parameters():
                p.requires_grad=True
        else:
            for n,p in self.named_parameters():
                p.requires_grad= any(nd in n for nd in do_first)
                
    def no_grad(self):
        for param in self.parameters():
            param.requires_grad=False

    def do_grad(self):
        for n,p in self.named_parameters():
            p.requires_grad=  not any(nd in n for nd in self.dont_do_grad)


class NeighborsNet(nn.Module):
    def __init__(self,num_classes,num_features=1024,num_neighbors=1,classifier_layer=None,intermidiate=None,dropout=0.2):
        super(NeighborsNet, self).__init__()
        self.num_classes=num_classes
        if classifier_layer is not None:
            self.num_features = classifier_layer.in_features
        else:
            self.num_features=num_features
        self.num_neighbors=num_neighbors
        layers=collections.OrderedDict()
        if dropout>0:
            layers['dropout']=nn.Dropout(p=dropout)

        if intermidiate is not None:
            layers['intermidiate']=nn.Linear(self.num_features*(2*self.num_neighbors+1), intermidiate)
            layers['relu']=nn.ReLU()
            layers['classifier']=nn.Linear(intermidiate, self.num_classes)
        else:
            layers['classifier']=nn.Linear(self.num_features*(2*self.num_neighbors+1), self.num_classes)
        if (classifier_layer is not None) and (intermidiate is None):
            _=layers['classifier'].bias.data.copy_((1.0+0.2*self.num_neighbors)*classifier_layer.bias.data)
            d = torch.cat([0.1*classifier_layer.weight.data for i in range(self.num_neighbors)]+\
                             [classifier_layer.weight.data]+\
                             [0.1*classifier_layer.weight.data for i in range(self.num_neighbors)],dim=1)
            _=layers['classifier'].weight.data.copy_(d)
        self.network= torch.nn.Sequential(layers)

        
    def forward(self, x):
        x = x.view((x.shape[0],-1))
        return self.network(x) 
    
    def parameter_scheduler(self,epoch):
        do_first=['classifier']
        if epoch>0:
            for n,p in self.named_parameters():
                p.requires_grad=True
        else:
            for n,p in self.named_parameters():
                p.requires_grad= any(nd in n for nd in do_first)
                
    def no_grad(self):
        for param in self.parameters():
            param.requires_grad=False

    def do_grad(self):
        for param in self.parameters():
            param.requires_grad=True

def mean_model(models):
    model = copy.deepcopy(models[0])
    params=[]
    for model_ in models:
        params.append(dict(model_.named_parameters()))

    param_dict=dict(model.named_parameters())

    for name in param_dict.keys():
        _=param_dict[name].data.copy_(torch.cat([param[name].data[...,None] for param in params],-1).mean(-1))
    return model