import time
import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models



class FeaturesExtractor(nn.Module):
    def __init__(self, encoder='resnet18', scale = 1):
        super(FeaturesExtractor, self).__init__()
        self.encoder = getattr(models, encoder)(pretrained=True)
        self.flatten = torch.nn.Flatten()

                
    def forward(self, x):
        x = self.encoder.conv1(x)

        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        x = self.encoder.layer1(x)
#         print(x.size())
        x = self.encoder.layer2(x)
#         print(x.size())
        x = self.encoder.layer3(x)
#         print(x.size())
        x = self.encoder.layer4(x)
#         print(x.size())
        x = self.encoder.avgpool(x)
        feature_vector = self.flatten(x)
          
        return feature_vector
    
        



        
def resnet18(device='cpu', **kwargs):
    model = FeaturesExtractor(encoder='resnet18')
    return model.to(device)
    
    
def resnet34(device='cpu', **kwargs):
    model = FeaturesExtractor(encoder='resnet34')
    return model.to(device)

def resnet50(device='cpu', **kwargs):
    model = FeaturesExtractor(encoder='resnet50')
    return model.to(device)




if __name__ == '__main__':

    import os, ssl
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)): 
        ssl._create_default_https_context = ssl._create_unverified_context

    model = resnet18()
    model.eval()
     
    x = torch.Tensor(1, 3, 224, 224)
    for i in range(10):
        t = time.time()
        y = model(x)
#         print(y.size())
        print(time.time() - t)
        break
   
    
    