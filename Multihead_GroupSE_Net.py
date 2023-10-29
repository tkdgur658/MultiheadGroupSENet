import torch
import torch.nn as nn
import math

class Conv_Block(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int, pool=True) -> None:
        super(Conv_Block, self).__init__()
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=3, stride=1, padding=1, bias=False, groups=num_output_features))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        if pool == True:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

def Conv1(in_channels, out_channels):
    return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2,padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Group_Conv_SELayer(nn.Module):
    def __init__(self, in_channels, groups, expansion=1):
        super(Group_Conv_SELayer, self).__init__()
        self.groups = groups
        self.size_for_group = int(in_channels/self.groups)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, groups, kernel_size=1, stride=1, bias=False, groups=groups),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.groups, self.groups * expansion, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.groups * expansion, self.groups, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        y = self.fc(x)
        y = y.view(batch_size, -1, 1, 1)
        y_repeat = y.repeat_interleave(self.size_for_group,1)
        return x * y_repeat.expand_as(x), y
        
class Multihead_GroupSE_Net(nn.Module):
    def __init__(self, in_channels=78, num_classes=1, num_init_features=8, blocks = [1,1,1,1], expansion=8):
        super(Multihead_GroupSE_Net,self).__init__()
        self.in_channels = in_channels
        self.head_layers = nn.ModuleList()
        self.head_skip_connections = nn.ModuleList()
        saved_num_features_list = []
        for i in range(1,in_channels+1):
            num_features = num_init_features
            saved_num_features_list.append(num_features)
            
            block_1 = Conv_Block(num_input_features=num_features, num_output_features=expansion)
            saved_num_features_list.append(expansion)
            
            num_features = num_features+expansion
            block_2 = Conv_Block(num_input_features=num_features,num_output_features=expansion)
            saved_num_features_list.append(expansion)
            
            num_features = num_features+expansion
            block_3 = Conv_Block(num_input_features=num_features,num_output_features=expansion)
            saved_num_features_list.append(expansion)
            
            num_features = num_features+expansion
            block_4 = Conv_Block(num_input_features=num_features,num_output_features=expansion, pool=False)
            
            self.head_layers.append(nn.ModuleList([
                Conv1(1,num_init_features), 
                block_1,
                block_2,
                block_3,
                block_4,]
            ))
            self.skip_connections = nn.ModuleList()
            scale_factor=1
            for i, num_features in enumerate(saved_num_features_list):
                for j in range(3-i):
                    pool_size = 2*(2**j)
                    self.skip_connections.append(nn.Sequential(
                        nn.AvgPool2d(pool_size),
                        nn.Conv2d(saved_num_features_list[i], saved_num_features_list[i], kernel_size=3, stride=1, padding=1, bias=False, groups=saved_num_features_list[i]),
                        nn.BatchNorm2d(saved_num_features_list[i]),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(saved_num_features_list[i], saved_num_features_list[i], kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(saved_num_features_list[i]),
                        nn.ReLU(inplace=True)
                            )
                        )

            self.head_skip_connections.append(self.skip_connections)
        self.group_conv_se = Group_Conv_SELayer(expansion*self.in_channels, self.in_channels)
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(num_features*in_channels, num_classes)
        
    def forward(self, x):
        for i in range(1,self.in_channels+1):
            an_input = torch.unsqueeze(x[:,i-1],1)
            stem, block_1, block_2, block_3, block_4 = self.head_layers[i-1]
            skip_connections = self.head_skip_connections[i-1]
            out_0 = stem(an_input)
            out_1 = block_1(out_0)
            out_2 = block_2(torch.cat([skip_connections[0](out_0), out_1], 1))
            out_3 = block_3(torch.cat([skip_connections[1](out_0), skip_connections[3](out_1), out_2], 1))            
            out_4 = block_4(torch.cat([skip_connections[2](out_0), skip_connections[4](out_1), skip_connections[5](out_2), out_3], 1))
            if i==1:
                out=out_4
            else:
                out = torch.cat([out,out_4], 1) 
        out, attention = self.group_conv_se(out)
        out = self.adaptive_avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
