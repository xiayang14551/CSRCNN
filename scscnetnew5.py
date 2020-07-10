import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
#######################################################################
# Define the RPP layers
class RPP(nn.Module):
    def __init__(self):
        super(RPP, self).__init__()
        self.part = 6
        add_block = []
        add_block += [nn.Conv2d(3840, 6, kernel_size=1, bias=False)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        norm_block = []
        norm_block += [nn.BatchNorm2d(3840)]
        norm_block += [nn.ReLU(inplace=True)]
        # norm_block += [nn.LeakyReLU(0.1, inplace=True)]
        norm_block = nn.Sequential(*norm_block)
        norm_block.apply(weights_init_kaiming)

        self.add_block = add_block
        self.norm_block = norm_block
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        w = self.add_block(x)
        p = self.softmax(w)
        y = []
        for i in range(self.part):
            p_i = p[:, i, :, :]
            p_i = torch.unsqueeze(p_i, 1)
            y_i = torch.mul(x, p_i)
            y_i = self.norm_block(y_i)
            y_i = self.avgpool(y_i)
            y.append(y_i)

        f = torch.cat(y, 2)
        return f
###########################################################
class ClassBlock1(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock1, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x
########################################################################
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.linear = nn.Linear(input_dim * 2, num_bottleneck)
        self.conv1x1 = conv1x1(input_dim * 2, num_bottleneck)

        add_block = []
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Dropout(p=0.5)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        xmax = self.maxpool(x)
        xavg = self.avgpool(x)
        x = torch.cat((xmax,xavg),dim=1)
        x = self.conv1x1(x)
        x = torch.squeeze(x)
        x = x.view(x.size(0), -1)
        xtr = x
        x = self.add_block(x)
        xid = self.classifier(x)
        return xid,xtr
########################################################################
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
####################scscnet网络结构#####################################
class scscnet(nn.Module):

    def __init__(self,class_num):
        super(scscnet, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.layer4[0].downsample[0].stride = (1,1)
        self.layer4[0].conv2.stride = (1,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool1x1 = nn.AdaptiveMaxPool2d((1,1))
        self.avgpoollayer1 = nn.AdaptiveAvgPool2d((24, 8))
        self.avgpoollayer2 = nn.AdaptiveAvgPool2d((24, 8))
        self.avgpoollayer3 = nn.AdaptiveAvgPool2d((24, 8))
        self.classifierlayer1 = ClassBlock(3840
                                           , class_num)
        self.classifierlayer2 = ClassBlock(3840
                                           , class_num)
        self.classifierlayer3 = ClassBlock(3840
                                           , class_num)
        self.classifierlayer4 = ClassBlock(3840
                                           , class_num)
        self.classifierlayer5 = ClassBlock(3840
                                           , class_num)
        self.classifierlayer6 = ClassBlock(3840
                                           , class_num)
        self.classifiernew = ClassBlock1(3840
                                         , class_num)

    def forward(self, x):
        x = x.view(-1, *x.size()[-3:])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        #平均池化到24*8*2048 x4的大小本身就为24*8
        x1_avg = self.avgpoollayer1(x1)
        x2_avg = self.avgpoollayer2(x2)
        x3_avg = self.avgpoollayer3(x3)
        x4_avg = x4

        # #拼接256+512+1024+2048=3840
        x_cat = torch.cat((x1_avg,x2_avg,x3_avg,x4_avg),1)
        x_rpp = self.avgpool(x_cat)
        #对feature map分块成P1-P6 每一块的维度为 4*8*3840
        p1 = x_rpp[:, :, 0, :]
        p2 = x_rpp[:, :, 1, :]
        p3 = x_rpp[:, :, 2, :]
        p4 = x_rpp[:, :, 3, :]
        p5 = x_rpp[:, :, 4, :]
        p6 = x_rpp[:, :, 5, :]

        #金字塔第一层 维度为24*8*3840
        py1 = x_cat
        py1id , py1tr = self.classifierlayer1(py1)
        #金字塔第二层 维度为20*8*3840
        py21 = torch.cat((p1, p2, p3, p4, p5), dim=2)
        py21id, py21tr = self.classifierlayer2(py21)
        py22 = torch.cat((p2, p3, p4, p5, p6), dim=2)
        py22id, py22tr = self.classifierlayer2(py22)
        #金字塔第三层 维度为16*8*3840
        py31 = torch.cat((p1, p2, p3, p4), dim=2)
        py31id, py31tr = self.classifierlayer3(py31)
        py32 = torch.cat((p2, p3, p4, p5), dim=2)
        py32id, py32tr = self.classifierlayer3(py32)
        py33 = torch.cat((p3, p4, p5, p6), dim=2)
        py33id, py33tr = self.classifierlayer3(py33)

        #金字塔第四层 维度为12*8*3840
        py41 = torch.cat((p1, p2, p3), dim=2)
        py41id, py41tr = self.classifierlayer4(py41)
        py42 = torch.cat((p2, p3, p4), dim=2)
        py42id, py42tr = self.classifierlayer4(py42)
        py43 = torch.cat((p3, p4, p5), dim=2)
        py43id, py43tr = self.classifierlayer4(py43)
        py44 = torch.cat((p4, p5, p6), dim=2)
        py44id, py44tr = self.classifierlayer4(py44)
        #金字塔第五层 维度为8*8*3840
        py51 = torch.cat((p1, p2), dim=2)
        py51id, py51tr = self.classifierlayer5(py51)
        py52 = torch.cat((p2, p3), dim=2)
        py52id, py52tr = self.classifierlayer5(py52)
        py53 = torch.cat((p3, p4), dim=2)
        py53id, py53tr = self.classifierlayer5(py53)
        py54 = torch.cat((p4, p5), dim=2)
        py54id, py54tr = self.classifierlayer5(py54)
        py55 = torch.cat((p5, p6), dim=2)
        py55id, py55tr = self.classifierlayer5(py55)
        #金字塔第六层  4*8*3840
        py61 = p1
        py61id, py61tr = self.classifierlayer6(py61)
        py62 = p2
        py62id, py62tr = self.classifierlayer6(py62)
        py63 = p3
        py63id, py63tr = self.classifierlayer6(py63)
        py64 = p4
        py64id, py64tr = self.classifierlayer6(py64)
        py65 = p5
        py65id, py65tr = self.classifierlayer6(py65)
        py66 = p6
        py66id, py66tr = self.classifierlayer6(py66)

        triplet1 = py1tr
        triplet2 = torch.cat((py21tr,py22tr),dim=1)
        triplet3 = torch.cat((py31tr,py32tr,py33tr),dim=1)
        triplet4 = torch.cat((py41tr,py42tr,py43tr,py44tr),dim=1)
        triplet5 = torch.cat((py51tr,py52tr,py53tr,py54tr,py55tr),dim=1)
        triplet6 = torch.cat((py61tr,py62tr,py63tr,py64tr,py65tr,py66tr),dim=1)

        featurenew = self.avgpool(x4)
        featurenew = torch.squeeze(featurenew)
        featurenew = featurenew.view(featurenew.size(0), -1)
        # featurenew = self.classifiernew(featurenew)

        return py1id,\
               py21id,py22id,\
               py31id,py32id,py33id,\
               py41id,py42id,py43id,py44id,\
               py51id,py52id,py53id,py54id,py55id,\
               py61id,py62id,py63id,py64id,py65id,py66id,\
               featurenew,triplet1,triplet2,triplet3,triplet4,triplet5,triplet6

    def rpp(self):
        self.avgpool = RPP()
        return self


net = scscnet(751)
net = net.rpp()
print(net)
input = Variable(torch.FloatTensor(8, 3, 384, 128))
output = net(input)
print(output.shape)