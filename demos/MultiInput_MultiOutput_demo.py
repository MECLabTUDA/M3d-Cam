from medcam import medcam
import torchvision.models as tvmodels
import torch
import torch.nn as nn
try:
    from wama.utils import show2D
    # https://github.com/WAMAWAMA/wama_medic
    # a medical image precessing toolbox
    flag = True
except:
    flag = False
    pass

def tensor2numpy(tensor):
    return tensor.data.cpu().numpy()

class Net1(nn.Module):
    """
    multi-input & multi-output（多输入多输出模型）
    """
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(tvmodels.resnet50(num_classes=1000, pretrained=False).children())[:-1])
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(2048+128, 2)
    def forward(self, input):
        """
        :param input1:shape in(x,3,x,x)
        :param input2:shape in(x,256)
        :return:
        """
        inputs1, inputs2 = input
        f1 = torch.squeeze(self.backbone(inputs1),3)
        f1 = torch.squeeze(f1,2)
        f2 = self.linear1(inputs2)
        f3 = torch.cat([f1,f2], 1)
        print(f1.shape, f2.shape, f3.shape)
        return [self.linear2(f3)]*3

class Net2(nn.Module):
    def __init__(self, model):
        """
        "coat" model （外套网络，用来改变输入输出）
        """
        super().__init__()
        self.model = model
        self.input = None
    def get_input(self, input):
        """
        for catching real input（负责导入真正的input）
        :param input:
        :return:
        """
        self.input = input
    def forward(self, inputs):
        """
        modify code to force output shape
        :param inputs: fake input , but shape of the fake_input will be used by medcam
        :param inputs: 假的input，但是形状会被medcam提取，用于reshape返回的attention map
        :return:
        """
        print(inputs.shape)
        if self.input is None:
            raise ValueError('self.input is None')

        out = self.model(self.input)
        return out[0]

# original network （构建原始网络）
input = [torch.ones([2,3,256,256]), torch.ones([2,256])]
model = Net1()
output = model(input)
_ = [print(i.shape) for i in output]

# use the "coat" network to packing the original network （构建外套网络）
model = Net2(model)
model.get_input(input)
output = model(torch.ones([2,3,128,128]))
print(output.shape)

# get layer name （提取层name）
name_list = [name for name,_ in model.named_parameters()]

# get and visualize the attention map （提取 attention map）
conv = name_list[156].split('.weight')[0]
print(conv) # model.backbone.7.2.conv3
model_c = medcam.inject(model, replace = True, label = 0 ,layer = conv)
model_c = medcam.inject(model, replace = True, label = 0)
model_c.get_input(input) # catch real input
attention = model_c(torch.ones([2,2, 1,1]))
print(attention.shape)
if flag:
    show2D(tensor2numpy(torch.squeeze(attention[0, 0, :, :])))

conv = name_list[126].split('.weight')[0]
print(conv) # model.backbone.6.5.conv3
model_c = medcam.inject(model, replace = True, label = 0 ,layer = conv)
model_c.get_input(input) # catch real input
attention = model_c(torch.ones([2,3,128,128]))
print(attention.shape)
if flag:
    show2D(tensor2numpy(torch.squeeze(attention[0, 0, :, :])))


conv = name_list[69].split('.weight')[0]
print(conv) # model.backbone.5.3.conv3
model_c = medcam.inject(model, replace = True, label = 0 ,layer = conv)
model_c.get_input(input) # catch real input
attention = model_c(torch.ones([2,3,128,128]))
print(attention.shape)
if flag:
    show2D(tensor2numpy(torch.squeeze(attention[0, 0, :, :])))

