import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= n_channels, out_channels= n_channels, kernel_size= 3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(n_channels, affine= True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels= n_channels, out_channels= n_channels, kernel_size= 3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(num_features= n_channels, affine= True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = x + residual
        return x

class TransNetEncoder(nn.Module):
    def __init__(self):
        super(TransNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1,padding=1, groups=16)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, groups=64)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.relu3 = nn.ReLU()
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.in3(x)
        x = self.relu3(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x

class TransNetDecoder(nn.Module):
    def __init__(self):
        super(TransNetDecoder, self).__init__()
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.deconv1(x)
        x = self.in1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.tanh(x)
        # x = F.interpolate(x, size=(512, 512), mode='nearest')  #jugaad
        return x

# encoder = TransNetEncoder()
# decoder = TransNetDecoder()

# # the weights of the last layers in encoders and the first layers in decoders are shared.
# decoder.res1.conv1.weight = encoder.res3.conv2.weight
# decoder.res1.in1.weight = encoder.res3.in2.weight
# decoder.res1.in1.bias = encoder.res3.in2.bias

class TransNet(nn.Module):
    def __init__(self):
        super(TransNet, self).__init__()
        self.encoder = TransNetEncoder()
        self.decoder = TransNetDecoder()
        self.decoder.res1.conv1.weight = self.encoder.res3.conv2.weight
        self.decoder.res1.in1.weight = self.encoder.res3.in2.weight
        self.decoder.res1.in1.bias = self.encoder.res3.in2.bias

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def test():
    x = torch.randn((1, 3, 512, 512))
    y = torch.randn((1, 3, 512, 512))
    model = TransNet()
    preds = model(x)
    # print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()