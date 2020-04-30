## import packages
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn

## set diretory and parameters
DATA_DIR = os.getcwd()
z_dim = 100

## init weight
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

## genertaor

class Generator(nn.Module):
    """
    input (N, noise dimension)
    output (N, 3, 64, 64)
    """
    def __init__(self, input):
        super(Generator, self).__init__()
        self.linear = nn.Sequential(
            # 100 to 8192 (512*4*4)
            nn.Linear(input, 8192, bias=False),
            nn.BatchNorm1d(8192),
            nn.ReLU())
        # 512,4,4 to 256,8,8
        self.convT = nn.Sequential(

            nn.ConvTranspose2d(512, 256, kernel_size=5, padding=2,
                               stride=2, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 256,8,8 to 128,16,16
            nn.ConvTranspose2d(256, 128, kernel_size=5, padding=2,
                               stride=2, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 128,16,16 to 64,32,32
            nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2,
                               stride=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64,32,32 to image 64,64,3
            nn.ConvTranspose2d(64, 3, kernel_size=5, padding=2,
                               stride=2, output_padding=1),
            nn.Tanh())

        self.apply(weights_init)
    def forward(self, x):
        y = self.linear(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.convT(y)
        return y


## load pretrained model
G = Generator(z_dim)
# below is the code for original generate images in my terminal
# G.load_state_dict(torch.load(os.path.join(DATA_DIR, './models/WGAN_G.pth')))

# if you clone the github run below, otherwise, change the path
G.load_state_dict(torch.load(os.path.join(DATA_DIR, 'Anime-Face-Generator-using-GAN-FinalProject-Group2/Code/WGAN_G.pth')))
G.eval()
G.cuda()

# generate images and save the result
n_col=5
n_row=5
z = Variable(torch.randn(n_col*n_row, z_dim)).cuda()
imgs_sample = (G(z).data + 1) / 2.0

# show image
grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=n_row)
plt.figure(figsize=(5, 5))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()

# # save image
save_dir = os.path.join(DATA_DIR, 'saved_images')  # u can set up your directory here
os.makedirs(save_dir, exist_ok=True)
filename = os.path.join(save_dir, f'WGAN_images.jpg')
torchvision.utils.save_image(imgs_sample, filename, nrow=n_row)
