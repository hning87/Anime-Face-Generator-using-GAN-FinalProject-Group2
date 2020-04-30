## import packages
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn

## set diretory and parameters
DATA_DIR = os.getcwd()
NOISE = 100

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
    input (N, in_dim)
    output (N, 3, 64, 64)
    """
    def __init__(self, in_dim):
        super(Generator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, 8192, bias=False), #(64*8*4*4)
            nn.BatchNorm1d(8192),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 5, 2,
                               padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2,
                                padding=2, output_padding = 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2,
                                padding = 2, output_padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 5, 2, padding=2, output_padding= 1 ),
            nn.Tanh())
        self.apply(weights_init)
    def forward(self, x):
        out = self.l1(x)
        out = out.view(out.size(0), -1, 4, 4)
        out = self.l2_5(out)
        return out


## load pretrained model
G = Generator(NOISE)
# below is the code for original generate images in my terminal
#G.load_state_dict(torch.load(os.path.join(DATA_DIR, './WGANGP_g_1D_5L.pth')))

# if you clone the github run below, otherwise, change the path
G.load_state_dict(torch.load(os.path.join(DATA_DIR, 'Anime-Face-Generator-using-GAN-FinalProject-Group2/Code/WGANGP_g_1D_5L.pth')))
G.eval()
G.cuda()

# generate images and save the result
n_col=5
n_row=5
z = Variable(torch.randn(n_col*n_row, NOISE)).cuda()
imgs_sample = (G(z).data + 1) / 2.0

# show image
grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=n_row)
plt.figure(figsize=(5, 5))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()

# # save image
save_dir = os.path.join(DATA_DIR, 'saved_images')  # u can set up your directory here
os.makedirs(save_dir, exist_ok=True)
filename = os.path.join(save_dir, f'WGANGP_images.jpg')
torchvision.utils.save_image(imgs_sample, filename, nrow=n_row)
