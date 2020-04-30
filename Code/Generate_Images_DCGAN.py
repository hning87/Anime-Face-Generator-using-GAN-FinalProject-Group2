## import packages
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn

## diretory and parameters
DATA_DIR = './'
z_dim = 100


## genertaor
class Generator(nn.Module):
    """
    input (N, in_dim)
    output (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False), # Output = (512, 4*4)
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            nn.ConvTranspose2d(dim * 8, dim * 4, 5, 2, padding=2, output_padding=1, bias=False), # Output = (256, 8*8)
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(dim * 4, dim * 2, 5, 2, padding=2, output_padding=1, bias=False), # Output = (128, 16*16)
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(dim * 2, dim, 5, 2, padding=2, output_padding=1, bias=False), # Output = (64, 32*32)
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1), # Output = (3, 64*64)
            nn.Tanh())
        self.apply(weights_init)
    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


## init weight
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

## load pretrained model
G = Generator(z_dim)
G.load_state_dict(torch.load(os.path.join(DATA_DIR, 'Anime-Face-Generator-using-GAN-FinalProject-Group2/Code/DCGAN_1D1G_g.pth')))
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
plt.title('DCGAN')
plt.show()

# # save image
save_dir = os.path.join(DATA_DIR, 'saved_images')  # u can set up your directory here
os.makedirs(save_dir, exist_ok=True)
filename = os.path.join(save_dir, f'DCGAN_images.jpg')
torchvision.utils.save_image(imgs_sample, filename, nrow=n_row)

