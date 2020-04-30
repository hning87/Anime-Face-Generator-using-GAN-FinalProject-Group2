import os
import cv2
import time
import random
import load_data
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


# %% --------------------------------------- Set-Up --------------------------------------------------------------------

torch.manual_seed(123)
np.random.seed(123)
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
workspace_dir = os.getcwd()
save_dir = os.path.join(workspace_dir, 'DCGAN_1D1G/epochs')
os.makedirs(save_dir, exist_ok=True)

# %% -------------------------------- Hyperparameter -------------------------------------------------------------------

BATCH_SIZE = 128
NOISE = 100
LR = 1e-4
N_EPOCHS = 50

# %% --------------------------------------- Load data -------------------------------------------------------------

dataset = np.load("imgs.npy")
class CustomTensorDataset(Dataset):
    def __init__(self, img, transform=None):

        self.img = img
        self.transform = transform
        self.num_samples = len(self.img)

    def __getitem__(self, index):
        imgs = self.img[index]
        imgs = self.transform(imgs)

        return imgs

    def __len__(self):
        return self.num_samples

transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
dataset = CustomTensorDataset(dataset, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# %% --------------------------------------- Loss history -------------------------------------------------------------

def train_history(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# %% ----------------------- Discriminator and Generator -------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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

class Discriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2), # Output = (3, 64*64)
            nn.Conv2d(dim, dim * 2, 5, 2, 2), # Output = (64, 32*32)
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim * 2, dim * 4, 5, 2, 2), # Output = (128, 16*16)
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim * 4, dim * 8, 5, 2, 2), # Output = (256, 8*8)
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim * 8, 1, 4), # Output = (512, 4*4)
            nn.Sigmoid())
        self.apply(weights_init)
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

# %% --------------------------------------- Training prep -------------------------------------------------------------
G = Generator(in_dim=NOISE).cuda()
D = Discriminator(3).cuda()
G.train()
D.train()

# Show the model structure
print(G)
print(D)

# loss criterion
criterion = nn.BCELoss()

# optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))


# %% --------------------------------------- Training  -----------------------------------------------------------------

# Training loop
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_times'] = []
train_hist['total_ptime'] = []


for e, epoch in enumerate(range(N_EPOCHS)):
    # Record the loss
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    start_time = time.time()
    for i, data in enumerate(dataloader):
        imgs = data
        imgs = imgs.cuda()

        '''Training Discriminator'''
        z = Variable(torch.randn(BATCH_SIZE, NOISE)).cuda()
        real_imgs = Variable(imgs).cuda()
        fake_imgs = G(z)

        # label
        real_label = torch.ones((BATCH_SIZE)).cuda()
        fake_label = torch.zeros((BATCH_SIZE)).cuda()

        # dis
        real_dis = D(real_imgs.detach())
        fake_dis = D(fake_imgs.detach())

        # compute loss
        r_loss = criterion(real_dis, real_label)
        f_loss = criterion(fake_dis, fake_label)
        loss_D = (r_loss + f_loss) / 2

        D_losses.append(loss_D)

        # update model
        D.zero_grad()
        loss_D.backward()
        opt_D.step()

        '''Training Generator'''
        # leaf
        z = Variable(torch.randn(BATCH_SIZE, NOISE)).cuda()
        fake_imgs = G(z)

        # dis
        fake_dis = D(fake_imgs)

        # compute loss
        loss_G = criterion(fake_dis, real_label)
        G_losses.append(loss_G)

        # update model
        G.zero_grad()
        loss_G.backward()
        opt_G.step()

        # log
        print(
            f'\rEpoch [{epoch + 1}/{N_EPOCHS}] {i + 1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}',
            end='')


    G.eval()
    fake_imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(save_dir, f'Epoch_{epoch + 1:03d}.jpg')
    torchvision.utils.save_image(fake_imgs_sample, filename, nrow=5)
    print(f' | Save some samples to {filename}.')

    # For printing loss
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_times'].append(per_epoch_ptime)

    # show generated image
    grid_img = torchvision.utils.make_grid(fake_imgs_sample.cpu(), nrow=10)
    plt.figure(figsize=(10, 10)) # print out 10*10 images
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
    G.train()
    if (e + 1) % 5 == 0:
        # Save the model
        torch.save(G.state_dict(), os.path.join(workspace_dir, f'DCGAN_1D1G_g.pth'))
        torch.save(D.state_dict(), os.path.join(workspace_dir, f'DCGAN_1D1G_d.pth'))

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

train_history(train_hist, save=True, path='./DCGAN_1D1G_train_hist.png')