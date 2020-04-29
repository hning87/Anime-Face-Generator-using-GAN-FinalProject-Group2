##
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from torch import optim
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
DATA_DIR =  os.getcwd()
save_dir = os.path.join(DATA_DIR, 'WGAN')
os.makedirs(save_dir, exist_ok=True)
torch.manual_seed(123)
np.random.seed(123)
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# %% --------------------------------hyperparameters ------------------------------------------------------------------

batch_size = 128
z_dim = 100
lr = 1E-4
n_epoch = 50
train_D = 2  # When train D more than G, change this number

# %% --------------------------------------- Load data -------------------------------------------------------------

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

dataset = np.load("imgs.npy")
transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5) ,(0.5, 0.5, 0.5))])

dataset = CustomTensorDataset(dataset, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

##

# init weight
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# %% ----------------------- Discriminator and Generator -------------------------------------------------------------
class Discriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, channel):
        super(Discriminator, self).__init__()
        self.ls = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 64,32,32 to output 128,16,16
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 128,16,16 to 256,8,8
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 256,8,8 to 512,4,4
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, kernel_size=4))
            # nn.Sigmoid()) # no sigmoid for WGAN
        self.apply(weights_init)
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

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

# %% --------------------------------------- Training prep -------------------------------------------------------------
G = Generator(input=z_dim).cuda()
D = Discriminator(3).cuda()
G.train()
D.train()

# optimizer
opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)

# noise
z_sample = Variable(torch.randn(100, z_dim)).cuda()

print('start training ...')
# train
for e, epoch in enumerate(range(n_epoch)):
    for i, data in enumerate(dataloader):
        imgs = data
        imgs = imgs.cuda()

        batch = imgs.size(0)

        # train D
        for _ in range(train_D):

            """ Train D """
            z = Variable(torch.randn(batch, z_dim)).cuda()
            # real image
            r_img = Variable(imgs).cuda()
            # generated image
            g_img = G(z)

            # get a score from D
            r_score = D(r_img.detach())
            g_score = D(g_img.detach())

            # compute loss
            D_loss = -torch.mean(r_score) + torch.mean(g_score)

            # update model
            D.zero_grad()
            D_loss.backward()
            opt_D.step()

            # weight clipping

            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

        """ train G """
        # generate image from noise z
        z = Variable(torch.randn(batch, z_dim)).cuda()
        g_img = G(z)

        # score
        g_score = D(g_img)

        # compute loss
        G_loss = -torch.mean(g_score)

        # update model
        G.zero_grad()
        G_loss.backward()
        opt_G.step()

        # print training log
        print(
            f'\rEpoch [{epoch + 1}/{n_epoch}] {i + 1}/{len(dataloader)} D_loss: {D_loss.item():.4f} G_loss: {G_loss.item():.4f}',
            end='')
    G.eval()
    g_imgs_sample = (G(z_sample).data + 1) * 0.5
    filename = os.path.join(save_dir, f'Epoch_{epoch + 1:03d}.jpg')
    torchvision.utils.save_image(g_imgs_sample, filename, nrow=10)
    print(f' | Save Generated Images to {filename}.')

    # show generated image
    grid_img = torchvision.utils.make_grid(g_imgs_sample.cpu(), nrow=10)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
    G.train()
    if (e + 1) % 5 == 0:
        torch.save(G.state_dict(), os.path.join(DATA_DIR, f'WGAN_G.pth'))
        torch.save(D.state_dict(), os.path.join(DATA_DIR, f'WGAN_D.pth'))

print('done')