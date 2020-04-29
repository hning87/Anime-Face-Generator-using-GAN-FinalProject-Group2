import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import torch.autograd as autograd
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch
import numpy as np
# %% --------------------------------------- Set-Up --------------------------------------------------------------------

torch.manual_seed(123)
np.random.seed(123)
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
workspace_dir = os.getcwd()
save_dir = os.path.join(workspace_dir, 'WGANGP_2D_5L/epochs')
os.makedirs(save_dir, exist_ok=True)

# %% -------------------------------- Hyperparameter -------------------------------------------------------------------

BATCH_SIZE = 128
NOISE = 100
LR = 1e-4
N_EPOCHS = 50
LAMBDA = 5

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

class Discriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, 64, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4))
        self.apply(weights_init)
    def forward(self, x):
        out = self.ls(x)
        out = out.view(-1)
        return out

# %% --------------------------------------- Training prep -------------------------------------------------------------
G = Generator(in_dim=NOISE).cuda()
D = Discriminator(3).cuda()
G.train()
D.train()

# loss criterion
criterion = nn.BCELoss()

# optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))

# gradient penalty
# code reference:https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dualgan/dualgan.py

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(FloatTensor(real_samples.shape[0]).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# %% --------------------------------------- Training  -----------------------------------------------------------------

for e, epoch in enumerate(range(N_EPOCHS)):
    for i, data in enumerate(dataloader):
        imgs = data.cuda()
        size = imgs.size(0)

        # train D
        for _ in range(2):
            noise = Variable(torch.randn(size, NOISE)).cuda()
            real_imgs = Variable(imgs).cuda()
            fake_imgs = G(noise)

            # label
            real_label = torch.ones((size)).cuda()
            fake_label = torch.zeros((size)).cuda()

            # dis
            real_logit = D(real_imgs.detach())
            fake_logit = D(fake_imgs.detach())

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(D, real_imgs.data, fake_imgs.data)

            # compute loss
            loss_D = -torch.mean(real_logit) + torch.mean(fake_logit) + LAMBDA * gradient_penalty

            # Forward + Backward + Optimize
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        # train G
        noise = Variable(torch.randn(size, NOISE)).cuda()
        fake_imgs = G(noise)
        fake_logit = D(fake_imgs)

        # compute loss
        loss_G = -torch.mean(fake_logit)

        # Forward + Backward + Optimize
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        # print training process
        print(
            f'\rEpoch [{epoch + 1}/{N_EPOCHS}] {i + 1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}',
            end='')

    G.eval()
    sample = Variable(torch.randn(100, NOISE)).cuda()
    fake_imgs_sample = (G(sample).data + 1) / 2.0
    filename = os.path.join(save_dir, f'Epoch_{epoch + 1:03d}.jpg')
    torchvision.utils.save_image(fake_imgs_sample, filename, nrow=10)
    print(f' | Save some samples to {filename}.')

    # show generated image
    grid_img = torchvision.utils.make_grid(fake_imgs_sample.cpu(), nrow=10)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

    # save models
    G.train()
    if (e + 1) % 5 == 0:
        torch.save(G.state_dict(), os.path.join(os.getcwd(), 'WGANGP_g_2D_5L.pth'))
        torch.save(D.state_dict(), os.path.join(os.getcwd(), 'WGANGP_d_2D_5L.pth'))

