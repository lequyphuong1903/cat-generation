import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import getdata
import train

image_size = 64
batch_size = 64
latent_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
DATA_DIR = './data'

train_ds = ImageFolder(DATA_DIR, transform = T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)
]))

train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 2, pin_memory = True)

# if __name__ == '__main__':
#     getdata.show_images.show_batch(train_dl)

device = getdata.get_device.get_default_device()

train_dl = getdata.get_device.DeviceDataLoader(train_dl, device)

fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

discriminator = train.model_net.discriminator
discriminator = getdata.get_device.to_device(discriminator, device)
generator = train.model_net.generator
generator = getdata.get_device.to_device(generator, device)

epochs = 50
lr = 0.0001

if __name__ == '__main__':
    history = train.fit.fit(epochs, lr, discriminator, generator, train_dl, batch_size, latent_size,fixed_latent, device)
    getdata.save_images.save_plot(history, 'losses')
    getdata.save_images.save_plot(history, 'scores')

torch.save(generator.state_dict(), 'G.pth')
torch.save(discriminator.state_dict(), 'D.pth')


