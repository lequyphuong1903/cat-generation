from torchvision.utils import save_image
from getdata.show_images import denorm
import os
import matplotlib.pyplot as plt
import train.model_net as m

sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)

def save_samples(index, latent_tensors):
    fake_images = m.generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)

def save_plot(history, type):
    if type == 'losses':
        plt.plot(history[0], '-')
        plt.plot(history[1], '-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Discriminator', 'Generator'])
        plt.title('Losses')
        plt.savefig('Losses.png')
        plt.close()
    elif type == 'scores':
        plt.plot(history[2], '-')
        plt.plot(history[3], '-')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Real', 'Fake'])
        plt.title('Scores')
        plt.savefig('Scores.png')
        plt.close()
