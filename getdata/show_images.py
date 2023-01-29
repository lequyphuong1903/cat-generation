from torchvision.utils import make_grid
import matplotlib.pyplot as plt

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def denorm(img_tensors):
  return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax = 64):
  fig, ax = plt.subplots(figsize = (8, 8))
  ax.set_xticks([])
  ax.set_yticks([])
  ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow = 8).permute(1, 2, 0))

def show_batch(dl, nmax = 64):
  for images, _ in dl:
    show_images(images, nmax)
    plt.show()
    break