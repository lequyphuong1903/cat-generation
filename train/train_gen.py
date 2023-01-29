import torch
import torch.nn.functional as F
import train.model_net as m

def train_generator(opt_g, batch_size, latent_size, device):
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = m.generator(latent)
    
    # Try to fool the discriminator
    preds = m.discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()