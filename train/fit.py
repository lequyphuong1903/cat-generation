from tqdm.notebook import tqdm
import torch
from train.train_disc import train_discriminator
from train.train_gen import train_generator
import getdata

def fit(epochs, lr, discriminator, generator, train_dl, batch_size, latent_size,fixed_latent, device, start_idx=1):
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d, batch_size, latent_size, device)
            # Train generator
            loss_g = train_generator(opt_g, batch_size, latent_size, device)
            
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
    
        # Save generated images
        getdata.save_images.save_samples(epoch+start_idx, fixed_latent)
    
    return losses_g, losses_d, real_scores, fake_scores