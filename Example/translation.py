from matplotlib import pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau




class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            input_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        layers = []
        for h_dim in reversed(hidden_dims):
            layers.append(nn.Linear(latent_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            latent_dim = h_dim
        layers.append(nn.Linear(hidden_dims[0], output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar

# Discriminator 
class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))

# Loss Functions
def vae_loss(reconstructed_x, original_x, mean, logvar):
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, original_x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + KLD

def discriminator_loss(real_logits, fake_logits):
    return -torch.mean(torch.log(torch.clamp(real_logits, min=1e-10)) + torch.log(torch.clamp(1 - fake_logits, min=1e-10)))

def generator_loss(fake_logits):
    return -torch.mean(torch.log(torch.clamp(fake_logits, min=1e-10)))


def contrastive_loss_separate(z, labels, temperature=0.5):


    z = F.normalize(z, p=2, dim=-1)

    
    logits = torch.matmul(z, z.T) / temperature
    exp_logits = torch.exp(logits)

    positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    
    positive_loss = -torch.log((exp_logits * positive_mask).sum(dim=-1) + 1e-10)
    negative_mask = 1 - positive_mask
    negative_loss = torch.log((exp_logits * negative_mask).sum(dim=-1) + 1e-10)
    
    return (positive_loss + negative_loss).mean()



def cycle_consistency_loss(original_x, cycled_x):
    return nn.functional.mse_loss(cycled_x, original_x, reduction='sum')




def train_model(path1,path2,vae_omic1, vae_omic2, discriminator_modality, discriminator_omic1, discriminator_omic2, 
                omic1_data, omic2_data, labels_omic1, labels_omic2, num_epochs=100, learning_rate=0.0001, batch_size=64):
    
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_omic1.to(device)
    vae_omic2.to(device)
    discriminator_modality.to(device)
    discriminator_omic1.to(device)
    discriminator_omic2.to(device)
    
    # Optimizers
    optimizer_E = optim.Adam(list(vae_omic1.encoder.parameters()) + list(vae_omic2.encoder.parameters()), lr=learning_rate)
    optimizer_G = optim.Adam(list(vae_omic1.decoder.parameters()) + list(vae_omic2.decoder.parameters()), lr=learning_rate)
    optimizer_D_modality = optim.Adam(discriminator_modality.parameters(), lr=learning_rate)
    optimizer_D_omic1 = optim.Adam(discriminator_omic1.parameters(), lr=learning_rate)
    optimizer_D_omic2 = optim.Adam(discriminator_omic2.parameters(), lr=learning_rate)

    # Create data loaders

    omic1_data_tensor = torch.tensor(omic1_data, dtype=torch.float32)
    omic2_data_tensor = torch.tensor(omic2_data, dtype=torch.float32)
    labels_omic1_tensor = torch.tensor(labels_omic1, dtype=torch.float32)
    labels_omic2_tensor = torch.tensor(labels_omic2, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(omic1_data_tensor, omic2_data_tensor, labels_omic1_tensor, labels_omic2_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epoch_losses = []

    
    # Training Loop
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_total_loss = 0
        for omic1_batch, omic2_batch, labels1_batch, labels2_batch in dataloader:
            omic1_batch, omic2_batch = omic1_batch.to(device), omic2_batch.to(device)
            labels1_batch, labels2_batch = labels1_batch.to(device), labels2_batch.to(device)
            
            optimizer_D_modality.zero_grad()
            
            z_omic1_mean, z_omic1_logvar = vae_omic1.encoder(omic1_batch)
            z_omic1 = vae_omic1.reparameterize(z_omic1_mean, z_omic1_logvar).detach()
            
            z_omic2_mean, z_omic2_logvar = vae_omic2.encoder(omic2_batch)
            z_omic2 = vae_omic2.reparameterize(z_omic2_mean, z_omic2_logvar).detach()
            
            d_loss_modality = discriminator_loss(discriminator_modality(z_omic1), discriminator_modality(z_omic2))
            d_loss_modality.backward()
            optimizer_D_modality.step()
            

            optimizer_D_omic1.zero_grad()
            real_validity_omic1 = discriminator_omic1(omic1_batch)  
            generated_omic1 = vae_omic1.decoder(vae_omic2.encoder(omic2_batch)[0])  
            fake_validity_omic1 = discriminator_omic1(generated_omic1.detach()) 
            d_loss_omic1 = discriminator_loss(real_validity_omic1, fake_validity_omic1)
            d_loss_omic1.backward()
            optimizer_D_omic1.step()

            optimizer_D_omic2.zero_grad()
            real_validity_omic2 = discriminator_omic2(omic2_batch)  
            generated_omic2 = vae_omic2.decoder(vae_omic1.encoder(omic1_batch)[0])  
            fake_validity_omic2 = discriminator_omic2(generated_omic2.detach())  
            d_loss_omic2 = discriminator_loss(real_validity_omic2, fake_validity_omic2)
            d_loss_omic2.backward()
            optimizer_D_omic2.step()


            
            optimizer_E.zero_grad()
            optimizer_G.zero_grad()
            
            z_omic1_mean, z_omic1_logvar = vae_omic1.encoder(omic1_batch)
            z_omic1 = vae_omic1.reparameterize(z_omic1_mean, z_omic1_logvar)
            omic1_reconstructed = vae_omic1.decoder(z_omic1)
            vae_loss_omic1 = vae_loss(omic1_reconstructed, omic1_batch, z_omic1_mean, z_omic1_logvar)
            
            z_omic2_mean, z_omic2_logvar = vae_omic2.encoder(omic2_batch)
            z_omic2 = vae_omic2.reparameterize(z_omic2_mean, z_omic2_logvar)
            omic2_reconstructed = vae_omic2.decoder(z_omic2)
            vae_loss_omic2 = vae_loss(omic2_reconstructed, omic2_batch, z_omic2_mean, z_omic2_logvar)
            

            # Cycle Consistency Loss

            omic1_to_omic2 = vae_omic2.decoder(z_omic1)
            omic2_to_omic1 = vae_omic1.decoder(z_omic2)
            
            
            cycle_omic1 = vae_omic1.decoder(vae_omic2.encoder(omic1_to_omic2)[0])
            cycle_omic2 = vae_omic2.decoder(vae_omic1.encoder(omic2_to_omic1)[0])
            
            loss_cycle = cycle_consistency_loss(omic1_batch, cycle_omic1) + cycle_consistency_loss(omic2_batch, cycle_omic2)
            
            # Contrastive Loss 

            loss_contrastive_z1 = contrastive_loss_separate(z_omic1, labels1_batch)
            loss_contrastive_z2 = contrastive_loss_separate(z_omic2, labels2_batch)

            loss_contrastive = loss_contrastive_z1 + loss_contrastive_z2

            
            # Generator Loss to fool discriminators
            g_loss_modality = generator_loss(discriminator_modality(z_omic1)) + generator_loss(discriminator_modality(z_omic2))
            g_loss_omic1 = generator_loss(discriminator_omic1(generated_omic1))
            g_loss_omic2 = generator_loss(discriminator_omic2(generated_omic2))

            
            # Total Loss
            total_loss = (vae_loss_omic1 + vae_loss_omic2 + loss_cycle + loss_contrastive + g_loss_modality + g_loss_omic1 + g_loss_omic2)
            
            total_loss.backward()
            optimizer_E.step()
            optimizer_G.step()
            
            epoch_total_loss += total_loss.item()

        epoch_losses.append(epoch_total_loss)

    
    print("Training complete.")

    # Save the trained models
    torch.save(vae_omic1.state_dict(), path1)
    torch.save(vae_omic2.state_dict(), path2)


    print("Models saved.")

    plt.figure(figsize=(10, 6)) 
    plt.plot(range(1, num_epochs + 1), epoch_losses, label='Total Loss')  # num_epochs + 1
   # plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()






#################################  Gene-DM Translation ################################################



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

omic1 = pd.read_excel('omic1.xlsx')
omic2 = pd.read_excel('omic2.xlsx')


X_gene = omic1.iloc[:, 1:-1].values
X_dm = omic2.iloc[:, 1:-1].values

scaler_gene = StandardScaler()
scaler_dm = StandardScaler()
X_gene = scaler_gene.fit_transform(X_gene)
X_dm = scaler_dm.fit_transform(X_dm)

labels_gene = omic1.iloc[:, -1].values  
labels_dna = omic2.iloc[:, -1].values  

latent_dim = 64
hidden_dims = [1024, 512, 256, 128]

vae_gene = VAE(input_dim=X_gene.shape[1], hidden_dims=hidden_dims, latent_dim=latent_dim).to(device)
vae_dm = VAE(input_dim=X_dm.shape[1], hidden_dims=hidden_dims, latent_dim=latent_dim).to(device)

discriminator_modality = Discriminator(latent_dim=latent_dim, hidden_dim=128).to(device)
discriminator_gene = Discriminator(latent_dim=X_gene.shape[1], hidden_dim=128).to(device)
discriminator_dm = Discriminator(latent_dim=X_dm.shape[1], hidden_dim=128).to(device)


if len(X_gene) > len(X_dm):
    indices = np.random.choice(len(X_gene), size=len(X_dm), replace=False)
    X_gene = X_gene[indices]
    labels_gene = labels_gene[indices]
elif len(X_dm) > len(X_gene):
    indices = np.random.choice(len(X_dm), size=len(X_gene), replace=False)
    X_dm = X_dm[indices]
    labels_dna = labels_dna[indices]


train_model(
            path1='vae_gene_GD.pth',
            path2='vae_dm_GD.pth',   
             vae_omic1=vae_gene, 
            vae_omic2=vae_dm, 
            discriminator_modality=discriminator_modality, 
            discriminator_omic1=discriminator_gene, 
            discriminator_omic2=discriminator_dm, 
            omic1_data=X_gene, 
            omic2_data=X_dm, 
            labels_omic1=labels_gene, 
            labels_omic2=labels_dna, 
            num_epochs=3000, 
            learning_rate=1e-4, 
            batch_size=64)




