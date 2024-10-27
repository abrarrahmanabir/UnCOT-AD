
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef



# Encoder
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

# Decoder
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
        x_recon = self.decoder(z)
        return x_recon

# Variational Autoencoder (VAE)
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



def generate_data(vae_source, vae_target, source_data, device):
    vae_source.eval()
    vae_target.eval()
    
    source_data = source_data.to(device)
    
    with torch.no_grad():
        # Encode the source data
        source_mean, source_logvar = vae_source.encoder(source_data)
        z = vae_source.reparameterize(source_mean, source_logvar)
        
        # Decode into the target modality
        translated_target_data = vae_target.decoder(z)
    
    return translated_target_data



class OmicsFusionModel(nn.Module):
    def __init__(self, input_dim_gene, input_dim_dna, projected_dim):
        super(OmicsFusionModel, self).__init__()
        self.gene_project = nn.Linear(input_dim_gene, projected_dim)
        self.dm_project = nn.Linear(input_dim_dna, projected_dim)
        
        self.m1 = nn.Parameter(torch.ones(projected_dim))  # Learnable vector for gene data
        self.m2 = nn.Parameter(torch.ones(projected_dim))  # Learnable vector for methylation data
        
    def forward(self, gene_data, dm_data):
        gene_projected = self.gene_project(gene_data)
        dm_projected = self.dm_project(dm_data)
        
        weighted_gene = gene_projected * self.m1
        weighted_dm = dm_projected * self.m2
        
        fused_representation = weighted_gene + weighted_dm
        return fused_representation


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Classifier, self).__init__()
        
        # Create hidden layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i])) 
        
        layers.append(nn.Linear(hidden_dims[-1], 1))  
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.network(x)
        return torch.sigmoid(x) 

    


def train_and_evaluate(X1, X2, y, model, classifier, num_epochs=100, learning_rate=0.0001):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize lists to store metrics for all folds
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_mcc_scores = []

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for train_index, val_index in skf.split(X1, y):



        X1_train, X1_val = X1[train_index], X1[val_index]
        X2_train, X2_val = X2[train_index], X2[val_index]
        y_train, y_val = y[train_index], y[val_index]
        

        if isinstance(X1_train, np.ndarray):
            X1_train_tensor = torch.from_numpy(X1_train).float().to(device)
            X1_val_tensor = torch.from_numpy(X1_val).float().to(device)
        elif isinstance(X1_train, torch.Tensor):
            X1_train_tensor = X1_train.float().to(device)
            X1_val_tensor = X1_val.float().to(device)
        else:
            raise TypeError("X1_train should be either a numpy.ndarray or torch.Tensor")
        
        # Generalized conversion to tensor for y_train and y_val
        if isinstance(y_train, np.ndarray):
            y_train_tensor = torch.from_numpy(y_train).float().to(device)
            y_val_tensor = torch.from_numpy(y_val).float().to(device)
        elif isinstance(y_train, torch.Tensor):
            y_train_tensor = y_train.float().to(device)
            y_val_tensor = y_val.float().to(device)
        else:
            raise TypeError("y_train should be either a numpy.ndarray or torch.Tensor")
        
        # Generalized conversion to tensor for X2_train and X2_val
        if isinstance(X2_train, np.ndarray):
            X2_train_tensor = torch.from_numpy(X2_train).float().to(device)
            X2_val_tensor = torch.from_numpy(X2_val).float().to(device)
        elif isinstance(X2_train, torch.Tensor):
            X2_train_tensor = X2_train.float().to(device)
            X2_val_tensor = X2_val.float().to(device)
        else:
            raise TypeError("X2_train should be either a numpy.ndarray or torch.Tensor")
        


        model.to(device)
        classifier.to(device)
        
        optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            classifier.train()
            
            optimizer.zero_grad()
            fused_data = model(X1_train_tensor, X2_train_tensor)
            outputs = classifier(fused_data)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        classifier.eval()

        with torch.no_grad():
            fused_val_data = model(X1_val_tensor, X2_val_tensor)
            val_outputs = classifier(fused_val_data).squeeze()

            predicted = (val_outputs > 0.5).long()
            
            accuracy = accuracy_score(y_val_tensor.cpu(), predicted.cpu())
            precision = precision_score(y_val_tensor.cpu(), predicted.cpu())
            recall = recall_score(y_val_tensor.cpu(), predicted.cpu())
            f1 = f1_score(y_val_tensor.cpu(), predicted.cpu())
            mcc = matthews_corrcoef(y_val_tensor.cpu(), predicted.cpu())  

            all_accuracies.append(accuracy)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)
            all_mcc_scores.append(mcc) 
    
    avg_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1_scores)
    avg_mcc = np.mean(all_mcc_scores)


    print(f"Average Accuracy across all folds: {avg_accuracy:.4f}")
    print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
    print(f"Average Precision across all folds: {avg_precision:.4f}")
    print(f"Average Recall across all folds: {avg_recall:.4f}")
    print(f"Average F1 Score across all folds: {avg_f1:.4f}")
    print(f"Average MCC across all folds: {avg_mcc:.4f}")


def train_and_evaluate_single_omics(X, y, classifier, num_epochs=30, learning_rate=0.001):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_mcc_scores = []

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        if isinstance(X_train, np.ndarray):
            X_train_tensor = torch.from_numpy(X_train).float().to(device)
            X_val_tensor = torch.from_numpy(X_val).float().to(device)
        elif isinstance(X_train, torch.Tensor):
            X_train_tensor = X_train.float().to(device)
            X_val_tensor = X_val.float().to(device)
        else:
            raise TypeError("X_train should be either a numpy.ndarray or torch.Tensor")
        
        if isinstance(y_train, np.ndarray):
            y_train_tensor = torch.from_numpy(y_train).float().to(device)
            y_val_tensor = torch.from_numpy(y_val).float().to(device)
        elif isinstance(y_train, torch.Tensor):
            y_train_tensor = y_train.float().to(device)
            y_val_tensor = y_val.float().to(device)
        else:
            raise TypeError("y_train should be either a numpy.ndarray or torch.Tensor")
        
        classifier.to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        for epoch in range(num_epochs):
            classifier.train()
            
            optimizer.zero_grad()
            outputs = classifier(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            
            loss.backward()
            optimizer.step()
        
        classifier.eval()
        
        with torch.no_grad():
            val_outputs = classifier(X_val_tensor).squeeze()
            
            predicted = (val_outputs > 0.5).long()
            
            accuracy = accuracy_score(y_val_tensor.cpu(), predicted.cpu())
            precision = precision_score(y_val_tensor.cpu(), predicted.cpu())
            recall = recall_score(y_val_tensor.cpu(), predicted.cpu())
            f1 = f1_score(y_val_tensor.cpu(), predicted.cpu())
            mcc = matthews_corrcoef(y_val_tensor.cpu(), predicted.cpu())  # MCC calculation

            all_accuracies.append(accuracy)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)
            all_mcc_scores.append(mcc)  
    avg_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1_scores)
    avg_mcc = np.mean(all_mcc_scores)

    


    print(f"Average Accuracy across all folds: {avg_accuracy:.4f}")
    print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
    print(f"Average Precision across all folds: {avg_precision:.4f}")
    print(f"Average Recall across all folds: {avg_recall:.4f}")
    print(f"Average F1 Score across all folds: {avg_f1:.4f}")
    print(f"Average MCC across all folds: {avg_mcc:.4f}")




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

dmdata = pd.read_excel('prot_data.xlsx')
genedata = pd.read_excel('genedata.xlsx')

X_gene = genedata.iloc[:, 1:-1].values
X_dm = dmdata.iloc[:, 1:-1].values

scaler_gene = StandardScaler()
scaler_dm = StandardScaler()
X_gene = scaler_gene.fit_transform(X_gene)
X_dm = scaler_dm.fit_transform(X_dm)

labels_gene = genedata.iloc[:, -1].values  
labels_dna = dmdata.iloc[:, -1].values  




print(f"Shape of DNA methylation data: {X_dm.shape}")
print(f"Shape of gene expression data: {X_gene.shape}")


input_dim_gene = X_gene.shape[1]
input_dim_methylation = X_dm.shape[1]
hidden_dims = [1024, 512, 256, 128]
latent_dim = 64
vae_gene = VAE(input_dim_gene, hidden_dims, latent_dim).to(device)
vae_methylation = VAE(input_dim_methylation, hidden_dims, latent_dim).to(device)
vae_gene.load_state_dict(torch.load('vae_gene_GP.pth'))
vae_methylation.load_state_dict(torch.load('vae_prot_GP.pth'))



vae_gene.eval()
vae_methylation.eval()
print("Models loaded for inference.")



print("\nEXPERIMENT : GENE + PREDICTED PROTEIN : \n")

predicted_dm_data = generate_data(vae_gene,vae_methylation, torch.tensor(X_gene,dtype=torch.float32),device)
fusion_model = OmicsFusionModel(input_dim_gene=X_gene.shape[1],input_dim_dna=X_dm.shape[1] ,projected_dim=128)
hidden_dims = [512, 256, 128, 64]
classifier = Classifier(input_dim=128, hidden_dims=hidden_dims)
train_and_evaluate(X_gene, predicted_dm_data, labels_gene, fusion_model, classifier,num_epochs=80)




print("\nEXPERIMENT : PROTEIN + PREDICTED GENE : \n")

predicted_gene_data = generate_data(vae_methylation, vae_gene, torch.tensor(X_dm, dtype=torch.float32), device)
fusion_model = OmicsFusionModel(input_dim_gene=predicted_gene_data.shape[1], input_dim_dna=X_dm.shape[1], projected_dim=128)
hidden_dims = [ 512, 256, 128, 64]
classifier = Classifier(input_dim=128, hidden_dims=hidden_dims)
train_and_evaluate(predicted_gene_data, X_dm, labels_dna, fusion_model, classifier,num_epochs=80,learning_rate=1e-3)


print("\nEXPERIMENT :  GENE : \n")

hidden_dims = [512, 256, 128, 64]
classifier = Classifier(input_dim=X_gene.shape[1], hidden_dims=hidden_dims)
train_and_evaluate_single_omics(X_gene, labels_gene, classifier)


print("\nEXPERIMENT : PROTEIN : \n")

hidden_dims = [512, 256, 128, 64]
classifier = Classifier(input_dim=X_dm.shape[1], hidden_dims=hidden_dims)
train_and_evaluate_single_omics(X_dm, labels_dna, classifier,num_epochs=10,learning_rate=1e-2)




def evaluate_cycle_reconstruction(vae_omic1, vae_omic2, X_omic1, X_omic2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_omic1_tensor = torch.tensor(X_omic1, dtype=torch.float32).to(device)
    X_omic2_tensor = torch.tensor(X_omic2, dtype=torch.float32).to(device)

    z_omic1_mean, _ = vae_omic1.encoder(X_omic1_tensor)
    z_omic2_mean, _ = vae_omic2.encoder(X_omic2_tensor)
    
    omic1_to_omic2 = vae_omic2.decoder(z_omic1_mean)
    omic2_to_omic1 = vae_omic1.decoder(z_omic2_mean)
    
    # # Debugging lines to check the shapes
    # print(f"Shape of z_omic1_mean: {z_omic1_mean.shape}")
    # print(f"Shape of z_omic2_mean: {z_omic2_mean.shape}")
    # print(f"Shape of omic1_to_omic2: {omic1_to_omic2.shape}")
    # print(f"Shape of omic2_to_omic1: {omic2_to_omic1.shape}")

    # Cycle through the opposite modality

    if omic1_to_omic2.shape[1] == X_omic2.shape[1]:
        omic1_cycle = vae_omic1.decoder(vae_omic2.encoder(omic1_to_omic2))
        reconstruction_error_omic1 = nn.functional.mse_loss(z_omic1_mean, omic1_cycle).item()
        print(f"Cycle reconstruction error (Omic1 -> Omic2 -> Omic1): {reconstruction_error_omic1}")
    else:
        print("Mismatch in Omic1 to Omic2 cycle dimensions.")

    if omic2_to_omic1.shape[1] == X_omic1.shape[1]:
        omic2_cycle =  vae_omic2.decoder(vae_omic1.encoder(omic2_to_omic1))
        reconstruction_error_omic2 = nn.functional.mse_loss(z_omic2_mean, omic2_cycle).item()
        print(f"Cycle reconstruction error (Omic2 -> Omic1 -> Omic2): {reconstruction_error_omic2}")
    else:
        print("Mismatch in Omic2 to Omic1 cycle dimensions.")

    return reconstruction_error_omic1, reconstruction_error_omic2


evaluate_cycle_reconstruction(vae_gene, vae_methylation, X_gene, X_dm)

from scipy.linalg import sqrtm

def calculate_fid(mu1, sigma1, mu2, sigma2):
    """ Calculate the FID score given mean and covariance of two distributions """
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def evaluate_fid(vae_omic1, vae_omic2, omic1_data, omic2_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_omic1.to(device)
    vae_omic2.to(device)

    vae_omic1.eval()
    vae_omic2.eval()
    with torch.no_grad():
        omic1_data_tensor = torch.tensor(omic1_data, dtype=torch.float32).to(device)
        omic2_data_tensor = torch.tensor(omic2_data, dtype=torch.float32).to(device)


        z_omic1_mean, _ = vae_omic1.encoder(omic1_data_tensor)
        predicted_omic2 = vae_omic2.decoder(z_omic1_mean)

        z_predicted_omic2_mean, _ = vae_omic2.encoder(predicted_omic2)
        z_real_omic2_mean, _ = vae_omic2.encoder(omic2_data_tensor)

        mu_predicted_omic2 = np.mean(z_predicted_omic2_mean.cpu().numpy(), axis=0)
        sigma_predicted_omic2 = np.cov(z_predicted_omic2_mean.cpu().numpy(), rowvar=False)

        mu_real_omic2 = np.mean(z_real_omic2_mean.cpu().numpy(), axis=0)
        sigma_real_omic2 = np.cov(z_real_omic2_mean.cpu().numpy(), rowvar=False)

        fid_score_omic1_to_omic2 = calculate_fid(mu_real_omic2, sigma_real_omic2, mu_predicted_omic2, sigma_predicted_omic2)
        print(f"FID Score from Omic1 to Omic2 in Latent Space: {fid_score_omic1_to_omic2:.4f}")


        z_omic2_mean, _ = vae_omic2.encoder(omic2_data_tensor)
        predicted_omic1 = vae_omic1.decoder(z_omic2_mean)

        z_predicted_omic1_mean, _ = vae_omic1.encoder(predicted_omic1)
        z_real_omic1_mean, _ = vae_omic1.encoder(omic1_data_tensor)

        mu_predicted_omic1 = np.mean(z_predicted_omic1_mean.cpu().numpy(), axis=0)
        sigma_predicted_omic1 = np.cov(z_predicted_omic1_mean.cpu().numpy(), rowvar=False)

        mu_real_omic1 = np.mean(z_real_omic1_mean.cpu().numpy(), axis=0)
        sigma_real_omic1 = np.cov(z_real_omic1_mean.cpu().numpy(), rowvar=False)

        fid_score_omic2_to_omic1 = calculate_fid(mu_real_omic1, sigma_real_omic1, mu_predicted_omic1, sigma_predicted_omic1)
        print(f"FID Score from Omic2 to Omic1 in Latent Space: {fid_score_omic2_to_omic1:.4f}")

    return fid_score_omic1_to_omic2, fid_score_omic2_to_omic1


evaluate_fid(vae_gene, vae_methylation, X_gene, X_dm)


################################################################
