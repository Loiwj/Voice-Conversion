# train_model.py
import os
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from pydub import AudioSegment

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to convert mp3 to wav
def convert_mp3_to_wav(mp3_file, output_wav):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(output_wav, format="wav")

# Dataset for voice files A and B
class VoiceDataset(Dataset):
    def __init__(self, source_file_mp3, target_file_mp3, transform=None):
        self.source_file_wav = source_file_mp3.replace(".mp3", ".wav")
        self.target_file_wav = target_file_mp3.replace(".mp3", ".wav")
        
        # Convert mp3 to wav if necessary
        if not os.path.exists(self.source_file_wav):
            convert_mp3_to_wav(source_file_mp3, self.source_file_wav)
        if not os.path.exists(self.target_file_wav):
            convert_mp3_to_wav(target_file_mp3, self.target_file_wav)
        
        self.transform = transform

    def __len__(self):
        return 1  # Only 1 pair of A and B files

    def __getitem__(self, _):
        # Load audio data and convert to mono
        source_waveform, _ = torchaudio.load(self.source_file_wav)
        target_waveform, _ = torchaudio.load(self.target_file_wav)

        if source_waveform.shape[0] == 2:  # If stereo
            source_waveform = torch.mean(source_waveform, dim=0, keepdim=True)  # Convert to mono
        if target_waveform.shape[0] == 2:  # If stereo
            target_waveform = torch.mean(target_waveform, dim=0, keepdim=True)  # Convert to mono

        return source_waveform, target_waveform

# Define Generator and Discriminator
class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            ResNetBlock(16),
            ResNetBlock(16),
        )
        self.final = nn.Sequential(
            nn.Conv1d(16, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.final(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(2, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
G_A2B = Generator().to(device)
G_B2A = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

# DataLoader
source_file_mp3 = "/kaggle/working/Voice-Conversion/A.mp3"  # Voice A
target_file_mp3 = "/kaggle/working/Voice-Conversion/B.mp3"  # Voice B

voice_dataset = VoiceDataset(source_file_mp3, target_file_mp3)
data_loader = DataLoader(voice_dataset, batch_size=1, shuffle=True)

# Optimizers
optimizer_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
adversarial_loss = nn.MSELoss().to(device)
cycle_loss = nn.L1Loss().to(device)

# Training loop
num_epochs = 200  
scaler = torch.cuda.amp.GradScaler()  # Mixed precision

for epoch in range(num_epochs):
    for i, (source_waveform, target_waveform) in enumerate(data_loader):
        source_waveform = source_waveform.to(device)
        target_waveform = target_waveform.to(device)

        with torch.cuda.amp.autocast():  # Mixed precision
            # Convert voice A to B
            fake_B = G_A2B(source_waveform)
            recon_A = G_B2A(fake_B)
            
            # Convert voice B to A
            fake_A = G_B2A(target_waveform)
            recon_B = G_A2B(fake_A)
            
            # Discriminator loss
            loss_D_A = adversarial_loss(D_A(source_waveform), torch.ones_like(D_A(source_waveform))) + \
                       adversarial_loss(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))
            loss_D_B = adversarial_loss(D_B(target_waveform), torch.ones_like(D_B(target_waveform))) + \
                       adversarial_loss(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))
            
            # Cycle loss
            loss_G = adversarial_loss(D_B(fake_B), torch.ones_like(D_B(fake_B))) + \
                     adversarial_loss(D_A(fake_A), torch.ones_like(D_A(fake_A))) + \
                     cycle_loss(recon_A, source_waveform) * 10.0 + \
                     cycle_loss(recon_B, target_waveform) * 10.0
        
        # Update weights
        optimizer_G.zero_grad()
        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.update()

        optimizer_D_A.zero_grad()
        scaler.scale(loss_D_A).backward()
        scaler.step(optimizer_D_A)
        scaler.update()

        optimizer_D_B.zero_grad()
        scaler.scale(loss_D_B).backward()
        scaler.step(optimizer_D_B)
        scaler.update()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss_G: {loss_G.item()}, Loss_D_A: {loss_D_A.item()}, Loss_D_B: {loss_D_B.item()}")

# Save models
torch.save(G_A2B.state_dict(), "G_A2B.pth")
torch.save(G_B2A.state_dict(), "G_B2A.pth")
