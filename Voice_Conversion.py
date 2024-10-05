import os
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

# Ensure torchaudio uses the correct backend
torchaudio.set_audio_backend("sox_io")

# Kiểm tra xem GPU có khả dụng không
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset để lấy các tệp âm thanh của giọng nói A và B
class VoiceDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform=None):
        self.source_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.mp3')]
        self.target_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.mp3')]
        self.transform = transform

    def __len__(self):
        return min(len(self.source_files), len(self.target_files))

    def __getitem__(self, idx):
        source_waveform, source_sr = torchaudio.load(self.source_files[idx])
        target_waveform, target_sr = torchaudio.load(self.target_files[idx])

        if self.transform:
            source_waveform = self.transform(source_waveform)
            target_waveform = self.transform(target_waveform)
        
        return source_waveform, target_waveform

# Load dataset
source_dir = "path_to_source_voice_A"
target_dir = "path_to_target_voice_B"
voice_dataset = VoiceDataset(source_dir, target_dir)
data_loader = DataLoader(voice_dataset, batch_size=4, shuffle=True)  # Tăng batch size

# Định nghĩa Generator cho mô hình CycleGAN với ResNet blocks
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
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            ResNetBlock(64),
            ResNetBlock(64),
            ResNetBlock(64)
        )
        self.final = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.final(x)
        return x

# Định nghĩa Discriminator với nhiều lớp hơn
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# Tạo các generator và discriminator
G_A2B = Generator().to(device)
G_B2A = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
adversarial_loss = nn.MSELoss().to(device)
cycle_loss = nn.L1Loss().to(device)

# Training loop
num_epochs = 200  # Tăng số lượng epoch
for epoch in range(num_epochs):
    for i, (source_waveform, target_waveform) in enumerate(data_loader):
        source_waveform = source_waveform.to(device)
        target_waveform = target_waveform.to(device)

        # Chuyển giọng A thành B
        fake_B = G_A2B(source_waveform)
        recon_A = G_B2A(fake_B)
        
        # Chuyển giọng B thành A
        fake_A = G_B2A(target_waveform)
        recon_B = G_A2B(fake_A)
        
        # Tính loss phân biệt
        loss_D_A = adversarial_loss(D_A(source_waveform), torch.ones_like(D_A(source_waveform))) + \
                adversarial_loss(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))
        loss_D_B = adversarial_loss(D_B(target_waveform), torch.ones_like(D_B(target_waveform))) + \
                adversarial_loss(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))
        
        # Tính loss chuyển đổi chu kỳ (Cycle Loss)
        loss_G = adversarial_loss(D_B(fake_B), torch.ones_like(D_B(fake_B))) + \
                adversarial_loss(D_A(fake_A), torch.ones_like(D_A(fake_A))) + \
                 cycle_loss(recon_A, source_waveform) * 10.0 + \
                 cycle_loss(recon_B, target_waveform) * 10.0
        
        # Cập nhật trọng số
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        optimizer_D_A.zero_grad()
        loss_D_A.backward()
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()
        loss_D_B.backward()
        optimizer_D_B.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss_G: {loss_G.item()}, Loss_D_A: {loss_D_A.item()}, Loss_D_B: {loss_D_B.item()}")
torch.save(G_A2B.state_dict(), "G_A2B.pth")
torch.save(G_B2A.state_dict(), "G_B2A.pth")

# Load mô hình đã huấn luyện
G_A2B.load_state_dict(torch.load("G_A2B.pth"))
G_A2B.eval()

# Chuyển đổi giọng nói của bạn (giọng nói A sang giọng nói B)
your_voice, fs = torchaudio.load("your_voice_A.mp3")
your_voice = your_voice.to(device)
converted_voice = G_A2B(your_voice)

# Lưu giọng nói đã chuyển đổi
torchaudio.save("your_converted_voice_B.wav", converted_voice.cpu(), fs)
