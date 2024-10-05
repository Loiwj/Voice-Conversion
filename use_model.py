# use_model.py
import torchaudio
import torch
import os
from model import Generator  # Assuming you will create a model.py for the Generator class

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_generator(generator_path, device):
    generator = Generator().to(device)  # Initialize the Generator model
    state_dict = torch.load(generator_path, map_location=device)

    # Remove prefix "module." if using DataParallel
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Ensure the state_dict keys match the model's keys
    model_state_dict = generator.state_dict()
    new_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state_dict and model_state_dict[key].shape == value.shape:
            new_state_dict[key] = value

    generator.load_state_dict(new_state_dict, strict=False)
    generator.eval()
    return generator

# Define paths for trained models
G_A2B_path = "/kaggle/working/Voice-Conversion/G_A2B.pth"
G_B2A_path = "/kaggle/working/Voice-Conversion/G_B2A.pth"

# Load trained models
G_A2B = load_trained_generator(G_A2B_path, device)  # Voice A to B
G_B2A = load_trained_generator(G_B2A_path, device)  # Voice B to A

# Convert voice A to B (assuming you have a voice to convert)
print("Loading your voice for conversion...")
your_voice, fs = torchaudio.load("/kaggle/working/Voice-Conversion/A.wav")  # Your voice to convert
# Check if the WAV file exists
wav_path = "/kaggle/working/Voice-Conversion/A.wav"
mp3_path = "/kaggle/working/Voice-Conversion/A.mp3"

if not os.path.exists(wav_path):
    if os.path.exists(mp3_path):
        print("WAV file not found. Converting MP3 to WAV...")
        your_voice, fs = torchaudio.load(mp3_path)
        torchaudio.save(wav_path, your_voice, fs)
    else:
        raise FileNotFoundError("Neither WAV nor MP3 file found for conversion.")
else:
    your_voice, fs = torchaudio.load(wav_path)

# Convert from stereo (2 channels) to mono (1 channel) if necessary
if your_voice.shape[0] == 2:  # If stereo
    your_voice = torch.mean(your_voice, dim=0, keepdim=True)  # Convert to mono

your_voice = your_voice.to(device)
print("Converting voice A to B...")
converted_voice = G_A2B(your_voice)

# Save the converted voice
torchaudio.save("your_converted_voice_B.wav", converted_voice.cpu(), fs)
print("Voice conversion completed and saved as 'your_converted_voice_B.wav'.")

