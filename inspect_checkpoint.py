import torch

# Path ke file checkpoint kamu
checkpoint_path = "saved_models/20250427_195123/best_model.pth"  # Sesuaikan kalau mau cek file lain

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Cek keys di dalam checkpoint
print("Keys di checkpoint:")
print(checkpoint.keys())

# Kalau ada 'state_dict', cek isinya
if 'state_dict' in checkpoint:
    print("\nIsi state_dict:")
    for key in checkpoint['state_dict'].keys():
        print(key)
else:
    print("\nLangsung isi model:")
    for key in checkpoint.keys():
        print(key)
