import torch
from models.enhancer import SpeechEnhancer
from utils.audio_processing import load_audio, save_audio

def enhance_audio(input_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeechEnhancer().to(device)
    model.load_state_dict(torch.load("models/enhancer.pth"))
    model.eval()
    
    audio = load_audio(input_path)
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        enhanced = model(audio_tensor)
    
    enhanced = enhanced.squeeze().cpu().numpy()
    save_audio(output_path, enhanced)