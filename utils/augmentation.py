import nlpaug.augmenter.audio as naa

def augment_audio(audio):
    aug = naa.NoiseAug(noise_factor=0.02)
    return aug.augment(audio)