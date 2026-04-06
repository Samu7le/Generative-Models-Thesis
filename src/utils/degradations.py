import torch

def add_gaussian_noise(image, noise_level):
    noise = torch.randn_like(image) * noise_level
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0., 1.)

def apply_noise_levels(image, levels=[0.1, 0.3, 0.5]):
    return {level: add_gaussian_noise(image, level) for level in levels}