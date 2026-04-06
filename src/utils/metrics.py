import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(original, reconstructed):
    """Calcola MSE, PSNR e SSIM tra due tensori PyTorch (1 immagine)."""
    # Convertiamo i tensori in array NumPy 2D
    orig_np = original.cpu().detach().numpy().squeeze()
    recon_np = reconstructed.cpu().detach().numpy().squeeze()

    # MSE (con skikit-image o numpy)
    mse_val = np.mean((orig_np - recon_np) ** 2)
    # PSNR (con skikit-image, data_range è 1.0)
    psnr_val = psnr(orig_np, recon_np, data_range=1.0)
    # SSIM (con skikit-image, data_range è 1.0)
    ssim_val = ssim(orig_np, recon_np, data_range=1.0)

    return {"MSE": mse_val, "PSNR": psnr_val, "SSIM": ssim_val}