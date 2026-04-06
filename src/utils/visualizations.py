import matplotlib.pyplot as plt

def plot_training_curves(losses_dict, save_path="results/training_curves.png"):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('Loss VAE', color=color1)
    # Creiamo l'asse X dinamicamente in base alla lunghezza della history
    x_vae = range(1, len(losses_dict['VAE']) + 1)
    line1, = ax1.plot(x_vae, losses_dict['VAE'], color=color1, label='VAE Loss (20 Epoche)')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  
    color2 = 'tab:orange'
    ax2.set_ylabel('Loss DDPM', color=color2)
    x_ddpm = range(1, len(losses_dict['DDPM']) + 1)
    line2, = ax2.plot(x_ddpm, losses_dict['DDPM'], color=color2, label='DDPM Loss (100 Epoche)')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Training Loss per Modello')
    fig.legend(handles=[line1, line2], loc="upper right", bbox_to_anchor=(0.9, 0.88))
    ax1.grid(True)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_inference_results_comparative(original, noisy_dict, 
                                       recon_vae, recon_diff_20, recon_diff_100, 
                                       metrics_vae, metrics_diff_20, metrics_diff_100, 
                                       times_vae, times_diff_20, times_diff_100, 
                                       save_path="results/inference_comparativa.png"):
    levels = list(noisy_dict.keys())
    # Riduciamo a 5 righe
    fig, axes = plt.subplots(nrows=5, ncols=len(levels), figsize=(14, 16))
    
    for i, level in enumerate(levels):
        # Riga 0: Originale
        axes[0, i].imshow(original.cpu().squeeze(), cmap='gray')
        axes[0, i].set_title("Originale")
        axes[0, i].axis('off')

        # Riga 1: Rumore
        axes[1, i].imshow(noisy_dict[level].cpu().squeeze(), cmap='gray')
        axes[1, i].set_title(f"Rumore (Livello {level})")
        axes[1, i].axis('off')

        # Riga 2: VAE (20 Epoche, definitiva)
        axes[2, i].imshow(recon_vae[level].cpu().squeeze(), cmap='gray')
        mv = metrics_vae[level]
        axes[2, i].set_title(f"VAE (20 Epoche)\nMSE:{mv['MSE']:.3f}|SSIM:{mv['SSIM']:.2f}\nTempo: {times_vae[level]:.4f}s", fontsize=9)
        axes[2, i].axis('off')

        # Riga 3: DDPM (20 Epoche)
        axes[3, i].imshow(recon_diff_20[level].cpu().squeeze(), cmap='gray')
        md20 = metrics_diff_20[level]
        axes[3, i].set_title(f"DDPM (20 Epoche)\nMSE:{md20['MSE']:.3f}|SSIM:{md20['SSIM']:.2f}\nTempo: {times_diff_20[level]:.4f}s", fontsize=9)
        axes[3, i].axis('off')

        # Riga 4: DDPM (100 Epoche)
        axes[4, i].imshow(recon_diff_100[level].cpu().squeeze(), cmap='gray')
        md100 = metrics_diff_100[level]
        axes[4, i].set_title(f"DDPM (100 Epoche)\nMSE:{md100['MSE']:.3f}|SSIM:{md100['SSIM']:.2f}\nTempo: {times_diff_100[level]:.4f}s", fontsize=9)
        axes[4, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()