import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt

# Assicurati di importare le tue classi originali
from models import DenoisingVAE, DDPM
from utils.metrics import calculate_metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in uso: {device}")

    # --- 1. SETUP DATASET (Solo Test Set) ---
    print("Caricamento ChestMNIST (Test Set)...")
    info = INFO['chestmnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    data_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = DataClass(split='test', transform=data_transform, download=True, root='./data') 
    
    # Batch size 1 per valutazione individuale
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # --- 2. INIZIALIZZAZIONE MODELLI E CARICAMENTO PESI ---
    print("Inizializzazione architetture...")
    vae = DenoisingVAE().to(device)
    ddpm = DDPM(time_steps=200).to(device)
    
    print("Caricamento pesi pre-addestrati...")
    try:
        vae.load_state_dict(torch.load("results/vae_20_epochs.pth", map_location=device))
        ddpm_20_weights = torch.load("results/ddpm_20_epochs.pth", map_location=device)
        ddpm_100_weights = torch.load("results/ddpm_100_epochs.pth", map_location=device)
        print("Pesi caricati con successo!")
    except FileNotFoundError as e:
        print(f"ERRORE: Impossibile trovare i file dei pesi in 'results/'. {e}")
        return

    # --- 3. VALUTAZIONE STATISTICA (FEDELTÀ INPUT PULITO) ---
    print("\nInizio valutazione statistica della Fedeltà (Input Pulito)...")
    
    model_stats = {
        "VAE_20": {"MSE": [], "PSNR": [], "SSIM": [], "Time": []},
        "DDPM_20": {"MSE": [], "PSNR": [], "SSIM": [], "Time": []},
        "DDPM_100": {"MSE": [], "PSNR": [], "SSIM": [], "Time": []}
    }

    # Numero di campioni da testare (es. 100 per consistenza statistica)
    num_samples = 100

    def evaluate_model(model_key, weights, is_vae=False):
        if is_vae:
            vae.load_state_dict(weights)
            vae.eval()
        else:
            ddpm.network.load_state_dict(weights)
            ddpm.eval()

        with torch.no_grad():
            for i, (img, _) in enumerate(test_loader):
                if i >= num_samples: break 
                img = img.to(device)

                start_t = time.perf_counter()
                if is_vae:
                    # Passaggio diretto nell'autoencoder (niente rumore)
                    recon, _, _ = vae(img)
                else:
                    # DDPM: partenza dal timestep 0 (nessun rumore iniziale, solo auto-codifica)
                    recon = ddpm.sample(img, start_t=0)
                
                elapsed = time.perf_counter() - start_t
                m = calculate_metrics(img, recon)
                
                model_stats[model_key]["MSE"].append(m["MSE"])
                model_stats[model_key]["PSNR"].append(m["PSNR"])
                model_stats[model_key]["SSIM"].append(m["SSIM"])
                model_stats[model_key]["Time"].append(elapsed)

    print("Valutazione VAE (20 epoche)...")
    evaluate_model("VAE_20", vae.state_dict(), is_vae=True) # Usa lo state_dict già caricato
    
    print("Valutazione DDPM (20 epoche)...")
    evaluate_model("DDPM_20", ddpm_20_weights, is_vae=False)
    
    print("Valutazione DDPM (100 epoche)...")
    evaluate_model("DDPM_100", ddpm_100_weights, is_vae=False)

    # --- 4. STAMPA TABELLA FEDELTÀ ---
    print("\n" + "="*70)
    print("TABELLA 5.1: FEDELTA' DI RICOSTRUZIONE (Input Pulito)")
    print("="*70)
    header = f"{'Modello':<12} | {'MSE (avg)':<10} | {'PSNR (avg)':<10} | {'SSIM (avg)':<10} | {'Time (avg)':<10}"
    print(header)
    print("-" * 70)
    
    for m_key in model_stats:
        res = model_stats[m_key]
        mse_m, mse_s = np.mean(res["MSE"]), np.std(res["MSE"])
        psnr_m, psnr_s = np.mean(res["PSNR"]), np.std(res["PSNR"])
        ssim_m, ssim_s = np.mean(res["SSIM"]), np.std(res["SSIM"])
        time_m = np.mean(res["Time"])
        
        row = f"{m_key:<12} | {mse_m:.4f}±{mse_s:.3f} | {psnr_m:.2f}±{psnr_s:.2f} | {ssim_m:.3f}±{ssim_s:.3f} | {time_m:.4f}s"
        print(row)

    # --- 5. PLOT QUALITATIVO (Input Pulito) ---
    print("\nGenerazione plot visivo comparativo...")
    
    # Prendiamo la prima immagine del test set per il plot
    test_img = test_dataset[0][0].unsqueeze(0).to(device)

    # Eseguiamo le 3 inferenze
    with torch.no_grad():
        # VAE
        vae.eval()
        recon_vae, _, _ = vae(test_img)
        
        # DDPM 20
        ddpm.network.load_state_dict(ddpm_20_weights)
        ddpm.eval()
        recon_ddpm_20 = ddpm.sample(test_img, start_t=0)
        
        # DDPM 100
        ddpm.network.load_state_dict(ddpm_100_weights)
        ddpm.eval()
        recon_ddpm_100 = ddpm.sample(test_img, start_t=0)

    # Trasferimento su CPU per matplotlib
    orig_np = test_img.cpu().squeeze().numpy()
    vae_np = recon_vae.cpu().squeeze().numpy()
    ddpm20_np = recon_ddpm_20.cpu().squeeze().numpy()
    ddpm100_np = recon_ddpm_100.cpu().squeeze().numpy()

    # Creazione della griglia 1x4
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    images = [orig_np, vae_np, ddpm20_np, ddpm100_np]
    titles = ['Originale (Pulito)', 'Ricostruzione VAE (20 Epoche)', 
              'Ricostruzione DDPM (20 Epoche)', 'Ricostruzione DDPM (100 Epoche)']

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join("results", "clean_inference.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Immagine salvata in: {save_path}")
    
    # Se sei su un notebook o vuoi vederla a schermo
    # plt.show() 

if __name__ == "__main__":
    main()