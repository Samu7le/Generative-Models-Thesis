import os
import time
import copy
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

from models import DenoisingVAE, vae_loss_function, DDPM
from utils.degradations import apply_noise_levels
from utils.metrics import calculate_metrics
from utils.visualizations import plot_training_curves, plot_inference_results_comparative

def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in uso: {device}")

    # --- 1. SETUP DATASET ---
    print("Caricamento ChestMNIST...")
    info = INFO['chestmnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    data_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = DataClass(split='train', transform=data_transform, download=True, root='./data')
    test_dataset = DataClass(split='test', transform=data_transform, download=True, root='./data') # Aggiunto test set
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # --- 2. INIZIALIZZAZIONE MODELLI ---
    epochs = 100
    vae = DenoisingVAE().to(device)
    ddpm = DDPM(time_steps=200).to(device)
    
    optimizer_vae = optim.Adam(vae.parameters(), lr=1e-3)
    optimizer_ddpm = optim.Adam(ddpm.network.parameters(), lr=1e-3)

    loss_history = {"VAE": [], "DDPM": []}
    
    ddpm_20_weights = None
    vae_final_weights = None

    # --- 3. LOOP DI TRAINING ---
    print(f"\n{'='*50}")
    print(f" INIZIO ADDESTRAMENTO CONGIUNTO (Fase 1: 20 Epoche)")
    print(f"{'='*50}")
    
    for epoch in range(epochs):
        if epoch < 20:
            vae.train()
            running_loss_vae = 0.0
            
        ddpm.train()
        running_loss_ddpm = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            if epoch < 20:
                optimizer_vae.zero_grad()
                vae_noisy_img = apply_noise_levels(images, levels=[0.3])[0.3]
                recon_vae, mu, log_var = vae(vae_noisy_img)
                loss_v = vae_loss_function(recon_vae, images, mu, log_var)
                loss_v.backward()
                optimizer_vae.step()
                running_loss_vae += loss_v.item()

            optimizer_ddpm.zero_grad()
            t = torch.randint(0, ddpm.time_steps, (images.size(0),), device=device, dtype=torch.long)
            epsilon = torch.randn_like(images)
            ddpm_noisy_img = ddpm.add_noise(images, t, epsilon)
            predicted_epsilon = ddpm(ddpm_noisy_img, t)
            loss_d = torch.mean((epsilon - predicted_epsilon) ** 2)
            loss_d.backward()
            optimizer_ddpm.step()
            running_loss_ddpm += loss_d.item()

        loss_history["DDPM"].append(running_loss_ddpm / len(train_loader))
        if epoch < 20:
            loss_history["VAE"].append(running_loss_vae / len(train_loader))
            print(f"Epoca [{epoch+1:3d}/{epochs}] | Loss VAE: {loss_history['VAE'][-1]:.4f} | Loss DDPM: {loss_history['DDPM'][-1]:.4f}")
        else:
            print(f"Epoca [{epoch+1:3d}/{epochs}] | Loss DDPM: {loss_history['DDPM'][-1]:.4f}")

        if epoch + 1 == 20:
            vae_final_weights = copy.deepcopy(vae.state_dict())
            ddpm_20_weights = copy.deepcopy(ddpm.network.state_dict())
            torch.save(vae_final_weights, "results/vae_20_epochs.pth")
            torch.save(ddpm_20_weights, "results/ddpm_20_epochs.pth")
            print(f"\n{'*'*20} TRAGUARDO 20 EPOCHE {'*'*20}\n")

    ddpm_100_weights = copy.deepcopy(ddpm.network.state_dict())
    torch.save(ddpm_100_weights, "results/ddpm_100_epochs.pth")
    plot_training_curves(loss_history)

    # --- 4. VALUTAZIONE STATISTICA SU TUTTO IL TEST SET ---
    print("\nInizio valutazione statistica sul dataset di test...")
    noise_levels = [0.1, 0.3, 0.5]
    livello_a_timestep = {0.1: int(ddpm.time_steps * 0.15), 0.3: int(ddpm.time_steps * 0.40), 0.5: int(ddpm.time_steps * 0.70)}
    
    # Struttura per accumulare i dati: stats[modello][livello][metrica] = []
    model_stats = {
        "VAE_20": {lvl: {"MSE": [], "PSNR": [], "SSIM": [], "Time": []} for lvl in noise_levels},
        "DDPM_20": {lvl: {"MSE": [], "PSNR": [], "SSIM": [], "Time": []} for lvl in noise_levels},
        "DDPM_100": {lvl: {"MSE": [], "PSNR": [], "SSIM": [], "Time": []} for lvl in noise_levels}
    }

    # Helper per i test
    def run_eval_batch(model_key, weights, is_vae=False):
        if is_vae:
            vae.load_state_dict(weights)
            vae.eval()
        else:
            ddpm.network.load_state_dict(weights)
            ddpm.eval()

        with torch.no_grad():
            # Testiamo su 100 campioni per bilanciare precisione e tempo
            for i, (img, _) in enumerate(test_loader):
                if i >= 100: break 
                img = img.to(device)
                noisy_imgs = apply_noise_levels(img, noise_levels)

                for lvl in noise_levels:
                    start_t = time.perf_counter()
                    if is_vae:
                        recon, _, _ = vae(noisy_imgs[lvl])
                    else:
                        recon = ddpm.sample(noisy_imgs[lvl], start_t=livello_a_timestep[lvl])
                    
                    elapsed = time.perf_counter() - start_t
                    m = calculate_metrics(img, recon)
                    
                    model_stats[model_key][lvl]["MSE"].append(m["MSE"])
                    model_stats[model_key][lvl]["PSNR"].append(m["PSNR"])
                    model_stats[model_key][lvl]["SSIM"].append(m["SSIM"])
                    model_stats[model_key][lvl]["Time"].append(elapsed)

    print("Valutazione VAE 20 epoche...")
    run_eval_batch("VAE_20", vae_final_weights, is_vae=True)
    print("Valutazione DDPM 20 epoche...")
    run_eval_batch("DDPM_20", ddpm_20_weights, is_vae=False)
    print("Valutazione DDPM 100 epoche...")
    run_eval_batch("DDPM_100", ddpm_100_weights, is_vae=False)

    # --- 5. STAMPA E SALVATAGGIO TABELLA FINALE ---
    log_path = "results/final_metrics_table.txt"
    with open(log_path, "w") as f:
        header = f"{'Modello':<12} | {'Noise':<5} | {'MSE (avg)':<10} | {'PSNR (avg)':<10} | {'SSIM (avg)':<10} | {'Time (avg)':<10}"
        print(f"\n{header}")
        f.write(header + "\n" + "-"*80 + "\n")
        
        for m_key in model_stats:
            for lvl in noise_levels:
                res = model_stats[m_key][lvl]
                mse_m, mse_s = np.mean(res["MSE"]), np.std(res["MSE"])
                psnr_m, psnr_s = np.mean(res["PSNR"]), np.std(res["PSNR"])
                ssim_m, ssim_s = np.mean(res["SSIM"]), np.std(res["SSIM"])
                time_m = np.mean(res["Time"])
                
                row = f"{m_key:<12} | {lvl:<5.1f} | {mse_m:.4f}±{mse_s:.3f} | {psnr_m:.2f}±{psnr_s:.2f} | {ssim_m:.3f}±{ssim_s:.3f} | {time_m:.4f}s"
                print(row)
                f.write(row + "\n")

    # --- 6. PLOT QUALITATIVO (Singola Immagine come da richiesta originale) ---
    print("\nGenerazione plot qualitativo finale...")
    # Riassegna per l'ultimo plot qualitativo (logica originale intatta)
    test_img_vis = test_dataset[0][0].unsqueeze(0).to(device)
    noisy_vis = apply_noise_levels(test_img_vis, noise_levels)
    
    # Riutilizziamo le funzioni helper per estrarre i dati per il plot
    def get_single_inference(weights, is_vae=False):
        if is_vae: vae.load_state_dict(weights); vae.eval()
        else: ddpm.network.load_state_dict(weights); ddpm.eval()
        
        r, m, t = {}, {}, {}
        with torch.no_grad():
            for lvl in noise_levels:
                start = time.perf_counter()
                out = vae(noisy_vis[lvl])[0] if is_vae else ddpm.sample(noisy_vis[lvl], start_t=livello_a_timestep[lvl])
                t[lvl] = time.perf_counter() - start
                r[lvl] = out
                m[lvl] = calculate_metrics(test_img_vis, out)
        return r, m, t

    rv, mv, tv = get_single_inference(vae_final_weights, is_vae=True)
    rd20, md20, td20 = get_single_inference(ddpm_20_weights, is_vae=False)
    rd100, md100, td100 = get_single_inference(ddpm_100_weights, is_vae=False)

    plot_inference_results_comparative(test_img_vis, noisy_vis, rv, rd20, rd100, mv, md20, md100, tv, td20, td100)
    
    print(f"\nEsperimento completo! Tabella salvata in: {log_path}")

if __name__ == "__main__":
    main()