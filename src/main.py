import os
import time
import copy
import torch
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
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

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
        # Il VAE si allena solo fino all'epoca 20
        if epoch < 20:
            vae.train()
            running_loss_vae = 0.0
            
        ddpm.train()
        running_loss_ddpm = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            # Training VAE (limitato)
            if epoch < 20:
                optimizer_vae.zero_grad()
                vae_noisy_img = apply_noise_levels(images, levels=[0.3])[0.3]
                recon_vae, mu, log_var = vae(vae_noisy_img)
                loss_v = vae_loss_function(recon_vae, images, mu, log_var)
                loss_v.backward()
                optimizer_vae.step()
                running_loss_vae += loss_v.item()

            # Training DDPM (completo)
            optimizer_ddpm.zero_grad()
            t = torch.randint(0, ddpm.time_steps, (images.size(0),), device=device, dtype=torch.long)
            epsilon = torch.randn_like(images)
            ddpm_noisy_img = ddpm.add_noise(images, t, epsilon)
            predicted_epsilon = ddpm(ddpm_noisy_img, t)
            loss_d = torch.mean((epsilon - predicted_epsilon) ** 2)
            loss_d.backward()
            optimizer_ddpm.step()
            running_loss_ddpm += loss_d.item()

        # --- GESTIONE DEI LOG E DEI SALVATAGGI ---
        loss_history["DDPM"].append(running_loss_ddpm / len(train_loader))

        if epoch < 20:
            loss_history["VAE"].append(running_loss_vae / len(train_loader))
            print(f"Epoca [{epoch+1:3d}/{epochs}] | Loss VAE: {loss_history['VAE'][-1]:.4f} | Loss DDPM: {loss_history['DDPM'][-1]:.4f}")
        else:
            print(f"Epoca [{epoch+1:3d}/{epochs}] | Loss DDPM: {loss_history['DDPM'][-1]:.4f}")

        # Traguardo 20 epoche
        if epoch + 1 == 20:
            print(f"\n{'-'*50}")
            print(" | 20 EPOCHE RAGGIUNTE |")
            print(" Salvataggio dello stato dei modelli su disco...")
            vae_final_weights = copy.deepcopy(vae.state_dict())
            ddpm_20_weights = copy.deepcopy(ddpm.network.state_dict())
            
            torch.save(vae_final_weights, "results/vae_20_epochs.pth")
            torch.save(ddpm_20_weights, "results/ddpm_20_epochs.pth")
            print(" VAE DISATTIVATO. Ha completato il suo addestramento.")
            print(f"{'-'*50}")
            print(f" INIZIO ADDESTRAMENTO ESCLUSIVO DDPM (Fase 2: 80 Epoche)")
            print(f"{'-'*50}\n")

    # Fine addestramento DDPM
    print(f"\n{'-'*50}")
    print(" | 100 EPOCHE RAGGIUNTE |")
    print(" Salvataggio dello stato finale del DDPM su disco...")
    ddpm_100_weights = copy.deepcopy(ddpm.network.state_dict())
    torch.save(ddpm_100_weights, "results/ddpm_100_epochs.pth")
    print(f"{'-'*50}\n")

    plot_training_curves(loss_history)

    # --- 4. INFERENZA COMPARATIVA ---
    print("\nInizio fase di inferenza e calcolo metriche...")
    test_image = train_dataset[0][0].unsqueeze(0).to(device)
    noise_levels = [0.1, 0.3, 0.5]
    noisy_images = apply_noise_levels(test_image, noise_levels)

    livello_a_timestep = {
        0.1: int(ddpm.time_steps * 0.15),
        0.3: int(ddpm.time_steps * 0.40),
        0.5: int(ddpm.time_steps * 0.70)
    }

    # Strutture dati per salvare i risultati
    recon_v, recon_d20, recon_d100 = {}, {}, {}
    met_v, met_d20, met_d100 = {}, {}, {}
    tempi_v, tempi_d20, tempi_d100 = {}, {}, {}

    # 4.1 Inferenza VAE (unico modello a 20 epoche)
    vae.load_state_dict(vae_final_weights)
    vae.eval()
    with torch.no_grad():
        for level in noise_levels:
            noisy_img = noisy_images[level]
            start_t = time.perf_counter()
            r_vae, _, _ = vae(noisy_img)
            tempi_v[level] = time.perf_counter() - start_t
            recon_v[level] = r_vae
            met_v[level] = calculate_metrics(test_image, r_vae)

    # Funzione helper per le due iterazioni del DDPM
    def inferenza_ddpm(pesi):
        ddpm.network.load_state_dict(pesi)
        ddpm.eval()
        r_d, m_d, t_d = {}, {}, {}
        with torch.no_grad():
            for level in noise_levels:
                noisy_img = noisy_images[level]
                start_t = time.perf_counter()
                start_step = livello_a_timestep[level]
                r_diff = ddpm.sample(noisy_img, start_t=start_step) 
                t_d[level] = time.perf_counter() - start_t
                r_d[level] = r_diff
                m_d[level] = calculate_metrics(test_image, r_diff)
        return r_d, m_d, t_d

    # 4.2 Inferenza DDPM a 20 e 100 epoche
    print("Eseguendo test DDPM a 20 epoche...")
    recon_d20, met_d20, tempi_d20 = inferenza_ddpm(ddpm_20_weights)

    print("Eseguendo test DDPM a 100 epoche...")
    recon_d100, met_d100, tempi_d100 = inferenza_ddpm(ddpm_100_weights)

    plot_inference_results_comparative(
        test_image, noisy_images, 
        recon_v, recon_d20, recon_d100, 
        met_v, met_d20, met_d100, 
        tempi_v, tempi_d20, tempi_d100
    )
    
    print("\nEsperimento completo! Grafici e pesi salvati in results/.")

if __name__ == "__main__":
    main()