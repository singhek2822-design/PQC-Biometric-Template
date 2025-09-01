
# GAN_attack.py
# Simulates GAN-based attacks on PQC-protected biometric templates.
# Generates synthetic biometric samples using a simple GAN (fingerprint/face)
# Attempts to match against protected templates
# Reports attack success rate and system robustness

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## Here I am trying to import PyTorch for the GAN demo.
## If PyTorch is not available, the code will just use numpy for random generation.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    print("PyTorch not available. GAN simulation will use numpy only.")

class SimpleGAN:
    """
    Minimal GAN for synthetic biometric generation (fingerprint/face)
    """
    def __init__(self, feature_dim=128, latent_dim=32, device='cpu'):
        # setting up the GAN architecture.
        # If PyTorch is available,  uses a simple generator and discriminator.
        # Otherwise, it just falls back to random noise for demo purposes.
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.device = device
        if torch:
            self.generator = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.ReLU(),
                nn.Linear(64, feature_dim)
            ).to(device)
            self.discriminator = nn.Sequential(
                nn.Linear(feature_dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Sigmoid()
            ).to(device)
            self.criterion = nn.BCELoss()
            self.optim_g = optim.Adam(self.generator.parameters(), lr=0.001)
            self.optim_d = optim.Adam(self.discriminator.parameters(), lr=0.001)
        else:
            # Fallback: random noise generator
            pass

    def train(self, real_data, epochs=100):
        # This function trains the GAN on the provided real data.
        # If PyTorch is not available, it just skips training and prints a message.
        if not torch:
            print("PyTorch not available. Skipping GAN training.")
            return
        real_data = torch.tensor(real_data, dtype=torch.float32).to(self.device)
        batch_size = min(32, real_data.shape[0])
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.choice(real_data.shape[0], batch_size)
            real_batch = real_data[idx]
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_batch = self.generator(z)
            d_real = self.discriminator(real_batch)
            d_fake = self.discriminator(fake_batch.detach())
            loss_d = self.criterion(d_real, torch.ones_like(d_real)) + \
                     self.criterion(d_fake, torch.zeros_like(d_fake))
            self.optim_d.zero_grad()
            loss_d.backward()
            self.optim_d.step()
            # Train generator
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_batch = self.generator(z)
            d_fake = self.discriminator(fake_batch)
            loss_g = self.criterion(d_fake, torch.ones_like(d_fake))
            self.optim_g.zero_grad()
            loss_g.backward()
            self.optim_g.step()
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: D_loss={loss_d.item():.4f}, G_loss={loss_g.item():.4f}")

    def generate(self, n_samples=10):
        # Here we are generating synthetic biometric samples using the GAN.
        # If PyTorch is not available, it just returns random noise samples.
        if torch:
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            samples = self.generator(z).detach().cpu().numpy()
        else:
            samples = np.random.normal(0, 1, (n_samples, self.feature_dim))
        return samples

## Attack Simulation 
def simulate_gan_attack(protected_templates, feature_dim=128, n_attack_samples=100):
    # Here it is simulating a GAN-based attack against PQC-protected templates.
    # The idea is to generate synthetic biometric samples and see if any of them
    # can match the protected templates above a certain similarity threshold.
    print("\n GAN-Based Attack Simulation ")
    # First,  prepares the GAN and train it on the protected templates.
    gan = SimpleGAN(feature_dim=feature_dim)
    gan.train(protected_templates, epochs=100)
    attack_samples = gan.generate(n_attack_samples)
    # For each attack sample, it computes the cosine similarity with all protected templates.
    # It keeps the best similarity for each attack sample.
    matches = []
    for atk in attack_samples:
        sims = [np.dot(atk, pt)/(np.linalg.norm(atk)*np.linalg.norm(pt)+1e-8) for pt in protected_templates]
        best_sim = max(sims)
        matches.append(best_sim)
    # Below function sets a threshold for what counts as a successful match.
    # For this demo, I use 0.8 as the threshold.
    threshold = 0.8
    success_count = sum([m > threshold for m in matches])
    success_rate = success_count / n_attack_samples
    print(f"Attack success rate: {success_rate:.2%} (threshold={threshold})")
    # Histogram of the cosine similarity scores for all attack samples.
    # This helps visualize how close the synthetic samples get to the real templates.
    # If called for PQC, color is blue; if called for classical, color is orange
    import inspect
    caller = inspect.stack()[1].function
    color = 'blue' if 'PQC' in caller or feature_dim == 256 else 'orange'
    label = 'PQC-protected' if color == 'blue' else 'Classical-protected'
    plt.hist(matches, bins=20, alpha=0.7, color=color, label=label)
    plt.title("GAN Attack: Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("gan_attack_histogram.png")
    print("Histogram saved as gan_attack_histogram.png")
    return success_rate, matches

if __name__ == "__main__":
    import os
    # PQC-protected templates
    template_path = "protected_templates.npy"
    if os.path.exists(template_path):
        print(f"\n--- PQC-Protected Templates ---")
        print(f"Loading PQC-protected templates from '{template_path}'...")
        pqc_templates = np.load(template_path)
        pqc_feature_dim = pqc_templates.shape[1]
        print(f"Loaded {pqc_templates.shape[0]} templates with {pqc_feature_dim} features each.")
        print("Running GAN attack on PQC-protected templates...")
        simulate_gan_attack(pqc_templates, feature_dim=pqc_feature_dim, n_attack_samples=100)
    else:
        print("No protected_templates.npy found. Skipping PQC attack.")

    # Classical-protected templates
    classical_files = [
        "classical_protected_templates_fingerprints.npy",
        "classical_protected_templates_faces_ATT.npy",
        "classical_protected_templates_faces_LFW.npy",
        "classical_protected_templates_NIST.npy"
    ]
    classical_templates = []
    for fname in classical_files:
        if os.path.exists(fname):
            arr = np.load(fname)
            classical_templates.append(arr)
            print(f"Loaded {arr.shape[0]} classical templates from '{fname}'.")
        else:
            print(f"File '{fname}' not found. Skipping.")
    if classical_templates:
        print(f"\n--- Classical-Protected Templates ---")
        protected_templates = np.concatenate(classical_templates, axis=0)
        feature_dim = protected_templates.shape[1]
        print(f"Total loaded classical templates: {protected_templates.shape[0]} with {feature_dim} features each.")
        print("Running GAN attack on classical-protected templates...")
        simulate_gan_attack(protected_templates, feature_dim=feature_dim, n_attack_samples=100)
    else:
        print("No classical protected template files found. Skipping classical attack.")
