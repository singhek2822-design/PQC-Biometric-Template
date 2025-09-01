# Post-Quantum Biometric Authentication System and Template Protection

This repository contains the full implementation for the dissertation project:

**“Quantum-Resilient Biometric Authentication: Secure Template Protection with Module-LWE Fuzzy Extractors”**

The system integrates biometric processing (fingerprint & face), Module-LWE fuzzy extractors, cancelable templates, and post-quantum cryptography (Kyber ML-KEM & Dilithium ML-DSA) into a robust authentication pipeline. It is benchmarked against a classical baseline (RSA, AES, SHA-256 BioHashing) for computational performance and resilience to adversarial attacks.

---

## Repository Structure

```
PQC-Biometric-Hardening-Project/
│── data/                     # Biometric datasets (place here after download)
│── logs/                     # Auto-generated experiment logs, CSVs, JSONs, PNGs
│── attack_simulation.py      # Adversarial testing modules (brute force, hill climbing, inversion, etc.)
│── GAN_attack.py             # GAN-based spoof generation and attack
│── tfhe_biometric_matcher.py # Homomorphic evaluation (simulated) (Helper file, not to be run)
│── main.py                   # Full PQC authentication pipeline
│── main_classical.py         # Classical baseline pipeline
│── Far_frr_AUC_pqc_metrics.py      # Clean metrics evaluation
│── metrics_PQCProtected_noisedata.py      # Noisy metrics evaluation
│── requirements.txt          # Library dependencies
│── README.md                 # Project documentation (this file)
```

---

## Development Environment

- **OS:** Windows 11 Pro (build 23H2)
- **Python:** 3.13.0
- **CPU:** Intel® i7-12700K (12c/20t, 3.6 GHz)
- **RAM:** 32 GB DDR5
- **GPU:** NVIDIA RTX 3060 Ti (unused in main experiments)

---

## Installation

Clone this repository:

```sh
git clone https://github.com/singhek2822-design/PQC-Biometric-Template.git
cd PQC-Biometric-Hardening-Project
```



## Execution Guide  

Follow these steps carefully to set up and run the project.  

### 1. Setup  

1. **Get the project files**  
   - Clone the GitHub repository (link above), or download the source code directory (link at bottom)  

2. **Prepare required folders**  
   - Create a folder named `logs/` inside the project root (`PQC-Biometric-Template`).  
   - Create a folder named `data/` and copy the biometric datasets from the Warwick OneDrive link into it.  
     - Expected structure:  
       ```
       data/Fingerprint-FVC/
       data/Faces-ATT/
       data/Faces-LFW/
       data/NIST-SD302/
       ```

3. **Install dependencies**  
   -Create and activate a virtual environment:

```sh
python -m venv pqc_env
# Linux/Mac:
source pqc_env/bin/activate
# Windows:
pqc_env\Scripts\activate
```

Install dependencies:

```sh
pip install -r requirements.txt
```

### 2. Running the Pipelines (in order)  

1. Run PQC Pipeline:  
       python main.py  

2. Evaluate PQC Metrics (clean data):  
       python Far_frr_AUC_pqc_evaluation.py  

3. Evaluate PQC Metrics (noisy data):  
       python metrics_PQCProtected_noisedata.py  

4. Run Classical Baseline:  
       python main_classical.py  

5. Run GAN Attack Simulation:  
       python GAN_attack.py  

6. Run Full Attack Simulations (last – takes ~1 hour):  
       python attack_simulation.py  

---

### 3. Important Notes  

- **Execution Time:** `attack_simulation.py` should be run last, as it is the most time-intensive (~1 hour).  
- **Common Error (ZeroDivisionError in `main.py`):**  
  If you encounter this error:  
  in `tfhe_biometric_matcher.py` (line 393), update the code around line 392 as follows:  

"""```python
# existing code...
if total_time > 0:
    print(f"  • Throughput: {len(templates)/total_time:.1f} templates/second")
else:
    print("  • Throughput: N/A (total_time is zero)")"""

- **Helper Scripts:** Files like logger_utils.py and tfhe_biometric_matcher.py are helper modules only. You do not need to run them separately.

- All logs, CSVs, JSONs, and plots are automatically stored in the logs/ and logs/evaluation directory with timestamps for reproducibility.

## Datasets

The following biometric datasets were used:
- **Fingerprints:** FVC2002, NIST SD302b
- **Faces:** LFW, Face-ATT

>  Place dataset files in the `/data/` folder before execution. Evaluation splits (development/test) are pre-defined and reproducible.

---

## Outputs

- **Logs:** Detailed traces per module (e.g., `logs/PQC_system.log`, `logs/classical_system.log`, `logs/attack_simulation.log`, `logs/gan_attack.log`,`logs/ckks_homomorphic_detailed_XX.log`, `logs/comprehensive_performance_evaluation.log`, `logs/module_lwe_detailed_analysis_XX.log`, `logs/quantum_safe_mfa_detailed.log`, `logs/evaluation/fixed_pqc_result_XX.log`, `logs/evaluation/fixed_summary_XX.log`)
- **CSVs:** Attack results (inversion, brute force, hill climbing) — e.g., `logs/attack_results.csv`,
- **PNGs:** Attack comparison plots & unlinkability histograms — e.g., `logs/attack_comparison.png`, `logs/gan_attack_histogram.png,`, `logs/evaluation/real_far_frr_analysis_xx.png`, `logs/evaluation/genuine_vs_imposter_xx.png`
- **JSONs:** FAR/FRR/EER/AUC metrics under clean + noisy conditions — e.g., `logs/metrics_clean.json`, `logs/metrics_noisy.json`, you will find these under logs-->evaluation
- **Checksums:** Each file is paired with a SHA-256 checksum for integrity verification.

---

## Key Files

- `main.py` — PQC-based biometric pipeline (feature extraction, Module-LWE fuzzy extractor, cancelable templates, PQC key exchange/signature, logging)
- `main_classical.py` — Classical baseline pipeline (feature extraction, BioHashing, RSA/AES, logging, MFA simulation, LaTeX table generation)
- `attack_simulation.py` — Advanced attack scenarios (noise, spoofing, adversarial attacks)
- `GAN_attack.py` — GAN-based adversarial attacks against templates
- `Far_frr_AUC-pqc_evaluation.py` - FAR, FRR, AUC computation for PQC-protected templates
- `metrics_PQCProtected_noisedata.py` — FAR, FRR, AUC computation for PQC-protected templates with noise 10%, 15% and 20%
- `tfhe_biometric_matcher.py` — Simulated homomorphic matching
- `requirements.txt` — All required Python packages
- `data/` — All biometric datasets (not included in repo)
- `logs/` — Output logs, CSVs, JSONs, PNGs

---

## Ethical Compliance

This project uses only secondary, public, and anonymised biometric datasets in accordance with research ethics guidelines. No new data collection from human participants was performed.

---

## Reproducibility Notes

- All random seeds are fixed across Python, NumPy, and cryptographic routines.
- Full environment specification is provided in `requirements.txt`.
- Configuration (hyperparameters, thresholds, dataset splits) is logged in JSON for exact reproducibility.

---

## External Links

- **GitHub Repo:** [https://github.com/singhek2822-design/PQC-Biometric-Template]
- **Data on Warwick OneDrive:** [https://livewarwickac-my.sharepoint.com/:u:/g/personal/u5647213_live_warwick_ac_uk/EaIcoF_YFcxEib5x9l1UxioBt9-f1YJkJ3JZnqflEOA6bQ?e=xaJuHt]

---

## Contact

For any questions, clarifications, or further details, please contact:

- Ekta Singh
- ekta.singh@warwick.ac.uk

---

Thank you for reviewing my work!

