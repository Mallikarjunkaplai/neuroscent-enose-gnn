# Electronic Nose & OpenPOM Olfactory Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![DGL](https://img.shields.io/badge/DGL-2.0.0-FF6F00?style=for-the-badge)
![DeepChem](https://img.shields.io/badge/DeepChem-MPNN-2E7D32?style=for-the-badge)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-00897B?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random_Forest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenPOM](https://img.shields.io/badge/OpenPOM_Ensemble-0.89_ROC--AUC-6A1B9A?style=for-the-badge)

An end-to-end machine learning and computational chemistry pipeline that bridges **simulated hardware gas sensing** with **deep molecular graph learning**. The system emulates an **8-channel Metal-Oxide (MOX) electronic nose** to classify real-world aromas from electrical resistance vectors and passes their constituent molecules through a **10-model OpenPOM Message Passing Neural Network (MPNN) ensemble** to predict human perception across **138 olfactory descriptors**.

---

## End-to-End System Architecture

```text
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                         1. PHYSICAL SENSING SIMULATION                       â”‚
â”‚   Real-World Aroma Sample â”€â”€â–º 8-Channel MOX Gas Sensor Array (sensor.py)     â”‚
â”‚   â€¢ Extracts RDKit descriptors (MolWt, MolLogP, Oxygen & Sulfur atom counts) â”‚
â”‚   â€¢ Injects realistic Gaussian sensor noise (2%â€“8%) & humidity drift (0.8â€“1.2) â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                       â”‚ 8-Channel Electrical Resistance Vector
                                       â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                       2. SIGNAL CLASSIFICATION ENGINE                        â”‚
â”‚   100-Tree Random Forest Classifier (train_nose.py â”€â”€â–º nose_model.pkl)       â”‚
â”‚   â€¢ Trained on 1,200 multi-condition synthetic sniffs across 6 aroma classes â”‚
â”‚   â€¢ Achieves 100.00% test accuracy on held-out noisy sensor signatures       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                       â”‚ Predicted Aroma Identity
                                       â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                     3. MOLECULAR GRAPH OLFACTION ENGINE                      â”‚
â”‚   Constituent SMILES Lookup (samples.json) â”€â”€â–º Molecular Graph Featurizer    â”‚
â”‚   10-Fold OpenPOM MPNN Ensemble (smell_model.py / checkpoint2.pt)            â”‚
â”‚   â€¢ 134 atom features | 6 bond features | 5 message-passing hops | Set2Set   â”‚
â”‚   â€¢ Averages multi-label predictions across all 10 checkpoints (138 odors)   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## Core Technical Highlights

* **Hybrid Tabular + Graph Architecture:** Uses a lightweight **Random Forest** for fixed-dimension Euclidean sensor arrays (`8 channels`) while leveraging **Graph Neural Networks (MPNN)** for non-Euclidean molecular topologies where atom ordering is arbitrary.
* **Physicochemical Sensor Modeling (`sensor.py`):** Simulates cross-reactive metal-oxide semiconductor responses driven by molecular weight, lipophilicity (`MolLogP`), and polar/reactive heteroatoms (`O`, `S`), complete with environmental humidity scaling and sensor noise.
* **10-Checkpoint Ensemble Inference (`smell_model.py`):** Reconstructs the exact OpenPOM architecture (`node_out_feats=100`, `edge_out_feats=100`, `num_step_message_passing=5`, `ffn_hidden_list=[392, 392]`) on CPU and aggregates predictions across all 10 cross-validated model folds (`0.89 ROC-AUC` benchmark).
* **Zero-Hardware Local Reproducibility:** Runs completely offline on CPU without requiring physical microcontroller hardware or CUDA GPUs, while remaining modular for real Kaggle/UCI sensor CSVs and GPU fine-tuning.

---

## Repository Structure

| File / Directory | System Role | Technical Details |
| :--- | :--- | :--- |
| **`app.py`** | Integrated CLI Application | Interactive end-to-end pipeline connecting live sensor simulation, Random Forest aroma classification, SMILES retrieval, and GNN odor profiling. |
| **`sensor.py`** | 8-Channel MOX Sensor Array | RDKit-powered virtual gas sensor array translating molecular structures into 8-dimensional electrical response vectors. |
| **`train_nose.py`** | E-Nose Model Trainer | Generates 1,200 synthetic sniffs (200 per aroma across varied noise/humidity states) and exports the trained classifier to `nose_model.pkl`. |
| **`smell_model.py`** | OpenPOM GNN Inference Wrapper | Loads all 10 ensemble checkpoints from `models/ensemble_models/` on CPU and ranks the top human smell descriptors for multi-molecule mixtures. |
| **`samples.json`** | Aroma-to-SMILES Knowledge Base | Curated mapping of 6 real-world aromas (*Coffee, Vanilla Candle, Mint, Lemon, Banana, Wood Smoke*) to verified KekulÃ© SMILES strings. |
| **`nose_model.pkl`** | Pre-Trained E-Nose Classifier | Serialized Scikit-Learn Random Forest model ready for instant inference. |
| **`predict_odors.py`** | Standalone SMILES Predictor | Original OpenPOM Tkinter GUI and CLI utility for single-molecule odor prediction. |
| **`models/ensemble_models/`** | Pre-Trained GNN Weights (Git LFS) | 10 PyTorch ensemble checkpoints (`experiments_1..10/checkpoint2.pt`) trained on ~5,000 GoodScents & Leffingwell molecules. |
| **`examples/`** | Upstream Benchmark Notebooks | Reference notebooks for 10-fold cross-validation (`ensemble_cv.ipynb`), hyperparameter tuning, and GPU training. |

---

## Verified Benchmark & Sample Output

* **Electronic Nose Classifier (`train_nose.py`):** `100.00%` test accuracy across 240 held-out noisy and humidity-shifted test sniffs.
* **Live Pipeline Execution (`app.py` â€” Sample: Mint):**

```text
==================================================
  ELECTRONIC NOSE & OPENPOM INTEGRATED PIPELINE   
==================================================

Available aroma samples to sniff:
  [1] Coffee
  [2] Vanilla Candle
  [3] Mint
  [4] Lemon
  [5] Banana
  [6] Wood Smoke

Select a sample number to sniff (1-6) [Default: 5 - Banana]: 3

---> Sniffing 'Mint' with 8-channel MOX sensor array...

--- 8-Channel Sensor Signal Output ---
  Sensor 1: 10.824
  Sensor 2: 18.901
  Sensor 3: 6.929
  Sensor 4: 1.000
  Sensor 5: 3.526
  Sensor 6: 2.568
  Sensor 7: 14.573
  Sensor 8: 1.333

--- Electronic Nose Classifier Output ---
  Predicted Identity : Mint
  Confidence Score   : 100.0%

--- Running OpenPOM GNN Model on Constituent Molecules ---
  - CC1CCC(C(C)C)C(=O)C1
  - CC1=CCC(C(C)C)C(=O)C1
  - CC(=O)OC1CCC(C(C)C)CC1

--- Top Predicted Smell Descriptors for 'Mint' ---
  herbal         : 0.8374
  mint           : 0.7400
  sweet          : 0.6133
  fresh          : 0.6112
  camphoreous    : 0.5776
  woody          : 0.5589
  cooling        : 0.5524
  green          : 0.5084
  spicy          : 0.4972
  fruity         : 0.4696
==================================================
```

---

## Quickstart & Local Setup

### 1. Clone the Repository (with Git LFS for Model Checkpoints)
```bash
git lfs install
git clone [https://github.com/Mallikarjunkaplai/neuroscent-enose-gnn.git](https://github.com/Mallikarjunkaplai/neuroscent-enose-gnn.git)
cd neuroscent-enose-gnn
```

### 2. Create a Python 3.10 Virtual Environment & Install Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install torch==2.1.2 torchvision==0.16.2 --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install dgl==2.0.0 -f [https://data.dgl.ai/wheels/repo.html](https://data.dgl.ai/wheels/repo.html)
pip install deepchem rdkit scikit-learn numpy
pip install -e .
```

### 3. Train the Electronic Nose & Run the Integrated Pipeline
```powershell
python train_nose.py
python app.py
```

---

## Future Roadmap

* **Real-World Sensor Ingestion:** Plug in empirical MOX sensor drift logs (e.g., UCI Gas Sensor Array / Kaggle E-Nose datasets) directly into `train_nose.py`.
* **Batch Fragrance Screening:** Extend `smell_model.py` for high-throughput CSV screening of essential oil and perfumery SMILES libraries.
* **CUDA Fine-Tuning:** Fine-tune custom MPNN checkpoints on domain-specific sensory datasets using GPU acceleration.

---

## Author

**Mallikarjun Kaplai**  
B.Tech in Computer Science & Engineering (Data Science) â€” J.B. Institute of Engineering & Technology  
* **GitHub:** [@Mallikarjunkaplai](https://github.com/Mallikarjunkaplai)  
* **LinkedIn:** [linkedin.com/in/mallikarjun-kaplai-245259337](https://www.linkedin.com/in/mallikarjun-kaplai-245259337)
