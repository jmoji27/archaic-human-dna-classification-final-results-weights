# Classification of Archaic Human Genomics Using Deep Neural Networks

### Problem Definition
Modern Humans (Homo Sapiens) and Neanderthals (Homo neanderthalensis) diverged approximately 500,000 to 750,000 years ago. Subsequently, Neanderthals and Denisovans diverged around 381,000 to 473,000 years ago, both going extinct roughly 40,000 years ago. Because these hominin groups are evolutionarily closely related, distinguishing their short genomic sequences represents a challenging classification task.

Classification is heavily impacted by ancient DNA degradation (postmortem fragmentation, chemical modifications driven by hydrolytic cleavage and oxidation) and historical introgression. For instance, modern Eurasian populations contain 1–4% Neanderthal DNA, while Pacific Islanders and Southeast Asians carry up to 5–6% Denisovan DNA.

This project frames the problem in two ways:
1. **Binary Classification:** Deciding between two species (Human vs. Denisovan, Human vs. Neanderthal, or Denisovan vs. Neanderthal).
2. **Multiclass Classification:** Simultaneously distinguishing between Modern Human, Neanderthal, and Denisovan sequences.

---

## Model Architectures & Pipelines

All models process input DNA sequences by transforming nucleotide base pairs (A, C, G, T) into one-hot encoded vectors of dimension 4. Unknown bases (N) are encoded as a zero vector.

* A ➔ [1, 0, 0, 0]
* C ➔ [0, 1, 0, 0]
* G ➔ [0, 0, 1, 0]
* T ➔ [0, 0, 0, 1]
* N ➔ [0, 0, 0, 0]

### 1. Convolutional Neural Network (CNN)
Optimized for detecting short local sequence motifs without sequential modeling dependencies.
* **Feature Extraction:** Four 1D convolutional layers compressing channels ($128 \rightarrow 64 \rightarrow 64 \rightarrow 32$) with decreasing kernel sizes (19, 15, 5). Intermediate MaxPool is avoided to preserve fine spatial data on short sequences.
* **Variable Length Handling:** Parallel Global Max Pooling and Global Average Pooling concatenated into a fixed 64-dimensional vector.
* **Classifier Head:** Two dense layers ($64 \rightarrow 64 \rightarrow 256$) with high dropout rates to mitigate overfitting, evaluated using Focal Loss ($\gamma = 2.0$).

### 2. Recurrent Neural Network (BiLSTM)
Selected to capture long-range contextual relationships spanning across the sequence length.
* **Architecture:** Uses Bidirectional LSTM cells with three gating mechanisms (Forget, Input, Output gates) to prevent vanishing gradients.
* **Dynamic Capacity:** Automatically scales from 1 layer (64 units) for simpler binary tasks to 2 layers (128 units per direction) for complex tasks.

### 3. DanQ (Hybrid CNN-BiLSTM)
A hybrid architecture combining the localized feature extraction of a CNN with the long-range sequential reasoning of a BiLSTM.
* **Pipeline:** Two sequential 1D Convolution blocks extract local patterns, followed by a Max Pooling layer (size=2, stride=2) to reduce sequence length before feeding into a 1-layer BiLSTM.
* **Attention Pooling:** Normalizes spatial scores using a softmax mask to output a richer 256-dimensional summary vector to the final classification layer.

---

## Dataset Construction & Fragment Lengths

Data was filtered using a Mapping Quality score ($\text{MAPQ} \ge 30$) to keep a balanced threshold of data quality (1 in 1,000 alignment error probability).

### Fragment Length Distribution Across Species
Sequence-length distributions differed significantly across species due to differences in DNA preservation quality and sequencing coverage. Modern human reads show stable long lengths, Denisovan reads skew short, and Neanderthal reads rarely cross 85 bp.

<p align="center">
  <img src="results/denisovan_length_dist.png" width="30%" alt="Denisovan Fragment Length Distribution" />
  <img src="results/human_length_dist.png" width="30%" alt="Humans Fragment Length Distribution" />
  <img src="results/neanderthal_length_dist.png" width="30%" alt="Neanderthals Fragment Length Distribution" />
</p>

### Dataset Statistics

| Dataset Setting | Task Type | Total Samples (Train / Val / Test) | Class Distribution | Mean Length |
| :--- | :--- | :--- | :--- | :--- |
| **Human vs. Denisovan (Original)** | Binary | 2,019,025 / 148,815 / 50,906 | Human: ~59% <br> Denisovan: ~41% | ~52.5 bp |
| **Human vs. Denisovan (Longerbp)** | Binary | 47,825 / 4,902 / 1,889 | Human: ~51% <br> Denisovan: ~49% | ~48.0 bp |
| **Multiclass Dataset** | 3-Class | 72,786 / 7,403 / 2,837 | Human: 34% <br> Denisovan: 32% <br> Neanderthal: 34% | ~49.3 bp |
| **Human vs. Neanderthal** | Binary | 50,062 / 5,002 / 1,928 | Human: 50% <br> Neanderthal: 50% | ~51.5 bp |
| **Denisovan vs. Neanderthal** | Binary | 47,753 / 4,896 / 1,869 | Denisovan: ~48% <br> Neanderthal: ~52% | ~50.1 bp |

---

## Performance Results

### Model Accuracy Summary

| Dataset Setting | CNN Accuracy | BiLSTM Accuracy | DanQ Accuracy | Best Model |
| :--- | :---: | :---: | :---: | :---: |
| **Human vs. Denisovan** | **0.6420** | 0.4206 | 0.5367 | **CNN** |
| **Human vs. Neanderthal** | **0.7293** | 0.7132 | 0.7101 | **CNN** |
| **Denisovan vs. Neanderthal** | 0.7833 | 0.7833 | **0.7849** | **DanQ** |
| **Multiclass (All Three)** | 0.5731 | 0.5643 | **0.5791** | **DanQ** |
| **Bottleneck Experiment** | **0.6379** | 0.6374 | 0.6310 | **CNN** |
| **Longerbp Dataset** | 0.5894 | 0.5993 | **0.6057** | **DanQ** |

---

## Confusion Matrices & Evaluation

### 1. Human vs. Denisovan (Original) & Human vs. Neanderthal
* **Human vs. Denisovan (CNN):** Shows asymmetric performance. It correctly identifies 61.6% of Humans but misclassifies 38.4% as Denisovans, reflecting massive genomic similarity.
* **Human vs. Neanderthal (CNN):** Distinguishes Neanderthal sequences more reliably (77.4% recall) than Human sequences (68.5% recall).

<p align="center">
  <img src="results/cnn_original_confusion_matrix.png" width="45%" alt="CNN Original Confusion Matrix" />
  <img src="results/cnn_human_neanderthal_confusion_matrix.png" width="45%" alt="CNN Human vs Neanderthal Confusion Matrix" />
</p>

### 2. Denisovan vs. Neanderthal & Multiclass Classification
* **Denisovan vs. Neanderthal (DanQ):** This is the strongest binary result overall. Neanderthals achieve a very high recall of 91.9%, while Denisovan recall is lower at 64.2%.
* **Multiclass (DanQ):** Performs unevenly across classes. Neanderthals reach 85.5% recall, but Human (32.0%) and Denisovan (56.2%) sequences are highly confused.

<p align="center">
  <img src="results/danq_denisovan_neanderthal_confusion_matrix.png" width="45%" alt="DanQ Denisovan vs Neanderthal Confusion Matrix" />
  <img src="results/danq_multiclass_confusion_matrix.png" width="45%" alt="DanQ Multiclass Confusion Matrix" />
</p>

### 3. Bottleneck vs. Longer Base Pairs (Longerbp)
* **Bottleneck (CNN):** Near-chance performance on Denisovan sequences (52.5%), highlighting that the structural bottleneck limits feature extraction capacity.
* **Longerbp (DanQ):** Heavily biased toward the Human class (86.2% recall) while failing to leverage extra sequence length for Denisovans (27.5% recall).

<p align="center">
  <img src="results/cnn_bottleneck_confusion_matrix.png" width="45%" alt="CNN Bottleneck Confusion Matrix" />
  <img src="results/danq_longerbp_confusion_matrix.png" width="45%" alt="DanQ Longerbp Confusion Matrix" />
</p>

---

## Training Dynamics & Generalization

### 1. Loss & Accuracy Curves (Standard Implementations)
* **Human vs. Neanderthal (CNN):** Healthy generalization with clean convergence and a tiny validation loss gap.
* **Human vs. Denisovan (CNN):** Severe overfitting; validation loss plateaus high while validation accuracy experiences major spikes.
* **Multiclass (DanQ):** Progressive overfitting becomes visible from epoch 20 onward as the loss gap widens.

<p align="center">
  <img src="results/cnn_human_neanderthal_curves.png" width="90%" alt="CNN Human vs Neanderthal Loss and Accuracy" />
  <br/>
  <img src="results/cnn_original_curves.png" width="90%" alt="CNN Original Loss and Accuracy" />
  <br/>
  <img src="results/danq_multiclass_curves.png" width="90%" alt="DanQ Multiclass Loss and Accuracy" />
</p>

### 2. Additional Experiment: Effect of Removing Pooling Layers
Removing pooling layers altogether yielded much smoother training curves and slashed oscillations in the validation loss metrics, while final validation accuracy changed only marginally. For short sequences, spatial downsampling is completely unnecessary.

<p align="center">
  <img src="results/pooling_vs_nopooling_curves.png" width="90%" alt="Effect of Removing Pooling - Curves Comparison" />
</p>

---

## Environment Setup & How to Run

This project uses Docker to ensure completely reproducible test runs.

### 1. Prerequisites (NVIDIA Container Toolkit installation)
If your host system does not have the toolkit installed for GPU pass-through:

```bash
# Configure repository production keys
curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart docker engine
sudo systemctl restart docker

# Step 1: Build the environment container image
docker build -t dl-project-env .

# Step 2: Run the container mapping output directories
# If using an NVIDIA GPU:
docker run --gpus all -it -v "$(pwd)/results:/app/results" dl-project-env

# Or, if running purely on a CPU:
docker run -it -v "$(pwd)/results:/app/results" dl-project-env

# Step 3: Trigger pipeline tests from within the interactive container
./run_tests.sh
