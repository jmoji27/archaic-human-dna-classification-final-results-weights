# Classification of Archaic Human Genomics Using Deep Neural Networks

## Project Overview
[cite_start]This repository contains the course project for **Introduction to Deep Learning** at **Eskişehir Technical University** (Department of Computer Engineering)[cite: 2, 4, 7]. 

* [cite_start]**Instructor:** Prof. Dr. Cahit Perkgöz [cite: 5]
* [cite_start]**Group Members:** [cite: 8]
  * [cite_start]Jaime Jesus Mojica Dominguez [cite: 9]
  * [cite_start]Canay Moradi [cite: 10]
  * [cite_start]Şevval Gürler [cite: 11]

### Problem Definition
[cite_start]Modern Humans (*Homo Sapiens*) and Neanderthals (*Homo neanderthalensis*) diverged approximately 500,000 to 750,000 years ago[cite: 13]. [cite_start]Subsequently, Neanderthals and Denisovans diverged around 381,000 to 473,000 years ago [cite: 15][cite_start], both going extinct roughly 40,000 years ago[cite: 16]. [cite_start]Because these hominin groups are evolutionarily closely related, distinguishing their short genomic sequences represents a challenging classification task[cite: 17]. 

[cite_start]Classification is heavily impacted by **ancient DNA degradation** (postmortem fragmentation, chemical modifications driven by hydrolytic cleavage and oxidation) [cite: 18, 150] [cite_start]and **historical introgression**[cite: 24, 29]. [cite_start]For instance, modern Eurasian populations contain 1–4% Neanderthal DNA [cite: 23, 24][cite_start], while Pacific Islanders and Southeast Asians carry up to 5–6% Denisovan DNA[cite: 25].

[cite_start]This project frames the problem in two ways[cite: 30]:
1. [cite_start]**Binary Classification:** Deciding between two species (Human vs. Denisovan, Human vs. Neanderthal, or Denisovan vs. Neanderthal)[cite: 30].
2. [cite_start]**Multiclass Classification:** Simultaneously distinguishing between Modern Human, Neanderthal, and Denisovan sequences[cite: 31].

---

## Model Architectures & Pipelines

[cite_start]All models process input DNA sequences by transforming nucleotide base pairs (`A, C, G, T`) into one-hot encoded vectors of dimension 4[cite: 39]. [cite_start]Unknown bases (`N`) are encoded as a zero vector[cite: 44].

* [cite_start]`A` ➔ `[1, 0, 0, 0]` [cite: 40]
* [cite_start]`C` ➔ `[0, 1, 0, 0]` [cite: 41]
* [cite_start]`G` ➔ `[0, 0, 1, 0]` [cite: 42]
* [cite_start]`T` ➔ `[0, 0, 0, 1]` [cite: 43]
* [cite_start]`N` ➔ `[0, 0, 0, 0]` [cite: 44]

### 1. Convolutional Neural Network (CNN)
[cite_start]Optimized for detecting short local sequence motifs without sequential modeling dependencies[cite: 36, 37].
* [cite_start]**Feature Extraction:** Four 1D convolutional layers compressing channels ($128 \rightarrow 64 \rightarrow 64 \rightarrow 32$) with decreasing kernel sizes (19, 15, 5)[cite: 48, 50]. [cite_start]Intermediate MaxPool is avoided to preserve fine spatial data on short sequences[cite: 52].
* [cite_start]**Variable Length Handling:** Parallel Global Max Pooling and Global Average Pooling concatenated into a fixed 64-dimensional vector[cite: 54, 55, 56].
* [cite_start]**Classifier Head:** Two dense layers ($64 \rightarrow 64 \rightarrow 256$) with high dropout rates to mitigate overfitting [cite: 60, 61][cite_start], evaluated using Focal Loss ($\gamma = 2.0$)[cite: 66].

### 2. Recurrent Neural Network (BiLSTM)
[cite_start]Selected to capture long-range contextual relationships spanning across the sequence length[cite: 72, 74].
* [cite_start]**Architecture:** Uses Bidirectional LSTM cells with three gating mechanisms (Forget, Input, Output gates) to prevent vanishing gradients[cite: 78, 80]. 
* [cite_start]**Dynamic Capacity:** Automatically scales from 1 layer (64 units) for simpler binary tasks to 2 layers (128 units per direction) for complex tasks[cite: 86, 87].

### 3. DanQ (Hybrid CNN-BiLSTM)
[cite_start]A hybrid architecture combining the localized feature extraction of a CNN with the long-range sequential reasoning of a BiLSTM[cite: 95, 97].
* [cite_start]**Pipeline:** Two sequential 1D Convolution blocks extract local patterns [cite: 104, 108][cite_start], followed by a Max Pooling layer (size=2, stride=2) to reduce sequence length before feeding into a 1-layer BiLSTM[cite: 111, 113, 119].
* [cite_start]**Attention Pooling:** Normalizes spatial scores using a softmax mask to output a richer 256-dimensional summary vector to the final classification layer[cite: 121, 123, 125].

---

## Dataset Construction & Fragment Lengths

[cite_start]Data was filtered using a Mapping Quality score ($\text{MAPQ} \ge 30$) to keep a balanced threshold of data quality (1 in 1,000 alignment error probability)[cite: 22, 143, 153, 155].

### Fragment Length Distribution Across Species
[cite_start]Sequence-length distributions differed significantly across species due to differences in DNA preservation quality and sequencing coverage[cite: 158]. [cite_start]Modern human reads show stable long lengths [cite: 160][cite_start], Denisovan reads skew short [cite: 161][cite_start], and Neanderthal reads rarely cross 85 bp[cite: 162].

<p align="center">
  <img src="results/denisovan_length_dist.png" width="30%" alt="Denisovan Fragment Length Distribution" />
  <img src="results/human_length_dist.png" width="30%" alt="Humans Fragment Length Distribution" />
  <img src="results/neanderthal_length_dist.png" width="30%" alt="Neanderthals Fragment Length Distribution" />
</p>

### Dataset Statistics

| Dataset Setting | Task Type | Total Samples (Train / Val / Test) | Class Distribution | Mean Length |
| :--- | :--- | :--- | :--- | :--- |
| **Human vs. Denisovan (Original)** | Binary | [cite_start]2,019,025 / 148,815 / 50,906 [cite: 195] | [cite_start]Human: ~59% <br> Denisovan: ~41% [cite: 195] | [cite_start]~52.5 bp [cite: 195] |
| **Human vs. Denisovan (Longerbp)** | Binary | [cite_start]47,825 / 4,902 / 1,889 [cite: 199] | [cite_start]Human: ~51% <br> Denisovan: ~49% [cite: 199] | [cite_start]~48.0 bp [cite: 199] |
| **Multiclass Dataset** | 3-Class | [cite_start]72,786 / 7,403 / 2,837 [cite: 203] | [cite_start]Human: 34% <br> Denisovan: 32% <br> Neanderthal: 34% [cite: 203] | [cite_start]~49.3 bp [cite: 203] |
| **Human vs. Neanderthal** | Binary | [cite_start]50,062 / 5,002 / 1,928 [cite: 206] | [cite_start]Human: 50% <br> Neanderthal: 50% [cite: 206] | [cite_start]~51.5 bp [cite: 206] |
| **Denisovan vs. Neanderthal** | Binary | [cite_start]47,753 / 4,896 / 1,869 [cite: 210] | [cite_start]Denisovan: ~48% <br> Neanderthal: ~52% [cite: 210] | [cite_start]~50.1 bp [cite: 210] |

---

## Performance Results

### Model Accuracy Summary

| Dataset Setting | CNN Accuracy | BiLSTM Accuracy | DanQ Accuracy | Best Model |
| :--- | :---: | :---: | :---: | :---: |
| **Human vs. Denisovan** | **0.6420** | 0.4206 | 0.5367 | [cite_start]**CNN** [cite: 214] |
| **Human vs. Neanderthal** | **0.7293** | 0.7132 | 0.7101 | [cite_start]**CNN** [cite: 214] |
| **Denisovan vs. Neanderthal** | 0.7833 | 0.7833 | **0.7849** | [cite_start]**DanQ** [cite: 214] |
| **Multiclass (All Three)** | 0.5731 | 0.5643 | **0.5791** | [cite_start]**DanQ** [cite: 214] |
| **Bottleneck Experiment** | **0.6379** | 0.6374 | 0.6310 | [cite_start]**CNN** [cite: 214] |
| **Longerbp Dataset** | 0.5894 | 0.5993 | **0.6057** | [cite_start]**DanQ** [cite: 214] |

---

## Confusion Matrices & Evaluation

### 1. Human vs. Denisovan (Original) & Human vs. Neanderthal
* [cite_start]**Human vs. Denisovan (CNN):** Shows asymmetric performance[cite: 229]. [cite_start]It correctly identifies 61.6% of Humans but misclassifies 38.4% as Denisovans [cite: 229, 230][cite_start], reflecting massive genomic similarity[cite: 230].
* [cite_start]**Human vs. Neanderthal (CNN):** Distinguishes Neanderthal sequences more reliably (77.4% recall) than Human sequences (68.5% recall)[cite: 235].

<p align="center">
  <img src="results/cnn_original_confusion_matrix.png" width="45%" alt="CNN Original Confusion Matrix" />
  <img src="results/cnn_human_neanderthal_confusion_matrix.png" width="45%" alt="CNN Human vs Neanderthal Confusion Matrix" />
</p>

### 2. Denisovan vs. Neanderthal & Multiclass Classification
* [cite_start]**Denisovan vs. Neanderthal (DanQ):** This is the strongest binary result overall[cite: 263]. [cite_start]Neanderthals achieve a very high recall of 91.9%[cite: 264]. 
* [cite_start]**Multiclass (DanQ):** Performs unevenly across classes[cite: 271]. [cite_start]Neanderthals reach 85.5% recall [cite: 272][cite_start], but Human (32.0%) and Denisovan (56.2%) sequences are highly confused[cite: 272].

<p align="center">
  <img src="results/danq_denisovan_neanderthal_confusion_matrix.png" width="45%" alt="DanQ Denisovan vs Neanderthal Confusion Matrix" />
  <img src="results/danq_multiclass_confusion_matrix.png" width="45%" alt="DanQ Multiclass Confusion Matrix" />
</p>

### 3. Bottleneck vs. Longer Base Pairs (Longerbp)
* [cite_start]**Bottleneck (CNN):** Near-chance performance on Denisovan sequences (52.5%), highlighting that structural bottleneck limits feature extraction capacity[cite: 313, 314].
* [cite_start]**Longerbp (DanQ):** Heavily biased toward the Human class (86.2% recall) while failing to leverage extra sequence length for Denisovans (27.5% recall)[cite: 325, 326].

<p align="center">
  <img src="results/cnn_bottleneck_confusion_matrix.png" width="45%" alt="CNN Bottleneck Confusion Matrix" />
  <img src="results/danq_longerbp_confusion_matrix.png" width="45%" alt="DanQ Longerbp Confusion Matrix" />
</p>

---

## Training Dynamics & Generalization

### 1. Loss & Accuracy Curves (Standard Implementations)
* [cite_start]**Human vs. Neanderthal (CNN):** Healthy generalization with clean convergence and a tiny validation loss gap[cite: 345, 346].
* [cite_start]**Human vs. Denisovan (CNN):** Severe overfitting; validation loss plateaus high while validation accuracy experiences major spikes[cite: 376].
* [cite_start]**Multiclass (DanQ):** Progressive overfitting becomes visible from epoch 20 onward as the loss gap widens[cite: 439, 440].

<p align="center">
  <img src="results/cnn_human_neanderthal_curves.png" width="90%" alt="CNN Human vs Neanderthal Loss and Accuracy" />
  <br/>
  <img src="results/cnn_original_curves.png" width="90%" alt="CNN Original Loss and Accuracy" />
  <br/>
  <img src="results/danq_multiclass_curves.png" width="90%" alt="DanQ Multiclass Loss and Accuracy" />
</p>

### 2. Additional Experiment: Effect of Removing Pooling Layers
[cite_start]Removing pooling layers altogether yielded **much smoother training curves** and slashed oscillations in the validation loss metrics, while final validation accuracy changed only marginally[cite: 217]. [cite_start]For short sequences, spatial downsampling is completely unnecessary[cite: 218, 219].

<p align="center">
  <img src="results/pooling_vs_nopooling_curves.png" width="90%" alt="Effect of Removing Pooling - Curves Comparison" />
</p>

---

## Environment Setup & How to Run

[cite_start]This project uses Docker to ensure completely reproducible test runs[cite: 514]. 

### 1. Prerequisites (NVIDIA Container Toolkit installation)
[cite_start]If your host system does not have the toolkit installed for GPU pass-through[cite: 503]:

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
