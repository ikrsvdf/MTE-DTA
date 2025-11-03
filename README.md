<div align="center">

# MTE-DTA: 

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.13+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![rdkit](https://img.shields.io/badge/-rdkit_2023.3.2+-792ee5?logo=rdkit&logoColor=white)](https://anaconda.org/conda-forge/rdkit/)
[![torch-geometric](https://img.shields.io/badge/torch--geometric-2.3.1+-792ee5?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/en/latest/)

</div>

## 📌  Introduction 
**Motivation:** Parasites, including protozoa and helminths, are diverse and widely transmitted, causing numerous diseases and being major contributors to outbreaks in many regions worldwide. Although various antiparasitic drugs are currently in use, their efficacy is often limited by high costs, toxicity, significant side effects, and drug resistance. Therefore, developing novel, safe, and effective antiparasitic treatments has become an urgent and important challenge. Antiparasitic drugs typically function by interfering with key proteins within parasites; therefore, identifying potential drug targets and predicting their binding affinity to candidate compounds are crucial steps in antiparasitic drug discovery.

**Results:** To overcome the limitations of existing drug–target affinity (DTA) prediction models in modality alignment and feature fusion, we propose MTE-DTA, a Transformer-based multi-scale framework for drug–target affinity prediction. The model innovatively integrates an imbalanced Transformer encoder with a bilinear attention mechanism to capture fine-grained interactions between drug substructures and protein residues across multiple scales, thereby achieving more accurate and robust affinity estimation. Experimental results on our newly curated parasite-related drug–target affinity dataset as well as on several public benchmark datasets demonstrate that MTE-DTA consistently outperforms state-of-the-art methods. Moreover, the model exhibits strong generalization ability across various real-world scenarios.
Specifically, we employ a dual-branch architecture that utilizes multi-modal information of drugs and proteins to enhance the feature interaction between proteins and drug substructures from both global and local perspectives. Firstly, in the global feature extraction module, we use a parallel architecture to independently extract the global features of drugs and proteins. Subsequently, in the multi-level feature extraction module, we employ an imbalanced Transformer encoder and a bilinear attention mechanism to meticulously capture hierarchical interaction features between protein residues and drug substructures. 
This approach effectively aligns drug features and protein features while preventing biased learning in the model. Finally, a linear attention mechanism is adopted to balance the drug-protein interaction features at different scales, thereby enhancing the accuracy of the model in the prediction of downstream tasks.

## 🚀 Architecture
<img width="2200" height="947" alt="模型框架" src="https://github.com/user-attachments/assets/8fa35495-6410-4fe5-869b-3cd19ca66089" />


## :blue_heart: Installation
First, you need to clone our code to your operating system.

```
git clone https://github.com/ikrsvdf/MTE-DTA.git
cd MTE-DTA
```


## :computer: The environment of MTE-DTA
Before running the code, you need to configure the environment, which mainly consists of the commonly used torch==2.0.1+cu118 ,rdkit==2025.9.1, torch-geometric==2.6.1 and other basic packages.
```
python==3.11.13
torch==2.0.1+cu118
torch-geometric==2.6.1
scipy==1.16.2
rdkit==2025.9.1
pandas==2.3.3
networkx==3.3
numpy==1.25.2
```
Of course you can also directly use the following to create a new environment:
```
conda create -n MTE-DTA python==3.11.13
conda activate 3.11.13
```
our code is based on python 3.11.13 and CUDA 11.8 and is used on a linux system. Note that while our code does not require a large amount of running memory, it is best to use more than 24G of RAM if you need to run it.
## :books: Dataset
Three datasets, DAVIS, KIBA, and Parasite, have all been placed in the data folder. First, for each protein sequence, a 300-length segment is extracted based on its binding site. Then, the ESM2 model is used to convert each amino acid into a high-dimensional vector representation, which is implemented using `test_pocket_embed.py`and saved as a .npy file with the key being the UniProt ID and the value being a tensor of shape (300, 1280). Next, the contact map for each protein is generated using `esm-2_graph.py`and saved as a .npy file. Finally, ChemBERTa is used to generate the embedded representation of SMILES molecules, implemented with `smiles_embedding.py`, outputting a 768-dimensional molecular embedding vector.
## :gear: Pre-trained model
Since the parameter files for the other pre-trained models are rather large, we will not give them here, you can download them according to the link below and save them to the appropriate location in the MTE-DTA folder. [ESM-2](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt),[ChemBERTa](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR)

## :chart_with_upwards_trend: Training
Once you have configured the base environment and dataset as well as some pre-trained models, you are ready to train the models.

```
python main_trainer.py
```



