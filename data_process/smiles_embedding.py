# -*- coding: gbk -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from rdkit import Chem
from torch import nn

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./chemBERTa")
model = AutoModelForMaskedLM.from_pretrained("./chemBERTa")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model = model.eval()

# Load data
df = pd.read_csv('./data/Davis/drug_davis.csv')
df = df.drop_duplicates(subset=["Ligand SMILES"])
smiles_list = df['Ligand SMILES'].tolist()
drugbank_ids = df['Ligand SMILES'].tolist()
print(f"Total SMILES: {len(smiles_list)}")

# Initialize dictionary to store embeddings
compounds_dict = {}
ln = nn.LayerNorm(768).to(device)

# Process each SMILES string
for drugbank_id, smiles in tqdm(zip(drugbank_ids, smiles_list), total=len(smiles_list)):
    tokens = tokenizer.tokenize(smiles)
    string = ''.join(tokens)

    if len(string) > 512:
        # Handle long sequences
        j = 0
        flag = True
        output = torch.zeros(1, 768).to(device)

        while flag:
            input = tokenizer(string[j:min(len(string), j + 511)],
                              return_tensors='pt').to(device)
            if len(string) <= j + 511:
                flag = False

            with torch.no_grad():
                hidden_states = model(**input, return_dict=True,
                                      output_hidden_states=True).hidden_states
                output_hidden_state = torch.cat([
                    (hidden_states[-1] + hidden_states[1]).mean(dim=1),
                    (hidden_states[-2] + hidden_states[2]).mean(dim=1)
                ], dim=1)  # first last layers average add
                output_hidden_state = ln(output_hidden_state)
                print(output_hidden_state.shape)

            output = torch.cat((output, output_hidden_state), dim=0)
            j += 256

        output = output[1:-1].mean(dim=0).unsqueeze(dim=0).to('cpu').data.numpy()
    else:
        # Handle short sequences
        input = tokenizer(smiles, return_tensors='pt').to(device)
        with torch.no_grad():
            hidden_states = model(**input, return_dict=True,
                                  output_hidden_states=True).hidden_states
            # print(hidden_states)
            output_hidden_state = torch.cat([
                (hidden_states[-1] + hidden_states[1]).mean(dim=1),
                (hidden_states[-2] + hidden_states[2]).mean(dim=1)
            ], dim=1)  # first last layers average add
            # print(output_hidden_state.shape)
            output_hidden_state = ln(output_hidden_state) # 768
            # print(output_hidden_state.shape)
        output = output_hidden_state.to('cpu').data.numpy() # 768
        # print(output.shape)


    compounds_dict[drugbank_id] = output

# Save embeddings
np.save('./data/Davis/new_drug_embedding.npy',
        compounds_dict, allow_pickle=True)
print('SMILES embeddings generated successfully')
