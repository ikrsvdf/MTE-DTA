# -*- coding: gbk -*-
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
./data/Davis/
# Initialize ESM2 model and tokenizer
model_name = "./esm2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def sequence_to_vector(sequence):
    """Convert protein sequence to embedding vector (seq_len, hidden_dim)"""
    inputs = tokenizer(sequence, return_tensors="pt", padding=True,
                       truncation=True, max_length=300)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
    return embeddings.numpy()  # Return as NumPy array


def pad_sequence(sequence, target_length=300):
    """Pad sequence with 'X' if shorter than target length"""
    if len(sequence) < target_length:
        sequence += "X" * (target_length - len(sequence))
    return sequence


# Read protein sequence data
df = pd.read_csv("./data/Davis/protein_davis.csv")

# 根据截取的300长度的氨基酸序列进行处理processed_sequence
required_columns = ["UniProtID", "processed_sequence"]
if all(col in df.columns for col in required_columns):
    df_unique = df.drop_duplicates(subset=["UniProtID"])
    # Process sequences (只对不足300的进行填充)
    df_unique["final_sequence"] = df_unique["processed_sequence"].apply(pad_sequence)

    # Compute embeddings
    sequence_embeddings = {
        row["UniProtID"]: sequence_to_vector(row["final_sequence"])
        for _, row in df_unique.iterrows()
    }

    # Save embeddings
    save_path = "./data/Davis/sequence_300_acid.npy"
    np.save(save_path, sequence_embeddings, allow_pickle=True)
    print(f"Protein sequence embeddings have been saved to {save_path}")
else:
    missing_cols = [col for col in required_columns if col not in df.columns]
    raise KeyError(f"DataFrame is missing required columns: {missing_cols}")
