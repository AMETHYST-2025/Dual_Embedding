import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
from tensorflow import keras
from datasets import Dataset

# Set seeds for reproducibility
random_seed = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(random_seed)

# -------------------- Parameters --------------------
data_path = "Tg_score.csv"          # Path to the dataset CSV file
score_column = "Exp_Tg"             # Column containing the target scores
model_name = "Save_models/PolyBERT" # Pretrained model path or name
batch_size = 16                      # Batch size for training
num_epochs = 5                       # Number of training epochs
save_dir = "Save_models/PolyBERT_Tg_Exp" # Directory to save fine-tuned model
os.makedirs(save_dir, exist_ok=True)

# -------------------- Load data --------------------
df = pd.read_csv(data_path).dropna(subset=[score_column, 'SMILES_A', 'SMILES_B'])
df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle data

# -------------------- Prepare training examples --------------------
train_examples = [
    InputExample(texts=[row['SMILES_A'], row['SMILES_B']], label=float(row[score_column]))
    for _, row in df.iterrows()
]

# -------------------- Load model --------------------
model = SentenceTransformer(model_name)

# -------------------- Define loss function and dataloader --------------------
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model)

# -------------------- Training --------------------
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=10,
    output_path=save_dir
)

# -------------------- Save the fine-tuned model --------------------
model.save(save_dir)
print(f"Model saved in: {save_dir}")

# -------------------- Simple visualization --------------------
# Select top 100 SMILES pairs by score
sample_smiles = df[['SMILES_A', 'SMILES_B', score_column]].sort_values(by=score_column, ascending=False).head(100)

# Load models before and after fine-tuning
before_model = SentenceTransformer(model_name)
after_model = SentenceTransformer(save_dir)

# Define cosine similarity function
cosine = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
before_sims, after_sims, true_scores = [], [], []

# Compute embeddings and cosine similarities
for _, row in sample_smiles.iterrows():
    emb1_before = before_model.encode(row['SMILES_A'])
    emb2_before = before_model.encode(row['SMILES_B'])
    emb1_after = after_model.encode(row['SMILES_A'])
    emb2_after = after_model.encode(row['SMILES_B'])

    before_sims.append(cosine(emb1_before, emb2_before))
    after_sims.append(cosine(emb1_after, emb2_after))
    true_scores.append(row[score_column])

# Plot the similarity comparison
plt.figure(figsize=(10, 6))
plt.plot(true_scores, before_sims, 'o', label='Before fine-tuning', alpha=0.7)
plt.plot(true_scores, after_sims, 's', label='After fine-tuning', alpha=0.7)
plt.xlabel("Target score (property similarity)")
plt.ylabel("Cosine similarity")
plt.title("Fine-tuning comparison: SMILES pairs with high "+score_column+" similarity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
