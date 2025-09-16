import pandas as pd
import numpy as np
import random
from tensorflow import keras
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, InputExample, losses, evaluation, datasets
import torch
from torch import nn

random_seed = 42
keras.utils.set_random_seed(random_seed)
def set_seed(seed):
    random.seed(seed)                      # Python's random module
    np.random.seed(seed)                   # NumPy
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed(seed)           # PyTorch GPU
    torch.cuda.manual_seed_all(seed)       # All GPUs (if using multi-GPU)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False    # Disables auto-optimization for reproducibility
set_seed(random_seed)

dataset_combinaison_embedding = "Data/Dataset_copolymers_combination_scores.csv"
dir_save = "Save_model"
DF_cosine_similarity = pd.read_csv(dataset_combinaison_embedding)
DF_cosine_similarity = DF_cosine_similarity.sample(frac=1, random_state=random_seed)
score_label1 ="PB_Tg_copolymers_MM_01" #Min-max norm
score_label2 ="PB_Tg_copolymers_Exp_01" #Min-max norm
DF_scores = DF_cosine_similarity[['SMILES_A', 'SMILES_B', score_label1, score_label2]][:]


model = SentenceTransformer('kuelumbus/polyBERT') #DeepChem/ChemBERTa-77M-MLM     #kuelumbus/polyBERT
# Combinaison des loss
class CombinedLoss(nn.Module):
    def __init__(self, model, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.cosine_loss = losses.CosineSimilarityLoss(model=model)
        self.mse_loss = losses.MSELoss(model=model)
        self.alpha = alpha

    def forward(self, sentence_features, labels):
        labels = torch.tensor(labels, dtype=torch.float, device=sentence_features[0]['input_ids'].device)
        F1_label = labels[:, 0]  # Première valeur
        F2_label = labels[:, 1]  # Deuxième valeur

        loss1 = self.cosine_loss(sentence_features, F1_label)
        loss2 = self.cosine_loss(sentence_features, F2_label)

        return self.alpha[0] * loss1 + self.alpha[1] * loss2

# SMILES_A, SMILES_B, score_label1, score_label2
train_examples = []
for i, row in DF_scores.iterrows():
    train_examples.append(
        InputExample(
            texts=[row['SMILES_A'], row['SMILES_B']],
            label=[float(row[score_label1]), float(row[score_label2])]
        )
    )



# DataLoader
train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=16)
train_loss = CombinedLoss(model=model, alpha=[1, 0])
model_name = "PB_Tg_MM" # for alpha=[1, 0] and "PB_Tg_Exp" for alpha=[0, 1]
# Entraînement du modèle
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=10,
    output_path='./'+model_name
)

Polymer_1= "[*]CCCCCCCCCCCCCCCCCCCCC([*])COc1ccc(-c2ccc(SSc3ccc([N+](=O)[O-])cc3)s2)cc1"
Polymer_2= "[*]CCCCCCCCSSc1ccc2c(c1)C(=O)N(C(=O)CCCCCCCCC(=O)OC([*])=O)C2=O"
embedding1 = model.encode(Polymer_1) #model_reload
embedding2 = model.encode(Polymer_2) #model_reload
cosine_sim = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
print('0.75->',cosine_sim)

