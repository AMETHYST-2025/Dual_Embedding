import pandas as pd
from tensorflow import keras
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from itertools import combinations
from scipy.stats import ks_2samp
keras.utils.set_random_seed(42)
polyBERT_ = SentenceTransformer('kuelumbus/polyBERT')
numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)
dir = "/Homopolymers/"
def PolyBERT(molecule):
    embeddings = polyBERT_.encode(molecule)
    embeddings = str(embeddings)
    embeddings = embeddings.replace(' ',',')
    embeddings = embeddings.replace('[,', '[')
    embeddings = embeddings.replace(',]', ']')
    embeddings = embeddings.replace(',,,,', ',')
    embeddings = embeddings.replace(',,,', ',')
    embeddings = embeddings.replace(',,', ',')
    return embeddings

def plot_data_structure(df,col,label):
    df = df[col][:]
    df.hist(figsize=(12, 8), bins=30, edgecolor="black")
    plt.tight_layout()  # Ajuste les marges pour éviter que les labels soient coupés
    plt.savefig(
        dir + "Histogramme/histogram_{}.png".format(label),
        format="png", dpi=600)
    plt.show()

def convert(x):
    return 10**x

def min_max_score(x, min, max,a):
    lambd = 1-a
    return 1 - lambd*(x/max)
def exponential_score(x, a, max):
    # Calcul de λ
    lambd =  np.log(a) / max
    # Calcul de f(x)
    fx = np.exp(lambd * x)
    return fx


def cosine(A,B):
    return dot(A, B) / (norm(A) * norm(B))

def log_dif(a,b):
    return 10**abs(np.log10(a) - np.log10(b))

def Extract_embedding_finetuning_dataset():
    k=22
    link_QSPR_data = dir + "42004_2024_1305_MOESM3_ESM.xlsx"
    link_QSPR_data_train = dir+"QSPR_data_train.csv"
    link_QSPR_data_test = dir + "QSPR_data_test.csv"
    link_QSPR_data_embedding = dir + "QSPR_data_embedding.csv"
    link_QSPR_data_combination_embedding = dir + "QSPR_data_embedding_combinations.csv"
    link_QSPR_data_combination_embedding_log = dir + "QSPR_data_embedding_combinations_log10.csv"

    DF_QSPR = pd.read_excel(link_QSPR_data)
    print(DF_QSPR)
    DF_QSPR['logTg'] = DF_QSPR['logTg'].apply(float)
    DF_QSPR['Tg'] = DF_QSPR['logTg'].apply(convert)
    #DF_QSPR['SMILES'] = DF_QSPR['SMILES'].apply(clean_smiles)
    DF_QSPR['PolyBERT'] = DF_QSPR['SMILES'].apply(PolyBERT)
    DF_QSPR_train = DF_QSPR[DF_QSPR['Serie'].str.contains('T', na=False)]
    DF_QSPR_test = DF_QSPR[DF_QSPR['Serie'].str.contains('P', na=False)]
    #DF_QSPR_train.to_csv(link_QSPR_data_train)
    #DF_QSPR_test.to_csv(link_QSPR_data_test)
    DF_QSPR_embedding = DF_QSPR_train.sample(n=200, random_state=42)

    #Kolmogorov-Smirnov test, used to evaluate the similarity of the distributions between the training set of the prediction model and the fine-tuning set of the embedding.
    print(ks_2samp(list(DF_QSPR_train['Tg'][:]), list(DF_QSPR_embedding['Tg'][:])))
    #DF_QSPR_embedding.to_csv(link_QSPR_data_embedding)


    if k==2: #Creation of the dataset for fine-tuning the embedding.
        # ---------------------------Polymers-------------------------
        DF = pd.read_csv(link_QSPR_data_embedding)
        print(DF)
        print(len(DF))
        DF['PolyBERT'] = DF['PolyBERT'].apply(eval)
        # Generate all unique 2-by-2 combinations (order not considered).
        pairs = list(combinations(DF.itertuples(index=False), 2))
        # Construct a new DataFrame containing SMILES_A, SMILES_B, and delta_Tg.
        df_pairs = pd.DataFrame({
            'SMILES_A': [p[0].SMILES for p in pairs],
            'SMILES_B': [p[1].SMILES for p in pairs],
            #'Delta_Tg': [abs(p[0].Tg - p[1].Tg) for p in pairs],
            'Delta_Tg': [log_dif(p[0].Tg, p[1].Tg) for p in pairs],
            'cosine_sim': [cosine(p[0].PolyBERT, p[1].PolyBERT) for p in pairs]
        })
        # ---------------------------------------------------------------
        df_pairs = df_pairs[df_pairs["Delta_Tg"] < 500]
        plot_data_structure(df_pairs, ['cosine_sim'], 'cosine_sim')
        plot_data_structure(df_pairs, ['Delta_Tg'], 'Delta_Tg')

        distances = np.array(list(df_pairs['Delta_Tg']))
        d_min, d_max = distances.min(), distances.max()
        print(d_min,d_max)

        df_pairs['min_max_score_a_01'] = df_pairs['Delta_Tg'].apply(lambda x: min_max_score(x, min=d_min, max=d_max, a=0.01))
        df_pairs['min_max_score_a_001'] = df_pairs['Delta_Tg'].apply(lambda x: min_max_score(x, min=d_min, max=d_max, a=0.001))

        df_pairs['exponential_score_a_01'] = df_pairs['Delta_Tg'].apply(lambda x: exponential_score(x, a=0.01, max=d_max))
        df_pairs['exponential_score_a_001'] = df_pairs['Delta_Tg'].apply(lambda x: exponential_score(x, a=0.001, max=d_max))
        plot_data_structure(df_pairs, ['min_max_score_a_01'], 'Tg_min_max_score_a_01')
        plot_data_structure(df_pairs, ['exponential_score_a_01'], 'Tg_exponential_score_a_01')
        df_pairs.to_csv(link_QSPR_data_combination_embedding_log)

Extract_embedding_finetuning_dataset()





