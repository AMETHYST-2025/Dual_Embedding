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
dir = "./Polyimides/"

numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)


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
    plt.tight_layout()
    plt.savefig(dir + "Histogramme/histogram_{}.png".format(label),
        format="png", dpi=600)
    plt.show()


def min_max_score(x, min, max,a):
    # Calculation of λ
    lambd = 1-a
    # Calculation of f(x)
    return 1 - lambd*(x/max)
def exponential_score(x, a, max):
    # Calculation of λ
    lambd =  np.log(a) / max
    # Calculation of f(x)
    fx = np.exp(lambd * x)
    return fx

def cosine(A,B):
    return dot(A, B) / (norm(A) * norm(B))

def Extract_embedding_finetune_dataset():
    k=21
    link_Polyimides_syn_subset = dir + "Polyimides_synthetic_sub_set.csv"
    link_Polyimides_syn_embedding = dir + "Polyimides_synthetic_embedding.csv"
    link_Polyimides_syn_combination_embedding = dir + "Polyimides_combination_synthetic_embedding.csv"

    data1 = pd.read_csv(link_Polyimides_syn_subset)#1000
    data2 = pd.read_csv(link_Polyimides_syn_embedding)#200
    print(ks_2samp(list(data1['Tg'][:]), list(data2['Tg'][:])))

    #--------------------------------------------------------------
    """DF_Polyimides = pd.read_csv(link_Polyimides_data)
    print(DF_Polyimides)
    DF_Polyimides.rename(columns={'Tg, K': 'Tg'}, inplace=True)
    DF_Polyimides = DF_Polyimides.groupby('SMILES', as_index=False)['Tg'].mean()
    print(DF_Polyimides)
    #DF_Polyimides = DF_Polyimides[DF_Polyimides.columns][:2]
    DF_Polyimides['PolyBERT'] = DF_Polyimides['SMILES'].apply(PolyBERT)
    print(DF_Polyimides)
    DF_Polyimides.to_csv(link_Polyimides_mean)"""

    """DF_Polyimides_syn = pd.read_csv(link_Polyimides_synthetic_data)
    DF_Polyimides_syn.rename(columns={'Ic1ccc(nc1)Cc1cccc(c1)Cc1ccc(cn1)N1C(=O)c2c(C1=O)cc(cc2)c1ccc2c(c1)C(=O)N(C2=O)I': 'SMILES', '495.0': 'Tg'}, inplace=True)
    print(DF_Polyimides_syn)
    DF_Polyimides_syn = DF_Polyimides_syn.sample(n=1200, random_state=42)
    DF_Polyimides_syn = DF_Polyimides_syn.groupby('SMILES', as_index=False)['Tg'].mean()
    DF_Polyimides_syn['PolyBERT'] = DF_Polyimides_syn['SMILES'].apply(PolyBERT)
    print(DF_Polyimides_syn)
    DF_Polyimides_syn_save = DF_Polyimides_syn[:1000]
    DF_Polyimides_syn_embedding = DF_Polyimides_syn[1000:1200]
    DF_Polyimides_syn_save.to_csv(link_Polyimides_syn_subset)
    DF_Polyimides_syn_embedding.to_csv(link_Polyimides_syn_embedding)"""

    #DF_benchmak_test, DF_benchmak_finetune = train_test_split(DF_benchmak, test_size=0.3, random_state=42)
    #DF_benchmak_test.to_csv(dataset_for_test)
    #DF_benchmak_finetune.to_csv(link_benchmark_file_for_finetuning)

    if k==2:
        #------------------------------------------------------------
        DF = pd.read_csv(link_Polyimides_syn_embedding)
        print(DF)
        print(len(DF))
        DF['PolyBERT'] = DF['PolyBERT'].apply(eval)
        # Generate all unique 2-by-2 combinations (order not considered).
        pairs = list(combinations(DF.itertuples(index=False), 2))
        # Construct a new DataFrame containing SMILES_A, SMILES_B, and delta_Tg.
        df_pairs = pd.DataFrame({
            'SMILES_A': [p[0].SMILES for p in pairs],
            'SMILES_B': [p[1].SMILES for p in pairs],
            'Delta_Tg': [abs(p[0].Tg - p[1].Tg) for p in pairs],
            'cosine_sim': [cosine(p[0].PolyBERT, p[1].PolyBERT) for p in pairs]
        })
        # ---------------------------------------------------------------
        df_pairs = df_pairs[df_pairs["Delta_Tg"] < 200]
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
        df_pairs.to_csv(link_Polyimides_syn_combination_embedding)

Extract_embedding_finetune_dataset()






