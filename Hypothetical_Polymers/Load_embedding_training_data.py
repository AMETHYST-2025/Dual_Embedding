import pandas as pd
from tensorflow import keras
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math
from sentence_transformers import SentenceTransformer
from itertools import combinations
from scipy.stats import ks_2samp
keras.utils.set_random_seed(42)
polyBERT_ = SentenceTransformer('kuelumbus/polyBERT')

dir= "/Hypothetical_Polymers/"
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
    plt.tight_layout()  # Ajuste les marges pour éviter que les labels soient coupés
    plt.savefig(dir + "Histogramme/histogram_{}.png".format(label),
        format="png", dpi=600)
    plt.show()


def min_max_score(x, min, max,a):
    # Calcul de λ
    lambd = 1-a
    # Calcul de f(x)
    return 1 - lambd*(x/max)
def exponential_score(x, a, max):
    # Calcul de λ
    lambd =  np.log(a) / max
    # Calcul de f(x)
    fx = np.exp(lambd * x)
    return fx

def F_3_Tg(a,b):
    delta = abs(a - b)
    mean = (a+b)/2
    if delta <= mean:
        return delta/mean
    else:
        return 0

dir_save_model = dir+"Embedding/"
model_reload = SentenceTransformer(dir_save_model +"PB_Tg_copolymers_Exp_01") #PB_Tg_copolymers_Min_Max_01
def finetune_PolyBERT(smiles):
    embedding = model_reload.encode(smiles)
    return embedding
def cosine_sim(A,B, model):
    if model == "PB":
        PB_A = PolyBERT(A)[0]
        PB_B = PolyBERT(B)[0]
    if model == "finetuned":
        PB_A = finetune_PolyBERT(A)
        PB_B = finetune_PolyBERT(B)
    return dot(PB_A, PB_B) / (norm(PB_A) * norm(PB_B))

def cosine(A,B):
    return dot(A, B) / (norm(A) * norm(B))
def dis_tg2(A,B):
    return math.sqrt(abs(A**2-B**2)) #  métrique de Minkowski modifiée
def log_distance(A,B):
    return abs(np.log(A/B))
def Extract_embedding_finetuning_dataset():
    k=23
    link_data_embedding = dir + "Dataset_copolymers_for_embedding_manyfeatures.csv"
    link_data_combination_embedding = dir + "Dataset_copolymers_combination_embedding.csv"
    link_data_all = dir + "DF_Sub_CP_all_features_Full_MorganFP.csv"
    data1 = pd.read_csv(link_data_embedding)
    data2 = pd.read_csv(link_data_all)
    #Kolmogorov-Smirnov test, used to evaluate the similarity of the distributions between the training set of the prediction model and the fine-tuning set of the embedding.
    print(ks_2samp(list(data1['Tg (°C)'][:]), list(data2['Tg (°C)'][:])))

    if k==2: #Creation of the dataset for fine-tuning the embedding.
        #------------------------------------------------------------
        DF = pd.read_csv(link_data_embedding)
        DF = DF.rename(columns={'Tg (°C)': 'Tg'})
        DF['PolyBERT'] = DF['smiles'].apply(PolyBERT)
        DF['PolyBERT'] = DF['PolyBERT'].apply(eval)
        # Generate all unique 2-by-2 combinations (order not considered).
        pairs = list(combinations(DF.itertuples(index=False), 2))
        # Construct a new DataFrame containing SMILES_A, SMILES_B, and delta_Tg.
        df_pairs = pd.DataFrame({
            'SMILES_A': [p[0].smiles for p in pairs],
            'SMILES_B': [p[1].smiles for p in pairs],
            'Delta_Tg': [abs(p[0].Tg - p[1].Tg) for p in pairs],
            'DeltaLn_Tg': [log_distance(p[0].Tg, p[1].Tg) for p in pairs],
            'Delta_Tg2': [dis_tg2(p[0].Tg, p[1].Tg) for p in pairs],
            'Delta_Td': [abs(p[0].Td - p[1].Td) for p in pairs],
            'Delta_Cp': [abs(p[0].Cp - p[1].Cp) for p in pairs],
            'Delta_Eea': [abs(p[0].Eea - p[1].Eea) for p in pairs],
            'cosine_sim': [cosine(p[0].PolyBERT, p[1].PolyBERT) for p in pairs]

        })
        # ---------------------------------------------------------------
        #df_pairs = df_pairs[df_pairs["Delta_Tg"] < 300]
        plot_data_structure(df_pairs, ['Delta_Eea'], 'Delta_Eea')
        plot_data_structure(df_pairs, ['Delta_Td'], 'Delta_Td')
        plot_data_structure(df_pairs, ['Delta_Cp'], 'Delta_Cp')
        plot_data_structure(df_pairs, ['Delta_Tg2'], 'Delta_Tg2')
        plot_data_structure(df_pairs, ['DeltaLn_Tg'], 'DeltaLn_Tg')

        distances = np.array(list(df_pairs['DeltaLn_Tg']))
        d_min, d_max = distances.min(), distances.max()
        df_pairs['min_max_Ln_a_01'] = df_pairs['DeltaLn_Tg'].apply(lambda x: min_max_score(x, min=d_min, max=d_max, a=0.01))
        df_pairs['exponential_Ln_a_01'] = df_pairs['DeltaLn_Tg'].apply(lambda x: exponential_score(x, a=0.01, max=d_max))
        plot_data_structure(df_pairs, ['min_max_Ln_a_01'], 'Tg_min_max_Ln_a_01')
        plot_data_structure(df_pairs, ['exponential_Ln_a_01'], 'Tg_exponential_Ln_a_01')


        distances = np.array(list(df_pairs['Delta_Tg']))
        d_min, d_max = distances.min(), distances.max()
        df_pairs['min_max_score_a_01'] = df_pairs['Delta_Tg'].apply(lambda x: min_max_score(x, min=d_min, max=d_max, a=0.01))
        df_pairs['exponential_score_a_01'] = df_pairs['Delta_Tg'].apply(lambda x: exponential_score(x, a=0.01, max=d_max))
        plot_data_structure(df_pairs, ['min_max_score_a_01'], 'Tg_min_max_score_a_01')
        plot_data_structure(df_pairs, ['exponential_score_a_01'], 'Tg_exponential_score_a_01')

        distances = np.array(list(df_pairs['Delta_Tg2']))
        d_min, d_max = distances.min(), distances.max()
        df_pairs['min_max_score2_a_01'] = df_pairs['Delta_Tg2'].apply(lambda x: min_max_score(x, min=d_min, max=d_max, a=0.01))
        df_pairs['exponential_score2_a_01'] = df_pairs['Delta_Tg2'].apply(lambda x: exponential_score(x, a=0.01, max=d_max))
        plot_data_structure(df_pairs, ['min_max_score2_a_01'], 'Tg_min_max_score2_a_01')
        plot_data_structure(df_pairs, ['exponential_score2_a_01'], 'Tg_exponential_score2_a_01')

        distances = np.array(list(df_pairs['Delta_Eea']))
        d_min, d_max = distances.min(), distances.max()
        df_pairs['min_max_Eea_a_01'] = df_pairs['Delta_Eea'].apply(lambda x: min_max_score(x, min=d_min, max=d_max, a=0.01))
        df_pairs['exponential_Eea_a_01'] = df_pairs['Delta_Eea'].apply(lambda x: exponential_score(x, a=0.01, max=d_max))
        plot_data_structure(df_pairs, ['min_max_Eea_a_01'], 'Tg_min_max_Eea_a_01')
        plot_data_structure(df_pairs, ['exponential_Eea_a_01'], 'Tg_exponential_Eea_a_01')

        distances = np.array(list(df_pairs['Delta_Td']))
        d_min, d_max = distances.min(), distances.max()
        df_pairs['min_max_Td_a_01'] = df_pairs['Delta_Td'].apply(lambda x: min_max_score(x, min=d_min, max=d_max, a=0.01))
        df_pairs['exponential_Td_a_01'] = df_pairs['Delta_Td'].apply(lambda x: exponential_score(x, a=0.01, max=d_max))
        plot_data_structure(df_pairs, ['min_max_Td_a_01'], 'Tg_min_max_Td_a_01')
        plot_data_structure(df_pairs, ['exponential_Td_a_01'], 'Tg_exponential_Td_a_01')

        distances = np.array(list(df_pairs['Delta_Cp']))
        d_min, d_max = distances.min(), distances.max()
        df_pairs['min_max_Cp_a_01'] = df_pairs['Delta_Cp'].apply(lambda x: min_max_score(x, min=d_min, max=d_max, a=0.01))
        df_pairs['exponential_Cp_a_01'] = df_pairs['Delta_Cp'].apply(lambda x: exponential_score(x, a=0.01, max=d_max))
        plot_data_structure(df_pairs, ['min_max_Cp_a_01'], 'Tg_min_max_Cp_a_01')
        plot_data_structure(df_pairs, ['exponential_Cp_a_01'], 'Tg_exponential_Cp_a_01')

        df_pairs.to_csv(link_data_combination_embedding)

Extract_embedding_finetuning_dataset()

def plot_embedding_comparison():
    link_data_combination_embedding = dir+"Dataset_copolymers_combination_embedding_log.csv"
    df_pairs= pd.read_csv(link_data_combination_embedding)
    df_pairs = df_pairs.sample(frac=1, random_state=42)
    #df_pairs = df_pairs[df_pairs.columns][:1000]
    df_pairs = df_pairs[df_pairs["Delta_Tg"] < 1]
    df_pairs = df_pairs[df_pairs["cosine_sim"] > 0]
    df_pairs = df_pairs.rename(columns={'cosine_sim': 'PB'})
    df_pairs = df_pairs.sort_values(by='PB')
    df_pairs["PB_Tg_Exp"] = df_pairs.apply(lambda row: cosine_sim(row['SMILES_A'], row['SMILES_B'], "finetuned"),axis=1)
    print(len(df_pairs))
    plt.figure(figsize=(5, 4))
    for col in ['PB',"PB_Tg_Exp"]:  # df_pairs.columns[3:]
        plt.plot(range(len(df_pairs)), df_pairs[col], marker='o', linestyle='', label=col)
    #plt.title('PB and PB_Tg_Exp')
    plt.xlabel('couple(P1,P2), |Tg1-Tg2|< 1°C')
    plt.ylabel("Cosine similarity (P1,P2)")
    plt.legend()  # Affiche la légende
    plt.grid(True)
    plt.show()
#plot_embedding_comparison()




