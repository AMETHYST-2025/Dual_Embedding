import pandas as pd
from tensorflow import keras
import re
import numpy as np
from sklearn.utils import shuffle
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ks_2samp
keras.utils.set_random_seed(42)
polyBERT_ = SentenceTransformer('kuelumbus/polyBERT')
numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)

dir= "/Benchmark_Polymers/"

def PolyBERT(molecule):
    psmiles_strings = [molecule]
    embeddings = polyBERT_.encode(psmiles_strings)
    return embeddings

def plot_data_structure(df,col,label):
    df = df[col][:]
    df.hist(figsize=(12, 8), bins=30, edgecolor="black")
    plt.tight_layout()  # Ajuste les marges pour éviter que les labels soient coupés
    plt.savefig(dir+"histogram_{}.png".format(label), format="png", dpi=600)
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

dir_save_model = dir+"finetune_PolyBERT"
def finetune_PolyBERT(smiles):
    model_reload = SentenceTransformer(dir_save_model +"/PB_Benchmark_Min_Max_01")
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

def Extract_embedding_finetuning_dataset():
    k=21
    link_copolymers_data = dir + "DF_Copolymers.csv"
    link_benchmark_file = dir + "ci1c01031_si_003.xlsx"
    link_benchmark_file_for_finetuning = dir + "DF_Benchmark_finetuning.csv"
    combinations_for_embedding_finetuning_merge = dir + "DF_Benchmark_merge_embedding_combinations.csv"
    dataset_for_test = dir + "DF_Benchmark_test.csv"
    evaluation_result = dir + "DF_results.csv"

    DF_benchmak = pd.read_excel(link_benchmark_file)
    DF_benchmak_test, DF_benchmak_finetune = train_test_split(DF_benchmak, test_size=0.3, random_state=42)
    # Kolmogorov-Smirnov test, used to evaluate the similarity of the distributions between the training set of the prediction model and the fine-tuning set of the embedding.
    print(ks_2samp(list(DF_benchmak_test['MD'][:]), list(DF_benchmak_finetune['MD'][:])))

    #DF_benchmak_test.to_csv(dataset_for_test)
    #DF_benchmak_finetune.to_csv(link_benchmark_file_for_finetuning)

    if k==2:# Creation of the dataset for fine-tuning the embedding.
        # -----------------------benchmark Polymers------------------
        DF_E = pd.read_csv(link_benchmark_file_for_finetuning)
        DF_E = DF_E.rename(columns={'SMILES_RepeatUnit_1': 'smiles'})
        DF_E = DF_E.rename(columns={'MD': 'Tg'})
        DF_E = DF_E[['smiles','Tg']][:]
        #----------------------Hypothetical Polymers-----------------
        DF_CP = pd.read_csv(link_copolymers_data)
        DF_CP = DF_CP.rename(columns={'Tg (°C)': 'Tg'})
        DF_CP = DF_CP[['smiles', 'Tg']][:]
        #------------------------Merge data--------------------------
        DF_E = DF_E.sample(n=100, random_state=42)
        DF_CP = DF_CP.sample(n=100, random_state=42)
        DF = pd.concat([DF_E, DF_CP], ignore_index=True)
        #------------------------------------------------------------
        # Generate all unique 2-by-2 combinations (order not considered).
        pairs = list(combinations(DF.itertuples(index=False), 2))
        # Construct a new DataFrame containing SMILES_A, SMILES_B, and delta_Tg.
        df_pairs = pd.DataFrame({
            'SMILES_A': [p[0].smiles for p in pairs],
            'SMILES_B': [p[1].smiles for p in pairs],
            'Delta_Tg': [abs(p[0].Tg - p[1].Tg) for p in pairs],
            'cosine_sim': [cosine_sim(p[0].smiles, p[1].smiles, "PB") for p in pairs]
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

        df_pairs.to_csv(combinations_for_embedding_finetuning_merge)
    if k==3: #benchmark evaluation
        col_name = ['MD']
        DF_benchmak = pd.read_excel(link_benchmark_file)
        for n in DF_benchmak.columns:
            if 'Repeat' in n:
                col_name.append(n)
        col_name.remove('SMILES_RepeatUnit_1')
        DF_benchmak = DF_benchmak[col_name][:]
        print(DF_benchmak)
        partial_DF_benchmak = pd.read_csv(dataset_for_test)
        partial_DF_benchmak = partial_DF_benchmak[col_name][:]
        print(partial_DF_benchmak)
        DF_Evaluation = pd.DataFrame(columns=['Embeddings_Models','RMSE','partial_RMSE','R2','partial_R2', 'MAE', 'partial_MAE', 'delta_RMSE'])

        for col in DF_benchmak.columns[1:]:
            rmse = np.sqrt(mean_squared_error(DF_benchmak['MD'], DF_benchmak[col]))
            r2 = r2_score(DF_benchmak['MD'], DF_benchmak[col])
            mae = mean_absolute_error(DF_benchmak['MD'], DF_benchmak[col])

            p_rmse = np.sqrt(mean_squared_error(partial_DF_benchmak['MD'], partial_DF_benchmak[col]))
            p_r2 = r2_score(partial_DF_benchmak['MD'], partial_DF_benchmak[col])
            p_mae = mean_absolute_error(partial_DF_benchmak['MD'], partial_DF_benchmak[col])

            new_row = ({'Embeddings_Models': col, 'RMSE': rmse, 'partial_RMSE': p_rmse, 'R2': r2,'partial_R2': p_r2, 'MAE': mae, 'partial_MAE': p_mae,'delta_RMSE': rmse-p_rmse})
            DF_Evaluation.loc[len(DF_Evaluation)] = new_row
        DF_Evaluation = DF_Evaluation.sort_values(by='RMSE')
        DF_Evaluation.to_csv(evaluation_result)

Extract_embedding_finetuning_dataset()


def plot_embedding_comparison():
    combinations_for_embedding_finetuning = dir + "DF_Benchmark_embedding_combinations.csv"
    df_pairs= pd.read_csv(combinations_for_embedding_finetuning)
    df_pairs = df_pairs.sample(frac=1, random_state=42)
    #df_pairs = df_pairs[df_pairs.columns][:1000]
    df_pairs = df_pairs[df_pairs["Delta_Tg"] > 450]
    df_pairs = df_pairs.sort_values(by='cosine_sim')
    df_pairs['cosine_sim_finetuned_PB'] = df_pairs.apply(lambda row: cosine_sim(row['SMILES_A'], row['SMILES_B'], "finetuned"),axis=1)
    print(len(df_pairs))
    plt.figure(figsize=(10, 6))
    for col in ['cosine_sim','cosine_sim_finetuned_PB']:  # df_pairs.columns[3:]
        plt.plot(range(len(df_pairs)), df_pairs[col], marker='o', linestyle='', label=col)
    plt.title('Tracé des différentes valeurs de cosine_sim(PB(A),PB(B))')
    plt.xlabel('coupe(A,B), |Tg(A)-Tg(B)|>450')
    plt.ylabel('Cosine_sim des embeddings de A et B')
    plt.legend()  # Affiche la légende
    plt.grid(True)
    plt.show()
#plot_embedding_comparison()




