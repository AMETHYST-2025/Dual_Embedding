#***********************************************************************
# This code develops the prediction of polymer Tg based on
# the modified PolyBERT embedding.
#***********************************************************************
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Dense, Flatten, Activation, ZeroPadding2D
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import re
from keras import backend as K
from keras.layers import concatenate
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
random_seed = 42
keras.utils.set_random_seed(random_seed)
numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)

MODEL_PolyBERT_Benchmark_2branch_100HP_100BM = ['PB_Benchmark_merge_data','M1_merge_data_100_CP_100_BM.keras']
MODEL_PolyBERT_Benchmark_2branch_100HP_100BM_exp = ['PB_Benchmark_merge_data_Exp','M1_merge_data_100_CP_100_BM_Exp.keras']
MODEL_PolyBERT =['PB_Benchmark_merge_data','M1_PB_only.keras']
MODEL_PolyBERT_2 =['PB_Benchmark_merge_data', 'M1_PB_PB.keras']
embedding_model = MODEL_PolyBERT_2[0]
model_name= MODEL_PolyBERT_2[1]

dir= "/Benchmark_Polymers/"

link_model = dir+"Benchmark/save_model_M1/"+model_name
dir_save_model = dir + "finetune_PolyBERT"
polyBERT_ = SentenceTransformer('kuelumbus/polyBERT')
model_reload = SentenceTransformer(dir_save_model +"/"+embedding_model) # model to fine-tuned

def PolyBERT(molecule):
    psmiles_strings = [molecule]
    embeddings = polyBERT_.encode(psmiles_strings)
    return embeddings
def finetune_PolyBERT(smiles):
    embedding = model_reload.encode(smiles)
    return embedding
def EVAL(value):
    #print(value)
    h = value
    h = h.replace(' ', '')
    h = h.replace('array(', '')
    h = h.replace('dtype=float32),', '')
    h = h.replace('nan', '0')
    h = h.replace('NaN', '0')
    h = h.replace('inf', '10000')
    H = eval(h)
    return H

def sub_function(Data_reshape,Dim,pos):
    all_vec = []
    for vec in Data_reshape:
        if pos == 1: vec = eval(vec)
        if pos == 2: vec = EVAL(vec)
        if pos == 3:
            vec = vec.replace(' ', ',')
            vec = vec.replace(',,,,', ',')
            vec = vec.replace(',,,', ',')
            vec = vec.replace(',,', ',')
            vec = vec.replace('[,', '[')
            vec = vec.replace(',]', ']')
            vec = eval(vec)
            #print(len(vec))
        all_vec.append(vec)
    all_vec = np.array(all_vec)
    all_vec = np.nan_to_num(all_vec, posinf=10000, neginf=-10000)
    all_vec = all_vec.reshape((Dim[0], (Dim[1]), (Dim[2])))
    return all_vec
def structure_data(DF):
    Data_reshape_PolyBERT = DF['PolyBERT']  # MorganFP
    Data_reshape_PolyBERT_Tg = DF[embedding_model]

    all_vec_PB = sub_function(Data_reshape_PolyBERT, [len(DF),1,600],3)
    all_vec_PB_Tg = sub_function(Data_reshape_PolyBERT_Tg,[len(DF),1,600],0) #600

    return all_vec_PB, all_vec_PB_Tg, DF


#--------------------------------------------------------------------
link_benchmark_file_for_finetuning = dir + "Benchmark/DF_Benchmark_finetuning.csv"
link_benchmark_file_for_finetuning_PB = dir + "Benchmark/DF_Benchmark_finetuning_PB.csv"
dataset_for_test = dir + "Benchmark/DF_Benchmark_test.csv"
dataset_for_test_PB = dir + "Benchmark/DF_Benchmark_test_PB.csv"

DF_polymer = pd.read_csv(link_benchmark_file_for_finetuning_PB)
DF_polymer = DF_polymer.rename(columns={'SMILES_RepeatUnit_1': 'smiles', 'MD': 'Tg (°C)'})
DF_polymer[embedding_model] = DF_polymer['smiles'].apply(finetune_PolyBERT)
"""DF_polymer['PolyBERT'] = DF_polymer['smiles'].apply(PolyBERT)
DF_polymer.to_csv(link_benchmark_file_for_finetuning_PB)"""

DF_polymer_test = pd.read_csv(dataset_for_test_PB)
DF_polymer_test = DF_polymer_test.rename(columns={'SMILES_RepeatUnit_1': 'smiles', 'MD': 'Tg (°C)'})
DF_polymer_test[embedding_model] = DF_polymer_test['smiles'].apply(finetune_PolyBERT)
"""DF_polymer_test['PolyBERT'] = DF_polymer_test['smiles'].apply(PolyBERT)
DF_polymer_test.to_csv(dataset_for_test_PB)"""

DF_polymer = DF_polymer.sample(frac=1, random_state=random_seed).reset_index(drop=True)
DF_polymer_test = DF_polymer_test.sample(frac=1, random_state=random_seed).reset_index(drop=True) # test
print(DF_polymer)
print(DF_polymer_test)
#--------------------------------------------------------------------
def train(DF_polymer, DF_polymer_test):
    #-----------------------------------------------------MERGE_NN------------------------------------------------------
    unité_Tg = 'Tg (°C)'
    X_polyBERT, X_polyBERT_Tg, DF_polymer = structure_data(DF_polymer)
    Y_Tg = DF_polymer[unité_Tg]

    X_polyBERT_test, X_polyBERT_Tg_test, DF_polymer_test = structure_data(DF_polymer_test)
    Y_Tg_test = DF_polymer_test[unité_Tg]
    print(sum(list(Y_Tg_test)) / len(list(Y_Tg_test)))
    print(np.std(Y_Tg_test, ddof=1))
    # -------------------------------------FFNN-----------------------------------
    # ---------------------------------------------------------------
    """keras.utils.set_random_seed(42)
    K.clear_session()
    keras.utils.set_random_seed(42)
    input_PB = Input(shape=(1, 600,))
    input_finetuned_PB = Input(shape=(1, 600,))

    nbr_a = 32;
    nbr_b = 8
    # PolyBERT branch
    x_polybert = Dense(nbr_a, activation="relu")(input_PB)
    x_polybert = Dense(nbr_a, activation="relu")(x_polybert)

    # Fine-tuned PolyBERT branch
    x_poly_finetune = Dense(nbr_b, activation="relu")(input_finetuned_PB)
    x_poly_finetune = Dense(nbr_b, activation="relu")(x_poly_finetune)

    combined = concatenate([x_polybert, x_poly_finetune])  #
    y = Dense(2, activation="relu")(combined)
    y = Dense(1)(y)

    model = Model(inputs=[input_PB, input_finetuned_PB], outputs=y)"""
    #---------------------------------------------------------------
    model = load_model(link_model) # Load
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="mean_squared_error")
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    # ----------------------------------------------------------------------------
    model.summary()
    Model_val = model.fit(x=[X_polyBERT, X_polyBERT_Tg], y=Y_Tg, epochs=700,
                      batch_size=16,
                      validation_split=0.2,
                      callbacks=[early_stopping])
    # -----------------------------------------------------------------------------
    y_pred_train_ = model.predict([X_polyBERT, X_polyBERT_Tg])
    y_pred_test_ = model.predict([X_polyBERT_test, X_polyBERT_Tg_test])

    y_pred_train = [ele[0][0] for ele in y_pred_train_]
    y_pred_test = [ele[0][0] for ele in y_pred_test_]
    # ----------------------------------------------------------------------------
    loss = Model_val.history['loss']
    val_loss = Model_val.history['val_loss']
    # ----------------------------------------------------------------------------
    r2_train = r2_score(Y_Tg, y_pred_train)
    r2_test = r2_score(Y_Tg_test, y_pred_test)
    MAE_test = mean_absolute_error(Y_Tg_test, y_pred_test)
    RMSE_train = np.sqrt(mean_squared_error(Y_Tg, y_pred_train))
    RMSE_test = np.sqrt(mean_squared_error(Y_Tg_test, y_pred_test))
    print("model performance")
    """print("Train set R^2: %.2f" % r2_train)
    print("Train RMSE score: %.2f" % RMSE_train)"""
    print("Test set R^2: %.2f" % r2_test)
    print("Test RMSE score: %.2f" % RMSE_test)
    print("Test MAE: %.2f" % MAE_test)

if __name__ == '__main__':
    train(DF_polymer, DF_polymer_test)
