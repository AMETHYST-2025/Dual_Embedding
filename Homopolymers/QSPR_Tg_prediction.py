import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Dense, Flatten, Activation, ZeroPadding2D
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import re
import torch
from keras import backend as K
from keras.layers import concatenate
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import random
random_seed = 42
keras.utils.set_random_seed(random_seed)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)
embedding_model='PB_QSPR_Exp' #PB_Tg_PI_MM, PB_Tg_PI_Exp, PB_QSPR_log_Exp, PB_QSPR_log10_MM
dir = "/Homopolymers/"
model_reload = SentenceTransformer(dir + embedding_model)
model_reload.eval()
def finetune_PolyBERT(smiles):
    embedding = model_reload.encode(smiles, device="cpu")
    embedding = str(embedding)
    embedding = embedding.replace(' ', ',')
    embedding = embedding.replace('[,', '[')
    embedding = embedding.replace(',]', ']')
    embedding = embedding.replace(',,,,', ',')
    embedding = embedding.replace(',,,', ',')
    embedding = embedding.replace(',,', ',')
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
def structure_data(DF, unité):
    all_vec_Tg = []
    Data_reshape_PolyBERT = DF['PolyBERT']  # MorganFP
    Data_reshape_PolyBERT_Tg = DF[embedding_model] #embedding_model
    Data_reshape_Tg = DF[unité]

    for vec_Tg in Data_reshape_Tg:
        all_vec_Tg.append(vec_Tg)

    all_vec_Tg = np.array(all_vec_Tg)

    all_vec_PB = sub_function(Data_reshape_PolyBERT, [len(DF),1,600],1)
    all_vec_PB_Tg = sub_function(Data_reshape_PolyBERT_Tg,[len(DF),1,600],1) #600

    return all_vec_PB, all_vec_PB_Tg, all_vec_Tg, DF


#--------------------------------------------------------------------
link_QSPR_data_train = dir + "QSPR_data_train.csv"
link_QSPR_data_test = dir + "QSPR_data_test.csv"

QSPR_data_train = pd.read_csv(link_QSPR_data_train)
QSPR_data_train = QSPR_data_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)
#QSPR_data_train[embedding_model] = QSPR_data_train['SMILES'].apply(finetune_PolyBERT)
#QSPR_data_train.to_csv(link_QSPR_data_train)

QSPR_data_test = pd.read_csv(link_QSPR_data_test)
QSPR_data_test = QSPR_data_test.sample(frac=1, random_state=random_seed).reset_index(drop=True)
#QSPR_data_test[embedding_model] = QSPR_data_test['SMILES'].apply(finetune_PolyBERT)
#QSPR_data_test.to_csv(link_QSPR_data_test)


def train(DF_train, DF_test):
    #-----------------------------------------------------MERGE_NN------------------------------------------------------
    unité_Tg = 'logTg' #logTg, Tg
    X_polyBERT_train, X_polyBERT_Tg_train, Y_TG_train, DF_polyimides = structure_data(DF_train, unité_Tg) # PolyBERT
    y_train = DF_polyimides[unité_Tg]
    X_polyBERT_test, X_polyBERT_Tg_test, Y_TG_test, DF_polyimides = structure_data(DF_test, unité_Tg)  # PolyBERT
    y_test = DF_polyimides[unité_Tg]
    print(sum(list(y_test)) / len(list(y_test)))
    print(np.std(y_test, ddof=1))
    # -------------------------------------FFNN-----------------------------------
    K.clear_session()
    keras.utils.set_random_seed(42)
    input_PB = Input(shape=(1, 600, ))
    input_finetuned_PB = Input(shape=(1, 600, ))

    nbr_a = 32; nbr_b = 8
    # PolyBERT branch
    x_polybert = Dense(nbr_a, activation="relu")(input_PB)
    x_polybert = Dense(nbr_a, activation="relu")(x_polybert)

    # Fine-tuned PolyBERT branch
    x_poly_finetune = Dense(nbr_b, activation="relu")(input_finetuned_PB) #input_fine-tuned_PB
    x_poly_finetune = Dense(nbr_b, activation="relu")(x_poly_finetune)

    combined = concatenate([x_polybert, x_poly_finetune]) #
    y = Dense(2, activation="relu")(combined)
    y = Dense(1)(y)

    model = Model(inputs=[input_PB, input_finetuned_PB], outputs=y)
    optimizer = keras.optimizers.Adam(learning_rate=0.02) #0.001
    model.compile(optimizer=optimizer,
                  loss="mean_absolute_error") #mean_absolute_error, mean_squared_error
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    # ----------------------------------------------------------------------------
    model.summary()
    Model_val = model.fit(x=[X_polyBERT_train, X_polyBERT_Tg_train], y=y_train, epochs=700,
                      batch_size=13,
                      validation_split=0.2,
                      callbacks=[early_stopping])
    # -----------------------------------------------------------------------------
    y_pred_train_ = model.predict([X_polyBERT_train, X_polyBERT_Tg_train])
    y_pred_test_ = model.predict([X_polyBERT_test, X_polyBERT_Tg_test])

    y_pred_train = [ele[0][0] for ele in y_pred_train_]
    y_pred_test = [ele[0][0] for ele in y_pred_test_]
    # ----------------------------------------------------------------------------
    loss = Model_val.history['loss']
    val_loss = Model_val.history['val_loss']
    # ----------------------------------------------------------------------------
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    MAE_test = np.sqrt(mean_absolute_error(y_test, y_pred_test))
    #print("Train set R^2: %.2f" % r2_train)
    #print("Train RMSE score: %.2f" % RMSE_train)
    print("model performance")
    print("Test R^2: %.4f" % r2_test)
    print("Test RMSE score: %.4f" % RMSE_test)
    #print("Test MAE score: %.2f" % MAE_test)
if __name__ == '__main__':
    train(QSPR_data_train,QSPR_data_test)
