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
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
random_seed = 42
keras.utils.set_random_seed(random_seed)
numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)
embedding_model='PB_Tg_PI_Exp' #PB_Tg_PI_MM, PB_Tg_PI_Exp
dir = "./Polyimides/"
model_name = "M1_MAE_polyimides_PB_Exp.keras"
link_model = dir + "Model_M1/"+model_name
model_reload = SentenceTransformer(dir + embedding_model)
def finetune_PolyBERT(smiles):
    #model_reload = SentenceTransformer("DeepChem/ChemBERTa-77M-MLM")
    embedding = model_reload.encode(smiles)
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
link_Polyimides_mean = dir + "experimental_db_Polyimides_mean.csv"
link_Polyimides_syn_subset = dir + "Polyimides_synthetic_sub_set.csv"
link = link_Polyimides_mean

DF_polyimides = pd.read_csv(link)
#DF_polyimides[embedding_model] = DF_polyimides['SMILES'].apply(finetune_PolyBERT)
#DF_polyimides.to_csv(link)
DF_polyimides = DF_polyimides.sample(frac=1, random_state=random_seed).reset_index(drop=True)
print(DF_polyimides)
#--------------------------------------------------------------------

def train(DF_polyimides):
    #-----------------------------------------------------MERGE_NN------------------------------------------------------
    unité_Tg = 'Tg'
    X_polyBERT, X_polyBERT_Tg, Y_TG, DF_polyimides = structure_data(DF_polyimides, unité_Tg) # PolyBERT
    Y_TG = DF_polyimides[unité_Tg]
    SEED_list =[2] #[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 42, 43, 47, 53, 59, 61, 71, 73, 79, 83, 89, 97, 101, 103] # 25 premiers nombres premiers + 42.
    RMSE = []; R2 = []; MAE=[]
    """for SEED in SEED_list:
        (X_polyBERT_train, X_polyBERT_test,
         X_polyBERT_Tg_train, X_polyBERT_Tg_test,
         y_train, y_test) = train_test_split(X_polyBERT, X_polyBERT_Tg, Y_TG, test_size=0.2, random_state=SEED)"""
    # Define 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    SEED_list = list(range(10))
    for train_index, test_index in kf.split(X_polyBERT):
        # Split dataset
        X_polyBERT_train, X_polyBERT_test = X_polyBERT[train_index], X_polyBERT[test_index]
        X_polyBERT_Tg_train, X_polyBERT_Tg_test = X_polyBERT_Tg[train_index], X_polyBERT_Tg[test_index]
        y_train, y_test = Y_TG[train_index], Y_TG[test_index]
        # -------------------------------------FFNN-----------------------------------
        K.clear_session()
        keras.utils.set_random_seed(42)
        model = load_model(link_model) #Model to be fine-tuned
        optimizer = keras.optimizers.Adam(learning_rate=0.001) #0.001
        model.compile(optimizer=optimizer,
                      loss="mean_absolute_error") #mean_squared_error, mean_absolute_error
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=5,
            verbose=1,
            restore_best_weights=True
        )
        # ----------------------------------------------------------------------------
        model.summary()
        Model_val = model.fit(x=[X_polyBERT_train, X_polyBERT_Tg_train], y=y_train, epochs=700,
                          batch_size=16,
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
        """for i,j in zip(y_test, y_pred_test):
            print(i,j)
        plt.scatter(y_test, y_pred_test)
        plt.show()"""
        RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        MAE_test = np.sqrt(mean_absolute_error(y_test, y_pred_test))
        print("model performance")
        print("Test set R^2: %.2f" % r2_test)
        print("Test RMSE score: %.2f" % RMSE_test)
        print("Train set R^2: %.2f" % r2_train)
        print("Train RMSE score: %.2f" % RMSE_train)
        R2.append(r2_test)
        RMSE.append(RMSE_test)
        MAE.append(MAE_test)

    mean_R2 = np.mean(R2)
    std_R2 = np.std(R2)
    R2.append(mean_R2)
    R2.append(std_R2)
    print('R2_score:', R2)

    mean_RMSE = np.mean(RMSE)
    std_RMSE = np.std(RMSE)
    RMSE.append(mean_RMSE)
    RMSE.append(std_RMSE)
    print('RMSE:', RMSE)

    mean_MAE = np.mean(MAE)
    std_MAE = np.std(MAE)
    MAE.append(mean_MAE)
    MAE.append(std_MAE)
    print('MAE:', MAE)

    SEED_list.append('mean')
    SEED_list.append('std')
    DF_result = pd.DataFrame(columns=SEED_list)
    DF_result.loc[len(DF_result)] = R2
    DF_result.loc[len(DF_result)] = RMSE
    print(DF_result)
if __name__ == '__main__':
    train(DF_polyimides)
