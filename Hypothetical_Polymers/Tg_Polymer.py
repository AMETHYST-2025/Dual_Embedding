from keras import backend as K
from keras.layers import concatenate
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Dense, Flatten, Activation, ZeroPadding2D
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import re
from sentence_transformers import SentenceTransformer
random_seed = 42
keras.utils.set_random_seed(random_seed)
dir = "/Hypothetical_Polymers/"
# dataset from PolyBERT file "polyOne_hx"
dir_save_model = dir + "Embedding/"

finetuned_model_name ="PB_Tg_copolymers_MM_01" # PB_Tg_copolymers_Exp_01, PB_Tg_copolymers_MM_01
model_reload = SentenceTransformer(dir_save_model + finetuned_model_name)
def finetune_PolyBERT(smiles):
    embedding = model_reload.encode(smiles)
    embeddings = str(embedding)
    embeddings = embeddings.replace(' ',',')
    embeddings = embeddings.replace('[,', '[')
    embeddings = embeddings.replace(',]', ']')
    embeddings = embeddings.replace(',,,,', ',')
    embeddings = embeddings.replace(',,,', ',')
    embeddings = embeddings.replace(',,', ',')
    return embeddings
def convert(val):
    return val+273.15
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
    Data_reshape_PolyBERT = DF['PolyBERT']
    Data_reshape_PolyBERT_Tg = DF[finetuned_model_name]
    Data_reshape_PolyBERT_Eea = DF["PB_Eea_Exp"]
    Data_reshape_PolyBERT_Td = DF["PB_Td_MM"]
    Data_reshape_PolyBERT_Cp = DF["PB_Cp_Exp"]
    #Data_reshape_PolyBERT_Tg3 = DF["PB_Tg_copolymers_Exp_01"]

    all_vec_PB = sub_function(Data_reshape_PolyBERT, [len(DF),1,600],1)
    all_vec_PB_Tg = sub_function(Data_reshape_PolyBERT_Tg,[len(DF),1,600],1)
    all_vec_PB_Eea = sub_function(Data_reshape_PolyBERT_Eea, [len(DF), 1, 600], 1)
    all_vec_PB_Td = sub_function(Data_reshape_PolyBERT_Td, [len(DF), 1, 600], 1)
    all_vec_PB_Cp = sub_function(Data_reshape_PolyBERT_Cp, [len(DF), 1, 600], 1)

    return all_vec_PB, all_vec_PB_Tg, all_vec_PB_Eea, all_vec_PB_Td, all_vec_PB_Cp, DF


link_copolymer = dir + "Data.csv"
#--------------------------------------------------------------------
DF_copolymer = pd.read_csv(link_copolymer)
"""DF_copolymer['Tg'] = DF_copolymer['Tg (°C)'].apply(convert)
df=DF_copolymer[['Tg','Td','Cp','Eea']][:]
corr_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='grey', fmt=".2f", linewidths=1, annot_kws={"size": 14})
plt.title("Correlation Matrix")
plt.show()"""
#histogram
"""plt.hist(DF_copolymer['Eea'][:], bins=30, edgecolor='black')
plt.xlabel('Eea (eV)')# Cp (J/gK), Eea (eV)
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()"""
DF_copolymer = DF_copolymer[['smiles', 'Tg (°C)', 'Td', 'Cp', 'Eea', 'PolyBERT', "PB_Tg_copolymers_Exp_01", "PB_Tg_copolymers_MM_01", "PB_Tg2_MM", "PB_Tg2_Exp","PB_Eea_MM","PB_Eea_Exp", "PB_Td_Exp", "PB_Td_MM", "PB_Cp_MM", "PB_Cp_Exp","PB_ln_Exp","PB_ln_MM"]][:]
#DF_copolymer[finetuned_model_name] = DF_copolymer['smiles'].apply(finetune_PolyBERT)
#DF_copolymer.to_csv(link_copolymer)
#--------------------------------------------------------------------

print(DF_copolymer)
DF_copolymer = DF_copolymer.sample(frac=1, random_state=random_seed)
def train(DF_copolymer):
    #-----------------------------------------------------MERGE_NN------------------------------------------------------
    unité_Tg = 'Tg (°C)' #Tg (°C)
    X_polyBERT, X_polyBERT_Tg, X_polyBERT_Tg1, X_polyBERT_Tg2, X_polyBERT_Tg3, DF_copolymer = structure_data(DF_copolymer)
    Y_TG = DF_copolymer[unité_Tg]
    SEED_list = [2]# Since the embedding is fine-tuned on a portion of the training set, we consider only a single sample seed to avoid confusing the fine-tuning dataset of the embedding with the test set of the prediction model.
    RMSE = []; R2 = []
    for SEED in SEED_list:
        (X_polyBERT_train, X_polyBERT_test,
         X_polyBERT_Tg_train, X_polyBERT_Tg_test,
         X_polyBERT_Eea_train, X_polyBERT_Eea_test, # Eea
         X_polyBERT_Td_train, X_polyBERT_Td_test, # Td
         X_polyBERT_Cp_train, X_polyBERT_Cp_test, # Cp
         y_train, y_test) = train_test_split(X_polyBERT, X_polyBERT_Tg, X_polyBERT_Tg1, X_polyBERT_Tg2, X_polyBERT_Tg3, Y_TG, test_size=0.2, random_state=SEED)
        # -------------------------------------FFNN-----------------------------------
        print(sum(list(y_test)) / len(list(y_test)))
        print(np.std(y_test, ddof=1))
        K.clear_session()
        keras.utils.set_random_seed(42)
        inputA = Input(shape=(1, 600, )) #PB
        inputF = Input(shape=(1, 600, )) #PB fine-tuned with Tg
        inputF1 = Input(shape=(1, 600,)) #PB fine-tuned with Eea
        inputF2 = Input(shape=(1, 600,)) #PB fine-tuned with Td
        inputF3 = Input(shape=(1, 600,)) #PB fine-tuned with Cp
        nbr_a = 32; nbr_b = 8; nbr_aa = 64
        nbr_c = 2; nbr_cc = 128
        # The first branch operates on the first input
        x_polybert = Dense(nbr_a, activation="relu")(inputA)
        x_polybert = Dense(nbr_a, activation="relu")(x_polybert)

        # Fine-tuned PolyBERT
        x_poly_finetune = Dense(nbr_b, activation="relu")(inputF1)
        x_poly_finetune = Dense(nbr_b, activation="relu")(x_poly_finetune)

        # ----------------------------------------------------------------------
        """x_poly_finetune1 = Dense(nbr_b, activation="relu")(inputF1)
        x_poly_finetune1 = Dense(nbr_b, activation="relu")(x_poly_finetune1)

        x_poly_finetune2 = Dense(nbr_b, activation="relu")(inputF2)
        x_poly_finetune2 = Dense(nbr_b, activation="relu")(x_poly_finetune2)

        x_poly_finetune3 = Dense(nbr_b, activation="relu")(inputF3)
        x_poly_finetune3 = Dense(nbr_b, activation="relu")(x_poly_finetune3)"""
        # ----------------------------------------------------------------------

        combined = concatenate([x_polybert, x_poly_finetune]) #x_poly_finetune1, x_poly_finetune2, x_poly_finetune3
        y = Dense(nbr_c, activation="relu")(combined) #nbr_c
        y = Dense(1)(y)

        model = Model(inputs=[inputA, inputF, inputF1, inputF2, inputF3], outputs=y) #F1, F2, F3 = Eea, Td, Cp
        optimizer = keras.optimizers.Adam(learning_rate=0.0001) #0.0001
        model.compile(optimizer=optimizer,
                      loss="mean_absolute_error") #mean_squared_error, #mean_absolute_error
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=5,
            verbose=1,
            restore_best_weights=True
        )
        # ----------------------------------------------------------------------------
        model.summary()
        Model_val = model.fit(x=[X_polyBERT_train, X_polyBERT_Tg_train, X_polyBERT_Eea_train, X_polyBERT_Td_train, X_polyBERT_Cp_train], y=y_train, epochs=1000,
                          batch_size=3, #16
                          validation_split=0.2,
                          callbacks=[early_stopping])
        # -----------------------------------------------------------------------------
        y_pred_train_ = model.predict([X_polyBERT_train, X_polyBERT_Tg_train, X_polyBERT_Eea_train, X_polyBERT_Td_train, X_polyBERT_Cp_train])
        y_pred_test_ = model.predict([X_polyBERT_test, X_polyBERT_Tg_test, X_polyBERT_Eea_test, X_polyBERT_Td_test, X_polyBERT_Cp_test])
        y_pred_train = [ele[0] for ele in y_pred_train_]
        y_pred_test = [ele[0] for ele in y_pred_test_]
        # ----------------------------------------------------------------------------
        loss = Model_val.history['loss']
        val_loss = Model_val.history['val_loss']
        # ----------------------------------------------------------------------------
        list_label = list(DF_copolymer[unité_Tg])
        pourcentage = 0.1
        gap = pourcentage * (sum(list_label) / len(list_label))
        print(gap) # valeur de reference 12.2 °C, qui represente approximativement 10% de la valeur moyenne des EA
        # ----------------------------------------------------------------------------
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        max_ele = max(list(y_test)); min_ele = min(list(y_test))
        # ----------------------------------------------------------------------------
        # model evaluation
        print("0.1 x Moyenne: %.2f" % gap)
        print("model performance")
        print("Test set R^2: %.2f" % r2_test)
        print("Test RMSE score: %.2f" % RMSE_test)

        R2.append(r2_test)
        RMSE.append(RMSE_test)

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

    SEED_list.append('mean')
    SEED_list.append('std')
    DF_result = pd.DataFrame(columns=SEED_list)
    DF_result.loc[len(DF_result)] = R2
    DF_result.loc[len(DF_result)] = RMSE
    print(DF_result)
if __name__ == '__main__':
    train(DF_copolymer)
