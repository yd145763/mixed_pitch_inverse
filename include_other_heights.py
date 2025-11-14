


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import importlib.util
from matplotlib.ticker import StrMethodFormatter
import re
import ast



small_pitch_list = []
big_pitch_list = []
small_count_list = []
big_count_list = []
small_dc_list = []
big_dc_list = []
E20_list = []
E40_list = []
E60_list = []
E80_list = []


full_power_distribution = False
visualize_index = True



wg_length = 10e-6
offset = 5e-6

# Specify the path
path = "G:\\My Drive\\mixed_pitch_inverse_design\\all_gds\\"

# Get a list of all files in the directory
files = os.listdir(path)

# Filter the GDS files and extract their names without the extension
gds_files = [filename[:-4] for filename in files if filename.endswith('end.gds')]
gds_files.sort(key=str.lower)




df_results = pd.DataFrame(columns=[f'col{i}' for i in range(10)])
df_results.columns = ['sp', 'bp', 'sc', 'bc', 'sdc', 'bdc', 'E20', 'E40', 'E60', 'E80']



def getfeature(column_name, df_name, row_index, multiple):
    feature = df_name[column_name]
    feature = [i for i in feature]
    feature = feature[row_index]
    feature = int(feature*multiple)
    feature = str(feature)
    return feature

ran_file_list = []

print("loading ran files")
df0 = pd.read_csv("G:\\My Drive\\\mixed_pitch_inverse_design\\compiled.csv", index_col = 0)
print("loaded df0")
df1 = pd.read_csv("G:\\My Drive\\\mixed_pitch_inverse_design\\from mis.csv", index_col = 0)
print("loaded df1")
df2 = pd.read_csv("G:\\My Drive\\\mixed_pitch_inverse_design\\from6 march.csv", index_col = 0)
print("loaded df2")
df3 = pd.read_csv("G:\\My Drive\\\mixed_pitch_inverse_design\\from13 march.csv", index_col = 0)
print("loaded df3")
df4 = pd.read_csv("G:\\My Drive\\\mixed_pitch_inverse_design\\from24 march.csv", index_col = 0)
print("loaded df4")

df_con = pd.concat([df0, df1, df2, df3, df4], axis = 0)
df_dropped = df_con.drop_duplicates(keep = False)
df_dropped = df_dropped.reset_index()
df_ran = df_dropped.iloc[:, 1:]

print("loaded ran files")

from scipy.signal import find_peaks, peak_widths

indicator = []
for row in range(len(df_ran)):
    single_row = df_ran.iloc[row,:]
    e = single_row[-1]
    lst = ast.literal_eval(e)
    lst = lst[:430]
    lst = lst[::2]
    lst = np.array(lst)
    def findpeaks(e):
        

        # Step 1: Find peaks
        peaks, _ = find_peaks(e)
        
        # Step 2: Calculate FWHM (Full Width at Half Maximum)
        results_half = peak_widths(e, peaks, rel_height=0.5)
        
        a = results_half[2]
        a = a/len(e)*155
        b = results_half[3]
        b = b/len(e)*155
        results_half_abs = (results_half[0], results_half[1], a, b)
        maxpoints = e[peaks]
        maxpoint_index = np.argmax(maxpoints)
        maxpoint_x =(np.argmax(e))*(155/(len(e)))
        maxpoint = max(e)
        
        lefts = results_half_abs[2]
        left = lefts[maxpoint_index]
        rights = results_half_abs[3]
        right = rights[maxpoint_index]


        
        return maxpoint_x, maxpoint, left, right, e, peaks, results_half_abs
    
    maxpoint_x, maxpoint, left, right, e, peaks, results_half_abs = findpeaks(lst)
    
    peak_intensity_list = results_half_abs[1]
    
    q3 = np.percentile(peak_intensity_list, 75)
    max_point = max(peak_intensity_list)
    
    if max_point < 2*q3:
        indicator.append(0)
        print('removed', row)
    else:
        print(row)
        indicator.append(1)

df_ran['indicator'] = indicator
df_ran = df_ran[df_ran['indicator'] == 1]
df_ran = df_ran.drop(columns=['indicator'])
df_ran.index = range(len(df_ran))

import pandas as pd 



from sklearn.preprocessing import StandardScaler
y = []




X = []

columns = ['E20', 'E40', 'E60', 'E80']
indicators = [0.02, 0.04, 0.06, 0.08]

for col, ind in zip(columns, indicators):
    X_str = df_ran[col]
    
    
    for i in range(len(X_str)):
        print(col, i)
        E = X_str[i]
        lst = ast.literal_eval(E)
        lst = lst[:430]
        lst = lst[::2]
        lst = np.array(lst)
        label = np.full(len(lst), ind)
        combined = np.column_stack((lst, label))
        X.append(combined)
        y.append(df_ran.iloc[i, :6])

X = np.array(X)
y = np.array(y)
scaler_y = StandardScaler()
scaler_x = StandardScaler()
y_normalized = scaler_y.fit_transform(y)



X_normalized = X




from sklearn.model_selection import train_test_split



X_test = X_normalized[::10, :, :]
y_test = y_normalized[::10, :]

test_drop_index = np.arange(0, X_normalized.shape[0], 10)
all_indices = np.arange(X_normalized.shape[0])
mask = ~np.isin(all_indices, test_drop_index)
X_TV = X_normalized[mask]

y_TV = y_normalized[mask]


X_TV = np.array(X_TV)
y_TV = np.array(y_TV)
#X_TV, X_test, y_TV, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_TV, y_TV, test_size=0.25, random_state=42)


import tensorflow as tf### models


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                     Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                     RandomContrast, Rescaling, Resizing, Reshape)
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.callbacks import (Callback, CSVLogger, EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint, ReduceLROnPlateau)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Available: ", tf.test.is_gpu_available())

CONFIGURATION = {
    "SEQ_LENGTH": len(lst),
    "LEARNING_RATE": 0.01,
    "N_EPOCHS": 5,
    "DROPOUT_RATE": 0,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 2,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 1024,
    "N_DENSE_2": 256,
    "NUM_CLASSES": 6,
    "PROJ_DIM": 192,
    "CLASS_NAMES": ["Region A", "Region B", "Region C", "Region D"],
    "HIDDEN_SIZE": 2,
}


class PatchEncoder(Layer):
    def __init__(self, SEQ_LENGTH, HIDDEN_SIZE):
        super(PatchEncoder, self).__init__(name='patch_encoder')
        self.linear_projection = Dense(HIDDEN_SIZE)
        self.positional_embedding = Embedding(SEQ_LENGTH, HIDDEN_SIZE)
        self.SEQ_LENGTH = SEQ_LENGTH

    def call(self, x):
        embedding_input = tf.range(start=0, limit=self.SEQ_LENGTH, delta=1)
        position_embeddings = self.positional_embedding(embedding_input)
        position_embeddings = tf.reshape(position_embeddings, [1, self.SEQ_LENGTH, -1])
        position_embeddings = tf.tile(position_embeddings, [tf.shape(x)[0], 1, 1])
        output = self.linear_projection(x) + position_embeddings
        return output


class TransformerEncoder(Layer):
  def __init__(self, N_HEADS, HIDDEN_SIZE):
    super(TransformerEncoder, self).__init__(name='transformer_encoder')
    self.layer_norm_1 = LayerNormalization()
    self.layer_norm_2 = LayerNormalization()
    self.multi_head_att = MultiHeadAttention(N_HEADS, HIDDEN_SIZE)
    self.dense_1 = Dense(HIDDEN_SIZE, activation=tf.nn.elu)
    self.dense_2 = Dense(HIDDEN_SIZE, activation=tf.nn.elu)

  def call(self, input):
    x_1 = self.layer_norm_1(input)
    x_1 = self.multi_head_att(x_1, x_1)
    x_1 = Add()([x_1, input])
    x_2 = self.layer_norm_2(x_1)
    x_2 = self.dense_1(x_2)
    output = self.dense_2(x_2)
    output = Add()([output, x_1])
    return output

class ViT(Model):
    def __init__(self, N_HEADS, HIDDEN_SIZE, SEQ_LENGTH, N_LAYERS, N_DENSE_UNITS):
        super(ViT, self).__init__(name='vision_transformer')
        self.N_LAYERS = N_LAYERS
        self.patch_encoder = PatchEncoder(SEQ_LENGTH, HIDDEN_SIZE)
        self.trans_encoders = [TransformerEncoder(N_HEADS, HIDDEN_SIZE) for _ in range(N_LAYERS)]
        self.flatten = Flatten()
        self.dense_1 = Dense(N_DENSE_UNITS, activation=tf.nn.elu)
        self.dropout_1 = Dropout(CONFIGURATION['DROPOUT_RATE'])
        self.dense_2 = Dense(int(N_DENSE_UNITS/2), activation=tf.nn.elu)
        self.dropout_2 = Dropout(CONFIGURATION['DROPOUT_RATE'])
        self.dense_3 = Dense(CONFIGURATION["NUM_CLASSES"], activation=tf.nn.elu)

    def call(self, input, training=True):
        x = tf.reshape(input, (-1, CONFIGURATION["SEQ_LENGTH"], CONFIGURATION["HIDDEN_SIZE"]))  # Reshape to the correct shape
        x = self.patch_encoder(x)
        for i in range(self.N_LAYERS):
            x = self.trans_encoders[i](x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout_1(x, training=training)
        x = self.dense_2(x)
        x = self.dropout_2(x, training=training)
        return self.dense_3(x)

trans_layersS = [4] #1,2,3,4
layer_sizeS = [512] # 16. 32
batch_sizeS = [400]


for trans_layers in trans_layersS:
    for layer_size in layer_sizeS:
        for batch_size in batch_sizeS:
            batch_size = batch_size

    
            layer_size = layer_size
            trans_layers = trans_layers
    
            # Define the model
            vit = ViT(
                N_HEADS=2,
                HIDDEN_SIZE=CONFIGURATION["HIDDEN_SIZE"],
                SEQ_LENGTH=CONFIGURATION["SEQ_LENGTH"],
                N_LAYERS=trans_layers,  # Example: 1 transformer layer
                N_DENSE_UNITS=layer_size  # Example dense unit size
            )
            



            def mean_percentage_error(y_true, y_pred):
                return tf.reduce_mean(abs((y_pred - y_true) / (y_true + tf.keras.backend.epsilon())))
    
            # Compile the model
            vit.compile(optimizer=tf.keras.optimizers.Adadelta(CONFIGURATION["LEARNING_RATE"]),
                        loss='mean_absolute_error',
                        metrics=['mean_absolute_error'])
    
            # Define the EarlyStopping callback
            early_stopping_callback = EarlyStopping(
                monitor='val_mean_absolute_error',  # Monitor training loss
                min_delta=0.003,  # Minimum change to qualify as an improvement
                patience=30,  # Number of epochs with no improvement after which training will be stopped
                verbose=1,  # Verbosity mode
                mode='min',  # Maximize the monitored quantity
                restore_best_weights=True  # Whether to restore model weights to the best observed during training
            )
            

            history = vit.fit(X_train, y_train, epochs=1000,validation_data=(X_val, y_val), batch_size = batch_size, 
                                callbacks=[early_stopping_callback]
                                )
            df_loss = pd.DataFrame({
                'epoch': range(1, len(history.history['loss']) + 1),
                'train_loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            })

            #df_loss.to_csv('G:\\My Drive\\mixed_pitch_inverse_design\\loss_t'+str(trans_layers)+'_l'+str(layer_size)+'.csv', index=False)
            
            vit.summary()
            
            y_pred_normalized = vit.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_normalized)
            y_test_denormalized = scaler_y.inverse_transform(y_test)

            for j in range(y_pred.shape[1]):
                for i in range(y_pred.shape[0]):
                    ape = (y_pred[i,j] - y_test_denormalized[i,j])/y_test_denormalized[i,j]


            y_pred = pd.DataFrame(y_pred)
            #y_pred.to_csv('G:\\My Drive\\mixed_pitch_inverse_design\\pred_t'+str(trans_layers)+'_l'+str(layer_size)+'.csv')
            
            #vit.save('G:\\My Drive\\mixed_pitch_inverse_design\\model_t'+str(trans_layers)+'_l'+str(layer_size)+'.h5')

"""
vit.summary()

e = X_train[0,0,:,0]

import numpy as np

import random


def lorentzian_peak(x, x0, gamma, A):
    return A * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2)

def gaussian_peak(x, x0, gamma, A):
    sigma = gamma / 2.355  # Convert FWHM to standard deviation
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

df_lorentz = pd.DataFrame(columns=["fwhm", "peak", "position", 'sp', 'bp', 'sc', 'bc', 'sdc', 'bdc'])


for x0 in [90, 100, 110, 120, 130]:  # Rename scalar x to x0 directly
    for fwhm in [15, 20, 25, 30, 25, 40]:
        for Peak in [0.06, 0.08, 0.10, 0.12]:
            gamma = fwhm
            A = Peak
            x_vals = np.linspace(0, 155, 215)
            y = lorentzian_peak(x_vals, x0, gamma, A)
            y = [i+0.02 for i in y]
            y = [i*random.uniform(1, 1.2) for i in y]
    
            # Plot
            plt.plot(x_vals, y, label=f"x0={x0}, FWHM={fwhm}")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title("Lorentzian Peaks")
            plt.grid(True)
            plt.legend()
            plt.show()
    
            
            y = np.array(y)
            y_single = y.reshape(1,len(y),1)
            meow = vit.predict(y_single)
            meow_denormalized = scaler_y.inverse_transform(meow)
            meow_denormalized = meow_denormalized.reshape(meow_denormalized.shape[1])
            meow_denormalized = [i for i in meow_denormalized]
            row = [fwhm, Peak, x0] + meow_denormalized
            df_lorentz.loc[len(df_lorentz)] = row
            
            
            
            
df_gaussian = pd.DataFrame(columns=["fwhm", "peak", "position", 'sp', 'bp', 'sc', 'bc', 'sdc', 'bdc'])


for x0 in [90, 100, 110, 120, 130]:  # Rename scalar x to x0 directly
    for fwhm in [15, 20, 25, 30, 25, 40]:
        for Peak in [0.06, 0.08, 0.10, 0.12]:
            gamma = fwhm
            A = Peak
            x_vals = np.linspace(0, 155, 215)

            
            y = gaussian_peak(x_vals, x0, gamma, A)
            y = [i+0.02 for i in y]
            y = [i*random.uniform(1, 1.2) for i in y]
            
            # Plot
            fig = plt.figure(figsize=(4, 4))
            ax = plt.axes()
            ax.plot(x_vals, y)

            #graph formatting     
            ax.tick_params(which='major', width=2.00)
            ax.tick_params(which='minor', width=2.00)
            ax.xaxis.label.set_fontsize(15)
            ax.xaxis.label.set_weight("bold")
            ax.yaxis.label.set_fontsize(15)
            ax.yaxis.label.set_weight("bold")
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_yticklabels(ax.get_yticks(), weight='bold')
            ax.set_xticklabels(ax.get_xticks(), weight='bold')
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            plt.xlabel("x-position (μm)")
            plt.ylabel("E-Field (eV)")
            plt.legend(prop={'weight': 'bold','size': 10}, loc = "best")
            plt.title(f"Computed FWHM={fwhm}, Peak = {Peak}, x0={x0}\n")
            plt.show()
            plt.close()
            
            y = np.array(y)
            # Prepare for model input
            y_single = y.reshape(1, len(y), 1)
            meow = vit.predict(y_single)
            meow_denormalized = scaler_y.inverse_transform(meow)
            meow_denormalized = meow_denormalized.reshape(meow_denormalized.shape[1])
            meow_denormalized = [i for i in meow_denormalized]
            row = [fwhm, Peak, x0] + meow_denormalized
            df_gaussian.loc[len(df_gaussian)] = row

#df_lorentz.to_csv("G:\\My Drive\\mixed_pitch_inverse_design\\df_lorentznoisy.csv")
#df_gaussian.to_csv("G:\\My Drive\\mixed_pitch_inverse_design\\df_gaussiannoisy.csv")
"""