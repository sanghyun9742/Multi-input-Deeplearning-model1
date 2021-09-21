import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
from keras import models, layers
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5900)])
import numpy as np
import pandas as pd
import os
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw,MACCSkeys, AllChem
from tqdm import tqdm
from tensorflow.keras.layers import Embedding, Dense, LSTM,GRU,SimpleRNN,Bidirectional,concatenate,BatchNormalization,Flatten,MaxPooling2D,Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from deepchem.feat.smiles_tokenizer import SmilesTokenizer, BasicSmilesTokenizer
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dense, LSTM,GRU,SimpleRNN,Bidirectional,concatenate,BatchNormalization,Flatten,MaxPooling2D
import tensorflow.keras
train=pd.read_csv('./train.csv')
dev = pd.read_csv('./dev.csv')
test = pd.read_csv('./test.csv')
train.head()
dev.head()
train=pd.concat([train,dev])




def get_macckeys(bbb):
  print("get macckeys...")
  train_fps=[]
  for index, row in tqdm(bbb.iterrows()):
     mol = Chem.MolFromSmiles(row['SMILES'])
     fp = MACCSkeys.GenMACCSKeys(mol)
     train_fps.append(fp)
  return train_fps


def get_morgan(bbb):
  print("get morgan...")
  train_fps=[]
  for index, row in tqdm(bbb.iterrows()):
     mol = Chem.MolFromSmiles(row['SMILES'])
     fp = AllChem.GetMorganFingerprintAsBitVect(mol,4)
     train_fps.append(fp)
  return train_fps


def get_pattern(bbb):
  print("get pattern...")
  train_fps=[]
  for index, row in tqdm(bbb.iterrows()):
     mol = Chem.MolFromSmiles(row['SMILES'])
     fp = Chem.rdmolops.PatternFingerprint(mol)
     train_fps.append(fp)
  return train_fps
  
  
def get_rdkit(bbb):
  print("get rdkit...")
  train_fps=[]
  for index, row in tqdm(bbb.iterrows()):
     mol = Chem.MolFromSmiles(row['SMILES'])
     fp = Chem.RDKFingerprint(mol)
     train_fps.append(fp)
  return train_fps

def get_layerd(bbb):
  print("get layerd...")
  train_fps=[]
  for index, row in tqdm(bbb.iterrows()):
     mol = Chem.MolFromSmiles(row['SMILES'])
     fp = Chem.rdmolops.LayeredFingerprint(mol)
     train_fps.append(fp)
  return train_fps




def getgap():
    train['ST1_GAP(eV)'] = train['S1_energy(eV)'] - train['T1_energy(eV)']
    train['ST1_GAP(eV)']=np.array(train['ST1_GAP(eV)'])
    return train['ST1_GAP(eV)'] 



x_train1=get_macckeys(train)
x_train2=get_morgan(train)
x_train3=get_pattern(train)
x_train4=get_rdkit(train)
x_train5=get_layerd(train)



y_t=getgap()
aa, bb ,cc,dd,ee,y_t= shuffle(x_train1,x_train2, x_train3, x_train4,x_train5,y_t,random_state=42)
x_train1 = aa[:27000]
x_train2= bb[:27000]
x_train3 = cc[:27000]
x_train4= dd[:27000]
x_train5 = ee[:27000]
y_t1 = y_t[:27000]
x_val1 = aa[27000:]
x_val2 =bb[27000:]
x_val3 = cc[27000:]
x_val4= dd[27000:]
x_val5 = ee[27000:]
y_t2 = y_t[27000:]

print("++++++++++++++++++++++++++++++++++++++++++++")
print(np.shape(x_train1))
print(np.shape(x_train2))
print(np.shape(x_train3))
print(np.shape(x_train4))
print(np.shape(x_train5))
print("++++++++++++++++++++++++++++++++++++++++++++")


inputs1= Input(shape=(167,))
x=Dense(128,  kernel_initializer='normal', activation='relu')(inputs1)
outputs1=Dense(256, activation='relu')(x)

model1 = Model(inputs=inputs1, outputs=outputs1)


inputs2= Input(shape=(2048,))
x=Dense(128,  kernel_initializer='normal', activation='relu')(inputs2)
outputs2=Dense(256, activation='relu')(x)

model2 = Model(inputs=inputs2, outputs=outputs2)


inputs3= Input(shape=(2048,))
x=Dense(128,  kernel_initializer='normal', activation='relu')(inputs3)
outputs3=Dense(256, activation='relu')(x)

model3 = Model(inputs=inputs3, outputs=outputs3)


inputs4= Input(shape=(2048,))
x=Dense(128,  kernel_initializer='normal', activation='relu')(inputs4)
outputs4=Dense(256, activation='relu')(x)

model4 = Model(inputs=inputs4, outputs=outputs4)




inputs5= Input(shape=(2048,))
x=Dense(128,  kernel_initializer='normal', activation='relu')(inputs5)
outputs5=Dense(256, activation='relu')(x)

model5 = Model(inputs=inputs5, outputs=outputs5)


concatenated = concatenate([model1.output, model2.output,model3.output,model4.output,model5.output])
concatenated = Dense(128,  kernel_initializer='normal', activation='relu')(concatenated)
concat_out = Dense(1, kernel_initializer='normal')(concatenated)


model = models.Model([inputs1, inputs2,inputs3,inputs4,inputs5], concat_out)



model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit([x_train1,x_train2,x_train3,x_train4,x_train5],y_t1,validation_data=([x_val1,x_val2,x_val3,x_val4,x_val5],y_t2),epochs=12)



from sklearn.metrics import mean_absolute_error, mean_squared_error

prediction = model.predict([x_val1,x_val2,x_val3,x_val4,x_val5])
print(prediction[0])
mae = mean_absolute_error(y_t2, prediction)
mse = mean_squared_error(y_t2, prediction)
    
print('MAE score:', round(mae, 4))
print('MSE score:', round(mse,4))




