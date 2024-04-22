import h5py
import timm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from ZOO_lightning import LitNeuralNet

DTYPE = torch.float32
DEVICE = 'mps:0'

# Load the Data
SEED = 42
DATA_PATH = '/Volumes/ES-HDD-Documents/Documents/matlas_dwarfs/NGC4249_224_patches.h5'
WEIGHTS_PATH = './effnetb0_color_224px.ckpt'

CHECKPOINT_PATH = './ZOO-e100-lr0.0003-bs2-seed42/ZOO-epoch=11-train_loss=0.0171-valid_loss=0.0088.ckpt'
CHECKPOINT_PATH = './ZOO-e100-lr0.0003-bs2-seed42/ZOO-epoch=11-train_loss=0.0171-valid_loss=0.0088-ONE_channel.ckpt'
BATCH_SIZE = 10


with h5py.File(DATA_PATH, 'r') as f:
    # patches_rgi   = f['data'][:, 0, 0, 0] 
    patches_dwarf = f['dwarf'][:, 0, 0, 0] 
# patches_bianry = (np.sum( patches_dwarf, axis=(1,2,3) ) != 0).astype(int)

N = patches_dwarf.shape[0]
X, y = np.arange(0, N, 1), np.arange(0, N, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

arg_valid = np.sort(y_test)
np.savetxt(f'arg_valid_seed{SEED}.txt', arg_valid)

with h5py.File(DATA_PATH, 'r') as f:
    patches_rgi   = np.arcsinh(f['data'][arg_valid])
    patches_dwarf = f['dwarf'][arg_valid] 
patches_binary = (np.sum( patches_dwarf, axis=(1,2,3) ) != 0).astype(int)
print(patches_binary.sum())

# Load the Model
ZOO_model = timm.create_model('efficientnet_b0', pretrained=False)

# Load the state dict from the .ckpt file
original_state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))['state_dict']

adjusted_state_dict = {}
for key, value in original_state_dict.items():
    if key.startswith('encoder.'):
        adjusted_state_dict[key.replace('encoder.', '')] = value
    if key.startswith('head.1.0.'):
        adjusted_state_dict[key.replace('head.1.0.', 'classifier.')] = value
adjusted_state_dict.pop('classifier.weight', None)
adjusted_state_dict.pop('classifier.bias', None)

ZOO_model.load_state_dict(adjusted_state_dict, strict=False)

model = LitNeuralNet.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH, ZOO_model=ZOO_model)
model.eval()

all_preds = []
print(len(arg_valid))
for i in tqdm( range(len(arg_valid)//BATCH_SIZE), leave=True ):
    y_pred = model.forward(torch.tensor(patches_rgi[BATCH_SIZE*i:BATCH_SIZE*(i+1)]).to(DTYPE))
    probs  = torch.sigmoid(y_pred.squeeze())  
    preds  = (probs > 0.5).long().long().detach().cpu()   
    all_preds.extend(preds)
all_preds = np.array(all_preds)

print(all_preds.sum())

N_drop = len(all_preds)
arg_interesting = np.where( all_preds + patches_binary[:N_drop] != 0)
np.savetxt('arg_interesting.txt', arg_interesting)
print('Done!')