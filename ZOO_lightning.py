import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import timm

import torch
import torch.nn as nn
import torchvision
import torchmetrics
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import functional as TF
import torch.nn.functional as F

import wandb
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


DTYPE = torch.float32

### Initialize a W&B run ###
wandb.login()
wandb_logger = WandbLogger(project="dwarf_galaxies_ZOO_BinaryClassification", log_model="all")

### Pre-Processing function for SAM ###
def pre(img):
    img = np.arcsinh(img)
    return img

### Define Weighted Sampler for Unbalance dataset ###
def calculate_weights(labels):
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])
    return torch.from_numpy(samples_weight)

def get_weighted_sampler(labels):
    samples_weight = calculate_weights(labels)
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
    return sampler

### Data Augmentation ###
def segementation_transform(image):#, label):
    """Apply the same transform to both image and label."""
    # Random horizontal flipping
    if torch.rand(1) < 0.5:
        image = TF.hflip(image)
        # label = TF.hflip(label)
    
    # Random vertical flipping
    if torch.rand(1) < 0.5:
        image = TF.vflip(image)
        # label = TF.vflip(label)
    
    # Random rotation
    angle = torch.rand(1) * 270  # generates a random value between 0 and 270
    angle = torch.div(angle, 90, rounding_mode='floor') * 90  # snaps angle to 0, 90, 180, or 270 degrees
    image = TF.rotate(image, angle.item())
    # label = TF.rotate(label, angle.item())
    
    return image#, label

### Dataset Class ###
class SimpleDataset(Dataset):
    def __init__(self, inputs, labels, OUTPUT_SIZE, transform=None, preprocess=None):
        """
        Args:
            inputs (list or ndarray): The input features.
            labels (list or ndarray): The labels corresponding to the inputs.
            transform (callable, optional): Optional transform to be applied on a sample.
            preprocess (callable, optional): Optional preprocessing to be applied on a sample.
        """

        # If there's a preprocessing function, apply it
        if preprocess is not None:
            inputs = preprocess(inputs)

        self.inputs = inputs
        self.labels = labels
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image, label = torch.tensor(self.inputs[idx]).to(DTYPE), torch.tensor(self.labels[idx]).to(DTYPE)

        if self.transform is not None:
            image = self.transform(image)

        return image, label[None]
    
### Lightning Module ###
class LitNeuralNet(pl.LightningModule):
    def __init__(self, ZOO_model, LEARNING_RATE, OUTPUT_SIZE):
        super(LitNeuralNet, self).__init__()

        self.save_hyperparameters(ignore=['ZOO_model'])

        self.ZOO_model = ZOO_model
        self.fc1       = nn.Linear(1000, OUTPUT_SIZE) 

        self.LEARNING_RATE  = LEARNING_RATE

        self.train_step_outputs = []
        self.valid_step_outputs = []

        self.validation_preds   = []
        self.validation_targets = []

    def forward(self, x):
            x = self.ZOO_model(x)
            x = self.fc1(x)
            return x
    
    def training_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)

        self.train_step_outputs.append(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self(images)
        loss    = F.binary_cross_entropy_with_logits(outputs, labels)

        probs = torch.sigmoid(outputs.squeeze())  
        preds = (probs > 0.5).long()              
        self.validation_preds.extend(preds.detach().cpu())
        self.validation_targets.extend(labels.squeeze().detach().cpu())

        self.valid_step_outputs.append(loss.item())
        return loss
    
    def on_train_epoch_end(self):
        train_loss = sum(self.train_step_outputs) / len(self.train_step_outputs)
        self.log("train_loss", train_loss)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        valid_loss = sum(self.valid_step_outputs) / len(self.valid_step_outputs)
        self.log("valid_loss", valid_loss)
        self.valid_step_outputs.clear()

        # Compute confusion matrix
        print(np.array(self.validation_targets).shape)
        print(np.array(self.validation_preds).shape)
        cm   = confusion_matrix(self.validation_targets, self.validation_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Dwarf', 'Dwarf'])
        
        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(6,6))
        disp.plot(cmap='Blues', ax=ax)
        disp.im_.colorbar.remove()
        plt.tight_layout()

        # Log the confusion matrix plot to wandb
        self.logger.experiment.log({"confusion_matrix": [wandb.Image(fig, caption="Confusion Matrix")]})

        # Close the plot to prevent memory leaks
        plt.close(fig)
        self.validation_preds.clear()
        self.validation_targets.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LEARNING_RATE)

if __name__ == '__main__':
    # Hyper-parameters
    SEED = 42
    EPOCHS = 10
    BATCH_SIZE  = 2
    NUM_WORKERS = 7
    LEARNING_RATE = 1e-4
    OUTPUT_SIZE  = 1
    WEIGHTS_PATH = './effnetb0_color_224px.ckpt'

    # Configure the ModelCheckpoint callback
    # This example saves the top 1 models based on validation loss
    checkpoint_callback_valid = ModelCheckpoint(
        monitor='valid_loss',
        dirpath=f'./ZOO-e{EPOCHS}-lr{LEARNING_RATE}-bs{BATCH_SIZE}-seed{SEED}',
        filename='ZOO-{epoch:02d}-{train_loss:.4f}-{valid_loss:.4f}',
        save_top_k=1,
        mode='min',
    )

    # Load the Data
    data_path = '/Volumes/ES-HDD-Documents/Documents/matlas_dwarfs'

    with h5py.File(f'{data_path}/NGC4249_224_patches.h5', 'r') as f:
        patches_rgi   = f['data'][:5000] #[arg_dwarf[:5]]
        patches_dwarf = f['dwarf'][:5000] #[arg_dwarf[:5]]
    patches_bianry = (np.sum( patches_dwarf, axis=(1,2,3) ) != 0).astype(int)

    data_train, data_valid, label_train, label_valid = train_test_split(patches_rgi, patches_bianry, test_size=0.2, random_state=SEED)

    train_dataset = SimpleDataset(data_train, label_train, OUTPUT_SIZE, transform=segementation_transform, preprocess=pre)
    train_sampler = get_weighted_sampler(label_train)
    train_loader  = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, persistent_workers=True, sampler=train_sampler)

    valid_dataset = SimpleDataset(data_valid, label_valid, OUTPUT_SIZE, transform=None, preprocess=pre)
    valid_loader  = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, persistent_workers=True)

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

    model   = LitNeuralNet(ZOO_model, LEARNING_RATE, OUTPUT_SIZE)
    trainer = Trainer(callbacks=[checkpoint_callback_valid], logger=wandb_logger, max_epochs=EPOCHS, log_every_n_steps=min(label_train.shape[0], 50))
    trainer.fit(model, train_loader, valid_loader)
    wandb.finish()
