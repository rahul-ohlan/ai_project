import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from utils.data_utils import CustomTransform, read_files_batch, read_files_pert

class CLIPDataset(Dataset):
    """
    Dataset class for cell image data.

    This class handles the loading and preprocessing of cell image datasets, 
    including the initialization of dataset splits, normalization, and embedding creation.
    """
    
    def __init__(self, split, device, transform=None, **args):
        """
        Initialize the CellDataset instance.
        
        Args:
            args (argparse.Namespace): Arguments containing dataset configuration.
            device (torch.device): Device to load the data onto (e.g., 'cuda' or 'cpu').
        """
        super().__init__()

        self.data_path = args['data_dir']
        assert os.path.exists(self.data_path), f"Data path {self.data_path} does not exist."
        self.meta_data_csv = args['meta_data_csv']
        self.embeddings_file = args['embeddings_file'] # path to SMILES: embeddings pkl file


        self.meta_data_path = os.path.join(self.data_path,'metadata',self.meta_data_csv) # root directory
        self.embeddings_path = os.path.join(self.data_path,self.embeddings_file)

        assert os.path.exists(self.meta_data_path), f"Metadata file {self.meta_data_path} does not exist."
        assert os.path.exists(self.embeddings_path), f"Embeddings file {self.embeddings_path} does not exist."

        self.embeddings_dict= self.load_embeddings()

        self.split = split # train / test
        self.meta_data = pd.read_csv(self.meta_data_path)
        # DEBUG
        # self.meta_data = self.meta_data.sample(500)
        self.meta_data = self.meta_data.loc[self.meta_data['SPLIT'] == self.split].reset_index(drop=True)
        
        self.device = device
        self.transform = transform
        self.encoder = OneHotEncoder(sparse=False)
        unique_doses = self.meta_data['DOSE'].unique().reshape(-1, 1)
        self.encoder.fit(unique_doses)

    def load_embeddings(self):
        """
        Create and initialize the embeddings for molecules.
        """
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                self.embedding_dict = pickle.load(f)
        else:
            self.embedding_dict = []
        
        return self.embedding_dict

    def one_hot_encode_dosage(self, dose):
        """
        One-hot encode the dosage value using the fitted encoder.
        """
        dose_array = np.array(dose).reshape(-1, 1)
        encoded = self.encoder.transform(dose_array)
        return torch.tensor(encoded[0], dtype=torch.float32)


    def __len__(self):
        """
        Return the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.meta_data)
    
    def __getitem__(self,idx):

        sample_key = self.meta_data.iloc[idx]['SAMPLE_KEY']
        dosage = self.meta_data.iloc[idx]['DOSE']
        week, id_part, image_file = sample_key.split('_',2)
        image_path = os.path.join(self.data_path, week, id_part, image_file + '.npy')
        assert os.path.exists(image_path), f"Image file {image_path} does not exist."
        # load image
        image = torch.from_numpy(np.load(image_path))
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)  # Place channel dimension in front of the others

        if self.transform:
            image = self.transform(image)
        
        smiles = self.meta_data.iloc[idx]['SMILES']
        emb = self.embedding_dict.get(smiles,None)
        if emb is not None:
            emb = torch.tensor(emb, dtype=torch.float32)

        dosage_encoded = self.one_hot_encode_dosage(dosage)
        
        return {
            'image': image.to(self.device),
            'smiles_emb': emb.to(self.device),
            'smiles': smiles, 
            'dosage': dosage_encoded.to(self.device)
        }




class CLIPDataLoader:
    """

    This class handles the creation of data loaders for training and testing, 
    including the initialization of datasets and batch processing.
    """
    
    def __init__(self, device,transform=None, **args):
        """
        Initialize the CellDataLoader instance.
        
        Args:
            args (argparse.Namespace): Arguments containing dataloader configuration.
        """
        super().__init__()
        self.device = device
        self.args = args
        self.transform = transform
        self.init_dataset() # create datasets and return dataloaders
        
    def create_torch_datasets(self):
        """
        Create datasets compatible with the PyTorch training loop.
        
        Returns:
            tuple: Training and test datasets.
        """
        train_set = CLIPDataset(split='train', device=self.device, transform=self.transform, **self.args)
        test_set = CLIPDataset(split='test', device=self.device, transform=self.transform, **self.args)
        
        return train_set, test_set
        
    def init_dataset(self):
        """
        Initialize dataset and data loaders.
        """
        self.training_set, self.test_set = self.create_torch_datasets()
        
        self.loader_train = torch.utils.data.DataLoader(self.training_set, 
                                                        batch_size=self.args['batch_size'], 
                                                        shuffle=True, 
                                                        # num_workers=self.args.num_workers, 
                                                        drop_last=True)  

        self.loader_test = torch.utils.data.DataLoader(self.test_set, 
                                                       batch_size=self.args['batch_size'], 
                                                       shuffle=False, 
                                                    #    num_workers=self.args.num_workers, 
                                                       drop_last=False)

        self.loader_val = None   
        
    
    def train_dataloader(self):
        """
        Return the training data loader.
        
        Returns:
            DataLoader: Training data loader.
        """
        return self.loader_train
    
    def val_dataloader(self):
        """
        Return the validation data loader.
        
        Returns:
            DataLoader: Validation data loader.
        """
        if self.loader_val is None:
            raise NotImplementedError("Validation data loader not implemented.")
        return self.loader_val
    
    def test_dataloader(self):
        """
        Return the test data loader.
        
        Returns:
            DataLoader: Test data loader.
        """
        return self.loader_test
    
# DEBUG
s_args = {
    'data_dir': '../data/bbbc021_all',
    'meta_data_csv': 'bbbc021_df_all.csv',
    'embeddings_file': 'unique_smiles_morgan_fingerprints.pkl',
    'batch_size': 32,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_loader = CLIPDataLoader(device, CustomTransform(), **s_args)

train_loader = data_loader.train_dataloader()
test_loader = data_loader.test_dataloader()

train_batch = next(iter(train_loader))
test_batch = next(iter(test_loader))

print('this is DOSAGE----------------------;')
print(train_batch['dosage'])

meta_data = pd.read_csv(os.path.join(s_args['data_dir'],'metadata',s_args['meta_data_csv']))
sample_key = meta_data.iloc[0]['SAMPLE_KEY']
# sample_key = 'Week1_22123_1_11_3.0'
week, id_part, image_file = sample_key.split('_',2)
image_path = os.path.join(s_args['data_dir'], week, id_part, image_file + '.npy')
from utils.data_utils import CustomTransform
transform = CustomTransform()

image = torch.from_numpy(np.load(image_path))
image = torch.tensor(image, dtype=torch.float32)
image = transform(image)

print(image.size())


# # # check for embeddings
train_batch.keys()
train_batch['smiles']
# # check image encoder
train_batch['image'].size()
image_sample = train_batch['image'][0]
image_sample.size() # (96,3,3)
from models.clip.model import ImageEncoder
image_encoder = ImageEncoder()
image_encoder = image_encoder.to(device)

image_out = image_encoder(train_batch['image']) 
print(image_out.size()) # 32, 256 expected
