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
# from utils.data_utils import CustomTransform, read_files_batch, read_files_pert

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
        self.dosage_level = args['dosage_level']
        assert os.path.exists(self.data_path), f"Data path {self.data_path} does not exist."
        self.meta_data_csv = args['meta_data_csv']
        self.embeddings_file = args['embeddings_file'] # path to SMILES: embeddings pkl file
        self.add_dosage = args['add_dosage']


        self.meta_data_path = os.path.join(self.data_path,'metadata',self.meta_data_csv) # root directory
        self.embeddings_path = os.path.join(self.data_path,self.embeddings_file)

        assert os.path.exists(self.meta_data_path), f"Metadata file {self.meta_data_path} does not exist."
        assert os.path.exists(self.embeddings_path), f"Embeddings file {self.embeddings_path} does not exist."

        self.embeddings_dict= self.load_embeddings()
        self.meta_data = pd.read_csv(self.meta_data_path)
        self.split = split # train / test
        if(self.dosage_level is not None):
            print(f'Loading data for dosage level {self.dosage_level}')
            self.meta_data = pd.concat([self.meta_data[self.meta_data['DOSE'] == self.dosage_level], self.meta_data[self.meta_data['DOSE'] == 0.0]], ignore_index=True)
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

        if self.add_dosage is not None:
            dosage_encoded = self.one_hot_encode_dosage(dosage)
        else:
            dosage_encoded = None
        
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
    

# simCLR only need to load train set images that's all
class simCLRDataset(Dataset):
    def __init__(self, split, transform=None, **args):
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


        self.meta_data_path = os.path.join(self.data_path,'metadata',self.meta_data_csv) # root directory

        assert os.path.exists(self.meta_data_path), f"Metadata file {self.meta_data_path} does not exist."


        self.split = split # train / test
        self.meta_data = pd.read_csv(self.meta_data_path)
        # DEBUG
        # self.meta_data = self.meta_data.sample(500)
        self.meta_data = self.meta_data.loc[self.meta_data['SPLIT'] == self.split].reset_index(drop=True)
        
        self.transform = transform




    def __len__(self):
        """
        Return the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.meta_data)
    
    def __getitem__(self,idx):

        sample_key = self.meta_data.iloc[idx]['SAMPLE_KEY']
        week, id_part, image_file = sample_key.split('_',2)
        image_path = os.path.join(self.data_path, week, id_part, image_file + '.npy')
        assert os.path.exists(image_path), f"Image file {image_path} does not exist."
        # load image
        image = torch.from_numpy(np.load(image_path))
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)  # Place channel dimension in front of the others

        if self.transform:
            x_i, x_j = self.transform(image)
        


        
        return {
            'x_i': x_i,
            'x_j': x_j

        }

class simCLRDataLoader:
    def __init__(self, rank,transform=None, **args):
        """
        Initialize the CellDataLoader instance.
        
        Args:
            args (argparse.Namespace): Arguments containing dataloader configuration.
        """
        super().__init__()
        self.rank = rank
        self.args = args
        self.transform = transform
        self.init_dataset() # create datasets and return dataloaders
        
    def create_torch_datasets(self):
        """
        Create datasets compatible with the PyTorch training loop.
        
        Returns:
            tuple: Training and test datasets.
        """
        train_set = simCLRDataset(split='train', transform=self.transform, **self.args)
        # test_set = simCLRDataset(split='test', device=self.device, transform=self.transform, **self.args)
        
        # return train_set, test_set
        return train_set
        
    def init_dataset(self):
        """
        Initialize dataset and data loaders.
        """
        self.training_set = self.create_torch_datasets()
        
        if self.args['distributed']:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_set,
                                                                            num_replicas=self.args['world_size'],
                                                                            rank=self.rank,
                                                                            shuffle=True)
        else:
            train_sampler=None                                                             

        self.loader_train = torch.utils.data.DataLoader(
            self.training_set, 
            batch_size=self.args['batch_size'], 
            shuffle=(train_sampler is None), 
            sampler=train_sampler,
            drop_last=True)
        
        self.loader_val=None
        self.loader_test=None


        
    
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


# decoder dataset and dataloaders
    

class DecoderDataset(Dataset):
    def __init__(self, split, transform=None, **args):
        """
        Initialize the CellDataset instance.
        
        Args:
            args (argparse.Namespace): Arguments containing dataset configuration.
            device (torch.device): Device to load the data onto (e.g., 'cuda' or 'cpu').
        """
        super().__init__()

        self.data_path = args['data_dir']
        assert os.path.exists(self.data_path), f"Data path {self.data_path} does not exist."


        if split == "train":
            self.data = pd.read_pickle(os.path.join(self.data_path, args["train_file"]))
        
        elif split == "val":
            self.data = pd.read_pickle(os.path.join(self.data_path, args["val_file"]))

        elif split == "test":
            self.data = pd.read_pickle(os.path.join(self.data_path, args["test_file"]))


        # DEBUG
        # self.data = self.data.sample(500)
        
        self.transform = transform




    def __len__(self):
        """
        Return the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.data)
    
    def __getitem__(self,idx):

        image_id = self.data.iloc[idx]['image_id']
        image_file = os.path.join(self.data_path, image_id + '.npz')
        assert os.path.exists(image_file), f"Image file {image_file} does not exist."
        # load image
        image = np.load(image_file, allow_pickle=True)['sample']
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)  # Place channel dimension in front of the others
        smiles = self.data.iloc[idx]['SMILES']
        image_emb = self.data.iloc[idx, 3:].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)
        


        
        return {
            'SMILES': smiles,
            'image': image,
            'image_emb': image_emb
        }

class DecoderDataLoader:
    def __init__(self, rank,transform=None, **args):
        """
        Initialize the CellDataLoader instance.
        
        Args:
            args (argparse.Namespace): Arguments containing dataloader configuration.
        """
        super().__init__()
        self.rank = rank
        self.args = args
        self.transform = transform
        self.init_dataset() # create datasets and return dataloaders
        
    def create_torch_datasets(self):
        """
        Create datasets compatible with the PyTorch training loop.
        
        Returns:
            tuple: Training and test datasets.
        """
        train_set = DecoderDataset(split='train', transform=self.transform, **self.args)
        val_set = DecoderDataset(split='val', transform=self.transform, **self.args)
        test_set = DecoderDataset(split='test', transform=self.transform, **self.args)
        
        return train_set, val_set, test_set
        
    def init_dataset(self):
        """
        Initialize dataset and data loaders.
        """
        self.train_set, self.val_set, self.test_set = self.create_torch_datasets()
        
        if self.args['distributed']:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set,
                                                                            num_replicas=self.args['world_size'],
                                                                            rank=self.rank,
                                                                            shuffle=True)
        else:
            train_sampler=None                                                             

        self.loader_train = torch.utils.data.DataLoader(
            self.train_set, 
            batch_size=self.args['train_batch_size'], 
            shuffle=(train_sampler is None), 
            sampler=train_sampler,
            drop_last=True)
        
        self.loader_val= torch.utils.data.DataLoader(
            self.val_set, 
            batch_size=self.args['val_batch_size'], 
            shuffle=False, 
            drop_last=False)
        
        self.loader_test= torch.utils.data.DataLoader(
            self.test_set, 
            batch_size=self.args['test_batch_size'], 
            shuffle=False, 
            drop_last=False)


        
    
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
        if self.loader_val is None:
            raise NotImplementedError("Test data loader not implemented.")
        return self.loader_test

if __name__ == "__main__":
    # args = {
    #     'train_file': 'train_df.pkl',
    #     'val_file': 'val_df.pkl',
    #     'test_file': 'test_df.pkl',
    #     'train_batch_size': 32,
    #     'val_batch_size': 32,
    #     'test_batch_size': 32,
    #     'data_dir': '/rohlan/workspace/data',
    #     'distributed': False
    # }

    # decoder_loader = DecoderDataLoader(rank=0, **args)
    # train_loader = decoder_loader.train_dataloader()

    # batch = next(iter(train_loader))
    # batch.keys()
    # len(batch['SMILES'])
    # batch['image'].shape
    # batch['image_emb'].shape

    # data = np.load('/rohlan/workspace/data/24277-A03-1.npz', allow_pickle=True)
    # print(list(data.keys()))

    # print(data['channels'])
    # print(data['filenames'])
    # print(data['sample'].shape)

    # # cool all decoder dataloader working fine now!
    pass