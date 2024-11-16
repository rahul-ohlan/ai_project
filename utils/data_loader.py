import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle



# gene expression dataset
class GexDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_pickle(data_file).reset_index(drop=True)
        # DEBUG
        # self.data = self.data.head(2048)
        self.cell_line = self.data['cell']
        self.gex = self.data['gex']
        self.target = self.data['gex_median']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        gex_tensor = torch.tensor(self.gex[idx], dtype=torch.float32)
        target_tensor = torch.tensor(self.target[idx], dtype=torch.float32)
        
        return gex_tensor, target_tensor
    
    
# CLIP Dataset
class CLIPDataset(Dataset):
    def __init__(self, data_file, basal_embeds, selfies_embeds, ge_type="median"):

        self.gex_data = pd.read_pickle(data_file) # gene expression data
        # DEBUG
        # print("debugging loading only 2048 samples ...")
        # self.gex_data = self.gex_data.head(2048)
        self.gex_data = self.gex_data.drop_duplicates(subset="cell-pert_id").reset_index(drop=True) # since we need to consider only medians

        with open(f"{basal_embeds}", "rb") as f:
            self.basal_embeds = pickle.load(f) # {"cell_line": nd.array}

        self.selfies_embeds = torch.load(selfies_embeds) # .pth file -- {"pert_ids": [], "smiles": [], "selfies_embed": [], "selfies": [] }
        # DEBUG
        # self.data = self.data.head(2048)
        self.cell_lines = self.gex_data['cell']
        self.pert_ids = self.gex_data['pert_id']

        if ge_type=="raw":
            self.gex = self.gex_data['gex']
        elif ge_type == "median":
            self.gex = self.gex_data["gex_median"]

    def __len__(self):
        return len(self.gex_data)
    
    def __getitem__(self, idx):

        gex_tensor = torch.tensor(self.gex.iloc[idx], dtype=torch.float32)
        cell_line = self.cell_lines.iloc[idx]
        pert_id = self.pert_ids.iloc[idx]

        # find index of pert_id in selfies_embeds
        pert_id_idx = self.selfies_embeds["pert_ids"].index(pert_id)
        selfies_embed = torch.tensor(self.selfies_embeds["selfies_embed"][pert_id_idx], dtype=torch.float32).squeeze()
        smiles = self.selfies_embeds["smiles"][pert_id_idx]
        selfies = self.selfies_embeds["selfies"][pert_id_idx]
        basal_tensor = torch.tensor(self.basal_embeds[cell_line], dtype=torch.float32).squeeze()

        
        return {"gex": gex_tensor,
                "basal": basal_tensor,
                "selfies_embed": selfies_embed,
                "smiles": smiles,
                "selfies": selfies,
                "pert_id": pert_id,
                "cell_line": cell_line}