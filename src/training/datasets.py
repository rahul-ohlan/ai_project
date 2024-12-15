import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import mmread
from torchvision.transforms import Compose
from torch.utils.data import Dataset

############################################################################################################
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class CellPainting2(Dataset):
    def __init__(self, sample_index_file: str, image_dir: str, transforms=None):
        """
        Dataset class for CellPainting images and molecule fingerprints.
        
        Args:
            sample_index_file (str): Path to the CSV file containing sample keys and inchikeys.
            image_directory_path (str): Directory where images are stored as <SAMPLE-KEY>.npz.
            molecule_file (str): Path to the HDF5 file containing molecule data (index as inchikeys, rows as fingerprints).
            transforms (callable, optional): Transform to be applied to images.
        """
        # Check file paths
        assert os.path.exists(sample_index_file), "Sample index file not found."
        self.sample_index = pd.read_pickle(sample_index_file)
        self.sample_index = self.sample_index.drop_duplicates(subset=["INCHIKEY"]).reset_index(drop=True)


        self.image_directory = image_dir
        self.transforms = transforms
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx):
        # Get the sample key and associated inchikey
        sample_key = self.sample_index.iloc[idx]["image_id"]
        inchikey = self.sample_index.iloc[idx]["INCHIKEY"]
        
        # Retrieve the molecule fingerprint
        molecule_fp = self.sample_index.iloc[idx]["fingerprints"]
        smiles = self.sample_index.iloc[idx]["SMILES"]
        inchi_key = self.sample_index.iloc[idx]["INCHIKEY"]
        
        # Load the image
        image_path = os.path.join(self.image_directory, f"{sample_key}.npz")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found.")
        image = np.load(image_path, allow_pickle=True)["sample"].astype(np.float32)
        
        # Apply transforms to the image
        if self.transforms:
            image = self.transforms(image)
        
        return {
            "image": image,
            "molecule_fp": np.asarray(molecule_fp, dtype=np.float32),
            "smiles": smiles,
            "inchi_key": inchi_key
        }


############################################################################################################
class CellPainting(Dataset):
    def __init__(self, sample_index_file: str, image_directory_path: str = None, molecule_file: str = None, label_matrix_file: str = None,
                 label_row_index_file: str = None, label_col_index_file: str = None, auxiliary_labels=None,
                 transforms=None, group_views: bool = False,
                 subset: float = 1., num_classes: int = None, verbose: bool = False):
        """ Read samples from cellpainting dataset."""
        self.verbose = verbose
        self.molecules = False
        self.images = False

        assert (os.path.exists(sample_index_file))

        # Read sample index
        sample_index = pd.read_csv(sample_index_file, sep=",", header=0)
        sample_index.set_index(["SAMPLE_KEY"])

        sample_keys = sample_index['SAMPLE_KEY'].tolist()

        # read auxiliary labels if provided
        if auxiliary_labels is not None:
            pddata = pd.read_csv(auxiliary_labels, sep=",", header=0)
            self.auxiliary_data = pddata.as_matrix()[:, 2:].astype(np.float32)
            # threshold
            self.auxiliary_data[self.auxiliary_data < 0.75] = -1
            self.auxiliary_data[self.auxiliary_data >= 0.75] = 1
            self.auxiliary_assays = list(pddata)[2:]
            self.n_auxiliary_classes = len(self.auxiliary_assays)
            self.auxiliary_smiles = pddata["SMILES"].tolist()
        else:
            self.n_auxiliary_classes = 0

        if image_directory_path is not None:
            self.images = True
            assert (os.path.exists(image_directory_path))

            if group_views:
                sample_groups = sample_index.groupby(['PLATE_ID', 'WELL_POSITION'])
                sample_keys = list(sample_groups.groups.keys())
                sample_index = sample_groups
                self.sample_to_smiles = None  # TODO
            else:

                if auxiliary_labels is not None:
                    self.sample_to_smiles = dict(zip(sample_index.SAMPLE_KEY, [self.auxiliary_smiles.index(s) for s in sample_index.SMILES]))
                else:
                    self.sample_to_smiles = None

        if molecule_file:
            assert (os.path.exists(molecule_file))

            self.molecules = True
            molecule_df = pd.read_hdf(molecule_file, key="df")
            mol_keys = list(molecule_df.index.values)

        if (self.images and self.molecules) or self.molecules:
            keys = list(set(sample_keys) & set(mol_keys))
        elif self.images:
            keys = sample_keys



        if len(keys) == 0:
            raise Exception("Empty dataset!")
        else:
            self.log("Found {} samples".format(len(keys)))

        if subset != 1.:
            sample_keys = sample_keys[:int(len(sample_keys) * subset)]

        # Read Label Matrix if specified
        if label_matrix_file is not None:
            assert (os.path.exists(label_matrix_file))

            assert (os.path.exists(label_row_index_file))

            assert (os.path.exists(label_col_index_file))


            if label_row_index_file is not None and label_col_index_file is not None:
                col_index = pd.read_csv(label_col_index_file, sep=",", header=0)
                row_index = pd.read_csv(label_row_index_file, sep=",", header=0)
                label_matrix = mmread(label_matrix_file).tocsr()
                # --
                self.label_matrix = label_matrix
                self.row_index = row_index
                self.col_index = col_index
                if group_views:
                    self.label_dict = dict(
                        (key, sample_groups.get_group(key).iloc[0].ROW_NR_LABEL_MAT) for key in sample_keys)
                else:
                    self.label_dict = dict(zip(sample_index.SAMPLE_KEY, sample_index.ROW_NR_LABEL_MAT))
                self.n_classes = label_matrix.shape[1]
            else:
                raise Exception("If label is specified index files must be passed!")
        else:
            self.label_matrix = None
            self.row_index = None
            self.col_index = None
            self.label_dict = None
            self.n_classes = num_classes

        if auxiliary_labels is not None:
            self.n_classes += self.n_auxiliary_classes

        # expose everything important
        self.data_directory = image_directory_path
        self.sample_index = sample_index
        if self.molecules:
            self.molecule_objs = molecule_df
        self.keys = keys
        print(len(keys))
        self.n_samples = len(keys)
        self.sample_keys = list(keys)
        self.group_views = group_views
        self.transforms = transforms


        if self.images:
            # load first sample and check shape
            i = 0
            sample = self[i][0] if self.molecules else self[i] #getitem returns tuple of img and fp
            while sample["input"] is np.nan and i < len(self):
                sample = self[i][0] if self.molecules else self[i]
                i += 1

            if sample["input"] is not None and not np.nan:
                self.data_shape = sample["input"].shape
            else:
                self.data_shape = "Unknown"
            self.log("Discovered {} samples (subset={}) with shape {}".format(self.n_samples, subset, self.data_shape))


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        sample_key = self.keys[idx]

        if self.molecules:
            mol_fp = self.molecule_objs.loc[sample_key].values
            mol = {
                    "input" : mol_fp,
                    "ID" : sample_key
            }

        if self.images:
            img = self.read_img(sample_key)


        if self.molecules and self.images:
            return img, mol
        elif self.images:
            return img
        elif self.molecules:
            return mol


    @property
    def shape(self):
        return self.data_shape

    @property
    def num_classes(self):
        return self.n_classes

    def log(self, message):
        if self.verbose:
            print(message)


    def read_img(self, key):
        if self.group_views:
            X = self.load_view_group(key)
        else:
            filepath = os.path.join(self.data_directory, "{}.npz".format(key))
            if os.path.exists(filepath):
                X = self.load_view(filepath=filepath)

                index = int(np.where(self.sample_index["SAMPLE_KEY"]==key)[0])

            else:
                print("ERROR: Missing sample '{}'".format(key))
                return dict(input=np.nan, ID=key)

        if self.transforms:
            X = self.transforms(X)

        # get label
        if self.label_dict is not None:
            label_idx = self.label_dict[key]
            y = self.label_matrix[label_idx].toarray()[0].astype(np.float32)
            if self.sample_to_smiles is not None and key in self.sample_to_smiles:
                y = np.concatenate([y, self.auxiliary_data[self.sample_to_smiles[key], :]])

            return dict(input=X, target=y, ID=key)
        else:
            return dict(input=X, row_id=index, ID=key)


    def get_sample_keys(self):
        return self.sample_keys.copy()

    def load_view(self, filepath):
        """Load all channels for one sample"""
        npz = np.load(filepath, allow_pickle=True)
        image = npz["sample"].astype(np.float32)
        return image


    def load_view_group(self, groupkey):
        result = np.empty((1040, 2088 - 12, 5), dtype=np.uint8)
        viewgroup = self.sample_index.get_group(groupkey)
        for i, view in enumerate(viewgroup.sort_values("SITE", ascending=True).iterrows()):
            corner = (0 if int(i / 3) == 0 else 520, i % 3 * 692)
            filepath = os.path.join(self.data_directory, "{}.npz".format(view[1].SAMPLE_KEY))
            v = self.load_view(filepath=filepath)[:, 4:, :]
            # for j in range(v.shape[-1]):
            #    plt.imshow(v[:, :, j])
            #    plt.savefig("{}-{}-{}-{}.png".format(groupkey[0], groupkey[1], i, j))
            result[corner[0]:corner[0] + 520, corner[1]:corner[1] + 692, :] = v
        return result
