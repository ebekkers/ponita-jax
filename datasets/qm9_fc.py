import os
import requests
import zipfile
import pickle
import pandas as pd
import numpy as np
import jax.numpy as jnp

from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import rdchem
from tqdm import tqdm


class QM9Dataset(Dataset):
    # Conversion factors for targets
    HAR2EV = 27.211386246
    KCALMOL2EV = 0.04336414
    TOTAL_SIZE = 130831  # Total size of the dataset
    TRAIN_SIZE = 110000
    VAL_SIZE = 10000  # 110000:120000 for validation
    TEST_SIZE = 10831  # Remaining for test
    TARGETS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']
    UNCHARACTERIZED_URL = 'https://ndownloader.figshare.com/files/3195404'

    def __init__(self, root='./datasets/qm9_dataset_fc', sdf_file='gdb9.sdf', csv_file='gdb9.sdf.csv', max_atoms=29, target=None, split=None):
        self.root = root
        self.sdf_file = os.path.join(root, sdf_file)
        self.csv_file = os.path.join(root, csv_file)
        self.max_atoms = max_atoms
        self.split = split  # Can be 'train', 'val', or 'test'
        self.processed_file = os.path.join(root, 'processed_qm9_data.pkl')
        self.uncharacterized_file = os.path.join(root, 'uncharacterized.txt')
        self.qm9_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip'
        self.target_index = None if target is None else self.TARGETS.index(target)

        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)

        if os.path.isfile(self.processed_file):
            print("Loading processed data...")
            with open(self.processed_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.download_uncharacterized()
            self.ensure_data_downloaded()
            print("Processing data from scratch...")
            self.data = self.process()
            with open(self.processed_file, 'wb') as f:
                pickle.dump(self.data, f)

        if split:
            self.apply_split()
        
        # self.dataset_to_pytorch()

    def download_uncharacterized(self):
        """Download the uncharacterized.txt file."""
        if not os.path.isfile(self.uncharacterized_file):
            print("Downloading uncharacterized.txt...")
            response = requests.get(self.UNCHARACTERIZED_URL)
            response.raise_for_status()  # Ensure the request was successful
            with open(self.uncharacterized_file, 'wb') as f:
                f.write(response.content)

    def read_uncharacterized_indices(self):
        """Read indices from uncharacterized.txt file."""
        # with open(self.uncharacterized_file, 'r') as file:
            # indices = [int(line.strip()) - 1 for line in file if line.strip().isdigit()]  # Adjusting indices to 0-based
        with open(self.uncharacterized_file, 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]
        return set(skip)

    def apply_split(self):
        # Create the split based on the predefined sizes
        random_state = np.random.RandomState(seed=42)
        perm = random_state.permutation(np.arange(self.TOTAL_SIZE))
        train_idx, val_idx, test_idx = perm[:self.TRAIN_SIZE], perm[self.TRAIN_SIZE:self.TRAIN_SIZE + self.VAL_SIZE], perm[self.TRAIN_SIZE + self.VAL_SIZE:]

        if self.split == 'train':
            self.data = [self.data[i] for i in train_idx]
        elif self.split == 'val':
            self.data = [self.data[i] for i in val_idx]
        elif self.split == 'test':
            self.data = [self.data[i] for i in test_idx]
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'.")

    def download_file(self, url, filename):
        local_filename = os.path.join(self.root, filename)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename

    def extract_zip(self, file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        print(f"Extracted to {self.root}")

    def ensure_data_downloaded(self):
        # Check if SDF and CSV files are present, otherwise download and extract the dataset
        if not os.path.isfile(self.sdf_file) or not os.path.isfile(self.csv_file):
            print(f"SDF or CSV file not found, downloading and extracting QM9 dataset...")
            zip_file_path = self.download_file(self.qm9_url, 'qm9.zip')
            self.extract_zip(zip_file_path)
        else:
            print("SDF and CSV files found, no need to download.")

    def process(self):
        suppl = Chem.SDMolSupplier(self.sdf_file, removeHs=False, sanitize=False)
        df = pd.read_csv(self.csv_file)
        raw_targets = df.iloc[:, 1:].values  # Excludes the first ID column
        raw_targets = raw_targets.astype(np.float32)

        # Rearrange the targets: Move the first three columns to the end
        rearranged_targets = np.concatenate([raw_targets[:, 3:], raw_targets[:, :3]], axis=1)

        # Apply conversion factors
        conversion_factors = np.array([
            1., 1., self.HAR2EV, self.HAR2EV, self.HAR2EV, 1., self.HAR2EV, self.HAR2EV, self.HAR2EV,
            self.HAR2EV, self.HAR2EV, 1., self.KCALMOL2EV, self.KCALMOL2EV, self.KCALMOL2EV,
            self.KCALMOL2EV, 1., 1., 1.
        ], dtype=np.float32)

        targets = rearranged_targets * conversion_factors

        atom_types = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}  # Map atomic numbers to indices for one-hot encoding
        data_list = []

        skip_indices = self.read_uncharacterized_indices()

        for i, mol in enumerate(tqdm(suppl, desc="Processing Molecules")):
            # Regarding dtypes: pos is float, z is integer, the rest are one-hot encodings and can be stored as booleans
            if mol is None or i in skip_indices:  # Skip uncharacterized molecules
                continue
            # if mol is None: continue
            num_atoms = mol.GetNumAtoms()
            pos = np.zeros((self.max_atoms, 3), dtype=np.float32)
            x = np.zeros((self.max_atoms, len(atom_types)), dtype=bool)  # One-hot encoding
            z = np.zeros((self.max_atoms,), dtype=int) # Integer atom type
            mask = np.zeros((self.max_atoms,), dtype=bool)
            edge_adj = np.zeros((self.max_atoms, self.max_atoms), dtype=bool)
            edge_attr = np.zeros((self.max_atoms, self.max_atoms, 4), dtype=bool)  # 4 bond types

            bond_type_to_index = {
                Chem.rdchem.BondType.SINGLE: 0,
                Chem.rdchem.BondType.DOUBLE: 1,
                Chem.rdchem.BondType.TRIPLE: 2,
                Chem.rdchem.BondType.AROMATIC: 3,
            }

            for j in range(num_atoms):
                atom = mol.GetAtomWithIdx(j)
                pos[j] = mol.GetConformer().GetAtomPosition(j)
                x[j, atom_types[atom.GetAtomicNum()]] = 1
                z[j] = atom.GetAtomicNum()
                mask[j] = True

            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_type = bond_type_to_index[bond.GetBondType()]
                edge_adj[start, end] = edge_adj[end, start] = 1
                edge_attr[start, end, bond_type] = edge_attr[end, start, bond_type] = 1

            y = targets[i]
            name = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol)

            data_list.append({
                'batch': 0,
                'pos': pos,
                'x': x,
                'z': z,
                'mask': mask,
                'edge_adj': edge_adj,
                'edge_attr': edge_attr,
                'y': y,
                'name': name,
                'smiles': smiles,
                'idx': i
            })
        return data_list

    def convert_bool_to_float(self):
        """Converts 'x' and 'edge_attr' from boolean to float arrays."""
        for item in self.data:
            if item['x'].dtype == bool:
                item['x'] = item['x'].astype(np.float32)
            if item['edge_attr'].dtype == bool:
                item['edge_attr'] = item['edge_attr'].astype(np.float32)

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
        # return self.data[idx]
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.target_index is not None:
            if len(item['y']) > 1:
                item['y'] = item['y'][self.target_index:self.target_index+1]
            else:
                # The item is already updated
                pass
        return item
    

def collate_fn(batch):
    """Collate function for the QM9 dataset."""
    keys = ['batch', 'pos', 'x', 'z', 'mask', 'edge_adj', 'edge_attr', 'y']#, 'name', 'smiles', 'idx']

    batch_dict = {k: [d[k] for d in batch] for k in keys}
    for k in ['pos', 'x', 'z', 'mask', 'edge_adj', 'edge_attr', 'y']:
        batch_dict[k] = np.stack(batch_dict[k], axis=0)
    batch_dict['edge_index'] = batch_dict['mask']
    return batch_dict


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    batch_size = 96
    dataset = QM9Dataset(target='alpha', split='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn, num_workers=0)

    t0 = time.time()
    for batch in tqdm(dataloader):
        x, y, pos, edge_attr = batch['x'], batch['y'], batch['pos'], batch['edge_attr']
        print(x.shape, x.dtype)
        print(y.shape)
        print(pos.shape)
        print(edge_attr.shape)
        print(batch.keys())
        exit()
    t1 = time.time()
    print(t1-t0)