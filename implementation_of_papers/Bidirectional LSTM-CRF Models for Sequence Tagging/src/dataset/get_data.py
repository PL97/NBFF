import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import pandas as pd

import sys
sys.path.append(".")
from dataset.CONLL_2003 import DataSequence


def get_data(name="CONLL_2003", batch_size=128, random_seed=0, root='../data/', max_length=20):
    
    stats = {}
    if name == "CONLL_2003":
        # df = pd.read_csv("../../data/ner.csv")
        df = pd.read_csv("/home/le/NBFF/implementation_of_papers/data/ner.csv")
        df = df.sample(frac=0.001).reset_index(drop=True)
        train_df, tmp_df = train_test_split(df, test_size=0.2)
        train_df, tmp_df = train_df.reset_index(drop=True), tmp_df.reset_index(drop=True)
        val_df, test_df = train_test_split(tmp_df, test_size=0.5)
        val_df, test_df = val_df.reset_index(drop=True), test_df.reset_index(drop=True)
        
        # Training dataset
        train_loader = torch.utils.data.DataLoader(
                        DataSequence(train_df, max_length=max_length), \
                        batch_size=batch_size, shuffle=True, num_workers=8)
        
        # Validation dataset
        val_loader = torch.utils.data.DataLoader(
                        DataSequence(val_df, max_length=max_length), \
                        batch_size=batch_size, shuffle=False, num_workers=8)
        
        # Test dataset
        test_loader = torch.utils.data.DataLoader(
                        DataSequence(test_df), batch_size=batch_size, \
                        shuffle=False, num_workers=8)
        
        
        stats['vocab_size'] = len(train_loader.dataset.unique_vocab)
        stats['tag_to_ix'] = train_loader.dataset.labels_to_ids
        stats['max_length'] = max_length
        stats['ids_to_labels'] = train_loader.dataset.ids_to_labels
        
        
        train_loader = DataSequence(train_df, max_length=max_length)
        val_loader = DataSequence(train_df, max_length=max_length)
        test_loader = DataSequence(train_df, max_length=max_length)
    else:
        exit("dataset not found")
    
    return train_loader, val_loader, test_loader, stats