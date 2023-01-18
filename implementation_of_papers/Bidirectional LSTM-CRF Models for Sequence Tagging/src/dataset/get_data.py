import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict

import sys
sys.path.append(".")
from dataset.CONLL_2003 import DataSequence


def preprocessing(train_df):
    labels_orig = [i.split() for i in train_df['labels'].values.tolist()]
    texts_orig = [i.split() for i in train_df['text'].values.tolist()]
    unique_labels = set()
    unique_words = set()
    for lb, txt in zip(labels_orig, texts_orig):
        [unique_labels.add(i) for i in lb if i not in unique_labels]
        [unique_words.add(j) for j in txt if j not in unique_words]
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    word_to_ids = defaultdict(lambda: 0)
    ## unkown words are indexed as 0
    word_to_ids.update({k:v+1 for v, k in enumerate(sorted(unique_words))})
    
    
    ## add start and stop, pad tokens in the label maps, add unknown words to word map
    labels_to_ids.update({
        "<PAD>": len(unique_labels), \
        "<START>": len(unique_labels)+1, \
        "<STOP>": len(unique_labels)+2
    })
    
    ids_to_labels.update({
        len(unique_labels): "<PAD>", \
        len(unique_labels)+1: "<START>",  \
        len(unique_labels)+2: "<STOP>", 
    })
    
    word_to_ids.update({
        '<PAD>': len(word_to_ids)+1, \
        '<UNK>': 0
    })
    
    unique_words.add('<PAD>')
    unique_words.add('<UNK>')
    return labels_to_ids, ids_to_labels, word_to_ids, unique_labels


def get_data(name="CONLL_2003", batch_size=128, random_seed=0, root='../data/', max_length=50):
    
    stats = {}
    if name == "CONLL_2003":
        # df = pd.read_csv("../../data/ner.csv")
        df = pd.read_csv("/home/le/NBFF/implementation_of_papers/data/ner.csv")
        # df = df.sample(n=10).reset_index(drop=True)
        train_df, tmp_df = train_test_split(df, test_size=0.2)
        train_df, tmp_df = train_df.reset_index(drop=True), tmp_df.reset_index(drop=True)
        val_df, test_df = train_test_split(tmp_df, test_size=0.5)
        val_df, test_df = val_df.reset_index(drop=True), test_df.reset_index(drop=True)
        
        labels_to_ids, ids_to_labels, word_to_ids, unique_labels = preprocessing(train_df=df)
        
        # Training dataset
        train_loader = torch.utils.data.DataLoader(
                            DataSequence(train_df, max_length=max_length, \
                            labels_to_ids=labels_to_ids, ids_to_labels=ids_to_labels, \
                            word_to_ids=word_to_ids), \
                        batch_size=batch_size, shuffle=True, num_workers=8)
        
        # Validation dataset
        val_loader = torch.utils.data.DataLoader(
                            DataSequence(val_df, max_length=max_length, \
                            labels_to_ids=labels_to_ids, ids_to_labels=ids_to_labels, \
                            word_to_ids=word_to_ids), \
                        batch_size=batch_size, shuffle=False, num_workers=8)
        
        # Test dataset
        test_loader = torch.utils.data.DataLoader(
                            DataSequence(test_df, max_length=max_length, \
                            labels_to_ids=labels_to_ids, ids_to_labels=ids_to_labels, \
                            word_to_ids=word_to_ids), \
                        batch_size=batch_size, shuffle=False, num_workers=8)
                        
        stats['vocab_size'] = len(word_to_ids)
        stats['tag_to_ix'] = labels_to_ids
        stats['max_length'] = max_length
        stats['ids_to_labels'] = ids_to_labels
        stats['char_to_ix'] = word_to_ids
        stats['unique_labels'] = unique_labels
        
        
    else:
        exit("dataset not found")
    
    return train_loader, val_loader, test_loader, stats