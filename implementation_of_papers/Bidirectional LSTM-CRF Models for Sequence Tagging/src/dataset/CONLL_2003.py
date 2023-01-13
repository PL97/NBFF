import torch
from collections import defaultdict


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class DataSequence(torch.utils.data.Dataset):
    
    def __init__(self, df, labels_to_ids, ids_to_labels, word_to_ids, max_length=20):
        self.labels_to_ids, self.ids_to_labels, self.word_to_ids = labels_to_ids, ids_to_labels, word_to_ids
        self.max_length = max_length
        self.labels_orig = [i.split() for i in df['labels'].values.tolist()]
        self.texts_orig = [i.split() for i in df['text'].values.tolist()]

        ## get the index of the label and words in text
        self.labels = [list(map(lambda x: self.labels_to_ids[x], l)) for l in self.labels_orig]
        self.texts = [list(map(lambda x: self.word_to_ids[x], t)) for t in self.texts_orig]
        
        ## padding and truncate to ensure same length
        self.label_pad, self.text_pad = [], []
        for l, t in zip(self.labels, self.texts):
             ## padding or truncate data
            t_pad = [self.word_to_ids['<PAD>']]*self.max_length
            t_pad[:min(self.max_length, len(t))] = t[:min(self.max_length, len(t))]
            l_pad = [self.labels_to_ids['<PAD>']]*self.max_length
            l_pad[:min(self.max_length, len(l))] = l[:min(self.max_length, len(l))]
            self.text_pad.append(t_pad)
            self.label_pad.append(l_pad)
        self.texts = self.text_pad
        self.labels = self.label_pad
        
        # print(self.labels[0])
        # print(self.labels_orig[0])
        # print(self.texts)
        # print(self.texts_orig[0])
        # print(self.labels_to_ids)
        # print(self.word_to_ids)
        # exit("finished")
        
    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return torch.LongTensor(self.texts[idx])

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels
    
    
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("../../data/ner.csv")
    train = DataSequence(df)
    for x, y in train:
        print(x.shape)