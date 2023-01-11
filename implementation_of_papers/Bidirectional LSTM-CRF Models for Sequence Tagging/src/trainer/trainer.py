import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD, AdamW, Adam, LBFGS
from sklearn.metrics import precision_score, recall_score
import numpy as np
import wandb
from seqeval.metrics import classification_report

from utils.parse_summary import parse_summary

class trainer_base(pl.LightningModule):
    def __init__(self, model, criterion, args):
        super().__init__()
        self.args = args
        self.workspace = self.args.workspace
        self.model = model
        self.criterion = criterion
        
    def forward(self, X):
        score, pred = self.model(X)
        return pred
    
    def configure_optimizers(self):
        optimizer = AdamW([
                {'params': self.model.parameters(), 'lr': self.args.learning_rate}
                ])
        return optimizer
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y = y.view(-1).long()
        score, pred = self.model(X)
        loss = self.model.neg_log_likelihood(X, y)
        # loss = self.criterion(pred, y)
        self.log("train/loss", loss)
        return {'loss': loss, 'preds': pred, 'target': y}
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # this is the test loop
        X, y = batch
        y = y.long()
        loss, pred = self.model(X)
        # loss = self.criterion(pred, y)
        self.log("test/loss", loss)
        return {'loss': loss, 'preds': pred, 'target': y}
        
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        X, y = batch
        y = y.view(-1).long()
        loss, pred = self.model(X)
        # loss = self.criterion(pred, y)
        self.log("val/loss", loss, sync_dist=True)
        return {'loss': loss, 'preds': pred, 'target': y}
    
        
    @torch.no_grad()
    def training_epoch_end(self, outputs):
        print("start training")
        exit()
        preds, pred_orig, target, target_orig = [], [], [], []
        for out in outputs:
            preds.extend(out['preds'])
            target.extend([t.item() for t in out['target']])
        # update and log
        for p, t in zip(preds, target):
            if self.args.ids_to_labels[t] not in ['<START>', '<STOP', '<PAD>']:
                pred_orig.append(self.args.ids_to_labels[p])
                target_orig.append(self.args.ids_to_labels[t])
        report = classification_report([pred_orig], [target_orig])
        report_dict = parse_summary(report)
        print(report_dict)
        asdf
        self.log('train/macro_avg/precision', report_dict['macro avg']['precision'], sync_dist=True)
        self.log('train/macro_avg/recall', report_dict['macro avg']['recall'], sync_dist=True)
    
    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        preds, pred_orig, target, target_orig = [], [], [], []
        for out in outputs:
            preds.extend(out['preds'])
            target.extend([t.item() for t in out['target']])
        # update and log
        for p, t in zip(preds, target):
            if self.args.ids_to_labels[t] not in ['<START>', '<STOP', '<PAD>']:
                pred_orig.append(self.args.ids_to_labels[p])
                target_orig.append(self.args.ids_to_labels[t])
        report = classification_report([pred_orig], [target_orig])
        report_dict = parse_summary(report)
        print("validation=================")
        print(report_dict)
        self.log('val/macro_avg/precision', report_dict['macro avg']['precision'], sync_dist=True)
        self.log('val/macro_avg/recall', report_dict['macro avg']['recall'], sync_dist=True)
    
    # @torch.no_grad()
    # def test(self, dataloader):
    #     self.model.eval()
    #     m = nn.Softmax(dim=1)
    #     prediction = []
    #     labels = []
    #     for X, y in dataloader:
    #         X, y = X.to(self.device), y.to(self.device)
    #         prediction.extend((m(self.model(X))[:, 1].detach().cpu().numpy() >= 0.5).astype(int))
    #         labels.extend(y)
            
    #     prediction = np.stack(prediction, axis=0).reshape(-1, 1)
    #     labels = torch.stack(labels, axis=0).detach().cpu().numpy()
    #     TP = int(prediction.T@(labels==1).astype(int))
    #     precision = 1.0*TP/np.sum(prediction)
    #     recall = 1.0*TP/np.sum(labels==1)
    #     return precision, recall