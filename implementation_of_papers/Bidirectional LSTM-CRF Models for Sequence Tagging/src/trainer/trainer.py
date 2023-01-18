import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD, AdamW, Adam, LBFGS
from sklearn.metrics import precision_score, recall_score
import numpy as np
import wandb
from seqeval.metrics import classification_report
from tqdm import tqdm

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
 
     
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) ## optional
        
    
    def _shared_step(self, batch, batch_idx, prefix, log=True):
        X, y = batch
        y = y.long()
        loss = self.model.loss(X, y)
        score, pred = self.model(X)
        if log:
            self.log(f"{prefix}/loss", loss)
        return {'loss': loss, 'preds': pred, 'target': y}
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, prefix="train")

    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # this is the test loop
        return self._shared_step(batch, batch_idx, prefix="test")
        
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        return self._shared_step(batch, batch_idx, prefix="val")
    
    def _eval(self, outputs, prefix="train", log=True):
        preds, target, pred_orig, target_orig = [], [], [], []
        for out in outputs:
            preds.extend(out['preds'])
            target.extend(out['target'].detach().cpu().tolist())
        preds = [item for sublist in preds for item in sublist]
        target = [item for sublist in target for item in sublist]
        # update and log
        for p, t in zip(preds, target):
            if self.args.ids_to_labels[t] not in ['<START>', '<STOP>', '<PAD>']:
                pred_orig.append(self.args.ids_to_labels[p])
                target_orig.append(self.args.ids_to_labels[t])
        report = classification_report([pred_orig], [target_orig], zero_division=0)
        report_dict = parse_summary(report)
        if log:
            self.log(f'{prefix}/macro_avg/precision', report_dict['macro avg']['precision'])
            self.log(f'{prefix}/macro_avg/recall', report_dict['macro avg']['recall'], sync_dist=True)
            self.log(f'{prefix}/macro_avg/f1-score', report_dict['macro avg']['f1-score'], sync_dist=True)
        return report_dict
        
    @torch.no_grad()
    def training_epoch_end(self, outputs):
        self._eval(outputs, prefix="train")
        
    
    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        self._eval(outputs, prefix="val")
        
    
    @torch.no_grad()
    def test_epoch_end(self, outputs):
        aggregation = "macro avg"
        report_dict = self._eval(outputs, log=False)
        wandb.run.summary.update({
            f"test/{aggregation}/precision": report_dict['macro avg']['precision'], \
            f"test/{aggregation}/recall": report_dict['macro avg']['recall'], \
            f"test/{aggregation}/f1-score": report_dict['macro avg']['f1-score'] 
        }) 
        return report_dict
    
        
        
    # @torch.no_grad()
    # def test(self, dataloader, prefix="test", aggregation="macro avg"):
    #     self.model.eval()
    #     self.model = self.model.to(self.device)
    #     preds, target = ['None'] * 1
    #     for X, y in tqdm(dataloader):
    #         X, y = X.to(self.device), y.to(self.device)
    #         preds.add(self.model(X))
    #         target.add(y)
    #     print(preds)
    #     target = [t.detach().cpu().tolist() for t in target]
    #     preds = [item for sublist in preds for item in sublist]
    #     target = [item for sublist in target for item in sublist]
    #     report_dict = self._eval(preds=preds, target=target)
    #     wandb.run.summary.update({
    #         f"{prefix}/{aggregation}/precision": report_dict['macro avg']['precision'], \
    #         f"{prefix}/{aggregation}/recall": report_dict['macro avg']['recall'], \
    #         f"{prefix}/{aggregation}/f1-score": report_dict['macro avg']['f1-score'] 
    #     })  