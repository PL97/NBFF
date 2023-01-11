import pandas as pd
import torch.optim as optim
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
import argparse
import os
import numpy as np
import random
import sys
sys.path.append(".")

from model.BILSTM_CRF import BiLSTM_CRF
from dataset.CONLL_2003 import DataSequence
from trainer.trainer import trainer_base
from dataset.get_data import get_data


def setup():
    """parse arguments in commandline and return a args object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=2)
    parser.add_argument('--workspace', type=str, default="checkpoints/")
    parser.add_argument('--run_name', type=str, default="1")
    parser.add_argument('--dataset', type=str, default="CONLL_2003")
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--rounds', type=int, default=1000)
    
    
    args = parser.parse_args()
    args.workspace = os.path.join(args.workspace, args.dataset)
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    args.run_name = args.dataset
    
    
    ## create workspace
    try:
        os.removedirs(args.workspace)
        os.makedirs(args.workspace, exist_ok=True)
    except:
        os.makedirs(args.workspace, exist_ok=True)
    return args

def pytorchlightning_wandb_setup(args):
    """initialize wandb looger
    Args:
        args (_type_): argurments containing training hyparams
    Returns:
        wandb_logger: will be used for Pytorch lightning trainer
    """
    wandb_logger = WandbLogger(project="STN", \
                                name=args.run_name, \
                                save_dir=args.workspace)
    # wandb_logger.experiment.config.update({
    #     'dataset': args.dataset, \
    #     'rounds': args.rounds, \
    #     'lr': args.learning_rate, \
    #     'solver': "AdamW"
    # })
    # wandb_logger.watch(model, log="all")
    # wandb.define_metric("train/*", step_metric="trainer/global_step")
    # wandb.define_metric("val/*", step_metric="trainer/global_step")
    return wandb_logger




if __name__ == "__main__":
    
    args = setup()

    device = torch.device("cuda")
    
    trainloader, valloader, testloader, stats = get_data(name=args.dataset, batch_size=args.batch_size)
    args.ids_to_labels =stats['ids_to_labels']
    
    

    
    model = BiLSTM_CRF(vocab_size=stats['vocab_size'], \
                       tag_to_ix=stats['tag_to_ix'], \
                       embedding_dim=32, \
                       hidden_dim=500, \
                       word_dim=stats['max_length'], \
                       device=device)
    
    model = model.to(device)
    from torch.optim import SGD, AdamW, Adam, LBFGS
    optimizer = AdamW([
                {'params': model.parameters(), 'lr': args.learning_rate}
                ])
    for i in range(100):
        print("============")
        for x, y in trainloader:
            print(x.shape, y.shape)
            model.zero_grad()
            x = x.to(device)
            y = y.to(device)
            # print(x.shape, y.shape)
            # print(model(x))
            loss = model.neg_log_likelihood(x, y)
            loss.backward()
            optimizer.step()
            print(loss)
            asdf
        
    exit("finished")
    
        
    args.wandb_logger = pytorchlightning_wandb_setup(args=args)
    trainer = pl.Trainer(max_epochs=args.rounds, 
                        accelerator="gpu", 
                        devices=1, 
                        strategy = DDPStrategy(find_unused_parameters=False),
                        log_every_n_steps=1,
                        auto_scale_batch_size=True,
                        logger=args.wandb_logger)
    
    MyLightningModule = trainer_base(
                    model=model, criterion=None, args=args)
    trainer.fit(MyLightningModule, \
                train_dataloaders=trainloader, \
                val_dataloaders=valloader)