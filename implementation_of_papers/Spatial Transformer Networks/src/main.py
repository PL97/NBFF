import torch
import numpy as np
import random
import sys
import argparse
import os

import wandb
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from trainer.trainer import trainer_base
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

sys.path.append("./")
from model.Net import Net
from dataset.get_data import get_data
from utils.utils import visualize_stn, convert_image_np


def setup():
    """parse arguments in commandline and return a args object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=2)
    parser.add_argument('--workspace', type=str, default="checkpoints/")
    parser.add_argument('--run_name', type=str, default="1")
    parser.add_argument('--dataset', type=str, default="MNIST")
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--rounds', type=int, default=50)
    
    
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
    wandb_logger.experiment.config.update({
        'dataset': args.dataset, \
        'rounds': args.rounds, \
        'lr': args.learning_rate, \
        'solver': "AdamW"
    })
    wandb_logger.watch(model, log="all")
    wandb.define_metric("train/*", step_metric="trainer/global_step")
    wandb.define_metric("val/*", step_metric="trainer/global_step")
    return wandb_logger




if __name__ == "__main__":
    
    args = setup()

    device = torch.device("cuda")
    trainloader, valloader, testloader, stats = get_data(name=args.dataset, \
                                                        batch_size=args.batch_size, \
                                                        random_seed=args.random_seed)
    args.datastats = stats

    model = Net(input_channels=stats['input_channels'])
        
    criterion = nn.CrossEntropyLoss()

    args.wandb_logger = pytorchlightning_wandb_setup(args=args)
    trainer = pl.Trainer(max_epochs=args.rounds, 
                        accelerator="gpu", 
                        devices=1, 
                        strategy = DDPStrategy(find_unused_parameters=False),
                        log_every_n_steps=1,
                        auto_scale_batch_size=True,
                        logger=args.wandb_logger)
    
    MyLightningModule = trainer_base(
                    model=model, criterion=criterion, args=args)
    trainer.fit(MyLightningModule, \
                train_dataloaders=trainloader, \
                val_dataloaders=valloader)
    
    
    ## final evaluation on train, val, and test set
    train_precision, train_recall = MyLightningModule.test(trainloader)
    val_precision, val_recall = MyLightningModule.test(valloader)
    test_precision, test_recall = MyLightningModule.test(testloader)
    
    wandb.run.summary.update({"train_precision": train_precision, \
                                "train_recall": train_recall, \
                                "val_precision": val_precision, \
                                "val_recall": val_recall, \
                                "test_precision": test_precision, \
                                "test_recall": test_recall})
    wandb.finish()
    
    visualize_stn(name=args.dataset, model=model, test_loader=testloader, device=device)
    