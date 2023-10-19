import torch
import random
import argparse
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import os
import pwd
import yaml
import json
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.simulation import ARHMNLICADataset
import models.simulation as sim_models
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
# torch.set_float32_matmul_precision('high')


def main(args):
    # seed everything
    config = yaml.safe_load(open(args.config, 'r'))
    pl.seed_everything(args.seed)
    data = ARHMNLICADataset(data_path=config['dataset']['data_path'])
    n_validation = config['dataset']['n_validation']
    train_data, valid_data = random_split(
        data, [len(data) - n_validation, n_validation])

    train_loader = DataLoader(train_data,
                              shuffle=False,
                              batch_size=config['dataloader']['train_batch_size'],
                              num_workers=config['dataloader']['num_workers'],
                              pin_memory=config['dataloader']['pin_memory'])
    valid_loader = DataLoader(valid_data,
                              shuffle=False,
                              batch_size=config['dataloader']['valid_batch_size'],
                              num_workers=config['dataloader']['num_workers'],
                              pin_memory=config['dataloader']['pin_memory'])
    model_class = getattr(sim_models, config['model'])
    model = model_class(**config['model_kwargs'])
    model.A = data.A
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=config['trainer']['default_root_dir']
    )
    checkpoint_callback = ModelCheckpoint(monitor='val/mcc',
                                          save_top_k=1,
                                          mode='max')
    early_stop_callback = EarlyStopping(monitor="val/mcc",
                                        stopping_threshold=0.99,
                                        patience=1_000,
                                        verbose=False,
                                        mode="max")
    logger_list = [tb_logger]
    trainer = pl.Trainer(
        logger=logger_list,
        callbacks=[checkpoint_callback,early_stop_callback],
        **config['trainer'],)
    log_dir = Path(trainer.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(config, open(log_dir/'config.yaml', 'w'))
    trainer.fit(model, train_loader, valid_loader)
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c',
        '--config',
        type=str,
        required=True
    )

    argparser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=770
    )
    args = argparser.parse_args()
    main(args)
