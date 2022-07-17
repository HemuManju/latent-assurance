import os
from datetime import date
import itertools

import pandas as pd
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl

from src.dataset import motion_planner_dataset

from src.models.motion_planner import MotionPlanner

from src.visualization.visualize import plot_trends

import yaml
from utils import skip_run, get_num_gpus


with skip_run('skip', 'warm_starting') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/motion_planning.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today())

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint and logging
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename='warm_start',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(cfg['logs_path'], name='warm_start')

    # Network setup
    # TODO: Implement net
    net = None
    # output = net(net.example_input_array)
    # print(output)  # verification

    # Dataloader
    data_loader = motion_planner_dataset.webdataset_data_iterator(cfg)
    model = MotionPlanner(cfg, net, data_loader)

    # Run the trainer
    if cfg['check_point_restore_path'] is None:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
        )
    else:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=cfg['check_point_restore_path'],
            enable_progress_bar=False,
        )
    trainer.fit(model)
