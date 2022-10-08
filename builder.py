from typing import Callable

import torch
import torch.optim
from torch import nn
from torchpack.utils.typing import Dataset, Optimizer, Scheduler

__all__ = ['build_dataset', 'build_model', 'build_loss', 'build_optimizer','build_scheduler']


def build_dataset(cfg) -> Dataset:  

    if cfg.dataset.name == 'Toronto-3D':
        from datasets.toronto3d import Toronto3D
        dataset = Toronto3D(cfg.dataset)

    elif cfg.dataset.name == 'SensatUrban':
        from datasets.sensaturban import SensatUrban
        dataset = SensatUrban(cfg.dataset)

    elif cfg.dataset.name == 'SemanticKITTI':
        from datasets.semantickitti import SemanticKITTI
        dataset = SemanticKITTI(cfg.dataset)
    
    else:
        raise NotImplementedError(cfg.dataset.name)

    return dataset

def build_model(cfg) -> nn.Module:

    if cfg.model.name == 'RandLA-Net':
        from models.randlanet import RandLANet
        model = RandLANet(cfg.model)

    elif cfg.model.name == 'RandLA-Net1':
        from models.randlanet1 import RandLANet1
        model = RandLANet1(cfg.model)

    # elif cfg.model.name == 'RandLA-Net1p1':
    #     from models.randlanet1p1 import RandLANet1p1
    #     model = RandLANet1p1(cfg.model)

    # elif cfg.model.name == 'RandLA-Net1p2':
    #     from models.randlanet1p2 import RandLANet1p2
    #     model = RandLANet1p2(cfg.model)

    # elif cfg.model.name == 'RandLA-Net1p3':
    #     from models.randlanet1p3 import RandLANet1p3
    #     model = RandLANet1p3(cfg.model)

    # elif cfg.model.name == 'RandLA-Net2p1':
    #     from models.randlanet2p1 import RandLANet2p1
    #     model = RandLANet2p1(cfg.model)

    elif cfg.model.name == 'RandLA-Net2':
        from models.randlanet2 import RandLANet2
        model = RandLANet2(cfg.model)

    elif cfg.model.name == 'RandLA-Net2_xyzdere':
        from models.randlanet2_xyzdere import RandLANet2_xyzdere
        model = RandLANet2_xyzdere(cfg.model)

    elif cfg.model.name == 'RandLA-Net2_rgbdere':
        from models.randlanet2_rgbdere import RandLANet2_rgbdere
        model = RandLANet2_rgbdere(cfg.model)
    # elif cfg.model.name == 'KPFCNN':
    #     from models.kpconv import KPFCNN

    # elif cfg.model.name == 'SPVCNN':
    #     from models.spvcnn import SPVCNN
    #     model = SPVCNN(cfg.model)

    else:
        raise NotImplementedError(cfg.model.name)

    return model

def build_loss(cfg) -> Callable:
    if cfg.loss.name == 'wce':
        from loss import CrossEntropyLoss
        Loss = CrossEntropyLoss(loss_weight=cfg.loss.get('loss_weight',1.0))

    elif cfg.loss.name == 'focal':
        from loss import FocalLoss
        Loss =  FocalLoss(gamma=cfg.loss.get('gamma',2.0),
                          alpha=cfg.loss.get('alpha',0.25),
                          loss_weight=cfg.loss.get('loss_weight',1.0))

    else:
        raise NotImplementedError(cfg.loss.name)
    return Loss

def build_optimizer(model: nn.Module, cfg) -> Optimizer:
    if cfg.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=cfg.optimizer.lr, 
                                     weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg.optimizer.lr,
                                      weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.optimizer.lr,
                                    momentum=cfg.optimizer.momentum,
                                    weight_decay=cfg.optimizer.weight_decay,
                                    nesterov=cfg.optimizer.nesterov)
    else:
        raise NotImplementedError(cfg.optimizer.name)
    
    return optimizer

def build_scheduler(optimizer: Optimizer, cfg) -> Scheduler:
    if cfg.scheduler.name == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda epoch: 1)

    elif cfg.scheduler.name == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           cfg.scheduler.gamma)
    elif cfg.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=cfg.num_epochs)
    elif cfg.scheduler.name == 'cosine_warmup':
        from functools import partial
        from schedulers import cosine_schedule_with_warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=partial(cosine_schedule_with_warmup,
                                                                        num_epochs=cfg.num_epochs,
                                                                        iter_per_epoch=cfg.dataset.train_steps_per_epoch)
                                                     )
    else:
        raise NotImplementedError(cfg.scheduler.name)
    return scheduler
