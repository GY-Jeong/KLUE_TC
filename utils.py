import os
import random
import torch
import numpy as np
from torch import nn

from torch.optim import Adam, AdamW, SGD
from adamp import AdamP
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR, \
    CosineAnnealingWarmRestarts
from transformers import get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from dataloader import YNAT_dataset


def set_seeds(seed=42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))


def get_optimizer(model, args):
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamP':
        optimizer = AdamP(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()

    return optimizer


def get_scheduler(optimizer, args):
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=args.plateau_patience, factor=args.plateau_factor, mode='max',
                                      verbose=True)
    elif args.scheduler == 'linear_warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.total_steps)
    elif args.scheduler == 'step_lr':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'exp_lr':
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == 'cosine_annealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)
    elif args.scheduler == 'cosine_annealing_warmstart':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min,
                                                last_epoch=-1)

    return scheduler


def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name
        else args.model_name_or_path,
        use_fast=True,
    )

    return tokenizer


def load_model(args, model_name=None):
    if not model_name:
        model_name = args.model_name
    model_path = os.path.join(args.model_dir, model_name)
    load_state = torch.load(model_path)
    print("Loading Model from:", model_path)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        args.config_name
        if args.config_name
        else args.model_name_or_path,
    )

    config.num_labels = 7

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config
    ).to(args.device)

    model.load_state_dict(load_state['state_dict'], strict=True)

    print(model)

    print("Loading Model from:", model_path, "...Finished.")

    return model


def get_loaders(args, train, valid, is_inference=False):
    pin_memory = True
    train_loader, valid_loader = None, None

    if is_inference:
        test_dataset = YNAT_dataset(args, valid, is_inference)
        test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=args.num_workers, shuffle=False,
                                                  batch_size=args.batch_size, pin_memory=pin_memory)
        return test_loader

    if train is not None:
        train_dataset = YNAT_dataset(args, train, is_inference)
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=args.num_workers, shuffle=True,
                                                   batch_size=args.batch_size, pin_memory=pin_memory)
    if valid is not None:
        valid_dataset = YNAT_dataset(args, valid, is_inference)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, num_workers=args.num_workers, shuffle=False,
                                                   batch_size=args.batch_size, pin_memory=pin_memory)

    return train_loader, valid_loader


# loss계산하고 parameter update!
def compute_loss(preds, targets, args):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)
    """
    # print(preds, targets)
    loss = get_criterion(preds, targets, args)
    # 마지막 시퀀스에 대한 값만 loss 계산
    # loss = loss[:, -1]
    # loss = torch.mean(loss)
    return loss


def get_criterion(pred, target, args):
    if args.criterion == 'BCE':
        loss = nn.BCELoss(reduction="none")
    elif args.criterion == "BCELogit":
        loss = nn.BCEWithLogitsLoss(reduction="none")
    elif args.criterion == "MSE":
        loss = nn.MSELoss(reduction="none")
    elif args.criterion == "L1":
        loss = nn.L1Loss(reduction="none")
    elif args.criterion == "CE":
        loss = nn.CrossEntropyLoss()
    # NLL, CrossEntropy not available
    return loss(pred, target)
