#!/usr/bin/env python

import os
import json
import pprint as pp
import warnings
import random

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
from lower.run import lRun
from lower.train import Ltrain_epoch
from lower.train import get_inner_model as Lget_inner_model
from Alternate_train import Lalternate_val_dataset, Lalternate_training_datasets
from problems import TAOP

UAVNum = 6

def Run(opts):

    # pre-training for lower-layer model
    if opts.pre_trained:
        Lmodel_, Loptimizer, Lbaseline, Llr_scheduler, Lproblem, Ltb_logger, Lopts = lRun(opts.pre_train_epochs)
        Lmodel = Lget_inner_model(Lmodel_)
        Lmodel.load_state_dict({**Lmodel.state_dict(), **{}.get('model', {})})

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    # save model to outputs dir
    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = TAOP()

    # if u have run the model before, u can continue from resume path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    # multi-gpu
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, Lmodel, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state from trained model
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch! 定义学习率调整器
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    baselineUpdate = False  # update the lower-layer model in the baseline

    if opts.eval_only:
        validate(model, Lmodel, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            if isinstance(Lmodel,torch.nn.DataParallel):
                Lmodel = Lmodel.module
                Lmodel.load_state_dict({**Lmodel.state_dict(), **{}.get('model', {})})
            Htraining_dataloader = train_epoch(
                                        model,
                                        Lmodel,
                                        optimizer,
                                        baseline,
                                        lr_scheduler,
                                        epoch,
                                        val_dataset,
                                        problem,
                                        tb_logger,
                                        opts,
                                        baselineUpdate
                                    )
            baselineUpdate = False
            if epoch < 12:
                if (epoch+1) % UAVNum == 0:
                    Lval_dataset = Lalternate_val_dataset(model, Lmodel, val_dataset, opts)
                    Ltraining_datasets = Lalternate_training_datasets(model, Lmodel, Htraining_dataloader, opts)
                    for train_id in range(UAVNum):
                        Ltrain_epoch(
                            Lmodel,
                            Loptimizer,
                            Lbaseline,
                            Llr_scheduler,
                            ((epoch+1)//UAVNum - 1)*UAVNum + opts.pre_train_epochs + train_id,
                            Lval_dataset,
                            Lproblem,
                            Ltb_logger,
                            Lopts,
                            Ltrain=True,
                            Ltraining_datasets=Ltraining_datasets[train_id*opts.epoch_size:(train_id+1)*opts.epoch_size]
                        )
                    baselineUpdate = True
                else:
                    continue
            else:
                Lval_dataset = Lalternate_val_dataset(model, Lmodel, val_dataset, opts)
                Ltraining_datasets = Lalternate_training_datasets(model, Lmodel, Htraining_dataloader, opts)
                Ltraining_datasets = random.sample(Ltraining_datasets, opts.epoch_size)
                Ltrain_epoch(
                    Lmodel,
                    Loptimizer,
                    Lbaseline,
                    Llr_scheduler,
                    epoch+opts.pre_train_epochs,
                    Lval_dataset,
                    Lproblem,
                    Ltb_logger,
                    Lopts,
                    Ltrain=True,
                    Ltraining_datasets=Ltraining_datasets
                )
                baselineUpdate = True

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    Run(get_options())
