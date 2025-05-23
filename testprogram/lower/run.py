#!/usr/bin/env python

import os
import json
import pprint as pp
import numpy as np
import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from lower.nets.critic_network import CriticNetwork
from lower.options import Lget_options
from lower.train import Ltrain_epoch, validate, get_inner_model
from lower.reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from lower.nets.attention_model import AttentionModel
from lower.nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from lower.utils import torch_load_cpu, load_problem


def run(Lopts):

    # Pretty print the run args
    pp.pprint(vars(Lopts))

    # Set the random seed
    torch.manual_seed(Lopts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not Lopts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(Lopts.log_dir, "{}_{}".format(Lopts.problem, Lopts.graph_size), Lopts.run_name))

    os.makedirs(Lopts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(Lopts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(Lopts), f, indent=True)

    # Set the device
    Lopts.device = torch.device("cuda:0" if Lopts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(Lopts.problem)

    # Load data from load_path
    load_data = {}
    assert Lopts.load_path is None or Lopts.resume is None, "Only one of load path and resume can be given"
    load_path = Lopts.load_path if Lopts.load_path is not None else Lopts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(Lopts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        Lopts.embedding_dim,
        Lopts.hidden_dim,
        problem,
        n_encode_layers=Lopts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=Lopts.normalization,
        tanh_clipping=Lopts.tanh_clipping,
        checkpoint_encoder=Lopts.checkpoint_encoder,
        shrink_size=Lopts.shrink_size
    ).to(Lopts.device)

    if Lopts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if Lopts.baseline == 'exponential':
        baseline = ExponentialBaseline(Lopts.exp_beta)
    elif Lopts.baseline == 'critic' or Lopts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    Lopts.embedding_dim,
                    Lopts.hidden_dim,
                    Lopts.n_encode_layers,
                    Lopts.tanh_clipping
                )
                if Lopts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    Lopts.embedding_dim,
                    Lopts.hidden_dim,
                    Lopts.n_encode_layers,
                    Lopts.normalization
                )
            ).to(Lopts.device)
        )
    elif Lopts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, Lopts)
    else:
        assert Lopts.baseline is None, "Unknown baseline: {}".format(Lopts.baseline)
        baseline = NoBaseline()

    if Lopts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, Lopts.bl_warmup_epochs, warmup_exp_beta=Lopts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': Lopts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': Lopts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(Lopts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: Lopts.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=Lopts.graph_size, num_samples=Lopts.val_size, filename=Lopts.val_dataset, distribution=Lopts.data_distribution)

    if Lopts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(Lopts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if Lopts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        Lopts.epoch_start = epoch_resume + 1

    if Lopts.eval_only:
        validate(model, val_dataset, Lopts)
    else:
        for epoch in range(Lopts.epoch_start, Lopts.epoch_start + Lopts.n_epochs):
            Ltrain_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                Lopts
            )

    return model, optimizer, baseline, lr_scheduler, problem, tb_logger, Lopts

def lRun(pre_train_epochs):
    Lopts = Lget_options()
    Lopts.n_epochs = pre_train_epochs
    Lmodel, Loptimizer, Lbaseline, Llr_scheduler, Lproblem, Ltb_logger, Lopts = run(Lopts)
    Lbaseline.baseline.bl_vals = np.zeros(Lbaseline.baseline.bl_vals.shape, dtype=float)
    Lbaseline.baseline.mean = 0.
    return Lmodel, Loptimizer, Lbaseline, Llr_scheduler, Lproblem, Ltb_logger, Lopts
