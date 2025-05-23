#!/usr/bin/env python

import os
import json
import numpy as np
import torch
import torch.optim as optim
import argparse
import logging
from tensorboard_logger import Logger as TbLogger

from lower.nets.critic_network import CriticNetwork
from lower.options import Lget_options
from lower.train import Ltrain_epoch, validate, get_inner_model
from lower.reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from lower.nets.attention_model import AttentionModel
from lower.nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from lower.utils import torch_load_cpu, load_problem
from lower.utils.llm_integration import LLMInterface
from lower.utils.hierarchical_rl import HierarchicalRL, HighLevelPolicy, LowLevelPolicy
from lower.utils.dynamic_task_allocation import DynamicTaskAllocator, DroneState, TaskInfo
from lower.utils.task_integration import TaskIntegrationSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("integrated_run.log")
    ]
)
logger = logging.getLogger(__name__)

def run_integrated(args=None):
    """
    集成运行脚本，结合原始代码与新功能
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="集成运行脚本")
    
    # 原始模型参数
    parser.add_argument('--problem', default='op', help="要解决的问题，默认为'op'")
    parser.add_argument('--graph_size', type=int, default=80, help="问题图的大小")
    parser.add_argument('--batch_size', type=int, default=512, help="训练期间每批实例的数量")
    parser.add_argument('--epoch_size', type=int, default=1280000, help="训练期间每个epoch的实例数量")
    parser.add_argument('--val_size', type=int, default=10000, help="用于报告验证性能的实例数量")
    parser.add_argument('--val_dataset', type=str, default=None, help="用于验证的数据集文件")
    parser.add_argument('--model', default='attention', help="模型，'attention'（默认）或'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help="输入嵌入的维度")
    parser.add_argument('--hidden_dim', type=int, default=128, help="编码器/解码器中隐藏层的维度")
    parser.add_argument('--n_encode_layers', type=int, default=3, help="编码器/评论家网络中的层数")
    parser.add_argument('--n_epochs', type=int, default=100, help="训练的总轮数")
    parser.add_argument('--seed', type=int, default=1234, help="随机种子")
    parser.add_argument('--baseline', default='rollout', help="使用的基准线：'rollout'、'critic'或'exponential'")
    parser.add_argument('--eval_only', action='store_true', help="仅评估模型，不训练")
    
    # 集成功能参数
    parser.add_argument('--use_integration', action='store_true', help="是否使用集成功能")
    parser.add_argument('--llm_model', type=str, default="openlm/open-llama-3b", help="LLM模型名称或路径")
    parser.add_argument('--api_key', type=str, default=None, help="API密钥（如果使用远程API）")
    parser.add_argument('--api_url', type=str, default=None, help="API URL（如果使用远程API）")
    parser.add_argument('--num_drones', type=int, default=6, help="无人机数量")
    parser.add_argument('--num_regions', type=int, default=4, help="区域数量")
    parser.add_argument('--integration_episodes', type=int, default=10, help="集成训练的回合数")
    parser.add_argument('--max_steps', type=int, default=100, help="每个回合的最大步数")
    parser.add_argument('--save_dir', type=str, default="./saved_models", help="模型保存目录")
    parser.add_argument('--log_dir', type=str, default="./logs", help="日志目录")
    parser.add_argument('--load_episode', type=int, default=0, help="加载指定回合的模型，0表示不加载")
    
    # 解析参数
    opts = parser.parse_args(args)
    
    # 设置随机种子
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    
    # 创建保存目录
    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 首先加载原始问题模型部分
    logger.info("加载原始问题模型...")
    
    # 获取原始参数
    Lopts = Lget_options(args)
    Lopts.device = device
    
    # 加载问题
    problem = load_problem(Lopts.problem)
    
    # 初始化原始模型
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(Lopts.model, None)
    
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
    ).to(device)
    
    # 加载基准线
    if Lopts.baseline == 'exponential':
        baseline = ExponentialBaseline(Lopts.exp_beta)
    elif Lopts.baseline == 'critic' or Lopts.baseline == 'critic_lstm':
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
        baseline = NoBaseline()
    
    # 初始化优化器
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': Lopts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': Lopts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )
    
    # 初始化学习率调度器
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: Lopts.lr_decay ** epoch)
    
    # 加载验证数据集
    val_dataset = problem.make_dataset(
        size=Lopts.graph_size, num_samples=Lopts.val_size, filename=Lopts.val_dataset
    )
    
    # 如果开启了集成功能，则初始化任务集成系统
    if opts.use_integration:
        logger.info("初始化任务集成系统...")
        
        integration_system = TaskIntegrationSystem(
            llm_model_name=opts.llm_model,
            api_key=opts.api_key,
            api_url=opts.api_url,
            num_drones=opts.num_drones,
            num_regions=opts.num_regions,
            device=device,
            save_dir=opts.save_dir,
            log_dir=opts.log_dir
        )
        
        # 如果指定了加载回合，尝试加载模型
        if opts.load_episode > 0:
            success = integration_system.load_models(opts.load_episode)
            if success:
                logger.info(f"成功加载回合 {opts.load_episode} 的模型")
            else:
                logger.warning(f"加载回合 {opts.load_episode} 的模型失败，将使用初始模型")
        
        # 示例任务指令
        task_instructions = [
            """
            10:00 - 12:00，在校园收集空气质量数据，每30分钟采集一次
            """,
            """
            10:00 - 12:00，在校园收集噪声数据，每40分钟采集一次
            """,
            """
            10:00 - 12:00，在校园收集道路流量数据，每20分钟采集一次
            """,
            """
            09:30 - 11:00，在居民区收集温度数据，每20分钟采集一次
            """,
            """
            10:30 - 12:00，在居民区收集温度数据，每30分钟采集一次
            """,
            """
            10:00 - 11:30，在居民区收集温度数据，每25分钟采集一次
            """,
            """
            11:15 - 12:30，将收集的数据卸载至边缘服务器
            """
        ]
        
        # 训练任务集成系统
        logger.info(f"开始任务集成系统训练，共 {opts.integration_episodes} 个回合...")
        
        integration_rewards = []
        
        for episode in range(1, opts.integration_episodes + 1):
            logger.info(f"开始集成训练回合 {episode}/{opts.integration_episodes}")
            
            # 训练一个回合
            reward = integration_system.train_episode(
                task_instructions=task_instructions,
                max_steps=opts.max_steps
            )
            
            integration_rewards.append(reward)
            
            # 打印进度
            avg_reward = sum(integration_rewards) / len(integration_rewards)
            logger.info(f"回合 {episode} 完成，奖励: {reward:.2f}, 平均奖励: {avg_reward:.2f}")
            
            # 每5个回合评估一次原始模型性能
            if episode % 5 == 0 or episode == opts.integration_episodes:
                logger.info("评估原始模型性能...")
                avg_reward = validate(model, val_dataset, Lopts)
                logger.info(f"原始模型验证平均奖励: {avg_reward}")
        
        # 评估最终集成系统性能
        final_reward = integration_system.evaluate(
            task_instructions=task_instructions,
            max_steps=opts.max_steps
        )
        
        logger.info(f"集成系统训练完成，最终评估奖励: {final_reward:.2f}")
    
    # 如果不只是评估模式，则训练原始模型
    if not opts.eval_only:
        logger.info(f"开始原始模型训练，共 {Lopts.n_epochs} 个轮次...")
        
        # 初始化TensorBoard日志记录器
        tb_logger = None
        if not Lopts.no_tensorboard:
            tb_logger = TbLogger(os.path.join(Lopts.log_dir, "{}_{}".format(Lopts.problem, Lopts.graph_size), Lopts.run_name))
        
        for epoch in range(Lopts.epoch_start, Lopts.epoch_start + Lopts.n_epochs):
            logger.info(f"开始轮次 {epoch}")
            
            # 训练一个轮次
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
            
            # 保存模型
            if (Lopts.checkpoint_epochs != 0 and epoch % Lopts.checkpoint_epochs == 0) or epoch == Lopts.n_epochs - 1:
                logger.info('保存模型和状态...')
                torch.save(
                    {
                        'model': get_inner_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state_all(),
                        'baseline': baseline.state_dict()
                    },
                    os.path.join(Lopts.save_dir, 'epoch-{}.pt'.format(epoch))
                )
    else:
        # 仅评估原始模型
        logger.info("仅评估原始模型性能...")
        avg_reward = validate(model, val_dataset, Lopts)
        logger.info(f"原始模型验证平均奖励: {avg_reward}")
    
    logger.info("程序执行完成")
    
    return model, optimizer, baseline, lr_scheduler, problem, integration_system if opts.use_integration else None

if __name__ == "__main__":
    run_integrated() 