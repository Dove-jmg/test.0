#!/usr/bin/env python

import os
import json
import argparse
import logging
import time
import torch
import numpy as np
import warnings

from U4_upper.utils.llm_integration import LLMInterface
from U4_upper.utils.hierarchical_rl import U4HighLevelPolicy, U4LowLevelPolicy, U4HierarchicalRL
from U4_upper.utils.dynamic_task_allocation import U4DynamicTaskAllocator
from U4_upper.utils.task_integration import U4TaskIntegrationSystem
from options import get_options

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("u4_integrated_run.log")
    ]
)
logger = logging.getLogger(__name__)

def run_integrated(args=None):
    """
    U4_upper集成运行脚本
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="U4_upper集成运行脚本")
    
    # 模型和LLM参数
    parser.add_argument('--llm_model', type=str, default="openlm/open-llama-3b", help="LLM模型名称或路径")
    parser.add_argument('--api_key', type=str, default=None, help="API密钥（如果使用远程API）")
    parser.add_argument('--api_url', type=str, default=None, help="API URL（如果使用远程API）")
    parser.add_argument('--no_llm', action='store_true', help="不使用LLM组件")
    
    # 训练参数
    parser.add_argument('--num_regions', type=int, default=4, help="区域数量")
    parser.add_argument('--num_episodes', type=int, default=10, help="训练回合数")
    parser.add_argument('--max_steps', type=int, default=100, help="每个回合的最大步数")
    parser.add_argument('--load_episode', type=int, default=0, help="加载指定回合的模型，0表示不加载")
    parser.add_argument('--eval_only', action='store_true', help="仅评估模型，不训练")
    
    # 路径参数
    parser.add_argument('--save_dir', type=str, default="./saved_models/u4", help="模型保存目录")
    parser.add_argument('--log_dir', type=str, default="./logs/u4", help="日志目录")
    parser.add_argument('--task_file', type=str, default=None, help="任务指令文件路径")
    
    # 原始模型集成参数
    parser.add_argument('--integrate_original', action='store_true', help="是否与原始模型集成")
    parser.add_argument('--original_weight', type=float, default=0.5, help="原始模型权重")
    
    # 解析参数
    cli_args = parser.parse_args(args)
    
    # 获取原始选项
    orig_opts = get_options(args)
    
    # 设置随机种子
    torch.manual_seed(orig_opts.seed)
    np.random.seed(orig_opts.seed)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs(cli_args.save_dir, exist_ok=True)
    os.makedirs(cli_args.log_dir, exist_ok=True)
    
    # 示例任务指令
    default_task_instructions = [
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
    
    # 如果指定了任务文件，则从文件加载任务
    task_instructions = default_task_instructions
    if cli_args.task_file and os.path.exists(cli_args.task_file):
        try:
            with open(cli_args.task_file, 'r', encoding='utf-8') as f:
                file_tasks = f.readlines()
                # 过滤掉空行
                task_instructions = [task.strip() for task in file_tasks if task.strip()]
            logger.info(f"从文件加载了 {len(task_instructions)} 个任务指令")
        except Exception as e:
            logger.error(f"加载任务文件时出错: {e}")
    
    # 初始化任务集成系统
    integration_system = U4TaskIntegrationSystem(
        llm_model_name=cli_args.llm_model if not cli_args.no_llm else None,
        api_key=cli_args.api_key,
        api_url=cli_args.api_url,
        num_regions=cli_args.num_regions,
        device=device,
        save_dir=cli_args.save_dir,
        log_dir=cli_args.log_dir
    )
    
    # 如果指定了加载回合，尝试加载模型
    if cli_args.load_episode > 0:
        success = integration_system.load_models(cli_args.load_episode)
        if success:
            logger.info(f"成功加载回合 {cli_args.load_episode} 的模型")
        else:
            logger.warning(f"加载回合 {cli_args.load_episode} 的模型失败，将使用初始模型")
    
    # 如果只是评估模式
    if cli_args.eval_only:
        logger.info("仅评估模式")
        avg_reward = integration_system.evaluate(
            task_instructions=task_instructions,
            max_steps=cli_args.max_steps
        )
        logger.info(f"评估完成，平均奖励: {avg_reward:.2f}")
        return avg_reward
    
    # 训练任务集成系统
    logger.info(f"开始任务集成系统训练，共 {cli_args.num_episodes} 个回合...")
    
    episode_rewards = []
    
    for episode in range(1, cli_args.num_episodes + 1):
        logger.info(f"开始集成训练回合 {episode}/{cli_args.num_episodes}")
        
        # 训练一个回合
        reward = integration_system.train_episode(
            task_instructions=task_instructions,
            max_steps=cli_args.max_steps
        )
        
        episode_rewards.append(reward)
        
        # 打印进度
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        logger.info(f"回合 {episode} 完成，奖励: {reward:.2f}, 平均奖励: {avg_reward:.2f}")
    
    # 评估最终系统性能
    logger.info("评估最终系统性能...")
    final_reward = integration_system.evaluate(
        task_instructions=task_instructions,
        max_steps=cli_args.max_steps
    )
    
    logger.info(f"训练和评估完成，最终奖励: {final_reward:.2f}")
    
    # 如果要集成原始模型
    if cli_args.integrate_original:
        logger.info(f"与原始模型集成，权重: {cli_args.original_weight}")
        
        # 这里可以添加与原始模型的集成逻辑
        # 例如，将输出与原始模型的输出按权重合并
        
        logger.info("集成完成")
    
    # 保存最终结果
    results = {
        "episode_rewards": episode_rewards,
        "final_reward": final_reward,
        "num_episodes": cli_args.num_episodes,
        "max_steps": cli_args.max_steps,
        "llm_used": not cli_args.no_llm,
        "integrated_with_original": cli_args.integrate_original
    }
    
    with open(os.path.join(cli_args.save_dir, "u4_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"结果已保存到 {os.path.join(cli_args.save_dir, 'u4_results.json')}")
    
    return final_reward

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_integrated() 