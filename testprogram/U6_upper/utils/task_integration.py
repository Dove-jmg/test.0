import torch
import numpy as np
import os
import logging
import json
import time
import random
from typing import Dict, List, Tuple, Union, Any, Optional

from .llm_integration import LLMInterface, U6Task
from .hierarchical_rl import U6HighLevelPolicy, U6LowLevelPolicy
from .hierarchical_rl_impl import U6HierarchicalRL
from .dynamic_task_allocation_base import U6DroneState, U6TaskInfo
from .dynamic_task_allocation import U6DynamicTaskAllocator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class U6TaskIntegrationSystem:
    """U6_upper的任务集成系统，专为6无人机系统设计，支持协作任务处理"""
    
    def __init__(
        self,
        llm_model_name: str = "openlm/open-llama-3b",
        api_key: str = None,
        api_url: str = None,
        num_regions: int = 6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "./saved_models/u6",
        log_dir: str = "./logs/u6"
    ):
        """
        初始化任务集成系统
        
        Args:
            llm_model_name: LLM模型名称
            api_key: API密钥（如果使用远程API）
            api_url: API URL（如果使用远程API）
            num_regions: 区域数量
            device: 计算设备
            save_dir: 模型保存目录
            log_dir: 日志目录
        """
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建LLM接口
        logger.info("初始化LLM接口...")
        self.llm = LLMInterface(
            model_name=llm_model_name,
            api_key=api_key,
            api_url=api_url
        )
        
        # 创建分层强化学习模型
        logger.info("初始化分层强化学习模型...")
        self.high_level_policy = U6HighLevelPolicy(
            num_uavs=6,
            num_regions=num_regions,
            num_priority_levels=5
        )
        
        self.low_level_policy = U6LowLevelPolicy(
            state_dim=15,    # 无人机状态+任务特征
            action_dim=6,    # 执行动作数量
            hidden_dim=128,
            task_embed_dim=32
        )
        
        self.hierarchical_rl = U6HierarchicalRL(
            high_level_policy=self.high_level_policy,
            low_level_policy=self.low_level_policy,
            device=device
        )
        
        # 创建动态任务分配器
        logger.info("初始化动态任务分配器...")
        self.task_allocator = U6DynamicTaskAllocator(
            num_drones=6,
            num_regions=num_regions
        )
        
        # 任务处理状态
        self.tasks = []
        self.current_step = 0
        self.episode = 0
        
        logger.info("U6任务集成系统初始化完成")
    
    def process_task_instructions(self, task_instructions: List[str]) -> List[U6TaskInfo]:
        """
        处理任务指令，转换为结构化任务信息
        
        Args:
            task_instructions: 任务指令列表
            
        Returns:
            list: 任务信息对象列表
        """
        logger.info(f"处理{len(task_instructions)}个任务指令...")
        
        structured_tasks = []
        
        for i, instruction in enumerate(task_instructions):
            # 使用LLM解析任务
            task_info = self.llm.parse_task(instruction)
            
            # 如果解析失败，记录错误并跳过
            if "error" in task_info:
                logger.error(f"任务{i}解析失败: {task_info['error']}")
                continue
            
            # 为6无人机系统分解任务
            subtasks = self.llm.decompose_task_for_six_uavs(task_info)
            
            # 转换为U6TaskInfo对象
            for j, subtask in enumerate(subtasks):
                # 生成任务ID
                task_id = f"task_{i}_{j}"
                
                # 获取任务位置（区域中心点）
                region_id = subtask.get("分配区域", f"区域{j+1}")
                if isinstance(region_id, str) and region_id.startswith("区域"):
                    try:
                        region_num = int(region_id[2:]) - 1
                        region_id = min(max(0, region_num), 5)  # 确保在0-5范围内
                    except ValueError:
                        region_id = j % 6
                
                # 获取任务位置坐标
                if region_id >= len(self.task_allocator.region_centers):
                    position = (5.0, 5.0)  # 默认位置
                else:
                    position = self.task_allocator.region_centers[region_id]
                
                # 获取时间窗口
                time_window = subtask.get("时间窗口", {})
                start_time = time.time()
                end_time = start_time + 3600  # 默认1小时
                
                if isinstance(time_window, dict):
                    # 如果有具体的时间，解析时间字符串
                    start_str = time_window.get("开始时间", "")
                    end_str = time_window.get("结束时间", "")
                    
                    if start_str and ":" in start_str:
                        # 简单解析时间格式（如"10:00"）
                        hour, minute = map(int, start_str.split(":"))
                        current_time = time.localtime()
                        start_time = time.mktime((
                            current_time.tm_year, current_time.tm_mon, current_time.tm_mday,
                            hour, minute, 0, 0, 0, 0
                        ))
                    
                    if end_str and ":" in end_str:
                        # 简单解析时间格式（如"12:00"）
                        hour, minute = map(int, end_str.split(":"))
                        current_time = time.localtime()
                        end_time = time.mktime((
                            current_time.tm_year, current_time.tm_mon, current_time.tm_mday,
                            hour, minute, 0, 0, 0, 0
                        ))
                        
                        # 确保结束时间在开始时间之后
                        if end_time <= start_time:
                            end_time += 86400  # 添加一天
                
                # 判断是否需要协作
                requires_collaboration = False
                min_collaborators = 0
                collaboration_benefit = 1.0
                
                # 根据任务描述判断是否需要协作
                if "协作" in str(subtask.get("功能描述", "")) or "合作" in str(subtask.get("功能描述", "")):
                    requires_collaboration = True
                    min_collaborators = random.randint(1, 2)  # 随机需要1-2个协作者
                    collaboration_benefit = random.uniform(1.2, 1.5)  # 协作带来1.2-1.5倍效益
                
                # 创建任务信息对象
                task = U6TaskInfo(
                    task_id=task_id,
                    task_type=subtask.get("任务类型", "数据收集"),
                    position=position,
                    start_time=start_time,
                    end_time=end_time,
                    priority=subtask.get("优先级", 3),
                    data_size=10.0,  # 默认数据大小
                    region_id=region_id,
                    requires_sensors=[],  # 默认不需要特定传感器
                    requires_collaboration=requires_collaboration,
                    min_collaborators=min_collaborators,
                    collaboration_benefit=collaboration_benefit
                )
                
                structured_tasks.append(task)
                logger.info(f"创建任务: {task_id}, 类型: {task.task_type}, 优先级: {task.priority}, 需要协作: {requires_collaboration}")
        
        return structured_tasks
    
    def train_episode(
        self, 
        task_instructions: List[str],
        max_steps: int = 100,
        save: bool = True
    ) -> float:
        """
        训练一个回合
        
        Args:
            task_instructions: 任务指令列表
            max_steps: 最大步数
            save: 是否保存模型
            
        Returns:
            float: 回合总奖励
        """
        self.episode += 1
        logger.info(f"开始训练回合 {self.episode}, 最大步数: {max_steps}")
        
        # 处理任务指令
        structured_tasks = self.process_task_instructions(task_instructions)
        
        # 重置环境
        self.task_allocator = U6DynamicTaskAllocator(
            num_drones=6,
            num_regions=len(self.task_allocator.region_centers)
        )
        
        # 添加任务到分配器
        for task in structured_tasks:
            self.task_allocator.add_task(task)
        
        # 训练循环
        total_reward = 0.0
        
        for step in range(max_steps):
            self.current_step = step
            
            # 收集当前状态
            system_state = self.task_allocator.get_system_state()
            
            # 执行高层决策
            high_level_actions = []
            for task in self.task_allocator.unassigned_tasks[:]:  # 使用副本避免修改原列表
                # 准备特征向量
                task_features = task.get_feature_vector()
                
                # 选择一个无人机处理任务
                drone_features = self.task_allocator.drones[0].get_feature_vector()  # 默认第一个无人机
                region_features = np.array([0.5, 0.5, 0.25])  # 默认区域特征
                time_features = np.array([0.5, 1.0])  # 默认时间特征
                
                # 获取高层策略动作
                high_action = self.hierarchical_rl.high_level_step(
                    task_features=task_features,
                    uav_features=drone_features,
                    region_features=region_features,
                    time_features=time_features
                )
                
                high_level_actions.append((task, high_action))
            
            # 执行低层决策
            low_level_actions = []
            for i, drone in enumerate(self.task_allocator.drones):
                # 准备状态向量
                drone_state = drone.get_feature_vector()
                
                # 如果有正在执行的任务
                if drone.current_task:
                    task_features = drone.current_task.get_feature_vector()
                else:
                    # 默认任务特征
                    task_features = np.zeros(9)  # U6任务特征向量长度为9
                
                # 准备协作特征
                collaboration_features = np.zeros(6)  # 6个无人机的协作状态
                if drone.is_collaborating:
                    # 设置协作组中的无人机
                    for collab_id in drone.collaboration_group:
                        if 0 <= collab_id < 6:
                            collaboration_features[collab_id] = 1.0
                
                # 获取低层策略动作
                low_action = self.hierarchical_rl.low_level_step(
                    state=np.concatenate([drone_state, task_features]),
                    task_features=task_features,
                    collaboration_features=collaboration_features,
                    uav_id=i
                )
                
                low_level_actions.append((drone, low_action))
            
            # 将动作转换为任务分配器可接受的格式
            allocator_actions = []
            for drone, action in low_level_actions:
                allocator_actions.append(action["action_idx"])
            
            # 执行一个环境步骤
            next_system_state = self.task_allocator.step(allocator_actions)
            
            # 计算奖励
            reward = self.task_allocator.calculate_reward()
            total_reward += reward
            
            # 存储高层转换
            for task, high_action in high_level_actions:
                # 简化：使用相同的特征作为下一个状态
                self.hierarchical_rl.store_high_level_transition(
                    task_features=task.get_feature_vector(),
                    uav_features=self.task_allocator.drones[0].get_feature_vector(),
                    region_features=np.array([0.5, 0.5, 0.25]),
                    time_features=np.array([0.5, 1.0]),
                    action=high_action,
                    reward=reward,
                    next_task_features=task.get_feature_vector(),
                    next_uav_features=self.task_allocator.drones[0].get_feature_vector(),
                    next_region_features=np.array([0.5, 0.5, 0.25]),
                    next_time_features=np.array([0.5, 1.0]),
                    done=(step == max_steps - 1)
                )
            
            # 存储低层转换
            for i, (drone, low_action) in enumerate(low_level_actions):
                # 准备当前状态和下一个状态
                current_drone_state = drone.get_feature_vector()
                
                if drone.current_task:
                    current_task_features = drone.current_task.get_feature_vector()
                else:
                    current_task_features = np.zeros(9)
                
                # 准备当前协作特征
                current_collab_features = np.zeros(6)
                if drone.is_collaborating:
                    for collab_id in drone.collaboration_group:
                        if 0 <= collab_id < 6:
                            current_collab_features[collab_id] = 1.0
                
                next_drone_state = self.task_allocator.drones[i].get_feature_vector()
                
                if self.task_allocator.drones[i].current_task:
                    next_task_features = self.task_allocator.drones[i].current_task.get_feature_vector()
                else:
                    next_task_features = np.zeros(9)
                
                # 准备下一个协作特征
                next_collab_features = np.zeros(6)
                if self.task_allocator.drones[i].is_collaborating:
                    for collab_id in self.task_allocator.drones[i].collaboration_group:
                        if 0 <= collab_id < 6:
                            next_collab_features[collab_id] = 1.0
                
                # 存储转换
                self.hierarchical_rl.store_low_level_transition(
                    state=np.concatenate([current_drone_state, current_task_features]),
                    task_features=current_task_features,
                    collaboration_features=current_collab_features,
                    action=low_action,
                    reward=reward,
                    next_state=np.concatenate([next_drone_state, next_task_features]),
                    next_task_features=next_task_features,
                    next_collaboration_features=next_collab_features,
                    done=(step == max_steps - 1)
                )
            
            # 更新策略
            if step % 10 == 0:
                self.hierarchical_rl.update_high_level_policy()
                self.hierarchical_rl.update_low_level_policy()
            
            # 记录进度
            if step % 20 == 0 or step == max_steps - 1:
                completed = len(self.task_allocator.completed_tasks)
                total = len(self.task_allocator.all_tasks)
                collab_groups = len(self.task_allocator.collaboration_groups)
                logger.info(f"步骤 {step}/{max_steps}, 完成任务: {completed}/{total}, 协作组: {collab_groups}, 奖励: {reward:.2f}, 总奖励: {total_reward:.2f}")
        
        # 保存模型
        if save:
            self.save_models()
        
        avg_reward = total_reward / max_steps
        logger.info(f"回合 {self.episode} 完成，平均奖励: {avg_reward:.2f}")
        
        return avg_reward
    
    def evaluate(
        self,
        task_instructions: List[str],
        max_steps: int = 100
    ) -> float:
        """
        评估模型性能
        
        Args:
            task_instructions: 任务指令列表
            max_steps: 最大步数
            
        Returns:
            float: 评估总奖励
        """
        logger.info("开始评估...")
        
        # 处理任务指令
        structured_tasks = self.process_task_instructions(task_instructions)
        
        # 重置环境
        self.task_allocator = U6DynamicTaskAllocator(
            num_drones=6,
            num_regions=len(self.task_allocator.region_centers)
        )
        
        # 添加任务到分配器
        for task in structured_tasks:
            self.task_allocator.add_task(task)
        
        # 评估循环
        total_reward = 0.0
        
        for step in range(max_steps):
            self.current_step = step
            
            # 收集当前状态
            system_state = self.task_allocator.get_system_state()
            
            # 执行高层决策（不更新模型）
            high_level_actions = []
            for task in self.task_allocator.unassigned_tasks[:]:
                # 准备特征向量
                task_features = task.get_feature_vector()
                drone_features = self.task_allocator.drones[0].get_feature_vector()
                region_features = np.array([0.5, 0.5, 0.25])
                time_features = np.array([0.5, 1.0])
                
                # 获取高层策略动作
                high_action = self.hierarchical_rl.high_level_step(
                    task_features=task_features,
                    uav_features=drone_features,
                    region_features=region_features,
                    time_features=time_features
                )
                
                high_level_actions.append((task, high_action))
            
            # 执行低层决策（不更新模型）
            low_level_actions = []
            for i, drone in enumerate(self.task_allocator.drones):
                # 准备状态向量
                drone_state = drone.get_feature_vector()
                
                if drone.current_task:
                    task_features = drone.current_task.get_feature_vector()
                else:
                    task_features = np.zeros(9)
                
                # 准备协作特征
                collaboration_features = np.zeros(6)
                if drone.is_collaborating:
                    for collab_id in drone.collaboration_group:
                        if 0 <= collab_id < 6:
                            collaboration_features[collab_id] = 1.0
                
                # 获取低层策略动作
                low_action = self.hierarchical_rl.low_level_step(
                    state=np.concatenate([drone_state, task_features]),
                    task_features=task_features,
                    collaboration_features=collaboration_features,
                    uav_id=i
                )
                
                low_level_actions.append((drone, low_action))
            
            # 将动作转换为任务分配器可接受的格式
            allocator_actions = []
            for drone, action in low_level_actions:
                allocator_actions.append(action["action_idx"])
            
            # 执行一个环境步骤
            next_system_state = self.task_allocator.step(allocator_actions)
            
            # 计算奖励
            reward = self.task_allocator.calculate_reward()
            total_reward += reward
            
            # 记录进度
            if step % 20 == 0 or step == max_steps - 1:
                completed = len(self.task_allocator.completed_tasks)
                total = len(self.task_allocator.all_tasks)
                collab_groups = len(self.task_allocator.collaboration_groups)
                logger.info(f"评估步骤 {step}/{max_steps}, 完成任务: {completed}/{total}, 协作组: {collab_groups}, 奖励: {reward:.2f}")
        
        # 分析协作结果
        collab_tasks = [t for t in self.task_allocator.completed_tasks if t.requires_collaboration]
        if collab_tasks:
            successful_collabs = sum(1 for t in collab_tasks if t.collaboration_success)
            collab_ratio = successful_collabs / len(collab_tasks)
            logger.info(f"协作任务完成情况: {successful_collabs}/{len(collab_tasks)} 成功率: {collab_ratio:.2%}")
        
        avg_reward = total_reward / max_steps
        logger.info(f"评估完成，平均奖励: {avg_reward:.2f}")
        
        return avg_reward
    
    def generate_collaboration_strategy(self, drones: List[int], task_description: str) -> Dict[str, Any]:
        """
        使用LLM生成协作策略
        
        Args:
            drones: 参与协作的无人机ID列表
            task_description: 任务描述
            
        Returns:
            dict: 协作策略
        """
        # 收集无人机状态
        drone_states = []
        for drone_id in drones:
            drone = self.task_allocator.get_drone(drone_id)
            if drone:
                drone_states.append(drone.to_dict())
        
        # 为这些无人机创建一个任务
        task = {
            "任务描述": task_description,
            "无人机": drones,
            "协作要求": "需要协调多架无人机共同完成"
        }
        
        # 使用LLM生成协作策略
        strategy = self.llm.generate_collaboration_strategy(drone_states, [task])
        
        return strategy
    
    def save_models(self):
        """保存模型"""
        high_level_path = os.path.join(self.save_dir, f"u6_high_level_ep{self.episode}.pt")
        low_level_path = os.path.join(self.save_dir, f"u6_low_level_ep{self.episode}.pt")
        
        self.hierarchical_rl.save_models(high_level_path, low_level_path)
        
        # 保存元数据
        metadata = {
            "episode": self.episode,
            "timestamp": time.time(),
            "high_level_path": high_level_path,
            "low_level_path": low_level_path
        }
        
        with open(os.path.join(self.save_dir, "u6_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"已保存模型到 {self.save_dir}")
    
    def load_models(self, episode: int = None) -> bool:
        """
        加载模型
        
        Args:
            episode: 要加载的回合，None表示最新
            
        Returns:
            bool: 是否成功加载
        """
        # 如果指定了回合
        if episode is not None:
            high_level_path = os.path.join(self.save_dir, f"u6_high_level_ep{episode}.pt")
            low_level_path = os.path.join(self.save_dir, f"u6_low_level_ep{episode}.pt")
            
            if os.path.exists(high_level_path) and os.path.exists(low_level_path):
                self.hierarchical_rl.load_models(high_level_path, low_level_path)
                self.episode = episode
                return True
            else:
                logger.error(f"找不到回合 {episode} 的模型文件")
                return False
        
        # 加载最新的模型
        metadata_path = os.path.join(self.save_dir, "u6_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                high_level_path = metadata.get("high_level_path")
                low_level_path = metadata.get("low_level_path")
                
                if os.path.exists(high_level_path) and os.path.exists(low_level_path):
                    self.hierarchical_rl.load_models(high_level_path, low_level_path)
                    self.episode = metadata.get("episode", 0)
                    logger.info(f"成功加载回合 {self.episode} 的模型")
                    return True
                else:
                    logger.error("元数据中的模型路径无效")
                    return False
            
            except Exception as e:
                logger.error(f"加载模型元数据时出错: {e}")
                return False
        else:
            logger.error("找不到模型元数据文件")
            return False 