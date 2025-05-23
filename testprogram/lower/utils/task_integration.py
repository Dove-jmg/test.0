import torch
import numpy as np
import os
import logging
import json
from typing import Dict, List, Tuple, Union, Any, Optional
import math
import time
import random
import uuid

# 导入自定义模块
from lower.utils.llm_integration import LLMInterface, Task
from lower.utils.hierarchical_rl import HierarchicalRL, HighLevelPolicy, LowLevelPolicy
from lower.utils.dynamic_task_allocation import DynamicTaskAllocator, DroneState, TaskInfo

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskIntegrationSystem:
    """任务集成系统，整合LLM、分层强化学习和动态任务分配"""
    
    def __init__(
        self,
        llm_model_name: str = "openlm/open-llama-3b",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        num_drones: int = 6,
        num_regions: int = 4,
        high_level_lr: float = 1e-4,
        low_level_lr: float = 1e-4,
        gamma: float = 0.99,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "./saved_models",
        log_dir: str = "./logs"
    ):
        # 创建目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # 初始化LLM接口
        self.llm = LLMInterface(
            model_name=llm_model_name,
            api_key=api_key,
            api_url=api_url
        )
        
        # 初始化高层策略
        self.high_level_policy = HighLevelPolicy(
            task_feature_dim=64,
            uav_feature_dim=32,
            region_feature_dim=32,
            time_feature_dim=16,
            hidden_dim=256,
            num_uavs=num_drones,
            num_regions=num_regions,
            num_priority_levels=3
        )
        
        # 初始化低层策略
        self.low_level_policy = LowLevelPolicy(
            state_dim=12,
            action_dim=4,
            hidden_dim=128,
            task_embed_dim=32
        )
        
        # 初始化分层强化学习
        self.hrl = HierarchicalRL(
            high_level_policy=self.high_level_policy,
            low_level_policy=self.low_level_policy,
            high_level_lr=high_level_lr,
            low_level_lr=low_level_lr,
            gamma=gamma,
            device=device
        )
        
        # 初始化无人机状态
        self.drones = []
        for i in range(num_drones):
            drone = DroneState(
                drone_id=i,
                position=(random.uniform(0, 1), random.uniform(0, 1)),
                storage_capacity=100.0,
                remaining_storage=100.0,
                battery_level=100.0,
                max_flight_distance=2.0,
                speed=0.1
            )
            self.drones.append(drone)
        
        # 初始化动态任务分配器
        self.task_allocator = DynamicTaskAllocator(
            drones=self.drones,
            current_time=0.0
        )
        
        # 保存任务和状态信息
        self.tasks = {}
        self.task_queue = []
        self.current_time = 0.0
        self.episode = 0
        
        # 设置区域信息
        self.regions = {
            "residential": {"center": (0.25, 0.25), "radius": 0.2, "area": 0.13},
            "commercial": {"center": (0.75, 0.25), "radius": 0.2, "area": 0.13},
            "industrial": {"center": (0.25, 0.75), "radius": 0.2, "area": 0.13},
            "park": {"center": (0.75, 0.75), "radius": 0.2, "area": 0.13}
        }
        
        # 传感器信息
        self.sensors = {
            "temperature": {"count": 5, "data_size": 2.0},
            "air_quality": {"count": 5, "data_size": 5.0},
            "noise": {"count": 4, "data_size": 3.0},
            "traffic": {"count": 5, "data_size": 7.0}
        }
        
        logger.info("任务集成系统初始化完成")
    
    def process_task_instruction(self, instruction: str) -> List[Dict[str, Any]]:
        """处理任务指令，提取关键信息并分解任务"""
        # 使用LLM解析任务描述
        logger.info(f"解析任务指令: {instruction}")
        task_info = self.llm.parse_task(instruction)
        
        if "error" in task_info:
            logger.error(f"任务解析失败: {task_info['error']}")
            return []
        
        # 按时间拆分任务
        logger.info("按时间拆分任务")
        time_subtasks = self.llm.decompose_task_by_time(task_info)
        
        # 添加任务ID
        for i, subtask in enumerate(time_subtasks):
            subtask["id"] = f"task_{self.episode}_{i}_{str(uuid.uuid4())[:8]}"
        
        # 按区域聚合任务
        logger.info("按区域聚合任务")
        aggregated_tasks = self.llm.aggregate_tasks_by_region(time_subtasks)
        
        # 创建TaskInfo对象并添加到系统
        processed_tasks = []
        for agg_task in aggregated_tasks:
            # 从聚合任务中提取信息
            task_id = agg_task.get("id", f"agg_{self.episode}_{str(uuid.uuid4())[:8]}")
            task_type = agg_task.get("任务类型", "data_collection")
            
            # 获取区域中心作为任务位置
            region_name = agg_task.get("目标区域", "residential")
            region_info = self.regions.get(region_name, {"center": (0.5, 0.5)})
            position = region_info["center"]
            
            # 获取时间范围
            time_ranges = agg_task.get("时间范围列表", [])
            if not time_ranges:
                # 如果没有时间范围，使用默认值
                start_time = self.current_time
                end_time = self.current_time + 60.0
            else:
                # 使用第一个时间范围
                time_range = time_ranges[0]
                start_time_str = time_range.get("开始时间", "00:00")
                end_time_str = time_range.get("结束时间", "23:59")
                
                # 转换时间格式为浮点数（单位：分钟）
                try:
                    h, m = map(int, start_time_str.split(":"))
                    start_time = h * 60 + m
                    
                    h, m = map(int, end_time_str.split(":"))
                    end_time = h * 60 + m
                except:
                    start_time = self.current_time
                    end_time = self.current_time + 60.0
            
            # 获取传感器类型和数据大小
            sensor_type = agg_task.get("传感器类型", "temperature")
            sensor_info = self.sensors.get(sensor_type, {"data_size": 2.0})
            data_size = sensor_info["data_size"]
            
            # 获取优先级
            priority = agg_task.get("优先级", 1)
            
            # 获取采集间隔
            collection_interval = agg_task.get("采集频率", 30.0)
            if isinstance(collection_interval, str):
                try:
                    collection_interval = float(collection_interval.replace("分钟", ""))
                except:
                    collection_interval = 30.0
            
            # 创建TaskInfo对象
            task = TaskInfo(
                task_id=task_id,
                task_type=task_type,
                position=position,
                data_size=data_size,
                start_time=start_time,
                end_time=end_time,
                priority=priority,
                region=region_name,
                sensor_type=sensor_type,
                collection_interval=collection_interval,
                required_data_points=max(1, int((end_time - start_time) / collection_interval))
            )
            
            # 添加到任务分配器和本地存储
            self.task_allocator.add_task(task)
            self.tasks[task_id] = agg_task
            processed_tasks.append(agg_task)
        
        # 将处理后的任务添加到队列
        self.task_queue.extend(processed_tasks)
        
        logger.info(f"任务处理完成，共生成 {len(processed_tasks)} 个任务")
        return processed_tasks
    
    def optimize_task_execution(self, time_increment: float = 10.0, max_steps: int = 100):
        """使用分层RL优化任务执行"""
        # 获取当前未分配的任务
        pending_tasks = [task for task_id, task in self.task_allocator.tasks.items() 
                         if task.status == "pending"]
        
        if not pending_tasks:
            logger.info("没有待处理的任务")
            return
        
        # 开始RL训练循环
        total_reward = 0.0
        step = 0
        
        while step < max_steps:
            # 更新当前时间
            self.current_time += time_increment
            self.task_allocator.update_time(self.current_time)
            
            # 使用高层策略为每个未分配任务做决策
            for task in pending_tasks:
                # 准备任务特征
                task_features = self._prepare_task_features(task)
                
                # 准备无人机特征（使用所有无人机的平均特征简化）
                uav_features = self._prepare_average_uav_features()
                
                # 准备区域特征
                region_name = task.region
                region_info = self.regions.get(region_name, {"center": (0.5, 0.5), "area": 0.1})
                region_features = [
                    region_info["center"][0],
                    region_info["center"][1],
                    region_info.get("area", 0.1)
                ]
                
                # 准备时间特征
                time_features = [
                    self.current_time,
                    task.end_time - task.start_time
                ]
                
                # 使用高层策略获取动作
                action = self.hrl.high_level_step(
                    task_features=task_features,
                    uav_features=uav_features,
                    region_features=region_features,
                    time_features=time_features
                )
                
                # 应用高层动作
                assigned_uav_id = action["assigned_uav"]
                priority_level = action["priority_level"]
                
                # 更新任务优先级
                task.priority = priority_level + 1  # 转换为1-3的优先级
                
                # 尝试分配任务
                self.task_allocator.allocate_tasks(max_allocations=1)
                
                # 收集高层奖励
                high_reward = self._calculate_high_level_reward()
                
                # 存储高层转换记忆
                next_task_features = self._prepare_task_features(task)
                next_uav_features = self._prepare_average_uav_features()
                
                self.hrl.store_high_level_transition(
                    task_features=task_features,
                    uav_features=uav_features,
                    region_features=region_features,
                    time_features=time_features,
                    action=action,
                    reward=high_reward,
                    next_task_features=next_task_features,
                    next_uav_features=next_uav_features,
                    next_region_features=region_features,  # 简化，区域特征不变
                    next_time_features=time_features,  # 简化，时间特征不变
                    done=(task.status != "pending")
                )
            
            # 使用低层策略为每个已分配任务做决策
            for task_id, task in self.task_allocator.active_tasks.items():
                drone_id = task.assigned_drone_id
                if drone_id is None:
                    continue
                
                drone = self.task_allocator.drones[drone_id]
                
                # 准备状态特征
                state = self._prepare_drone_state(drone, task)
                
                # 准备任务特征
                task_features = self._prepare_task_features(task)
                
                # 使用低层策略获取动作
                action = self.hrl.low_level_step(
                    state=state,
                    task_features=task_features
                )
                
                # 应用低层动作
                action_idx = action["action_idx"]
                
                # 根据动作执行操作
                next_state = state
                reward = 0.0
                done = False
                
                if action_idx == 0:  # 执行任务
                    # 模拟任务执行
                    success = random.random() > 0.1  # 90%成功率
                    if success:
                        # 完成任务
                        self.task_allocator.complete_task(task_id)
                        reward = 10.0
                        done = True
                    else:
                        # 任务失败
                        reward = -5.0
                
                elif action_idx == 1:  # 调整顺序
                    # 重新规划路径
                    self.task_allocator.replan_route(drone_id)
                    reward = 0.0
                
                elif action_idx == 2:  # 暂停卸载
                    # 模拟数据卸载
                    drone.remaining_storage = drone.storage_capacity
                    reward = 5.0
                
                elif action_idx == 3:  # 不操作
                    # 什么都不做
                    reward = -0.1
                
                # 更新状态
                next_state = self._prepare_drone_state(drone, task)
                
                # 存储低层转换记忆
                self.hrl.store_low_level_transition(
                    state=state,
                    task_features=task_features,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_task_features=task_features,  # 简化，任务特征不变
                    done=done
                )
                
                total_reward += reward
            
            # 定期更新策略
            if step % 10 == 0:
                self.hrl.update_high_level_policy()
                self.hrl.update_low_level_policy()
            
            # 检查是否所有任务都已完成
            if not self.task_allocator.active_tasks and not pending_tasks:
                logger.info("所有任务已完成")
                break
            
            step += 1
        
        # 最终更新策略
        self.hrl.update_high_level_policy()
        self.hrl.update_low_level_policy()
        
        logger.info(f"优化完成，总奖励: {total_reward}，步数: {step}")
        return total_reward
    
    def _prepare_task_features(self, task: TaskInfo) -> List[float]:
        """准备任务特征向量"""
        # 任务类型编码
        task_type_code = 0.0
        if task.task_type == "data_collection":
            task_type_code = 1.0
        elif task.task_type == "data_offload":
            task_type_code = 2.0
        
        # 任务特征：[类型, 位置x, 位置y, 数据量, 时间窗口]
        features = [
            task_type_code,
            task.position[0],
            task.position[1],
            task.data_size / 10.0,  # 归一化数据大小
            (task.end_time - task.start_time) / 60.0  # 归一化时间窗口（以小时为单位）
        ]
        
        return features
    
    def _prepare_drone_state(self, drone: DroneState, task: TaskInfo) -> List[float]:
        """准备无人机状态向量"""
        # 计算距离任务位置的距离
        distance = math.sqrt(
            (task.position[0] - drone.position[0])**2 + 
            (task.position[1] - drone.position[1])**2
        )
        
        # 计算剩余时间
        remaining_time = max(0, task.end_time - self.current_time)
        
        # 状态向量: [位置x, 位置y, 距离, 剩余存储, 电量, 剩余时间, 数据大小, 优先级, 任务类型, 是否已分配, 是否在进行, 是否完成]
        state = [
            drone.position[0],
            drone.position[1],
            distance,
            drone.remaining_storage / drone.storage_capacity,
            drone.battery_level / 100.0,
            remaining_time / 60.0,
            task.data_size / 10.0,
            task.priority / 3.0,
            1.0 if task.task_type == "data_collection" else 0.0,
            1.0 if task.assigned_drone_id is not None else 0.0,
            1.0 if task.status == "in_progress" else 0.0,
            1.0 if task.status == "completed" else 0.0
        ]
        
        return state
    
    def _prepare_average_uav_features(self) -> List[float]:
        """准备所有无人机的平均特征"""
        # 计算平均位置
        avg_x = sum(drone.position[0] for drone in self.drones) / len(self.drones)
        avg_y = sum(drone.position[1] for drone in self.drones) / len(self.drones)
        
        # 计算平均存储容量和剩余存储
        avg_storage = sum(drone.storage_capacity for drone in self.drones) / len(self.drones)
        avg_remaining = sum(drone.remaining_storage for drone in self.drones) / len(self.drones)
        
        # 平均特征: [位置x, 位置y, 存储容量, 最大飞行距离]
        features = [
            avg_x,
            avg_y,
            avg_storage,
            sum(drone.max_flight_distance for drone in self.drones) / len(self.drones)
        ]
        
        return features
    
    def _calculate_high_level_reward(self) -> float:
        """计算高层奖励"""
        # 已分配任务的价值
        assigned_value = sum(
            self.task_allocator.get_task_value(task) 
            for task in self.task_allocator.active_tasks.values()
        )
        
        # 总任务价值
        total_value = self.task_allocator.total_task_value()
        
        # 无人机使用数量
        used_uavs = sum(1 for drone in self.drones if drone.current_task_id is not None)
        
        # 计算高层延迟（简化为当前时间与任务开始时间的差）
        high_delay = 0.0
        for task in self.task_allocator.active_tasks.values():
            high_delay += max(0, self.current_time - task.start_time)
        
        # 估计任务总时间
        total_time = sum(
            task.end_time - task.start_time 
            for task in self.task_allocator.tasks.values()
        )
        
        # 计算高层奖励
        reward = self.hrl.high_level_reward(
            assigned_value=assigned_value,
            total_value=total_value,
            num_uavs=used_uavs,
            max_uavs=len(self.drones),
            high_delay=high_delay,
            total_time=max(1.0, total_time)
        )
        
        return reward
    
    def _calculate_low_level_reward(self, drone: DroneState) -> float:
        """计算低层奖励"""
        # 获取无人机完成的任务价值
        completed_value = 0.0
        for task_id in drone.task_history:
            if task_id in self.task_allocator.completed_tasks:
                task = self.task_allocator.completed_tasks[task_id]
                completed_value += self.task_allocator.get_task_value(task)
        
        # 分配给无人机的任务总价值
        assigned_value = completed_value
        if drone.current_task_id and drone.current_task_id in self.task_allocator.tasks:
            task = self.task_allocator.tasks[drone.current_task_id]
            assigned_value += self.task_allocator.get_task_value(task)
        
        # 剩余存储空间
        storage_left = drone.remaining_storage
        max_storage = drone.storage_capacity
        
        # 任务延迟
        low_delay = 0.0
        total_time = 0.0
        
        for task_id in drone.task_history:
            if task_id in self.task_allocator.completed_tasks:
                task = self.task_allocator.completed_tasks[task_id]
                if task.completion_time and task.start_time:
                    low_delay += max(0, task.completion_time - task.start_time)
                    total_time += (task.end_time - task.start_time)
        
        # 计算低层奖励
        reward = self.hrl.low_level_reward(
            completed_value=completed_value,
            assigned_value=max(1.0, assigned_value),
            storage_left=storage_left,
            max_storage=max_storage,
            low_delay=low_delay,
            total_time=max(1.0, total_time)
        )
        
        return reward
    
    def save_models(self):
        """保存模型"""
        high_level_path = os.path.join(self.save_dir, f"high_level_policy_{self.episode}.pt")
        low_level_path = os.path.join(self.save_dir, f"low_level_policy_{self.episode}.pt")
        self.hrl.save_models(high_level_path, low_level_path)
        
        # 保存任务状态
        state_path = os.path.join(self.save_dir, f"task_state_{self.episode}.json")
        state = {
            "episode": self.episode,
            "current_time": self.current_time,
            "tasks": self.tasks,
            "task_allocator": self.task_allocator.to_dict()
        }
        
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"模型和状态已保存到 {self.save_dir}")
    
    def load_models(self, episode: int):
        """加载模型"""
        high_level_path = os.path.join(self.save_dir, f"high_level_policy_{episode}.pt")
        low_level_path = os.path.join(self.save_dir, f"low_level_policy_{episode}.pt")
        
        if os.path.exists(high_level_path) and os.path.exists(low_level_path):
            self.hrl.load_models(high_level_path, low_level_path)
            
            # 加载任务状态
            state_path = os.path.join(self.save_dir, f"task_state_{episode}.json")
            if os.path.exists(state_path):
                with open(state_path, "r") as f:
                    state = json.load(f)
                
                self.episode = state["episode"]
                self.current_time = state["current_time"]
                self.tasks = state["tasks"]
                
                # 重新构建任务分配器
                allocator_dict = state["task_allocator"]
                self.task_allocator = DynamicTaskAllocator.from_dict(allocator_dict)
            
            logger.info(f"模型和状态已从 {self.save_dir} 加载")
            return True
        else:
            logger.warning(f"未找到模型文件 {high_level_path} 或 {low_level_path}")
            return False
    
    def train_episode(self, task_instructions: List[str], max_steps: int = 100):
        """训练一个完整的回合"""
        self.episode += 1
        self.current_time = 0.0
        
        # 重置任务和无人机状态
        self.tasks = {}
        self.task_queue = []
        
        # 重置无人机位置
        for drone in self.drones:
            drone.position = (random.uniform(0, 1), random.uniform(0, 1))
            drone.remaining_storage = drone.storage_capacity
            drone.battery_level = 100.0
            drone.current_task_id = None
            drone.task_history = []
        
        # 重置任务分配器
        self.task_allocator = DynamicTaskAllocator(
            drones=self.drones,
            current_time=self.current_time
        )
        
        # 处理所有任务指令
        for instruction in task_instructions:
            self.process_task_instruction(instruction)
        
        # 优化任务执行
        total_reward = self.optimize_task_execution(time_increment=10.0, max_steps=max_steps)
        
        # 保存模型
        self.save_models()
        
        logger.info(f"回合 {self.episode} 训练完成，总奖励: {total_reward}")
        return total_reward
    
    def evaluate(self, task_instructions: List[str], max_steps: int = 100):
        """评估当前模型的性能"""
        # 备份当前状态
        current_episode = self.episode
        current_time = self.current_time
        current_tasks = self.tasks.copy()
        current_allocator = self.task_allocator
        
        # 重置状态用于评估
        self.current_time = 0.0
        self.tasks = {}
        self.task_queue = []
        
        # 重置无人机位置
        for drone in self.drones:
            drone.position = (random.uniform(0, 1), random.uniform(0, 1))
            drone.remaining_storage = drone.storage_capacity
            drone.battery_level = 100.0
            drone.current_task_id = None
            drone.task_history = []
        
        # 重置任务分配器
        self.task_allocator = DynamicTaskAllocator(
            drones=self.drones,
            current_time=self.current_time
        )
        
        # 处理所有任务指令
        for instruction in task_instructions:
            self.process_task_instruction(instruction)
        
        # 评估性能（不更新策略）
        with torch.no_grad():
            total_reward = self.optimize_task_execution(time_increment=10.0, max_steps=max_steps)
        
        # 恢复原始状态
        self.episode = current_episode
        self.current_time = current_time
        self.tasks = current_tasks
        self.task_allocator = current_allocator
        
        logger.info(f"评估完成，总奖励: {total_reward}")
        return total_reward 