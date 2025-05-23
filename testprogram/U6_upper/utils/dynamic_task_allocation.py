import torch
import numpy as np
import random
import logging
import time
from typing import Dict, List, Tuple, Union, Any, Optional
from collections import deque

from .dynamic_task_allocation_base import U6DroneState, U6TaskInfo

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class U6DynamicTaskAllocator:
    """U6版动态任务分配器，专为6无人机系统设计，支持协作任务处理"""
    
    def __init__(
        self,
        num_drones: int = 6,  # 固定为6个无人机
        num_regions: int = 6,
        initial_positions: List[Tuple[float, float]] = None,
        max_tasks_per_drone: int = 10,
        reallocation_frequency: int = 5,
        battery_threshold: float = 20.0,
        collaboration_distance_threshold: float = 5.0  # 协作距离阈值
    ):
        """
        初始化动态任务分配器
        
        Args:
            num_drones: 无人机数量
            num_regions: 区域数量
            initial_positions: 无人机初始位置列表
            max_tasks_per_drone: 每个无人机最大任务数
            reallocation_frequency: 重新分配任务的频率（步数）
            battery_threshold: 低电量阈值
            collaboration_distance_threshold: 协作距离阈值
        """
        self.num_drones = min(num_drones, 6)  # 确保不超过6个
        self.num_regions = num_regions
        self.max_tasks_per_drone = max_tasks_per_drone
        self.reallocation_frequency = reallocation_frequency
        self.battery_threshold = battery_threshold
        self.collaboration_distance_threshold = collaboration_distance_threshold
        
        # 设置默认初始位置
        if initial_positions is None:
            initial_positions = [
                (0.0, 0.0),
                (10.0, 0.0),
                (0.0, 10.0),
                (10.0, 10.0),
                (5.0, 0.0),
                (5.0, 10.0)
            ]
        
        # 确保初始位置数量与无人机数量一致
        initial_positions = initial_positions[:self.num_drones]
        while len(initial_positions) < self.num_drones:
            initial_positions.append((0.0, 0.0))
        
        # 初始化无人机
        self.drones = [
            U6DroneState(i, pos) for i, pos in enumerate(initial_positions)
        ]
        
        # 任务列表
        self.all_tasks = []
        self.unassigned_tasks = []
        self.completed_tasks = []
        
        # 区域划分（简单的网格划分）
        self.region_centers = self._initialize_regions(num_regions)
        
        # 协作组
        self.collaboration_groups = {}  # 键为主导无人机ID，值为协作组信息
        
        # 运行时计数器
        self.step_counter = 0
    
    def _initialize_regions(self, num_regions: int) -> List[Tuple[float, float]]:
        """初始化区域中心点"""
        if num_regions <= 1:
            return [(5.0, 5.0)]
        elif num_regions == 2:
            return [(2.5, 5.0), (7.5, 5.0)]
        elif num_regions == 4:
            return [(2.5, 2.5), (7.5, 2.5), (2.5, 7.5), (7.5, 7.5)]
        elif num_regions == 6:
            return [(2.5, 2.5), (7.5, 2.5), (2.5, 7.5), (7.5, 7.5), (5.0, 2.5), (5.0, 7.5)]
        else:
            # 生成网格中心点
            centers = []
            grid_size = int(np.ceil(np.sqrt(num_regions)))
            step = 10.0 / grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(centers) < num_regions:
                        centers.append((step * (i + 0.5), step * (j + 0.5)))
            return centers
    
    def add_task(self, task: U6TaskInfo):
        """添加新任务"""
        self.all_tasks.append(task)
        self.unassigned_tasks.append(task)
        logger.info(f"添加任务: {task.task_id}, 类型: {task.task_type}, 区域: {task.region_id}, 需要协作: {task.requires_collaboration}")
    
    def get_drone(self, drone_id: int) -> Optional[U6DroneState]:
        """获取指定ID的无人机"""
        if 0 <= drone_id < len(self.drones):
            return self.drones[drone_id]
        return None
    
    def get_task_by_id(self, task_id: str) -> Optional[U6TaskInfo]:
        """通过ID获取任务"""
        for task in self.all_tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def allocate_tasks(self):
        """分配未分配的任务给无人机"""
        # 如果没有未分配的任务，直接返回
        if not self.unassigned_tasks:
            return
        
        # 首先处理需要协作的任务
        collaborative_tasks = [t for t in self.unassigned_tasks if t.requires_collaboration]
        non_collaborative_tasks = [t for t in self.unassigned_tasks if not t.requires_collaboration]
        
        # 为协作任务分配无人机
        if collaborative_tasks:
            self._allocate_collaborative_tasks(collaborative_tasks)
        
        # 为非协作任务分配无人机
        if non_collaborative_tasks:
            self._allocate_non_collaborative_tasks(non_collaborative_tasks)
    
    def _allocate_collaborative_tasks(self, tasks: List[U6TaskInfo]):
        """分配需要协作的任务"""
        for task in tasks[:]:  # 使用副本避免在迭代过程中修改列表
            # 寻找可用的无人机组合
            available_drones = [
                drone for drone in self.drones 
                if not drone.is_busy and 
                not drone.is_collaborating and 
                drone.battery > self.battery_threshold and
                len(drone.task_queue) < self.max_tasks_per_drone
            ]
            
            # 如果可用无人机数量不足，跳过此任务
            if len(available_drones) <= task.min_collaborators:
                continue
            
            # 按与任务位置的距离对无人机排序
            drone_distances = [
                (
                    drone,
                    ((drone.position[0] - task.position[0]) ** 2 + 
                     (drone.position[1] - task.position[1]) ** 2) ** 0.5
                )
                for drone in available_drones
            ]
            drone_distances.sort(key=lambda x: x[1])
            
            # 选择主导无人机和协作者
            primary_drone = drone_distances[0][0]
            collaborators = [d[0] for d in drone_distances[1:task.min_collaborators+1]]
            
            # 分配任务给主导无人机
            task.assign_to_drone(primary_drone.drone_id)
            primary_drone.add_task(task)
            
            # 记录协作者
            for collab_drone in collaborators:
                task.add_collaborator(collab_drone.drone_id)
            
            # 创建协作组
            collab_group_ids = [d.drone_id for d in collaborators]
            primary_drone.start_collaboration(collab_group_ids, "primary")
            for collab_drone in collaborators:
                collab_drone.start_collaboration([primary_drone.drone_id] + 
                    [d.drone_id for d in collaborators if d.drone_id != collab_drone.drone_id],
                    "assistant"
                )
            
            # 记录协作组
            self.collaboration_groups[primary_drone.drone_id] = {
                "primary": primary_drone.drone_id,
                "collaborators": collab_group_ids,
                "task_id": task.task_id,
                "formation_time": time.time()
            }
            
            # 从未分配任务中移除
            self.unassigned_tasks.remove(task)
            
            logger.info(f"分配协作任务 {task.task_id} 给无人机 {primary_drone.drone_id}，协作者: {collab_group_ids}")
    
    def _allocate_non_collaborative_tasks(self, tasks: List[U6TaskInfo]):
        """分配不需要协作的任务"""
        # 为每个无人机分配任务
        for drone in self.drones:
            # 如果无人机忙或协作中或电量过低，跳过
            if drone.is_busy or drone.is_collaborating or drone.battery < self.battery_threshold:
                continue
            
            # 计算无人机可以接受的最大任务数
            max_new_tasks = self.max_tasks_per_drone - len(drone.task_queue)
            if max_new_tasks <= 0:
                continue
            
            # 对任务按优先级和与无人机的距离进行排序
            suitable_tasks = []
            for task in tasks:
                # 检查无人机是否有所需的传感器
                if all(sensor in drone.sensor_types for sensor in task.requires_sensors):
                    # 计算任务与无人机的距离
                    distance = ((task.position[0] - drone.position[0]) ** 2 + 
                              (task.position[1] - drone.position[1]) ** 2) ** 0.5
                    
                    # 计算任务分数（优先级高且距离近的任务得分高）
                    task_score = task.priority - 0.1 * distance
                    
                    suitable_tasks.append((task, task_score))
            
            # 按任务分数排序
            suitable_tasks.sort(key=lambda x: x[1], reverse=True)
            
            # 分配任务
            assigned_count = 0
            for task, _ in suitable_tasks:
                if assigned_count >= max_new_tasks:
                    break
                
                # 分配任务给无人机
                task.assign_to_drone(drone.drone_id)
                drone.add_task(task)
                
                if task in self.unassigned_tasks:
                    self.unassigned_tasks.remove(task)
                
                logger.info(f"分配任务 {task.task_id} 给无人机 {drone.drone_id}")
                assigned_count += 1
    
    def rebalance_tasks(self):
        """重新平衡无人机之间的任务分配"""
        # 仅在特定步数重新平衡任务
        if self.step_counter % self.reallocation_frequency != 0:
            return
        
        # 收集所有未开始的非协作任务
        tasks_to_redistribute = []
        for drone in self.drones:
            # 跳过正在协作的无人机
            if drone.is_collaborating:
                continue
                
            # 只考虑未开始的任务（当前正在执行的任务不重新分配）
            if not drone.is_busy:
                # 收集非协作任务
                non_collab_tasks = [t for t in drone.task_queue if not t.requires_collaboration]
                tasks_to_redistribute.extend(non_collab_tasks)
                
                # 从队列中移除这些任务
                drone.task_queue = deque([t for t in drone.task_queue if t.requires_collaboration])
        
        # 将这些任务标记为未分配
        self.unassigned_tasks.extend(tasks_to_redistribute)
        
        # 重新分配任务
        self.allocate_tasks()
    
    def manage_collaborations(self):
        """管理协作组的状态和生命周期"""
        # 检查每个协作组
        for primary_id, group_info in list(self.collaboration_groups.items()):
            primary_drone = self.get_drone(primary_id)
            task = self.get_task_by_id(group_info["task_id"])
            
            # 如果任务已完成或主导无人机不存在，解散协作组
            if task is None or task.is_completed or primary_drone is None:
                self._dissolve_collaboration_group(primary_id)
                continue
            
            # 检查协作者是否都在有效距离内
            if primary_drone.current_task == task and primary_drone.is_busy:
                all_in_range = True
                for collab_id in group_info["collaborators"]:
                    collab_drone = self.get_drone(collab_id)
                    if collab_drone is None:
                        all_in_range = False
                        break
                    
                    # 计算与主导无人机的距离
                    distance = ((primary_drone.position[0] - collab_drone.position[0]) ** 2 + 
                               (primary_drone.position[1] - collab_drone.position[1]) ** 2) ** 0.5
                    
                    if distance > self.collaboration_distance_threshold:
                        all_in_range = False
                        break
                
                # 更新协作成功状态
                if task.collaboration_success is None:
                    task.collaboration_success = all_in_range
    
    def _dissolve_collaboration_group(self, primary_id: int):
        """解散协作组"""
        if primary_id not in self.collaboration_groups:
            return
        
        group_info = self.collaboration_groups[primary_id]
        
        # 更新主导无人机状态
        primary_drone = self.get_drone(primary_id)
        if primary_drone:
            primary_drone.end_collaboration()
        
        # 更新所有协作者状态
        for collab_id in group_info["collaborators"]:
            collab_drone = self.get_drone(collab_id)
            if collab_drone:
                collab_drone.end_collaboration()
        
        # 移除协作组记录
        del self.collaboration_groups[primary_id]
        
        logger.info(f"解散协作组 (主导: {primary_id}, 协作者: {group_info['collaborators']})")
    
    def update_drone_states(self, timestep: float = 1.0):
        """更新所有无人机状态"""
        current_time = time.time()
        
        for drone in self.drones:
            # 如果无人机正在执行任务
            if drone.is_busy and drone.current_task:
                # 简化模型：假设任务执行时间固定
                task_execution_time = timestep
                
                # 任务完成
                if task_execution_time >= 1.0:  # 假设1.0表示完成
                    # 更新无人机状态
                    drone.update_storage(drone.current_task.data_size)
                    drone.battery -= 1.0  # 简化的电量消耗模型
                    
                    # 标记任务完成
                    task = drone.current_task
                    collaboration_success = None
                    
                    # 如果是协作任务，检查协作状态
                    if task.requires_collaboration:
                        collaboration_success = task.collaboration_success
                        
                    drone.complete_task()
                    if task not in self.completed_tasks:
                        self.completed_tasks.append(task)
                    
                    logger.info(f"无人机 {drone.drone_id} 完成任务 {task.task_id}" + 
                               (f"，协作{'成功' if collaboration_success else '失败'}" if collaboration_success is not None else ""))
            
            # 如果无人机空闲且有任务在队列中
            elif not drone.is_busy and drone.task_queue:
                next_task = drone.get_next_task()
                if next_task:
                    # 移动到任务位置（简化模型）
                    drone.update_position(next_task.position, current_time)
                    
                    # 开始执行任务
                    drone.start_task(next_task)
                    logger.info(f"无人机 {drone.drone_id} 开始执行任务 {next_task.task_id}")
            
            # 如果无人机正在协作但没有当前任务
            elif drone.is_collaborating and not drone.is_busy:
                # 如果是协作者，移动到主导无人机附近
                if drone.collaboration_role == "assistant":
                    # 找到对应的主导无人机
                    for primary_id, group_info in self.collaboration_groups.items():
                        if drone.drone_id in group_info["collaborators"]:
                            primary_drone = self.get_drone(primary_id)
                            if primary_drone:
                                # 移动到主导无人机附近
                                # 添加一点随机偏移以避免所有协作者位于同一位置
                                offset_x = random.uniform(-2.0, 2.0)
                                offset_y = random.uniform(-2.0, 2.0)
                                new_pos = (
                                    primary_drone.position[0] + offset_x,
                                    primary_drone.position[1] + offset_y
                                )
                                drone.update_position(new_pos, current_time)
                            break
            
            # 如果无人机空闲且没有任务
            else:
                # 更新空闲时间
                drone.idle_time += timestep
        
        # 管理协作组
        self.manage_collaborations()
        
        self.step_counter += 1
    
    def get_system_state(self) -> Dict[str, Any]:
        """获取整个系统的状态"""
        return {
            "drones": [drone.to_dict() for drone in self.drones],
            "total_tasks": len(self.all_tasks),
            "unassigned_tasks": len(self.unassigned_tasks),
            "completed_tasks": len(self.completed_tasks),
            "collaboration_groups": len(self.collaboration_groups),
            "step_counter": self.step_counter
        }
    
    def get_feature_vector(self) -> np.ndarray:
        """获取系统特征向量用于RL模型"""
        # 无人机特征
        drone_features = np.concatenate([drone.get_feature_vector() for drone in self.drones])
        
        # 任务分配状态
        tasks_per_drone = np.array([
            len(list(drone.task_queue)) + (1 if drone.is_busy else 0)
            for drone in self.drones
        ]) / self.max_tasks_per_drone  # 归一化
        
        # 协作状态
        collaboration_status = np.array([
            float(drone.is_collaborating)
            for drone in self.drones
        ])
        
        # 整体系统状态
        system_features = np.array([
            len(self.unassigned_tasks) / max(1, len(self.all_tasks)),
            len(self.completed_tasks) / max(1, len(self.all_tasks)),
            len(self.collaboration_groups) / max(1, self.num_drones),
            self.step_counter / 1000.0  # 归一化步数
        ])
        
        return np.concatenate([drone_features, tasks_per_drone, collaboration_status, system_features])
    
    def step(self, actions: List[int] = None):
        """
        执行一个时间步，可以包含决策动作
        
        Args:
            actions: 可选的动作列表，每个无人机一个动作
        """
        # 处理动作（如果提供）
        if actions:
            for i, action in enumerate(actions):
                if i < len(self.drones):
                    drone = self.drones[i]
                    
                    # 动作0：继续当前任务
                    # 动作1：切换到下一个任务
                    # 动作2：重新分配任务
                    # 动作3：提议协作
                    # 动作4：移动到指定区域
                    # 动作5：待命
                    
                    if action == 1 and drone.is_busy:
                        # 强制切换到下一个任务
                        if drone.current_task:
                            # 将当前任务放回队列末尾
                            drone.task_queue.append(drone.current_task)
                            drone.current_task = None
                            drone.is_busy = False
                    
                    elif action == 2:
                        # 触发重新分配
                        self.rebalance_tasks()
                    
                    elif action == 3 and not drone.is_collaborating:
                        # 提议协作 - 尝试找到附近的无人机组成协作组
                        nearby_drones = []
                        for other_drone in self.drones:
                            if other_drone.drone_id != drone.drone_id and not other_drone.is_collaborating:
                                # 计算距离
                                distance = ((drone.position[0] - other_drone.position[0]) ** 2 + 
                                          (drone.position[1] - other_drone.position[1]) ** 2) ** 0.5
                                
                                if distance <= self.collaboration_distance_threshold:
                                    nearby_drones.append((other_drone, distance))
                        
                        # 如果有附近的无人机
                        if nearby_drones:
                            # 按距离排序
                            nearby_drones.sort(key=lambda x: x[1])
                            # 选择最近的1-2个无人机
                            num_collaborators = min(2, len(nearby_drones))
                            collaborators = [d[0] for d in nearby_drones[:num_collaborators]]
                            
                            # 创建协作组（仅状态更新，不关联具体任务）
                            collab_ids = [d.drone_id for d in collaborators]
                            drone.start_collaboration(collab_ids, "primary")
                            for collab_drone in collaborators:
                                collab_drone.start_collaboration([drone.drone_id] + 
                                    [d.drone_id for d in collaborators if d.drone_id != collab_drone.drone_id],
                                    "assistant"
                                )
                            
                            # 记录协作组
                            self.collaboration_groups[drone.drone_id] = {
                                "primary": drone.drone_id,
                                "collaborators": collab_ids,
                                "task_id": None,  # 暂时没有关联任务
                                "formation_time": time.time()
                            }
                            
                            logger.info(f"无人机 {drone.drone_id} 发起协作，协作者: {collab_ids}")
                    
                    elif action == 4 and not drone.is_busy:
                        # 移动到区域中心
                        region_id = i % len(self.region_centers)
                        drone.update_position(self.region_centers[region_id])
                        
                    elif action == 5:
                        # 待命 - 不做任何操作
                        pass
        
        # 分配未分配的任务
        self.allocate_tasks()
        
        # 更新无人机状态
        self.update_drone_states()
        
        # 每隔一定步数重新平衡任务
        if self.step_counter % self.reallocation_frequency == 0:
            self.rebalance_tasks()
        
        # 返回当前系统状态
        return self.get_system_state()
    
    def calculate_reward(self) -> float:
        """计算当前系统状态的奖励值"""
        # 完成任务的奖励
        completion_ratio = len(self.completed_tasks) / max(1, len(self.all_tasks))
        completion_reward = 10.0 * completion_ratio
        
        # 任务分配平衡性奖励
        task_counts = [
            len(list(drone.task_queue)) + (1 if drone.is_busy else 0)
            for drone in self.drones
        ]
        if sum(task_counts) > 0:
            balance_metric = 1.0 - (max(task_counts) - min(task_counts)) / max(1, max(task_counts))
            balance_reward = 5.0 * balance_metric
        else:
            balance_reward = 0.0
        
        # 协作成功奖励
        collaboration_tasks = [t for t in self.completed_tasks if t.requires_collaboration]
        if collaboration_tasks:
            successful_collabs = sum(1 for t in collaboration_tasks if t.collaboration_success)
            collab_ratio = successful_collabs / len(collaboration_tasks)
            collaboration_reward = 8.0 * collab_ratio
        else:
            collaboration_reward = 0.0
        
        # 无人机电池状态奖励
        battery_levels = [drone.battery / 100.0 for drone in self.drones]
        battery_reward = 3.0 * sum(battery_levels) / len(self.drones)
        
        # 空闲时间惩罚
        idle_ratio = sum(drone.idle_time for drone in self.drones) / max(1.0, self.step_counter * len(self.drones))
        idle_penalty = -5.0 * idle_ratio
        
        # 总奖励
        total_reward = completion_reward + balance_reward + collaboration_reward + battery_reward + idle_penalty
        
        return total_reward 