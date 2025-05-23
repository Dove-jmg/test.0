import torch
import numpy as np
import random
import logging
import time
from typing import Dict, List, Tuple, Union, Any, Optional
from collections import deque

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class U4DroneState:
    """无人机状态类 - U4版本，专为4无人机系统设计"""
    
    def __init__(
        self,
        drone_id: int,
        position: Tuple[float, float],
        max_storage: float = 100.0,
        current_storage: float = 0.0,
        battery: float = 100.0,
        max_speed: float = 5.0,
        sensor_types: List[str] = None
    ):
        """
        初始化无人机状态
        
        Args:
            drone_id: 无人机ID (0-3)
            position: 无人机当前位置 (x, y)
            max_storage: 最大存储容量
            current_storage: 当前已使用存储
            battery: 电池电量百分比
            max_speed: 最大飞行速度
            sensor_types: 搭载的传感器类型列表
        """
        self.drone_id = drone_id
        self.position = position
        self.max_storage = max_storage
        self.current_storage = current_storage
        self.battery = battery
        self.max_speed = max_speed
        self.sensor_types = sensor_types or ["温度", "湿度", "空气质量"]
        
        # 任务相关状态
        self.current_task = None
        self.task_queue = deque()
        self.completed_tasks = []
        self.is_busy = False
        
        # 运行时间统计
        self.total_distance = 0.0
        self.total_flight_time = 0.0
        self.idle_time = 0.0
        self.last_update_time = time.time()
    
    def update_position(self, new_position: Tuple[float, float], timestamp: Optional[float] = None):
        """
        更新无人机位置
        
        Args:
            new_position: 新位置坐标 (x, y)
            timestamp: 时间戳，如果为None则使用当前时间
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 计算移动距离
        distance = ((new_position[0] - self.position[0]) ** 2 + 
                   (new_position[1] - self.position[1]) ** 2) ** 0.5
        
        # 更新总飞行距离
        self.total_distance += distance
        
        # 按距离消耗电量 (简化模型)
        self.battery -= distance * 0.1
        self.battery = max(0.0, self.battery)
        
        # 计算飞行时间
        flight_time = distance / self.max_speed
        self.total_flight_time += flight_time
        
        # 更新位置
        self.position = new_position
        self.last_update_time = timestamp
    
    def add_task(self, task):
        """添加任务到队列"""
        self.task_queue.append(task)
    
    def start_task(self, task):
        """开始执行任务"""
        self.current_task = task
        self.is_busy = True
        task.mark_in_progress()
    
    def complete_task(self):
        """完成当前任务"""
        if self.current_task:
            self.current_task.mark_completed()
            self.completed_tasks.append(self.current_task)
            self.current_task = None
            self.is_busy = False
    
    def update_storage(self, data_size: float):
        """更新存储空间使用"""
        self.current_storage += data_size
        self.current_storage = min(self.current_storage, self.max_storage)
    
    def get_remaining_storage(self) -> float:
        """获取剩余存储空间"""
        return self.max_storage - self.current_storage
    
    def get_next_task(self):
        """获取队列中的下一个任务"""
        if self.task_queue:
            return self.task_queue.popleft()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """将状态转换为字典表示"""
        return {
            "drone_id": self.drone_id,
            "position": self.position,
            "max_storage": self.max_storage,
            "current_storage": self.current_storage,
            "battery": self.battery,
            "max_speed": self.max_speed,
            "sensor_types": self.sensor_types,
            "is_busy": self.is_busy,
            "total_distance": self.total_distance,
            "total_flight_time": self.total_flight_time,
            "idle_time": self.idle_time,
            "completed_tasks_count": len(self.completed_tasks)
        }
    
    def get_feature_vector(self) -> np.ndarray:
        """获取特征向量用于RL模型"""
        return np.array([
            self.position[0], 
            self.position[1],
            self.current_storage / self.max_storage,
            self.battery / 100.0,
            float(self.is_busy)
        ])

class U4TaskInfo:
    """任务信息类 - U4版本"""
    
    def __init__(
        self,
        task_id: str,
        task_type: str,
        position: Tuple[float, float],
        start_time: float,
        end_time: float,
        priority: int = 3,
        data_size: float = 10.0,
        region_id: int = 0,
        requires_sensors: List[str] = None
    ):
        """
        初始化任务信息
        
        Args:
            task_id: 任务ID
            task_type: 任务类型（数据收集、监测等）
            position: 任务位置坐标
            start_time: 开始时间
            end_time: 结束时间
            priority: 优先级（1-5，5最高）
            data_size: 预计产生的数据量
            region_id: 区域ID
            requires_sensors: 所需传感器类型
        """
        self.task_id = task_id
        self.task_type = task_type
        self.position = position
        self.start_time = start_time
        self.end_time = end_time
        self.priority = priority
        self.data_size = data_size
        self.region_id = region_id
        self.requires_sensors = requires_sensors or []
        
        # 任务状态
        self.assigned_drone = None
        self.is_completed = False
        self.is_in_progress = False
        self.completion_time = None
        self.start_execution_time = None
    
    def assign_to_drone(self, drone_id: int):
        """分配任务给指定无人机"""
        self.assigned_drone = drone_id
    
    def mark_in_progress(self):
        """标记任务为进行中"""
        self.is_in_progress = True
        self.start_execution_time = time.time()
    
    def mark_completed(self, completion_time: Optional[float] = None):
        """标记任务为已完成"""
        self.is_completed = True
        self.is_in_progress = False
        self.completion_time = completion_time or time.time()
    
    def get_time_window_length(self) -> float:
        """获取时间窗口长度（秒）"""
        return self.end_time - self.start_time
    
    def is_expired(self, current_time: float) -> bool:
        """检查任务是否已过期"""
        return current_time > self.end_time and not self.is_completed
    
    def get_urgency(self, current_time: float) -> float:
        """计算任务紧急度（0-1之间，1表示最紧急）"""
        if current_time >= self.end_time:
            return 1.0
        
        time_window = self.get_time_window_length()
        if time_window <= 0:
            return 1.0
        
        time_left = self.end_time - current_time
        return 1.0 - (time_left / time_window)
    
    def to_dict(self) -> Dict[str, Any]:
        """将任务信息转换为字典表示"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "position": self.position,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "priority": self.priority,
            "data_size": self.data_size,
            "region_id": self.region_id,
            "requires_sensors": self.requires_sensors,
            "assigned_drone": self.assigned_drone,
            "is_completed": self.is_completed,
            "is_in_progress": self.is_in_progress,
            "completion_time": self.completion_time,
            "start_execution_time": self.start_execution_time
        }
    
    def get_feature_vector(self) -> np.ndarray:
        """获取特征向量用于RL模型"""
        # 任务类型编码为数字
        type_encoding = {
            "数据收集": 0,
            "监测": 1,
            "巡逻": 2,
            "数据卸载": 3
        }.get(self.task_type, 4)  # 默认为4
        
        return np.array([
            type_encoding / 4.0,  # 归一化任务类型
            self.position[0],
            self.position[1],
            self.data_size / 100.0,  # 归一化数据大小
            self.priority / 5.0,  # 归一化优先级
            self.region_id / 3.0  # 归一化区域ID（假设最多4个区域）
        ])


class U4DynamicTaskAllocator:
    """U4版动态任务分配器，专为4无人机系统设计"""
    
    def __init__(
        self,
        num_drones: int = 4,  # 固定为4个无人机
        num_regions: int = 4,
        initial_positions: List[Tuple[float, float]] = None,
        max_tasks_per_drone: int = 10,
        reallocation_frequency: int = 5,
        battery_threshold: float = 20.0
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
        """
        self.num_drones = min(num_drones, 4)  # 确保不超过4个
        self.num_regions = num_regions
        self.max_tasks_per_drone = max_tasks_per_drone
        self.reallocation_frequency = reallocation_frequency
        self.battery_threshold = battery_threshold
        
        # 设置默认初始位置
        if initial_positions is None:
            initial_positions = [
                (0.0, 0.0),
                (10.0, 0.0),
                (0.0, 10.0),
                (10.0, 10.0)
            ]
        
        # 确保初始位置数量与无人机数量一致
        initial_positions = initial_positions[:self.num_drones]
        while len(initial_positions) < self.num_drones:
            initial_positions.append((0.0, 0.0))
        
        # 初始化无人机
        self.drones = [
            U4DroneState(i, pos) for i, pos in enumerate(initial_positions)
        ]
        
        # 任务列表
        self.all_tasks = []
        self.unassigned_tasks = []
        self.completed_tasks = []
        
        # 区域划分（简单的网格划分）
        self.region_centers = self._initialize_regions(num_regions)
        
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
    
    def add_task(self, task: U4TaskInfo):
        """添加新任务"""
        self.all_tasks.append(task)
        self.unassigned_tasks.append(task)
        logger.info(f"添加任务: {task.task_id}, 类型: {task.task_type}, 区域: {task.region_id}")
    
    def get_drone(self, drone_id: int) -> Optional[U4DroneState]:
        """获取指定ID的无人机"""
        if 0 <= drone_id < len(self.drones):
            return self.drones[drone_id]
        return None
    
    def get_task_by_id(self, task_id: str) -> Optional[U4TaskInfo]:
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
        
        # 为每个无人机分配任务
        for drone in self.drones:
            # 如果无人机电量过低，跳过
            if drone.battery < self.battery_threshold:
                continue
            
            # 计算无人机可以接受的最大任务数
            max_new_tasks = self.max_tasks_per_drone - len(drone.task_queue)
            if max_new_tasks <= 0:
                continue
            
            # 对任务按优先级和与无人机的距离进行排序
            suitable_tasks = []
            for task in self.unassigned_tasks:
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
                self.unassigned_tasks.remove(task)
                
                logger.info(f"分配任务 {task.task_id} 给无人机 {drone.drone_id}")
                assigned_count += 1
    
    def rebalance_tasks(self):
        """重新平衡无人机之间的任务分配"""
        # 仅在特定步数重新平衡任务
        if self.step_counter % self.reallocation_frequency != 0:
            return
        
        # 收集所有未开始的任务
        tasks_to_redistribute = []
        for drone in self.drones:
            # 只考虑未开始的任务（当前正在执行的任务不重新分配）
            if not drone.is_busy:
                tasks_to_redistribute.extend(list(drone.task_queue))
                drone.task_queue.clear()
        
        # 将这些任务标记为未分配
        self.unassigned_tasks.extend(tasks_to_redistribute)
        
        # 重新分配任务
        self.allocate_tasks()
    
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
                    drone.complete_task()
                    logger.info(f"无人机 {drone.drone_id} 完成任务")
            
            # 如果无人机空闲且有任务在队列中
            elif not drone.is_busy and drone.task_queue:
                next_task = drone.get_next_task()
                if next_task:
                    # 移动到任务位置（简化模型）
                    drone.update_position(next_task.position, current_time)
                    
                    # 开始执行任务
                    drone.start_task(next_task)
                    logger.info(f"无人机 {drone.drone_id} 开始执行任务 {next_task.task_id}")
            
            # 如果无人机空闲且没有任务
            else:
                # 更新空闲时间
                drone.idle_time += timestep
        
        self.step_counter += 1
    
    def get_system_state(self) -> Dict[str, Any]:
        """获取整个系统的状态"""
        return {
            "drones": [drone.to_dict() for drone in self.drones],
            "total_tasks": len(self.all_tasks),
            "unassigned_tasks": len(self.unassigned_tasks),
            "completed_tasks": len(self.completed_tasks),
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
        
        # 整体系统状态
        system_features = np.array([
            len(self.unassigned_tasks) / max(1, len(self.all_tasks)),
            len(self.completed_tasks) / max(1, len(self.all_tasks)),
            self.step_counter / 1000.0  # 归一化步数
        ])
        
        return np.concatenate([drone_features, tasks_per_drone, system_features])
    
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
                    # 动作3：移动到指定区域
                    
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
                    
                    elif action == 3 and not drone.is_busy:
                        # 移动到区域中心
                        region_id = i % len(self.region_centers)
                        drone.update_position(self.region_centers[region_id])
        
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
        
        # 无人机电池状态奖励
        battery_levels = [drone.battery / 100.0 for drone in self.drones]
        battery_reward = 3.0 * sum(battery_levels) / len(self.drones)
        
        # 空闲时间惩罚
        idle_ratio = sum(drone.idle_time for drone in self.drones) / max(1.0, self.step_counter * len(self.drones))
        idle_penalty = -5.0 * idle_ratio
        
        # 总奖励
        total_reward = completion_reward + balance_reward + battery_reward + idle_penalty
        
        return total_reward 