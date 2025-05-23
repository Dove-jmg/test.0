import torch
import numpy as np
import heapq
from typing import Dict, List, Tuple, Union, Any, Optional
import logging
import copy
import math

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DroneState:
    """无人机状态类"""
    
    def __init__(
        self,
        drone_id: int,
        position: Tuple[float, float],
        storage_capacity: float,
        remaining_storage: float,
        battery_level: float,
        max_flight_distance: float,
        speed: float,
        current_task_id: Optional[str] = None
    ):
        self.drone_id = drone_id
        self.position = position
        self.storage_capacity = storage_capacity
        self.remaining_storage = remaining_storage
        self.battery_level = battery_level
        self.max_flight_distance = max_flight_distance
        self.speed = speed
        self.current_task_id = current_task_id
        self.task_history = []
        
    def update_position(self, new_position: Tuple[float, float], distance_traveled: float = None):
        """更新无人机位置"""
        if distance_traveled is None:
            # 计算移动距离
            distance_traveled = math.sqrt(
                (new_position[0] - self.position[0])**2 + 
                (new_position[1] - self.position[1])**2
            )
        
        # 更新位置
        self.position = new_position
        
        # 根据移动距离消耗电量（简化模型）
        energy_consumed = distance_traveled * 0.01  # 简化的能量消耗模型
        self.battery_level = max(0, self.battery_level - energy_consumed)
    
    def update_storage(self, data_size: float):
        """更新存储空间"""
        if data_size > self.remaining_storage:
            logger.warning(f"无人机 {self.drone_id} 存储空间不足! 需要: {data_size}, 剩余: {self.remaining_storage}")
            return False
        
        self.remaining_storage -= data_size
        return True
    
    def assign_task(self, task_id: str):
        """分配任务给无人机"""
        self.current_task_id = task_id
    
    def complete_task(self, task_id: str):
        """完成任务"""
        if self.current_task_id == task_id:
            self.task_history.append(task_id)
            self.current_task_id = None
            return True
        else:
            logger.warning(f"无人机 {self.drone_id} 当前未执行任务 {task_id}")
            return False
    
    def can_reach(self, target_position: Tuple[float, float]) -> bool:
        """检查无人机是否能够到达目标位置"""
        distance = math.sqrt(
            (target_position[0] - self.position[0])**2 + 
            (target_position[1] - self.position[1])**2
        )
        
        # 简化的能量检查模型
        estimated_energy_needed = distance * 0.01
        return (distance <= self.max_flight_distance and 
                estimated_energy_needed <= self.battery_level)
    
    def estimate_travel_time(self, target_position: Tuple[float, float]) -> float:
        """估计到达目标位置的时间（秒）"""
        distance = math.sqrt(
            (target_position[0] - self.position[0])**2 + 
            (target_position[1] - self.position[1])**2
        )
        return distance / self.speed
    
    def to_dict(self) -> Dict[str, Any]:
        """将无人机状态转换为字典"""
        return {
            "drone_id": self.drone_id,
            "position": self.position,
            "storage_capacity": self.storage_capacity,
            "remaining_storage": self.remaining_storage,
            "battery_level": self.battery_level,
            "max_flight_distance": self.max_flight_distance,
            "speed": self.speed,
            "current_task_id": self.current_task_id,
            "task_history": self.task_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DroneState':
        """从字典创建无人机状态"""
        drone = cls(
            drone_id=data["drone_id"],
            position=data["position"],
            storage_capacity=data["storage_capacity"],
            remaining_storage=data["remaining_storage"],
            battery_level=data["battery_level"],
            max_flight_distance=data["max_flight_distance"],
            speed=data["speed"],
            current_task_id=data["current_task_id"]
        )
        drone.task_history = data["task_history"]
        return drone

class TaskInfo:
    """任务信息类"""
    
    def __init__(
        self,
        task_id: str,
        task_type: str,
        position: Tuple[float, float],
        data_size: float,
        start_time: float,
        end_time: float,
        priority: int = 1,
        region: str = "default",
        sensor_type: Optional[str] = None,
        collection_interval: Optional[float] = None,
        required_data_points: Optional[int] = None
    ):
        self.task_id = task_id
        self.task_type = task_type  # 'data_collection' 或 'data_offload'
        self.position = position
        self.data_size = data_size
        self.start_time = start_time
        self.end_time = end_time
        self.priority = priority  # 1-3，数字越大优先级越高
        self.region = region
        self.sensor_type = sensor_type
        self.collection_interval = collection_interval
        self.required_data_points = required_data_points
        
        self.assigned_drone_id = None
        self.status = "pending"  # pending, in_progress, completed, failed
        self.completion_time = None
    
    def assign_to_drone(self, drone_id: int):
        """将任务分配给无人机"""
        self.assigned_drone_id = drone_id
        self.status = "in_progress"
    
    def mark_completed(self, completion_time: float):
        """标记任务为已完成"""
        self.status = "completed"
        self.completion_time = completion_time
    
    def mark_failed(self):
        """标记任务为失败"""
        self.status = "failed"
        self.assigned_drone_id = None
    
    def is_in_time_window(self, current_time: float) -> bool:
        """检查当前时间是否在任务时间窗口内"""
        return self.start_time <= current_time <= self.end_time
    
    def time_to_deadline(self, current_time: float) -> float:
        """计算距离截止时间的剩余时间"""
        return max(0, self.end_time - current_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """将任务信息转换为字典"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "position": self.position,
            "data_size": self.data_size,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "priority": self.priority,
            "region": self.region,
            "sensor_type": self.sensor_type,
            "collection_interval": self.collection_interval,
            "required_data_points": self.required_data_points,
            "assigned_drone_id": self.assigned_drone_id,
            "status": self.status,
            "completion_time": self.completion_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskInfo':
        """从字典创建任务信息"""
        task = cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            position=data["position"],
            data_size=data["data_size"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            priority=data["priority"],
            region=data["region"],
            sensor_type=data["sensor_type"],
            collection_interval=data["collection_interval"],
            required_data_points=data["required_data_points"]
        )
        task.assigned_drone_id = data["assigned_drone_id"]
        task.status = data["status"]
        task.completion_time = data["completion_time"]
        return task

class DynamicTaskAllocator:
    """动态任务分配器"""
    
    def __init__(
        self,
        drones: List[DroneState],
        initial_tasks: Optional[List[TaskInfo]] = None,
        current_time: float = 0.0,
        alpha: float = 0.6,  # 已完成任务价值权重
        beta: float = 0.2,   # 剩余存储空间权重
        gamma: float = 0.2   # 任务延迟时间权重
    ):
        self.drones = {drone.drone_id: drone for drone in drones}
        self.tasks = {}
        if initial_tasks:
            for task in initial_tasks:
                self.tasks[task.task_id] = task
        
        self.current_time = current_time
        
        # 奖励函数权重
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 任务队列
        self.pending_tasks = []  # 优先队列，实际为堆
        self.active_tasks = {}   # 正在进行的任务
        self.completed_tasks = {}  # 已完成的任务
        self.failed_tasks = {}   # 失败的任务
        
        # 重新构建任务队列
        self._rebuild_task_queue()
    
    def _rebuild_task_queue(self):
        """重新构建任务优先队列"""
        self.pending_tasks = []
        for task_id, task in self.tasks.items():
            if task.status == "pending" and task.is_in_time_window(self.current_time):
                # 计算任务优先级分数
                score = self._calculate_task_priority(task)
                # 使用负分数，因为heapq是最小堆
                heapq.heappush(self.pending_tasks, (-score, task_id))
    
    def _calculate_task_priority(self, task: TaskInfo) -> float:
        """计算任务优先级分数"""
        # 基础优先级得分（1-3）
        base_priority = task.priority * 10.0
        
        # 时间紧迫性得分
        time_urgency = 0.0
        if task.end_time > self.current_time:
            remaining_time = task.end_time - self.current_time
            time_window = task.end_time - task.start_time
            if time_window > 0:
                time_urgency = (1.0 - (remaining_time / time_window)) * 10.0
        else:
            time_urgency = 10.0  # 已经超时
        
        # 数据大小得分，较小的数据任务优先处理
        data_size_score = (1.0 / (task.data_size + 1.0)) * 5.0
        
        # 最终得分
        score = base_priority + time_urgency + data_size_score
        return score
    
    def add_task(self, task: TaskInfo):
        """添加新任务"""
        if task.task_id in self.tasks:
            logger.warning(f"任务 {task.task_id} 已存在，将会被覆盖")
        
        self.tasks[task.task_id] = task
        
        # 如果任务在当前时间窗口内且处于待定状态，则添加到优先队列
        if task.status == "pending" and task.is_in_time_window(self.current_time):
            score = self._calculate_task_priority(task)
            heapq.heappush(self.pending_tasks, (-score, task.task_id))
    
    def add_drone(self, drone: DroneState):
        """添加新无人机"""
        if drone.drone_id in self.drones:
            logger.warning(f"无人机 {drone.drone_id} 已存在，将会被覆盖")
        
        self.drones[drone.drone_id] = drone
    
    def update_time(self, new_time: float):
        """更新当前时间并重新评估任务队列"""
        if new_time < self.current_time:
            logger.warning(f"新时间 {new_time} 小于当前时间 {self.current_time}，将被忽略")
            return
        
        self.current_time = new_time
        
        # 检查是否有任务过期
        for task_id, task in list(self.tasks.items()):
            if task.status == "pending" and self.current_time > task.end_time:
                task.mark_failed()
                self.failed_tasks[task_id] = task
                logger.info(f"任务 {task_id} 已过期，标记为失败")
        
        # 重新构建任务队列
        self._rebuild_task_queue()
    
    def get_available_drones(self) -> List[DroneState]:
        """获取可用的无人机列表"""
        available_drones = []
        for drone_id, drone in self.drones.items():
            if drone.current_task_id is None:
                available_drones.append(drone)
        return available_drones
    
    def allocate_tasks(self, max_allocations: int = None):
        """根据优先级分配任务给可用的无人机"""
        available_drones = self.get_available_drones()
        allocations_made = 0
        
        while available_drones and self.pending_tasks and (max_allocations is None or allocations_made < max_allocations):
            # 获取优先级最高的任务
            _, task_id = heapq.heappop(self.pending_tasks)
            task = self.tasks[task_id]
            
            # 如果任务不再处于有效时间窗口，跳过
            if not task.is_in_time_window(self.current_time):
                continue
            
            # 找到最适合该任务的无人机
            best_drone = None
            best_score = float('-inf')
            
            for drone in available_drones:
                # 检查无人机是否能执行该任务
                if not self._can_drone_execute_task(drone, task):
                    continue
                
                # 计算分配分数
                score = self._calculate_allocation_score(drone, task)
                
                if score > best_score:
                    best_score = score
                    best_drone = drone
            
            # 如果找到合适的无人机，分配任务
            if best_drone:
                self._assign_task_to_drone(best_drone, task)
                available_drones.remove(best_drone)
                allocations_made += 1
            else:
                # 如果没有合适的无人机，将任务放回队列
                score = self._calculate_task_priority(task)
                heapq.heappush(self.pending_tasks, (-score, task_id))
                break
    
    def _can_drone_execute_task(self, drone: DroneState, task: TaskInfo) -> bool:
        """检查无人机是否能执行给定任务"""
        # 检查存储空间
        if task.data_size > drone.remaining_storage:
            return False
        
        # 检查是否能到达任务位置
        if not drone.can_reach(task.position):
            return False
        
        # 检查时间约束
        travel_time = drone.estimate_travel_time(task.position)
        estimated_arrival_time = self.current_time + travel_time
        
        # 如果预计到达时间已经超过任务结束时间，则无法执行
        if estimated_arrival_time > task.end_time:
            return False
        
        return True
    
    def _calculate_allocation_score(self, drone: DroneState, task: TaskInfo) -> float:
        """计算将任务分配给无人机的得分"""
        # 距离得分 - 距离越短越好
        distance = math.sqrt(
            (task.position[0] - drone.position[0])**2 + 
            (task.position[1] - drone.position[1])**2
        )
        max_distance = drone.max_flight_distance
        distance_score = (1.0 - min(1.0, distance / max_distance)) * 10.0
        
        # 存储容量得分 - 任务所需存储比例越小越好
        storage_score = (1.0 - min(1.0, task.data_size / drone.storage_capacity)) * 5.0
        
        # 时间得分 - 预计到达时间与任务开始时间的符合程度
        travel_time = drone.estimate_travel_time(task.position)
        estimated_arrival_time = self.current_time + travel_time
        
        time_score = 0.0
        if estimated_arrival_time <= task.start_time:
            # 能在任务开始前到达
            time_score = 5.0
        else:
            # 在任务时间窗口内到达，分数随着接近截止时间而减少
            remaining_window = task.end_time - estimated_arrival_time
            total_window = task.end_time - task.start_time
            if total_window > 0 and remaining_window > 0:
                time_score = (remaining_window / total_window) * 5.0
        
        # 总分
        return distance_score + storage_score + time_score
    
    def _assign_task_to_drone(self, drone: DroneState, task: TaskInfo):
        """将任务分配给无人机"""
        drone.assign_task(task.task_id)
        task.assign_to_drone(drone.drone_id)
        self.active_tasks[task.task_id] = task
        logger.info(f"已将任务 {task.task_id} 分配给无人机 {drone.drone_id}")
    
    def complete_task(self, task_id: str, completion_time: Optional[float] = None):
        """标记任务为已完成"""
        if task_id not in self.tasks:
            logger.warning(f"任务 {task_id} 不存在")
            return False
        
        task = self.tasks[task_id]
        if task.status != "in_progress":
            logger.warning(f"任务 {task_id} 不在进行中，当前状态: {task.status}")
            return False
        
        # 设置完成时间
        if completion_time is None:
            completion_time = self.current_time
        
        # 更新任务状态
        task.mark_completed(completion_time)
        
        # 更新无人机状态
        drone_id = task.assigned_drone_id
        if drone_id in self.drones:
            drone = self.drones[drone_id]
            drone.complete_task(task_id)
            
            # 如果是数据收集任务，减少无人机存储空间
            if task.task_type == "data_collection":
                drone.update_storage(task.data_size)
            # 如果是数据卸载任务，增加无人机存储空间
            elif task.task_type == "data_offload":
                drone.remaining_storage = min(drone.storage_capacity, 
                                             drone.remaining_storage + task.data_size)
        
        # 从活动任务移动到已完成任务
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        self.completed_tasks[task_id] = task
        
        logger.info(f"任务 {task_id} 已完成")
        return True
    
    def fail_task(self, task_id: str, reason: str = "unknown"):
        """将任务标记为失败"""
        if task_id not in self.tasks:
            logger.warning(f"任务 {task_id} 不存在")
            return False
        
        task = self.tasks[task_id]
        drone_id = task.assigned_drone_id
        
        # 更新任务状态
        task.mark_failed()
        
        # 更新无人机状态
        if drone_id in self.drones:
            drone = self.drones[drone_id]
            if drone.current_task_id == task_id:
                drone.current_task_id = None
        
        # 从活动任务移动到失败任务
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        self.failed_tasks[task_id] = task
        
        logger.info(f"任务 {task_id} 已失败，原因: {reason}")
        return True
    
    def replan_route(self, drone_id: int):
        """重新规划指定无人机的路径（适用于当前任务被中断的情况）"""
        if drone_id not in self.drones:
            logger.warning(f"无人机 {drone_id} 不存在")
            return False
        
        drone = self.drones[drone_id]
        
        # 如果无人机当前有任务，取消该任务
        if drone.current_task_id:
            task_id = drone.current_task_id
            if task_id in self.tasks:
                self.fail_task(task_id, reason="任务被中断进行重新规划")
        
        # 现在无人机应该是空闲的，分配新任务
        self.allocate_tasks(max_allocations=1)
        
        return True
    
    def calculate_reward(
        self,
        completed_value: float,
        total_value: float,
        storage_left: float,
        max_storage: float,
        delay_time: float,
        total_time: float
    ) -> float:
        """计算奖励函数"""
        reward = (
            self.alpha * (completed_value / total_value) + 
            self.beta * (storage_left / max_storage) - 
            self.gamma * (delay_time / total_time)
        )
        return reward
    
    def get_task_value(self, task: TaskInfo) -> float:
        """获取任务的价值（根据优先级和数据大小）"""
        return task.priority * task.data_size
    
    def total_task_value(self) -> float:
        """计算所有任务的总价值"""
        return sum(self.get_task_value(task) for task in self.tasks.values())
    
    def completed_task_value(self) -> float:
        """计算已完成任务的总价值"""
        return sum(self.get_task_value(task) for task in self.completed_tasks.values())
    
    def get_delay_time(self) -> float:
        """计算所有已完成任务的延迟时间总和"""
        delay = 0.0
        for task in self.completed_tasks.values():
            if task.completion_time and task.start_time:
                # 如果在开始时间之前完成，没有延迟
                if task.completion_time <= task.start_time:
                    continue
                
                # 计算延迟时间
                expected_completion = task.start_time
                actual_completion = task.completion_time
                task_delay = actual_completion - expected_completion
                
                delay += max(0, task_delay)
        
        return delay
    
    def get_average_reward(self) -> float:
        """计算当前状态下的平均奖励"""
        completed_value = self.completed_task_value()
        total_value = self.total_task_value()
        
        # 如果没有任务，返回0
        if total_value == 0:
            return 0.0
        
        # 计算所有无人机的平均剩余存储空间比例
        total_storage_capacity = sum(drone.storage_capacity for drone in self.drones.values())
        total_remaining_storage = sum(drone.remaining_storage for drone in self.drones.values())
        storage_ratio = total_remaining_storage / total_storage_capacity if total_storage_capacity > 0 else 1.0
        
        # 计算延迟时间
        delay_time = self.get_delay_time()
        
        # 估计任务总时间
        total_time = 0.0
        for task in self.tasks.values():
            if task.end_time and task.start_time:
                total_time += (task.end_time - task.start_time)
        
        return self.calculate_reward(
            completed_value,
            total_value,
            total_remaining_storage,
            total_storage_capacity,
            delay_time,
            max(1.0, total_time)  # 避免除以0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """将分配器状态转换为字典"""
        return {
            "current_time": self.current_time,
            "drones": {id: drone.to_dict() for id, drone in self.drones.items()},
            "tasks": {id: task.to_dict() for id, task in self.tasks.items()},
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DynamicTaskAllocator':
        """从字典创建分配器"""
        drones = [DroneState.from_dict(drone_data) for drone_data in data["drones"].values()]
        tasks = [TaskInfo.from_dict(task_data) for task_data in data["tasks"].values()]
        
        allocator = cls(
            drones=drones,
            initial_tasks=tasks,
            current_time=data["current_time"],
            alpha=data["alpha"],
            beta=data["beta"],
            gamma=data["gamma"]
        )
        
        return allocator 