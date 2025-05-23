import torch
import requests
import json
import os
import logging
from typing import Dict, List, Tuple, Union, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMInterface:
    """大语言模型接口类，用于任务分解和处理"""
    
    def __init__(self, model_name: str = "openlm/open-llama-3b", api_key: str = None, api_url: str = None):
        """
        初始化LLM接口
        
        Args:
            model_name: 使用的开源模型名称
            api_key: API密钥（如果使用远程API）
            api_url: API URL（如果使用远程API）
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        
        # 根据设置决定是否使用本地模型或远程API
        self.use_api = (api_key is not None and api_url is not None)
        
        if not self.use_api:
            try:
                # 尝试导入transformers以使用本地模型
                from transformers import AutoModelForCausalLM, AutoTokenizer
                logger.info(f"正在加载本地模型: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                # 如果CUDA可用，则使用GPU
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                logger.info(f"本地模型 {model_name} 加载完成")
            except ImportError as e:
                logger.error(f"无法导入transformers，请安装或使用API模式: {e}")
                raise
        else:
            logger.info(f"将使用远程API: {api_url}")
    
    def parse_task(self, task_description: str) -> Dict[str, Any]:
        """
        解析任务描述，提取关键信息
        
        Args:
            task_description: 任务的文本描述
            
        Returns:
            dict: 包含解析后任务信息的字典
        """
        prompt = f"""
        请解析以下任务描述，并提取关键信息：
        
        {task_description}
        
        请以JSON格式返回以下信息：
        1. 任务类型（数据收集、数据卸载等）
        2. 时间范围（开始时间和结束时间）
        3. 目标区域
        4. 监测指标（如适用）
        5. 传感器类型（如适用）
        6. 采集频率（如适用）
        7. 其他任何相关参数
        """
        
        response = self._get_llm_response(prompt)
        
        try:
            # 尝试从响应中提取JSON部分
            json_str = self._extract_json(response)
            task_info = json.loads(json_str)
            return task_info
        except json.JSONDecodeError as e:
            logger.error(f"无法解析LLM响应为JSON: {e}")
            logger.debug(f"原始响应: {response}")
            # 返回一个带有错误信息的基础字典
            return {
                "error": "解析失败",
                "original_response": response,
                "task_description": task_description
            }
    
    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON部分"""
        try:
            # 尝试查找JSON部分（通常在```json和```之间，或者直接是一个JSON对象）
            if "```json" in text and "```" in text.split("```json")[1]:
                return text.split("```json")[1].split("```")[0].strip()
            elif text.strip().startswith("{") and text.strip().endswith("}"):
                return text.strip()
            else:
                # 查找可能是JSON的部分（从第一个{到最后一个}之间的内容）
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    return text[start:end]
                else:
                    raise ValueError("无法在文本中找到JSON部分")
        except Exception as e:
            logger.error(f"提取JSON时出错: {e}")
            raise
    
    def decompose_task_by_time(self, task_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        按时间拆分任务
        
        Args:
            task_info: 任务信息字典
            
        Returns:
            list: 子任务列表
        """
        if "时间范围" not in task_info or not task_info["时间范围"]:
            return [task_info]  # 如果没有时间信息，则返回原任务
        
        start_time = task_info.get("时间范围", {}).get("开始时间")
        end_time = task_info.get("时间范围", {}).get("结束时间")
        frequency = task_info.get("采集频率")
        
        if not all([start_time, end_time, frequency]):
            return [task_info]  # 缺少必要信息，返回原任务
        
        # 构建提示，让LLM将任务按时间段分解
        prompt = f"""
        请将以下任务按照采集频率分解为多个时间子任务：
        
        任务类型: {task_info.get('任务类型')}
        开始时间: {start_time}
        结束时间: {end_time}
        采集频率: {frequency}
        目标区域: {task_info.get('目标区域')}
        
        请以JSON数组格式返回子任务列表，每个子任务应该包含独立的开始时间和结束时间。
        """
        
        response = self._get_llm_response(prompt)
        
        try:
            json_str = self._extract_json(response)
            subtasks = json.loads(json_str)
            # 确保每个子任务都包含原任务的必要信息
            for subtask in subtasks:
                # 复制原任务信息，但使用新的时间范围
                for key, value in task_info.items():
                    if key != "时间范围" and key not in subtask:
                        subtask[key] = value
            return subtasks
        except Exception as e:
            logger.error(f"分解任务时出错: {e}")
            return [task_info]  # 出错时返回原任务
    
    def aggregate_tasks_by_region(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按区域聚合任务
        
        Args:
            tasks: 任务列表
            
        Returns:
            list: 聚合后的任务列表
        """
        if not tasks:
            return []
        
        # 将任务列表转换为JSON字符串
        tasks_json = json.dumps(tasks, ensure_ascii=False, indent=2)
        
        # 构建提示，让LLM按区域聚合任务
        prompt = f"""
        请将以下任务列表按照区域聚合，同一区域的同类任务应该合并：
        
        {tasks_json}
        
        请以JSON数组格式返回聚合后的任务列表，每个聚合任务应该包含：
        1. 任务类型
        2. 目标区域
        3. 时间范围列表（保留原子任务的时间信息）
        4. 其他相关参数（如监测指标、传感器类型等）
        5. 子任务列表（原始任务的引用）
        """
        
        response = self._get_llm_response(prompt)
        
        try:
            json_str = self._extract_json(response)
            aggregated_tasks = json.loads(json_str)
            return aggregated_tasks
        except Exception as e:
            logger.error(f"聚合任务时出错: {e}")
            return tasks  # 出错时返回原任务列表
    
    def optimize_task_order(self, tasks: List[Dict[str, Any]], drone_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        优化任务执行顺序
        
        Args:
            tasks: 任务列表
            drone_info: 无人机信息（位置、速度、电量等）
            
        Returns:
            list: 优化后的任务列表
        """
        tasks_json = json.dumps(tasks, ensure_ascii=False, indent=2)
        drone_json = json.dumps(drone_info, ensure_ascii=False, indent=2)
        
        prompt = f"""
        请为无人机优化以下任务的执行顺序，考虑位置距离、时间窗口和任务优先级：
        
        无人机信息：
        {drone_json}
        
        任务列表：
        {tasks_json}
        
        请以JSON数组格式返回优化后的任务执行顺序，每个任务应包含原任务ID及优先级评分（0-100）。
        """
        
        response = self._get_llm_response(prompt)
        
        try:
            json_str = self._extract_json(response)
            optimized_order = json.loads(json_str)
            
            # 重新排序原任务列表
            ordered_tasks = []
            task_map = {task.get("id", i): task for i, task in enumerate(tasks)}
            
            for item in optimized_order:
                task_id = item.get("id")
                if task_id in task_map:
                    task = task_map[task_id].copy()
                    # 添加优先级信息
                    task["优先级评分"] = item.get("优先级评分", 0)
                    ordered_tasks.append(task)
            
            # 添加未包含在优化顺序中的任务
            for task_id, task in task_map.items():
                if not any(item.get("id") == task_id for item in optimized_order):
                    ordered_tasks.append(task)
            
            return ordered_tasks
        except Exception as e:
            logger.error(f"优化任务顺序时出错: {e}")
            return tasks  # 出错时返回原任务列表
    
    def _get_llm_response(self, prompt: str) -> str:
        """获取LLM的响应"""
        if self.use_api:
            return self._call_api(prompt)
        else:
            return self._call_local_model(prompt)
    
    def _call_api(self, prompt: str) -> str:
        """调用远程API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return f"错误: {str(e)}"
    
    def _call_local_model(self, prompt: str) -> str:
        """使用本地模型生成响应"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=2048,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 移除提示部分，只保留生成的内容
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
        except Exception as e:
            logger.error(f"本地模型推理失败: {e}")
            return f"错误: {str(e)}"


# 任务类，用于表示经过分解和聚合后的任务
class Task:
    def __init__(self, task_id: str, task_info: Dict[str, Any]):
        self.id = task_id
        self.info = task_info
        self.priority = task_info.get("优先级评分", 50)  # 默认优先级为50
        self.completed = False
        self.in_progress = False
        self.subtasks = []
    
    def add_subtask(self, subtask: 'Task'):
        """添加子任务"""
        self.subtasks.append(subtask)
    
    def mark_completed(self):
        """标记任务为已完成"""
        self.completed = True
        self.in_progress = False
    
    def mark_in_progress(self):
        """标记任务为进行中"""
        self.in_progress = True
    
    def to_dict(self) -> Dict[str, Any]:
        """将任务转换为字典表示"""
        result = {
            "id": self.id,
            "info": self.info,
            "priority": self.priority,
            "completed": self.completed,
            "in_progress": self.in_progress
        }
        if self.subtasks:
            result["subtasks"] = [subtask.to_dict() for subtask in self.subtasks]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """从字典创建任务对象"""
        task = cls(data["id"], data["info"])
        task.priority = data["priority"]
        task.completed = data["completed"]
        task.in_progress = data["in_progress"]
        
        if "subtasks" in data:
            for subtask_data in data["subtasks"]:
                task.add_subtask(cls.from_dict(subtask_data))
        
        return task 