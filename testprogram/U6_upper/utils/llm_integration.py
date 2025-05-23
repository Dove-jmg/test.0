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
    """大语言模型接口类，用于任务分解和处理 - U6_upper版本"""
    
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
        解析任务描述，提取关键信息 - 适配U6_upper模型
        
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
        4. 无人机数量（6个无人机）
        5. 监测指标（如适用）
        6. 传感器类型（如适用）
        7. 采集频率（如适用）
        8. 其他任何相关参数
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
    
    def decompose_task_for_six_uavs(self, task_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        专为6无人机系统分解任务
        
        Args:
            task_info: 任务信息字典
            
        Returns:
            list: 分解后的子任务列表，适合6无人机执行
        """
        prompt = f"""
        请将以下任务为6个无人机系统分解。每个无人机可以负责不同区域或功能。
        
        任务信息：
        {json.dumps(task_info, ensure_ascii=False, indent=2)}
        
        请返回一个包含子任务的JSON数组，每个子任务应包含：
        1. 无人机ID（0-5）
        2. 分配区域
        3. 任务类型
        4. 时间窗口
        5. 优先级（1-5）
        6. 功能描述
        """
        
        response = self._get_llm_response(prompt)
        
        try:
            json_str = self._extract_json(response)
            subtasks = json.loads(json_str)
            
            # 确保正好有6个子任务
            if len(subtasks) != 6:
                logger.warning(f"LLM返回了{len(subtasks)}个子任务，但需要6个。调整子任务数量。")
                if len(subtasks) > 6:
                    subtasks = subtasks[:6]
                else:
                    # 如果少于6个，复制现有任务填充
                    while len(subtasks) < 6:
                        new_task = subtasks[len(subtasks) % len(subtasks)].copy()
                        new_task["无人机ID"] = len(subtasks)
                        subtasks.append(new_task)
            
            return subtasks
        except Exception as e:
            logger.error(f"为6无人机分解任务时出错: {e}")
            # 创建一个默认的六无人机任务分配
            default_tasks = []
            for i in range(6):
                default_tasks.append({
                    "无人机ID": i,
                    "分配区域": f"区域{i+1}",
                    "任务类型": task_info.get("任务类型", "数据收集"),
                    "时间窗口": task_info.get("时间范围", {"开始时间": "10:00", "结束时间": "12:00"}),
                    "优先级": 3,
                    "功能描述": f"无人机{i}的默认任务"
                })
            return default_tasks
    
    def optimize_six_uav_schedule(self, tasks: List[Dict[str, Any]], environment_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        优化6无人机的任务调度
        
        Args:
            tasks: 任务列表
            environment_info: 环境信息（天气、拥堵等）
            
        Returns:
            list: 优化后的任务列表
        """
        tasks_json = json.dumps(tasks, ensure_ascii=False, indent=2)
        env_json = json.dumps(environment_info, ensure_ascii=False, indent=2)
        
        prompt = f"""
        请为6无人机系统优化以下任务调度，考虑环境因素：
        
        环境信息：
        {env_json}
        
        任务列表：
        {tasks_json}
        
        请返回优化后的任务列表，包括：
        1. 执行顺序调整
        2. 任务优先级调整
        3. 每个无人机的具体执行路径
        4. 预计完成时间
        5. 协作策略（无人机之间如何协作）
        """
        
        response = self._get_llm_response(prompt)
        
        try:
            json_str = self._extract_json(response)
            optimized_tasks = json.loads(json_str)
            return optimized_tasks
        except Exception as e:
            logger.error(f"优化6无人机调度时出错: {e}")
            return tasks
    
    def generate_collaboration_strategy(self, uav_states: List[Dict[str, Any]], tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成6无人机的协作策略
        
        Args:
            uav_states: 无人机状态列表
            tasks: 任务列表
            
        Returns:
            dict: 协作策略
        """
        uav_json = json.dumps(uav_states, ensure_ascii=False, indent=2)
        tasks_json = json.dumps(tasks, ensure_ascii=False, indent=2)
        
        prompt = f"""
        请为以下6无人机系统生成协作策略，以最高效完成任务：
        
        无人机状态：
        {uav_json}
        
        任务列表：
        {tasks_json}
        
        请返回详细的协作策略，包括：
        1. 无人机分组方案
        2. 通信协议
        3. 任务交接点
        4. 异常处理策略
        5. 资源共享方案
        """
        
        response = self._get_llm_response(prompt)
        
        try:
            json_str = self._extract_json(response)
            strategy = json.loads(json_str)
            return strategy
        except Exception as e:
            logger.error(f"生成协作策略时出错: {e}")
            return {
                "error": "生成协作策略失败",
                "原因": str(e),
                "原始响应": response
            }
    
    def analyze_task_results(self, task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析任务执行结果
        
        Args:
            task_results: 任务执行结果列表
            
        Returns:
            dict: 分析结果
        """
        results_json = json.dumps(task_results, ensure_ascii=False, indent=2)
        
        prompt = f"""
        请分析以下6无人机系统的任务执行结果：
        
        {results_json}
        
        请提供以下分析结果：
        1. 总体完成率
        2. 每个无人机的效率
        3. 任务延迟情况
        4. 资源使用情况
        5. 协作效果评估
        6. 改进建议
        """
        
        response = self._get_llm_response(prompt)
        
        try:
            json_str = self._extract_json(response)
            analysis = json.loads(json_str)
            return analysis
        except Exception as e:
            logger.error(f"分析任务结果时出错: {e}")
            return {
                "error": "分析失败",
                "原因": str(e),
                "原始响应": response
            }
    
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


# U6任务类，专为6无人机系统设计
class U6Task:
    def __init__(self, task_id: str, uav_id: int, task_info: Dict[str, Any]):
        self.id = task_id
        self.uav_id = uav_id
        self.info = task_info
        self.priority = task_info.get("优先级", 3)
        self.completed = False
        self.in_progress = False
        self.completion_time = None
        self.collaboration_group = task_info.get("协作组", [])  # 协作无人机ID列表
        
    def mark_completed(self, completion_time=None):
        """标记任务为已完成"""
        self.completed = True
        self.in_progress = False
        self.completion_time = completion_time
        
    def mark_in_progress(self):
        """标记任务为进行中"""
        self.in_progress = True
        
    def add_collaborator(self, uav_id: int):
        """添加协作无人机"""
        if uav_id not in self.collaboration_group:
            self.collaboration_group.append(uav_id)
        
    def to_dict(self) -> Dict[str, Any]:
        """将任务转换为字典表示"""
        return {
            "id": self.id,
            "uav_id": self.uav_id,
            "info": self.info,
            "priority": self.priority,
            "completed": self.completed,
            "in_progress": self.in_progress,
            "completion_time": self.completion_time,
            "collaboration_group": self.collaboration_group
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'U6Task':
        """从字典创建任务对象"""
        task = cls(data["id"], data["uav_id"], data["info"])
        task.priority = data["priority"]
        task.completed = data["completed"] 
        task.in_progress = data["in_progress"]
        task.completion_time = data.get("completion_time")
        task.collaboration_group = data.get("collaboration_group", [])
        return task 