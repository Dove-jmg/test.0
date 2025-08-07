#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验结果可视化工具
用于绘制训练过程中各项指标的变化趋势图
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Optional

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

class ExperimentVisualizer:
    """实验结果可视化类"""
    
    def __init__(self, data_dir: str):
        """
        初始化可视化器
        
        Args:
            data_dir: 数据文件所在目录
        """
        self.data_dir = data_dir
        self.data_dict: Dict[str, pd.DataFrame] = {}
        self.available_metrics = [
            "actor_loss", "avg_cost", "grad_norm", "grad_norm_clipped", 
            "learnrate_pg0", "val_avg_reward"
        ]
    
    def load_data(self, metrics: Optional[List[str]] = None) -> None:
        """
        加载指定指标的数据
        
        Args:
            metrics: 要加载的指标列表，若为None则加载所有可用指标
        """
        if metrics is None:
            metrics = self.available_metrics
        
        for metric in metrics:
            csv_path = os.path.join(self.data_dir, f"run-.-tag-{metric}.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    # 确保Step列为数值类型
                    df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
                    # 过滤无效Step值
                    df = df.dropna(subset=['Step'])
                    self.data_dict[metric] = df
                    print(f"成功加载 {metric} 数据，共 {len(df)} 条记录")
                except Exception as e:
                    print(f"加载 {metric} 数据失败: {e}")
            else:
                print(f"未找到 {metric} 的CSV文件: {csv_path}")
    
    def plot_single_metric(self, metric: str, save_path: Optional[str] = None) -> None:
        """
        绘制单个指标的趋势图
        
        Args:
            metric: 要绘制的指标名称
            save_path: 图表保存路径，若为None则不保存
        """
        if metric not in self.data_dict:
            print(f"未加载 {metric} 数据，请先调用load_data()")
            return
        
        df = self.data_dict[metric]
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Step', y='Value', data=df, color='blue')
        
        plt.title(f'{self._get_metric_display_name(metric)} 训练趋势', fontsize=16)
        plt.xlabel('训练步数', fontsize=14)
        plt.ylabel(self._get_metric_display_name(metric), fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"图表已保存至 {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, metrics: List[str], save_path: Optional[str] = None) -> None:
        """
        绘制多个指标的对比图
        
        Args:
            metrics: 要对比的指标列表
            save_path: 图表保存路径，若为None则不保存
        """
        missing_metrics = [m for m in metrics if m not in self.data_dict]
        if missing_metrics:
            print(f"未加载以下指标数据: {', '.join(missing_metrics)}")
            return
        
        plt.figure(figsize=(12, 6))
        
        color_map = {
            "actor_loss": "blue",
            "avg_cost": "orange",
            "grad_norm": "green",
            "grad_norm_clipped": "red",
            "learnrate_pg0": "purple",
            "val_avg_reward": "brown"
        }
        
        for metric in metrics:
            df = self.data_dict[metric]
            sns.lineplot(
                x='Step', 
                y='Value', 
                data=df, 
                color=color_map.get(metric, 'gray'),
                label=self._get_metric_display_name(metric)
            )
        
        plt.title('训练指标对比', fontsize=16)
        plt.xlabel('训练步数', fontsize=14)
        plt.ylabel('指标值', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"图表已保存至 {save_path}")
        
        plt.show()
    
    def plot_metrics_grid(self, metrics: List[str], save_path: Optional[str] = None) -> None:
        """
        绘制多个指标的网格图
        
        Args:
            metrics: 要绘制的指标列表
            save_path: 图表保存路径，若为None则不保存
        """
        missing_metrics = [m for m in metrics if m not in self.data_dict]
        if missing_metrics:
            print(f"未加载以下指标数据: {', '.join(missing_metrics)}")
            return
        
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)  # 最多两列
        n_rows = (n_metrics + n_cols - 1) // n_cols  # 计算行数
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for i, metric in enumerate(metrics):
            df = self.data_dict[metric]
            ax = axes[i]
            
            sns.lineplot(x='Step', y='Value', data=df, ax=ax, color='blue')
            ax.set_title(self._get_metric_display_name(metric), fontsize=14)
            ax.set_xlabel('训练步数', fontsize=12)
            ax.set_ylabel(self._get_metric_display_name(metric), fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"图表已保存至 {save_path}")
        
        plt.show()
    
    def plot_rolling_average(self, metric: str, window: int = 10, save_path: Optional[str] = None) -> None:
        """
        绘制指标的滚动平均值
        
        Args:
            metric: 要绘制的指标名称
            window: 滚动窗口大小
            save_path: 图表保存路径，若为None则不保存
        """
        if metric not in self.data_dict:
            print(f"未加载 {metric} 数据，请先调用load_data()")
            return
        
        df = self.data_dict[metric].copy()
        df['rolling_mean'] = df['Value'].rolling(window=window).mean()
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Step', y='Value', data=df, alpha=0.3, label='原始数据')
        sns.lineplot(x='Step', y='rolling_mean', data=df, color='red', label=f'{window}步滚动平均')
        
        plt.title(f'{self._get_metric_display_name(metric)} 滚动平均值', fontsize=16)
        plt.xlabel('训练步数', fontsize=14)
        plt.ylabel(self._get_metric_display_name(metric), fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"图表已保存至 {save_path}")
        
        plt.show()
    
    def plot_convergence_rate(self, metric: str, save_path: Optional[str] = None) -> None:
        """
        绘制指标的收敛速率
        
        Args:
            metric: 要分析的指标名称
            save_path: 图表保存路径，若为None则不保存
        """
        if metric not in self.data_dict:
            print(f"未加载 {metric} 数据，请先调用load_data()")
            return
        
        df = self.data_dict[metric].copy()
        
        # 计算收敛速率（导数）
        df['convergence_rate'] = df['Value'].diff() / df['Step'].diff()
        
        plt.figure(figsize=(12, 6))
        
        # 绘制原始指标
        plt.subplot(2, 1, 1)
        sns.lineplot(x='Step', y='Value', data=df, color='blue')
        plt.title(f'{self._get_metric_display_name(metric)} 训练趋势', fontsize=14)
        plt.ylabel(self._get_metric_display_name(metric), fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 绘制收敛速率
        plt.subplot(2, 1, 2)
        sns.lineplot(x='Step', y='convergence_rate', data=df, color='red')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title(f'{self._get_metric_display_name(metric)} 收敛速率', fontsize=14)
        plt.xlabel('训练步数', fontsize=12)
        plt.ylabel('收敛速率', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"图表已保存至 {save_path}")
        
        plt.show()
    
    def _get_metric_display_name(self, metric: str) -> str:
        """获取指标的显示名称"""
        display_names = {
            "actor_loss": "Actor损失",
            "avg_cost": "平均成本",
            "grad_norm": "梯度范数",
            "grad_norm_clipped": "裁剪后的梯度范数",
            "learnrate_pg0": "学习率",
            "val_avg_reward": "验证集平均奖励"
        }
        return display_names.get(metric, metric)

def main():
    """主函数，演示如何使用可视化工具"""
    # 请修改为你的数据目录
    data_dir = r"C:\Users\Administrator\Desktop\相关代码文件\新建文件夹\run-.-tag-"
    
    # 创建可视化器实例
    visualizer = ExperimentVisualizer(data_dir)
    
    # 加载所有可用指标数据
    visualizer.load_data()
    
    # 为每个指标绘制单独的趋势图
    for metric in visualizer.available_metrics:
        visualizer.plot_single_metric(metric, save_path=f"plots/{metric}_trend.png")
    
    # 绘制多个指标的对比图
    visualizer.plot_metrics_comparison(
        ["actor_loss", "avg_cost"], 
        save_path="plots/loss_vs_cost.png"
    )
    
    visualizer.plot_metrics_comparison(
        ["grad_norm", "grad_norm_clipped"], 
        save_path="plots/grad_norm_comparison.png"
    )
    
    # 绘制所有指标的网格图
    visualizer.plot_metrics_grid(
        visualizer.available_metrics,
        save_path="plots/all_metrics_grid.png"
    )
    
    # 为每个指标绘制滚动平均值
    for metric in visualizer.available_metrics:
        visualizer.plot_rolling_average(metric, window=20, save_path=f"plots/{metric}_rolling.png")
    
    # 为每个指标绘制收敛速率
    for metric in visualizer.available_metrics:
        visualizer.plot_convergence_rate(metric, save_path=f"plots/{metric}_convergence.png")

if __name__ == "__main__":
    main()