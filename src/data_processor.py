"""
数据处理模块
支持多维度风控画像数据的加载、预处理和课程学习数据划分
"""

import json
import os
import random
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer


class RiskProfilingDataProcessor:
    """风控画像数据处理器"""
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 定义所有支持的评估维度
        self.all_dimensions = {
            "competitiveness": "产品竞争力",
            "innovation": "产品创新性", 
            "diversity": "产品多样性",
            "sales_performance": "产品销售情况",
            "market_position": "市场地位",
            "financial_health": "财务健康度",
            "operational_efficiency": "运营效率",
            "risk_management": "风险管控能力",
            "growth_potential": "成长潜力",
            "customer_satisfaction": "客户满意度"
        }
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_dimension_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载单个维度的数据文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        except Exception as e:
            print(f"加载数据文件 {file_path} 失败: {e}")
            return []
    
    def load_all_data(self, data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
        """加载所有维度的数据"""
        all_data = {}
        
        # 扫描数据目录
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(data_dir, file_name)
                
                # 尝试从文件名推断维度
                dimension = None
                for dim_key, dim_name in self.all_dimensions.items():
                    if dim_key in file_name.lower() or dim_name in file_name:
                        dimension = dim_key
                        break
                
                if dimension:
                    data = self.load_dimension_data(file_path)
                    all_data[dimension] = data
                    print(f"加载 {dimension} 数据: {len(data)} 条样本")
        
        return all_data
    
    def create_multitask_prompt(self, 
                              company_info: str, 
                              dimensions: List[str],
                              mode: str = "train") -> str:
        """创建多任务学习的prompt"""
        
        dimension_names = [self.all_dimensions[dim] for dim in dimensions]
        dimension_str = "、".join(dimension_names)
        
        if mode == "train":
            prompt_template = f"""作为风控专家，请对以下企业在{dimension_str}等维度进行全面评估。

企业信息：
{company_info}

请按照以下要求进行评估：
1. 对每个维度给出0-9的评分（0分最低，9分最高）
2. 提供详细的分析理由
3. 分析过程要体现专业的风控思维

评估维度："""
            
            for i, dimension in enumerate(dimensions):
                prompt_template += f"\n{i+1}. {self.all_dimensions[dimension]}"
                
        else:  # inference mode
            prompt_template = f"""作为风控专家，请对以下企业在{dimension_str}等维度进行全面评估。

企业信息：
{company_info}

请对每个维度进行详细分析并给出0-9的评分。"""
        
        return prompt_template
    
    def format_multitask_response(self, 
                                sample_data: Dict[str, Any], 
                                dimensions: List[str]) -> str:
        """格式化多任务响应"""
        response_parts = []
        
        for i, dimension in enumerate(dimensions):
            if dimension in sample_data:
                data = sample_data[dimension]
                think = data.get('think', '')
                score = data.get('score', 0)
                reason = data.get('reason', '')
                
                response_part = f"""{i+1}. {self.all_dimensions[dimension]}评估：

分析过程：{think}

评分：{score}/9

理由：{reason}"""
                response_parts.append(response_part)
        
        return "\n\n".join(response_parts)
    
    def create_curriculum_stages(self, 
                               all_data: Dict[str, List[Dict[str, Any]]], 
                               curriculum_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建课程学习的不同阶段数据"""
        
        stages = []
        
        for stage_config in curriculum_config['stages']:
            stage_name = stage_config['name']
            stage_tasks = stage_config['tasks']
            data_ratio = stage_config['data_ratio']
            
            # 收集该阶段涉及的所有数据
            stage_samples = []
            
            # 找到所有任务的共同样本（基于企业信息匹配）
            common_companies = self._find_common_companies(all_data, stage_tasks)
            
            for company_info in common_companies:
                # 构建多任务样本
                multitask_sample = {
                    'company_info': company_info,
                    'dimensions': {}
                }
                
                # 收集该企业在各个维度的数据
                for task in stage_tasks:
                    if task in all_data:
                        task_data = self._find_company_data(all_data[task], company_info)
                        if task_data:
                            multitask_sample['dimensions'][task] = task_data
                
                if multitask_sample['dimensions']:
                    stage_samples.append(multitask_sample)
            
            # 应用数据比例
            num_samples = int(len(stage_samples) * data_ratio)
            stage_samples = random.sample(stage_samples, min(num_samples, len(stage_samples)))
            
            stages.append({
                'name': stage_name,
                'tasks': stage_tasks,
                'samples': stage_samples,
                'num_samples': len(stage_samples)
            })
            
            print(f"课程学习阶段 {stage_name}: {len(stage_samples)} 个样本, 涉及任务: {stage_tasks}")
        
        return stages
    
    def _find_common_companies(self, 
                             all_data: Dict[str, List[Dict[str, Any]]], 
                             tasks: List[str]) -> List[str]:
        """找到多个任务中的共同企业"""
        if not tasks:
            return []
        
        # 获取第一个任务的所有企业信息
        first_task = tasks[0]
        if first_task not in all_data:
            return []
        
        common_companies = set()
        for sample in all_data[first_task]:
            company_info = sample.get('company_info', '')
            if company_info:
                common_companies.add(company_info)
        
        # 找交集
        for task in tasks[1:]:
            if task in all_data:
                task_companies = set()
                for sample in all_data[task]:
                    company_info = sample.get('company_info', '')
                    if company_info:
                        task_companies.add(company_info)
                common_companies = common_companies.intersection(task_companies)
        
        return list(common_companies)
    
    def _find_company_data(self, 
                          task_data: List[Dict[str, Any]], 
                          company_info: str) -> Optional[Dict[str, Any]]:
        """在任务数据中找到特定企业的数据"""
        for sample in task_data:
            if sample.get('company_info', '') == company_info:
                return sample
        return None
    
    def prepare_training_data(self, 
                            stage_samples: List[Dict[str, Any]], 
                            stage_tasks: List[str]) -> List[Dict[str, Any]]:
        """准备训练数据"""
        training_samples = []
        
        for sample in stage_samples:
            company_info = sample['company_info']
            dimensions_data = sample['dimensions']
            
            # 创建输入prompt
            input_text = self.create_multitask_prompt(company_info, stage_tasks, mode="train")
            
            # 创建目标输出
            output_text = self.format_multitask_response(dimensions_data, stage_tasks)
            
            # 组合完整的训练文本
            full_text = f"{input_text}\n\n{output_text}"
            
            training_samples.append({
                'input_text': input_text,
                'output_text': output_text,
                'full_text': full_text,
                'tasks': stage_tasks,
                'company_info': company_info
            })
        
        return training_samples
    
    def tokenize_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """对单个样本进行tokenize"""
        full_text = sample['full_text']
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 创建labels（用于计算loss）
        labels = tokenized['input_ids'].clone()
        
        # 只对输出部分计算loss
        input_text = sample['input_text']
        input_tokens = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )
        
        input_length = input_tokens['input_ids'].shape[1]
        labels[:, :input_length] = -100  # 忽略输入部分的loss
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }
    
    def create_dataset(self, training_samples: List[Dict[str, Any]]) -> Dataset:
        """创建Hugging Face Dataset"""
        tokenized_samples = []
        
        for sample in training_samples:
            tokenized_sample = self.tokenize_sample(sample)
            tokenized_samples.append(tokenized_sample)
        
        # 转换为Dataset格式
        dataset_dict = {
            'input_ids': [sample['input_ids'] for sample in tokenized_samples],
            'attention_mask': [sample['attention_mask'] for sample in tokenized_samples], 
            'labels': [sample['labels'] for sample in tokenized_samples]
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def create_data_ratio_experiments(self, 
                                   all_data: Dict[str, List[Dict[str, Any]]], 
                                   ratios: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0]) -> Dict[str, Any]:
        """创建不同数据配比的实验数据"""
        experiments = {}
        
        # 使用所有可用的维度
        available_tasks = list(all_data.keys())
        common_companies = self._find_common_companies(all_data, available_tasks)
        
        for ratio in ratios:
            exp_name = f"ratio_{ratio:.1f}"
            
            # 构建该比例下的样本
            num_samples = int(len(common_companies) * ratio)
            selected_companies = random.sample(common_companies, num_samples)
            
            exp_samples = []
            for company_info in selected_companies:
                multitask_sample = {
                    'company_info': company_info,
                    'dimensions': {}
                }
                
                for task in available_tasks:
                    task_data = self._find_company_data(all_data[task], company_info)
                    if task_data:
                        multitask_sample['dimensions'][task] = task_data
                
                if multitask_sample['dimensions']:
                    exp_samples.append(multitask_sample)
            
            experiments[exp_name] = {
                'ratio': ratio,
                'tasks': available_tasks,
                'samples': exp_samples,
                'num_samples': len(exp_samples)
            }
        
        return experiments 