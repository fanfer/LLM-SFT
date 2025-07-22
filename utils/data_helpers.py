"""
数据处理辅助工具
提供数据验证、格式转换、数据增强等功能
"""

import json
import os
import re
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path


class DataValidator:
    """数据验证器"""
    
    def __init__(self):
        self.required_fields = ['prompt', 'company_info', 'think', 'score', 'reason']
        self.score_range = (0, 9)
    
    def validate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证单个样本"""
        errors = []
        
        # 检查必需字段
        for field in self.required_fields:
            if field not in sample:
                errors.append(f"缺少必需字段: {field}")
            elif not sample[field]:
                errors.append(f"字段为空: {field}")
        
        # 验证评分
        if 'score' in sample:
            score = sample['score']
            if not isinstance(score, (int, float)):
                errors.append(f"评分类型错误: {type(score)}, 应为数字")
            elif not (self.score_range[0] <= score <= self.score_range[1]):
                errors.append(f"评分超出范围: {score}, 应在 {self.score_range} 之间")
        
        # 验证文本长度
        text_fields = ['company_info', 'think', 'reason']
        for field in text_fields:
            if field in sample and len(sample[field]) < 10:
                errors.append(f"字段 {field} 内容过短: {len(sample[field])} 字符")
        
        return len(errors) == 0, errors
    
    def validate_dataset(self, data: List[Dict[str, Any]], dimension: str = None) -> Dict[str, Any]:
        """验证整个数据集"""
        results = {
            'total_samples': len(data),
            'valid_samples': 0,
            'invalid_samples': 0,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        valid_samples = []
        all_scores = []
        text_lengths = []
        
        for i, sample in enumerate(data):
            is_valid, errors = self.validate_sample(sample)
            
            if is_valid:
                results['valid_samples'] += 1
                valid_samples.append(sample)
                if 'score' in sample:
                    all_scores.append(sample['score'])
                if 'company_info' in sample:
                    text_lengths.append(len(sample['company_info']))
            else:
                results['invalid_samples'] += 1
                results['errors'].append({
                    'sample_index': i,
                    'errors': errors
                })
        
        # 生成统计信息
        if all_scores:
            results['statistics']['score_distribution'] = {
                'mean': np.mean(all_scores),
                'std': np.std(all_scores),
                'min': min(all_scores),
                'max': max(all_scores),
                'unique_scores': list(set(all_scores))
            }
        
        if text_lengths:
            results['statistics']['text_length'] = {
                'mean': np.mean(text_lengths),
                'std': np.std(text_lengths),
                'min': min(text_lengths),
                'max': max(text_lengths)
            }
        
        # 检查数据平衡性
        if all_scores:
            score_counts = pd.Series(all_scores).value_counts()
            if score_counts.std() > score_counts.mean() * 0.5:
                results['warnings'].append("数据分布不平衡，某些评分的样本数量过少")
        
        if dimension:
            results['dimension'] = dimension
        
        return results 