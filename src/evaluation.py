"""
评估模块
实现多维度风控画像模型的评估指标和分析工具
"""

import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer


class RiskProfilingEvaluator:
    """风控画像模型评估器"""
    
    def __init__(self, tokenizer: AutoTokenizer, dimensions: Dict[str, str]):
        self.tokenizer = tokenizer
        self.dimensions = dimensions
        self.dimension_names = list(dimensions.keys())
        
    def extract_scores_from_response(self, response: str) -> Dict[str, float]:
        """从模型回复中提取各维度的评分"""
        scores = {}
        
        # 定义评分提取的正则表达式模式
        patterns = [
            r'评分[：:]\s*(\d+(?:\.\d+)?)[/／]?(?:9|10)?',
            r'得分[：:]\s*(\d+(?:\.\d+)?)[/／]?(?:9|10)?',
            r'分数[：:]\s*(\d+(?:\.\d+)?)[/／]?(?:9|10)?',
            r'(\d+(?:\.\d+)?)\s*[/／]\s*9',
            r'(\d+(?:\.\d+)?)\s*分',
        ]
        
        # 按维度提取评分
        for dimension_key, dimension_name in self.dimensions.items():
            dimension_score = None
            
            # 查找包含该维度名称的段落
            dimension_section = self._find_dimension_section(response, dimension_name)
            
            if dimension_section:
                # 在该段落中查找评分
                for pattern in patterns:
                    matches = re.findall(pattern, dimension_section)
                    if matches:
                        try:
                            score = float(matches[0])
                            if 0 <= score <= 9:  # 确保分数在有效范围内
                                dimension_score = score
                                break
                        except (ValueError, IndexError):
                            continue
            
            # 如果没有找到，尝试在整个回复中查找
            if dimension_score is None:
                for pattern in patterns:
                    matches = re.findall(pattern, response)
                    if matches:
                        try:
                            score = float(matches[0])
                            if 0 <= score <= 9:
                                dimension_score = score
                                break
                        except (ValueError, IndexError):
                            continue
            
            scores[dimension_key] = dimension_score if dimension_score is not None else 0.0
        
        return scores
    
    def _find_dimension_section(self, response: str, dimension_name: str) -> str:
        """查找特定维度的评估段落"""
        lines = response.split('\n')
        section_lines = []
        in_section = False
        
        for line in lines:
            if dimension_name in line:
                in_section = True
                section_lines.append(line)
            elif in_section:
                if any(other_dim in line for other_dim in self.dimensions.values() if other_dim != dimension_name):
                    break
                section_lines.append(line)
        
        return '\n'.join(section_lines)
    
    def extract_reasoning_quality(self, response: str) -> Dict[str, float]:
        """评估推理质量"""
        quality_scores = {}
        
        for dimension_key, dimension_name in self.dimensions.items():
            dimension_section = self._find_dimension_section(response, dimension_name)
            
            # 评估推理质量的多个维度
            reasoning_score = self._evaluate_reasoning_section(dimension_section)
            quality_scores[dimension_key] = reasoning_score
        
        return quality_scores
    
    def _evaluate_reasoning_section(self, section: str) -> float:
        """评估单个维度的推理质量"""
        if not section:
            return 0.0
        
        quality_indicators = {
            '逻辑连贯性': 0.0,
            '专业术语使用': 0.0,
            '论证充分性': 0.0,
            '具体性': 0.0
        }
        
        # 逻辑连贯性：检查逻辑连接词
        logic_words = ['因此', '所以', '由于', '因为', '导致', '说明', '表明', '体现']
        logic_count = sum(1 for word in logic_words if word in section)
        quality_indicators['逻辑连贯性'] = min(logic_count / 3.0, 1.0)
        
        # 专业术语使用：检查风控相关术语
        professional_terms = ['风险', '合规', '财务', '市场', '竞争力', '创新', '营收', '利润', '增长', '管理']
        term_count = sum(1 for term in professional_terms if term in section)
        quality_indicators['专业术语使用'] = min(term_count / 5.0, 1.0)
        
        # 论证充分性：检查字符长度和结构
        content_length = len(section)
        quality_indicators['论证充分性'] = min(content_length / 200.0, 1.0)
        
        # 具体性：检查是否包含具体数据或例子
        specific_indicators = ['%', '元', '万', '亿', '倍', '增长', '下降', '提升']
        specific_count = sum(1 for indicator in specific_indicators if indicator in section)
        quality_indicators['具体性'] = min(specific_count / 3.0, 1.0)
        
        # 综合评分
        return sum(quality_indicators.values()) / len(quality_indicators)
    
    def calculate_score_metrics(self, 
                              predicted_scores: List[Dict[str, float]], 
                              ground_truth_scores: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """计算评分准确性指标"""
        metrics = {}
        
        for dimension in self.dimension_names:
            pred_values = [scores.get(dimension, 0.0) for scores in predicted_scores]
            true_values = [scores.get(dimension, 0.0) for scores in ground_truth_scores]
            
            # 计算各种指标
            mae = mean_absolute_error(true_values, pred_values)
            mse = mean_squared_error(true_values, pred_values)
            rmse = np.sqrt(mse)
            
            # 相关性分析
            pearson_corr, pearson_p = pearsonr(true_values, pred_values)
            spearman_corr, spearman_p = spearmanr(true_values, pred_values)
            
            # 分箱准确性（按分数段分类的准确性）
            bin_accuracy = self._calculate_bin_accuracy(true_values, pred_values)
            
            # 平均绝对百分比误差
            mape = np.mean(np.abs((np.array(true_values) - np.array(pred_values)) / 
                                 (np.array(true_values) + 1e-8))) * 100
            
            metrics[dimension] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'bin_accuracy': bin_accuracy,
                'mape': mape
            }
        
        return metrics
    
    def _calculate_bin_accuracy(self, true_values: List[float], pred_values: List[float]) -> float:
        """计算分箱准确性"""
        bins = [0, 2, 4, 6, 8, 10]  # 0-1, 2-3, 4-5, 6-7, 8-9 分档
        
        true_bins = np.digitize(true_values, bins) - 1
        pred_bins = np.digitize(pred_values, bins) - 1
        
        return np.mean(true_bins == pred_bins)
    
    def calculate_consistency_metrics(self, 
                                   predicted_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """计算多维度评分的一致性指标"""
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame(predicted_scores)
        
        # 计算维度间相关性
        correlation_matrix = df.corr()
        
        # 计算评分方差（衡量评分的稳定性）
        score_variance = df.var(axis=1).mean()
        
        # 计算评分范围（衡量模型是否充分利用评分区间）
        score_ranges = {}
        for dimension in self.dimension_names:
            if dimension in df.columns:
                scores = df[dimension].dropna()
                score_ranges[dimension] = scores.max() - scores.min()
        
        avg_score_range = np.mean(list(score_ranges.values()))
        
        # 计算评分分布的熵（衡量评分多样性）
        entropy_scores = {}
        for dimension in self.dimension_names:
            if dimension in df.columns:
                scores = df[dimension].dropna()
                hist, _ = np.histogram(scores, bins=10, range=(0, 9))
                prob = hist / hist.sum()
                prob = prob[prob > 0]  # 移除零概率
                entropy = -np.sum(prob * np.log2(prob))
                entropy_scores[dimension] = entropy
        
        avg_entropy = np.mean(list(entropy_scores.values()))
        
        return {
            'inter_dimension_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
            'score_variance': score_variance,
            'average_score_range': avg_score_range,
            'average_entropy': avg_entropy,
            'dimension_correlations': correlation_matrix.to_dict(),
            'score_ranges': score_ranges,
            'entropy_scores': entropy_scores
        }
    
    def generate_evaluation_report(self, 
                                 predicted_scores: List[Dict[str, float]], 
                                 ground_truth_scores: List[Dict[str, float]], 
                                 predicted_reasoning: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """生成完整的评估报告"""
        
        # 计算评分指标
        score_metrics = self.calculate_score_metrics(predicted_scores, ground_truth_scores)
        
        # 计算一致性指标
        consistency_metrics = self.calculate_consistency_metrics(predicted_scores)
        
        # 计算推理质量（如果提供）
        reasoning_quality = None
        if predicted_reasoning:
            reasoning_quality = {}
            for dimension in self.dimension_names:
                dimension_reasoning = [r.get(dimension, '') for r in predicted_reasoning]
                quality_scores = [self._evaluate_reasoning_section(reasoning) for reasoning in dimension_reasoning]
                reasoning_quality[dimension] = {
                    'average_quality': np.mean(quality_scores),
                    'quality_std': np.std(quality_scores),
                    'quality_scores': quality_scores
                }
        
        # 整体性能总结
        overall_mae = np.mean([metrics['mae'] for metrics in score_metrics.values()])
        overall_pearson = np.mean([metrics['pearson_correlation'] for metrics in score_metrics.values()])
        overall_bin_accuracy = np.mean([metrics['bin_accuracy'] for metrics in score_metrics.values()])
        
        report = {
            'overall_performance': {
                'mean_absolute_error': overall_mae,
                'pearson_correlation': overall_pearson,
                'bin_accuracy': overall_bin_accuracy,
                'num_samples': len(predicted_scores)
            },
            'dimension_metrics': score_metrics,
            'consistency_metrics': consistency_metrics,
            'reasoning_quality': reasoning_quality,
            'summary': {
                'best_dimension': min(score_metrics.keys(), key=lambda x: score_metrics[x]['mae']),
                'worst_dimension': max(score_metrics.keys(), key=lambda x: score_metrics[x]['mae']),
                'most_consistent_scores': min(score_metrics.keys(), key=lambda x: score_metrics[x]['rmse']),
                'highest_correlation': max(score_metrics.keys(), key=lambda x: score_metrics[x]['pearson_correlation'])
            }
        }
        
        return report
    
    def visualize_results(self, 
                         predicted_scores: List[Dict[str, float]], 
                         ground_truth_scores: List[Dict[str, float]], 
                         save_path: str = None) -> None:
        """可视化评估结果"""
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Risk Profiling Model Evaluation Results', fontsize=16)
        
        # 转换数据格式
        pred_df = pd.DataFrame(predicted_scores)
        true_df = pd.DataFrame(ground_truth_scores)
        
        # 1. 各维度MAE对比
        maes = []
        dims = []
        for dimension in self.dimension_names:
            if dimension in pred_df.columns and dimension in true_df.columns:
                mae = mean_absolute_error(true_df[dimension], pred_df[dimension])
                maes.append(mae)
                dims.append(self.dimensions[dimension])
        
        axes[0, 0].bar(dims, maes)
        axes[0, 0].set_title('Mean Absolute Error by Dimension')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 预测值vs真实值散点图
        all_pred = []
        all_true = []
        for dimension in self.dimension_names:
            if dimension in pred_df.columns and dimension in true_df.columns:
                all_pred.extend(pred_df[dimension].tolist())
                all_true.extend(true_df[dimension].tolist())
        
        axes[0, 1].scatter(all_true, all_pred, alpha=0.6)
        axes[0, 1].plot([0, 9], [0, 9], 'r--', lw=2)
        axes[0, 1].set_xlabel('True Scores')
        axes[0, 1].set_ylabel('Predicted Scores')
        axes[0, 1].set_title('Predicted vs True Scores')
        
        # 3. 相关性热力图
        correlations = []
        for dimension in self.dimension_names:
            if dimension in pred_df.columns and dimension in true_df.columns:
                corr, _ = pearsonr(true_df[dimension], pred_df[dimension])
                correlations.append(corr)
        
        corr_data = pd.DataFrame({
            'Dimension': [self.dimensions[d] for d in self.dimension_names if d in pred_df.columns],
            'Correlation': correlations
        })
        
        sns.barplot(data=corr_data, x='Dimension', y='Correlation', ax=axes[0, 2])
        axes[0, 2].set_title('Pearson Correlation by Dimension')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. 误差分布
        errors = np.array(all_pred) - np.array(all_true)
        axes[1, 0].hist(errors, bins=20, alpha=0.7)
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        
        # 5. 分数分布对比
        axes[1, 1].hist(all_true, bins=10, alpha=0.7, label='True', range=(0, 9))
        axes[1, 1].hist(all_pred, bins=10, alpha=0.7, label='Predicted', range=(0, 9))
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Score Distribution Comparison')
        axes[1, 1].legend()
        
        # 6. 维度间相关性
        if len(pred_df.columns) > 1:
            corr_matrix = pred_df.corr()
            sns.heatmap(corr_matrix, annot=True, ax=axes[1, 2], cmap='coolwarm', center=0)
            axes[1, 2].set_title('Inter-dimension Correlation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_models(self, 
                      model_results: Dict[str, Dict[str, Any]], 
                      save_path: str = None) -> pd.DataFrame:
        """比较多个模型的性能"""
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {'Model': model_name}
            
            # 添加整体性能指标
            if 'overall_performance' in results:
                row.update({
                    'Overall_MAE': results['overall_performance']['mean_absolute_error'],
                    'Overall_Correlation': results['overall_performance']['pearson_correlation'],
                    'Overall_Bin_Accuracy': results['overall_performance']['bin_accuracy']
                })
            
            # 添加各维度的MAE
            if 'dimension_metrics' in results:
                for dimension in self.dimension_names:
                    if dimension in results['dimension_metrics']:
                        row[f'{dimension}_MAE'] = results['dimension_metrics'][dimension]['mae']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if save_path:
            comparison_df.to_csv(save_path, index=False)
        
        return comparison_df 