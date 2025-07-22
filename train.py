#!/usr/bin/env python3
"""
风控画像大模型训练主脚本
支持课程学习和数据配比实验
"""

import os
import sys
import yaml
import json
import random
import argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import RiskProfilingDataProcessor
from model_framework import MultiTaskLoRAModel, CurriculumLearningTrainer, DataRatioExperiment
from evaluation import RiskProfilingEvaluator


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_sample_data(output_dir: str):
    """创建示例数据（用于演示）"""
    
    # 创建数据目录
    data_dir = os.path.join(output_dir, 'data', 'train')
    os.makedirs(data_dir, exist_ok=True)
    
    # 示例企业信息
    companies = [
        "A科技公司是一家专注于人工智能技术研发的高新技术企业，成立于2018年，注册资本5000万元，主要产品包括智能语音识别系统、计算机视觉解决方案等。公司拥有研发人员120人，年营收2亿元，近三年年均增长率30%。",
        "B制造公司是传统制造业企业，成立于1995年，注册资本1亿元，主要生产汽车零部件，拥有员工800人，年营收8亿元，近年来受行业下行影响，营收增长放缓。",
        "C服务公司是一家金融服务企业，成立于2010年，注册资本3000万元，主要提供投资咨询和资产管理服务，员工200人，年营收1.5亿元，客户主要为高净值个人和机构投资者。"
    ]
    
    # 生成各维度的示例数据
    dimensions = {
        "competitiveness": "产品竞争力",
        "innovation": "产品创新性"
    }
    
    for dim_key, dim_name in dimensions.items():
        samples = []
        
        for i, company_info in enumerate(companies):
            # 生成随机评分和理由
            score = random.randint(3, 8)
            
            think = f"分析该企业的{dim_name}，需要考虑技术实力、市场地位、产品质量等因素。"
            reason = f"基于企业的技术水平、市场表现和发展前景，给出{dim_name}评分。"
            
            sample = {
                "prompt": f"请对以下企业的{dim_name}进行评估",
                "company_info": company_info,
                "think": think,
                "score": score,
                "reason": reason
            }
            samples.append(sample)
        
        # 保存到文件
        file_path = os.path.join(data_dir, f"{dim_key}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"创建示例数据: {file_path}")


def run_curriculum_learning(config: dict, args):
    """运行课程学习训练"""
    print("=" * 60)
    print("开始课程学习训练")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(config['experiment']['seed'])
    
    # 初始化模型框架
    print("初始化模型框架...")
    model_framework = MultiTaskLoRAModel(config)
    
    # 初始化数据处理器
    print("初始化数据处理器...")
    data_processor = RiskProfilingDataProcessor(
        tokenizer=model_framework.tokenizer,
        max_length=config['training']['max_seq_length']
    )
    
    # 加载训练数据
    print("加载训练数据...")
    all_data = data_processor.load_all_data(config['data']['train_data_path'])
    
    if not all_data:
        print("未找到训练数据，创建示例数据...")
        create_sample_data(".")
        all_data = data_processor.load_all_data(config['data']['train_data_path'])
    
    # 创建课程学习阶段
    print("创建课程学习阶段...")
    curriculum_stages = data_processor.create_curriculum_stages(
        all_data, 
        config['curriculum_learning']
    )
    
    # 初始化课程学习训练器
    curriculum_trainer = CurriculumLearningTrainer(model_framework, config)
    
    # 开始训练
    training_results = curriculum_trainer.train_with_curriculum(
        curriculum_stages, 
        data_processor
    )
    
    print("课程学习训练完成！")
    print(f"总共完成 {training_results['total_stages']} 个阶段")
    
    return training_results


def run_ratio_experiments(config: dict, args):
    """运行数据配比实验"""
    print("=" * 60)
    print("开始数据配比实验")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(config['experiment']['seed'])
    
    # 初始化模型框架
    model_framework = MultiTaskLoRAModel(config)
    
    # 初始化数据处理器
    data_processor = RiskProfilingDataProcessor(
        tokenizer=model_framework.tokenizer,
        max_length=config['training']['max_seq_length']
    )
    
    # 加载数据
    all_data = data_processor.load_all_data(config['data']['train_data_path'])
    
    if not all_data:
        print("未找到训练数据，创建示例数据...")
        create_sample_data(".")
        all_data = data_processor.load_all_data(config['data']['train_data_path'])
    
    # 创建不同数据配比的实验
    ratios = args.ratios if args.ratios else [0.2, 0.4, 0.6, 0.8, 1.0]
    ratio_experiments = data_processor.create_data_ratio_experiments(all_data, ratios)
    
    # 运行实验
    ratio_experiment_runner = DataRatioExperiment(model_framework, config)
    experiment_results = ratio_experiment_runner.run_ratio_experiments(
        ratio_experiments, 
        data_processor
    )
    
    print("数据配比实验完成！")
    print(f"完成 {len(experiment_results)} 个实验")
    
    return experiment_results


def run_evaluation(config: dict, args):
    """运行模型评估"""
    print("=" * 60)
    print("开始模型评估")
    print("=" * 60)
    
    if not args.model_path:
        print("请指定要评估的模型路径 (--model_path)")
        return
    
    # 初始化模型框架
    model_framework = MultiTaskLoRAModel(config)
    
    # 加载训练好的模型
    print(f"加载模型: {args.model_path}")
    model_framework.load_model(args.model_path)
    
    # 初始化数据处理器
    data_processor = RiskProfilingDataProcessor(
        tokenizer=model_framework.tokenizer,
        max_length=config['training']['max_seq_length']
    )
    
    # 加载测试数据
    test_data_path = config['data'].get('test_data_path', config['data']['train_data_path'])
    all_data = data_processor.load_all_data(test_data_path)
    
    # 初始化评估器
    evaluator = RiskProfilingEvaluator(
        tokenizer=model_framework.tokenizer,
        dimensions=data_processor.all_dimensions
    )
    
    # 准备评估数据
    eval_samples = []
    for dimension, samples in all_data.items():
        eval_samples.extend(samples[:10])  # 每个维度取10个样本进行评估
    
    # 运行推理
    predicted_scores = []
    ground_truth_scores = []
    
    for sample in eval_samples:
        company_info = sample['company_info']
        available_dims = [dim for dim in data_processor.all_dimensions.keys() if dim in all_data]
        
        # 创建prompt
        prompt = data_processor.create_multitask_prompt(
            company_info, 
            available_dims, 
            mode="inference"
        )
        
        # 生成回复
        response = model_framework.generate_response(prompt)
        
        # 提取预测评分
        pred_scores = evaluator.extract_scores_from_response(response)
        predicted_scores.append(pred_scores)
        
        # 提取真实评分
        true_scores = {}
        for dim in available_dims:
            if dim in all_data:
                dim_data = data_processor._find_company_data(all_data[dim], company_info)
                if dim_data:
                    true_scores[dim] = dim_data.get('score', 0)
        ground_truth_scores.append(true_scores)
    
    # 生成评估报告
    evaluation_report = evaluator.generate_evaluation_report(
        predicted_scores, 
        ground_truth_scores
    )
    
    # 保存评估结果
    report_path = os.path.join(args.model_path, "evaluation_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
    
    print(f"评估报告已保存到: {report_path}")
    print(f"整体MAE: {evaluation_report['overall_performance']['mean_absolute_error']:.4f}")
    print(f"整体相关性: {evaluation_report['overall_performance']['pearson_correlation']:.4f}")
    
    return evaluation_report


def main():
    parser = argparse.ArgumentParser(description='风控画像大模型训练')
    parser.add_argument('--config', default='config/training_config.yaml', help='配置文件路径')
    parser.add_argument('--mode', choices=['curriculum', 'ratio_exp', 'eval'], 
                       default='curriculum', help='运行模式')
    parser.add_argument('--model_path', help='用于评估的模型路径')
    parser.add_argument('--ratios', nargs='+', type=float, 
                       help='数据配比实验的比例列表')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    # 根据模式运行不同的任务
    if args.mode == 'curriculum':
        results = run_curriculum_learning(config, args)
    elif args.mode == 'ratio_exp':
        results = run_ratio_experiments(config, args)
    elif args.mode == 'eval':
        results = run_evaluation(config, args)
    else:
        print(f"未知的运行模式: {args.mode}")
        return
    
    print("训练任务完成！")


if __name__ == "__main__":
    main() 