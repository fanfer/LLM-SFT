"""
模型框架模块
实现基于LoRA的Qwen模型微调和多任务学习
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)
import wandb
import os


class MultiTaskLoRAModel:
    """多任务LoRA模型封装"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config['model']
        self.lora_config = config['lora']
        self.training_config = config['training']
        
        # 初始化tokenizer和模型
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        self._setup_model()
    
    def _setup_model(self):
        """设置模型和tokenizer"""
        print("正在加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['model_name'],
            cache_dir=self.model_config.get('cache_dir'),
            trust_remote_code=self.model_config.get('trust_remote_code', True)
        )
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print("正在加载基础模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config['model_name'],
            cache_dir=self.model_config.get('cache_dir'),
            trust_remote_code=self.model_config.get('trust_remote_code', True),
            torch_dtype=getattr(torch, self.model_config.get('torch_dtype', 'float16')),
            device_map=self.model_config.get('device_map', 'auto'),
            load_in_8bit=True,  # 使用8bit量化节省显存
        )
        
        # 设置LoRA配置
        print("配置LoRA...")
        peft_config = LoraConfig(
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['lora_alpha'],
            target_modules=self.lora_config['target_modules'],
            lora_dropout=self.lora_config['lora_dropout'],
            bias=self.lora_config['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        
        # 应用LoRA
        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()
    
    def get_training_arguments(self, output_dir: str) -> TrainingArguments:
        """获取训练参数"""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.training_config['num_train_epochs'],
            per_device_train_batch_size=self.training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
            warmup_ratio=self.training_config['warmup_ratio'],
            lr_scheduler_type=self.training_config['lr_scheduler_type'],
            logging_steps=self.training_config['logging_steps'],
            eval_steps=self.training_config.get('eval_steps', 500),
            save_steps=self.training_config['save_steps'],
            save_total_limit=self.training_config['save_total_limit'],
            fp16=self.training_config.get('fp16', False),
            bf16=self.training_config.get('bf16', True),
            gradient_checkpointing=self.training_config.get('gradient_checkpointing', True),
            dataloader_pin_memory=self.training_config.get('dataloader_pin_memory', False),
            remove_unused_columns=self.training_config.get('remove_unused_columns', False),
            report_to="wandb",
            logging_dir=f"{output_dir}/logs",
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    
    def create_trainer(self, 
                      train_dataset, 
                      eval_dataset, 
                      output_dir: str) -> Trainer:
        """创建Trainer"""
        
        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.peft_model,
            padding=True,
            return_tensors="pt"
        )
        
        # 训练参数
        training_args = self.get_training_arguments(output_dir)
        
        # 创建Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        return trainer
    
    def save_model(self, output_dir: str):
        """保存模型"""
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"模型已保存到: {output_dir}")
    
    def load_model(self, model_path: str):
        """加载已训练的模型"""
        self.peft_model = PeftModel.from_pretrained(
            self.model, 
            model_path,
            torch_dtype=getattr(torch, self.model_config.get('torch_dtype', 'float16'))
        )
        print(f"模型已从 {model_path} 加载")
    
    def generate_response(self, 
                         prompt: str, 
                         max_new_tokens: int = 512,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> str:
        """生成回复"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.peft_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码输出
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除输入部分，只返回生成的内容
        generated_part = full_response[len(prompt):].strip()
        return generated_part


class CurriculumLearningTrainer:
    """课程学习训练器"""
    
    def __init__(self, model_framework: MultiTaskLoRAModel, config: Dict[str, Any]):
        self.model_framework = model_framework
        self.config = config
        self.curriculum_config = config['curriculum_learning']
        
    def train_with_curriculum(self, 
                            curriculum_stages: List[Dict[str, Any]], 
                            data_processor) -> Dict[str, Any]:
        """使用课程学习进行训练"""
        
        training_history = []
        
        for stage_idx, stage in enumerate(curriculum_stages):
            stage_name = stage['name']
            stage_tasks = stage['tasks']
            stage_samples = stage['samples']
            
            print(f"\n开始课程学习阶段 {stage_idx + 1}: {stage_name}")
            print(f"任务: {stage_tasks}")
            print(f"样本数量: {len(stage_samples)}")
            
            # 准备该阶段的训练数据
            training_samples = data_processor.prepare_training_data(stage_samples, stage_tasks)
            
            # 划分训练和验证数据
            split_idx = int(len(training_samples) * 0.9)
            train_samples = training_samples[:split_idx]
            eval_samples = training_samples[split_idx:]
            
            # 创建数据集
            train_dataset = data_processor.create_dataset(train_samples)
            eval_dataset = data_processor.create_dataset(eval_samples)
            
            print(f"训练样本: {len(train_samples)}, 验证样本: {len(eval_samples)}")
            
            # 设置该阶段的输出目录
            stage_output_dir = os.path.join(
                self.config['training']['output_dir'], 
                f"stage_{stage_idx + 1}_{stage_name}"
            )
            
            # 初始化wandb run
            wandb.init(
                project=self.config['experiment']['wandb_project'],
                name=f"{self.config['experiment']['name']}_stage_{stage_idx + 1}",
                config={
                    'stage': stage_name,
                    'tasks': stage_tasks,
                    'num_samples': len(stage_samples),
                    **self.config
                },
                reinit=True
            )
            
            # 创建trainer并训练
            trainer = self.model_framework.create_trainer(
                train_dataset, 
                eval_dataset, 
                stage_output_dir
            )
            
            # 训练
            train_result = trainer.train()
            
            # 保存该阶段的模型
            self.model_framework.save_model(stage_output_dir)
            
            # 记录训练历史
            stage_history = {
                'stage_name': stage_name,
                'stage_tasks': stage_tasks,
                'num_samples': len(stage_samples),
                'train_loss': train_result.training_loss,
                'output_dir': stage_output_dir
            }
            training_history.append(stage_history)
            
            wandb.finish()
            
            print(f"阶段 {stage_name} 训练完成，训练损失: {train_result.training_loss:.4f}")
        
        return {
            'stages': training_history,
            'total_stages': len(curriculum_stages)
        }


class DataRatioExperiment:
    """数据配比实验"""
    
    def __init__(self, model_framework: MultiTaskLoRAModel, config: Dict[str, Any]):
        self.model_framework = model_framework
        self.config = config
        
    def run_ratio_experiments(self, 
                            ratio_experiments: Dict[str, Any], 
                            data_processor) -> Dict[str, Any]:
        """运行不同数据配比的实验"""
        
        experiment_results = {}
        
        for exp_name, exp_data in ratio_experiments.items():
            ratio = exp_data['ratio']
            tasks = exp_data['tasks']
            samples = exp_data['samples']
            
            print(f"\n开始数据配比实验: {exp_name} (比例: {ratio})")
            print(f"任务: {tasks}")
            print(f"样本数量: {len(samples)}")
            
            # 准备训练数据
            training_samples = data_processor.prepare_training_data(samples, tasks)
            
            # 划分数据
            split_idx = int(len(training_samples) * 0.9)
            train_samples = training_samples[:split_idx]
            eval_samples = training_samples[split_idx:]
            
            # 创建数据集
            train_dataset = data_processor.create_dataset(train_samples)
            eval_dataset = data_processor.create_dataset(eval_samples)
            
            # 设置输出目录
            exp_output_dir = os.path.join(
                self.config['training']['output_dir'], 
                f"ratio_experiment_{exp_name}"
            )
            
            # 初始化wandb
            wandb.init(
                project=self.config['experiment']['wandb_project'],
                name=f"{self.config['experiment']['name']}_{exp_name}",
                config={
                    'experiment': exp_name,
                    'ratio': ratio,
                    'tasks': tasks,
                    'num_samples': len(samples),
                    **self.config
                },
                reinit=True
            )
            
            # 训练
            trainer = self.model_framework.create_trainer(
                train_dataset, 
                eval_dataset, 
                exp_output_dir
            )
            
            train_result = trainer.train()
            
            # 保存模型
            self.model_framework.save_model(exp_output_dir)
            
            # 记录结果
            experiment_results[exp_name] = {
                'ratio': ratio,
                'tasks': tasks,
                'num_samples': len(samples),
                'train_loss': train_result.training_loss,
                'output_dir': exp_output_dir
            }
            
            wandb.finish()
            
            print(f"实验 {exp_name} 完成，训练损失: {train_result.training_loss:.4f}")
        
        return experiment_results 