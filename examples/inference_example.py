#!/usr/bin/env python3
"""
模型推理示例
展示如何使用训练好的风控画像模型进行企业评估
"""

import os
import sys
import yaml
import json

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_framework import MultiTaskLoRAModel
from data_processor import RiskProfilingDataProcessor
from evaluation import RiskProfilingEvaluator


def load_config(config_path: str = "../config/training_config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """主函数"""
    
    # 示例企业信息
    sample_companies = [
        {
            "name": "创新科技公司",
            "info": "某人工智能创业公司，成立于2020年，专注于计算机视觉和自然语言处理技术。公司拥有博士团队15人，硕士30人，获得A轮融资5000万元。主要产品包括智能客服系统和图像识别API，已服务500+企业客户，年营收3000万元，同比增长150%。"
        },
        {
            "name": "传统制造企业", 
            "info": "某汽车零部件制造企业，成立于1998年，注册资本2亿元。主要生产发动机零部件和刹车系统，拥有员工1200人，年产值15亿元。客户包括一汽、上汽等主机厂，但受新能源汽车冲击，传统业务增长放缓，正在转型升级。"
        },
        {
            "name": "金融服务机构",
            "info": "某地方性商业银行，成立于2005年，注册资本50亿元。主要业务包括个人银行、企业银行和投资银行。拥有网点200个，员工5000人，资产规模2000亿元。近年来加大金融科技投入，推出手机银行和线上贷款产品。"
        }
    ]
    
    print("🚀 风控画像模型推理示例")
    print("=" * 60)
    
    # 1. 加载配置
    print("📋 加载配置...")
    config = load_config()
    
    # 2. 初始化模型（这里使用基础模型，实际使用时应加载训练好的模型）
    print("🤖 初始化模型...")
    model = MultiTaskLoRAModel(config)
    
    # 注意：在实际使用中，您需要加载训练好的模型：
    # model.load_model("./outputs/stage_3_full_scoring")
    
    # 3. 初始化数据处理器和评估器
    data_processor = RiskProfilingDataProcessor(
        tokenizer=model.tokenizer,
        max_length=config['training']['max_seq_length']
    )
    
    evaluator = RiskProfilingEvaluator(
        tokenizer=model.tokenizer,
        dimensions=data_processor.all_dimensions
    )
    
    # 4. 进行推理
    print("🔍 开始企业评估...")
    
    # 选择要评估的维度
    target_dimensions = ["competitiveness", "innovation", "financial_health", "growth_potential"]
    
    for i, company in enumerate(sample_companies):
        print(f"\n--- 评估企业 {i+1}: {company['name']} ---")
        
        # 创建多任务prompt
        prompt = data_processor.create_multitask_prompt(
            company_info=company['info'],
            dimensions=target_dimensions,
            mode="inference"
        )
        
        print(f"📝 输入prompt:")
        print(f"{prompt[:200]}...")
        
        # 生成评估结果
        print(f"⏳ 生成评估中...")
        
        try:
            response = model.generate_response(
                prompt=prompt,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9
            )
            
            print(f"📊 评估结果:")
            print(response)
            
            # 提取评分
            scores = evaluator.extract_scores_from_response(response)
            
            print(f"\n🎯 提取的评分:")
            for dim_key, score in scores.items():
                dim_name = data_processor.all_dimensions[dim_key]
                print(f"  {dim_name}: {score}/9")
            
            # 评估推理质量
            reasoning_quality = evaluator.extract_reasoning_quality(response)
            avg_quality = sum(reasoning_quality.values()) / len(reasoning_quality) if reasoning_quality else 0
            
            print(f"📈 推理质量评分: {avg_quality:.2f}/1.0")
            
        except Exception as e:
            print(f"❌ 评估失败: {e}")
        
        print("-" * 60)
    
    # 5. 批量评估示例
    print("\n🔄 批量评估示例")
    print("=" * 60)
    
    batch_results = []
    
    for company in sample_companies:
        # 创建prompt
        prompt = data_processor.create_multitask_prompt(
            company_info=company['info'],
            dimensions=target_dimensions,
            mode="inference"
        )
        
        try:
            # 生成回复
            response = model.generate_response(prompt, max_new_tokens=512, temperature=0.5)
            
            # 提取评分
            scores = evaluator.extract_scores_from_response(response)
            
            batch_results.append({
                'company_name': company['name'],
                'scores': scores,
                'response': response
            })
            
        except Exception as e:
            print(f"批量评估 {company['name']} 失败: {e}")
    
    # 6. 结果分析
    print("📈 批量评估结果分析:")
    
    if batch_results:
        # 创建结果表格
        import pandas as pd
        
        score_data = []
        for result in batch_results:
            row = {'企业名称': result['company_name']}
            for dim_key, score in result['scores'].items():
                dim_name = data_processor.all_dimensions[dim_key]
                row[dim_name] = score
            score_data.append(row)
        
        df = pd.DataFrame(score_data)
        print(df.to_string(index=False))
        
        # 保存结果
        output_file = "inference_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细结果已保存到: {output_file}")
    
    # 7. 自定义评估示例
    print("\n🎨 自定义评估示例")
    print("=" * 60)
    
    # 用户可以输入自己的企业信息
    custom_company_info = """
    某新零售电商平台，成立于2019年，专注于生鲜食品配送。
    公司采用"前置仓+即时配送"模式，在一二线城市布局500个前置仓。
    拥有员工2000人，注册用户300万，日订单量10万单。
    年GMV达到50亿元，但由于配送成本高，尚未实现盈利。
    近期获得B轮融资10亿元，计划扩大市场覆盖和提升运营效率。
    """
    
    print("📝 自定义企业信息:")
    print(custom_company_info.strip())
    
    # 可以选择特定维度进行评估
    custom_dimensions = ["competitiveness", "innovation", "operational_efficiency", "financial_health"]
    
    prompt = data_processor.create_multitask_prompt(
        company_info=custom_company_info.strip(),
        dimensions=custom_dimensions,
        mode="inference"
    )
    
    try:
        response = model.generate_response(prompt, max_new_tokens=800, temperature=0.6)
        
        print("🎯 评估结果:")
        print(response)
        
        scores = evaluator.extract_scores_from_response(response)
        
        print("\n📊 评分汇总:")
        for dim_key, score in scores.items():
            dim_name = data_processor.all_dimensions[dim_key]
            print(f"  {dim_name}: {score}/9")
        
    except Exception as e:
        print(f"❌ 自定义评估失败: {e}")
    
    print("\n✅ 推理示例完成！")
    print("\n💡 使用提示:")
    print("1. 在实际使用中，请先加载训练好的模型:")
    print("   model.load_model('./outputs/your_trained_model')")
    print("2. 可以调整generation参数来控制输出质量")
    print("3. 建议对重要评估运行多次取平均值")
    print("4. 可以根据业务需求调整评估维度")


if __name__ == "__main__":
    main() 