# 大模型训练框架

基于LoRA微调Qwen模型的。支持课程学习和多任务学习。


## 🔧 环境配置

### 系统要求
- Python 3.8+
- CUDA 11.8+ (如果使用GPU)
- 8GB+ 显存 (推荐16GB以上)

### 安装依赖
```bash
# 克隆项目
git clone <your-repo-url>
cd lora-sft

# 安装依赖
pip install -r requirements.txt

# 如果使用conda
conda create -n risk-profiling python=3.9
conda activate risk-profiling
pip install -r requirements.txt
```

## 📊 数据格式

### 训练数据格式
每个维度的数据文件应为JSON格式，包含以下字段：

```json
[
  {
    "prompt": "请进行评估",
    "company_info": "详细信息描述...",
    "think": "分析思路和过程...",
    "score": 7,
    "reason": "评分理由和依据..."
  }
]
```

### 数据目录结构
```
data/
├── train/
│   ├── competitiveness.json    
│   ├── innovation.json         
│   ├── diversity.json          
│   └── ...
├── eval/
│   └── ...
└── test/
    └── ...
```

## 🚀 快速开始

### 1. 课程学习训练
```bash
python train.py --mode curriculum --config config/training_config.yaml
```

### 2. 数据配比实验
```bash
python train.py --mode ratio_exp --ratios 0.2 0.4 0.6 0.8 1.0
```

### 3. 模型评估
```bash
python train.py --mode eval --model_path ./outputs/stage_3_full_scoring
```

## ⚙️ 配置说明

### 主要配置文件: `config/training_config.yaml`

#### 模型配置
```yaml
model:
  model_name: "Qwen/Qwen-7B-Chat"  # 基础模型
  cache_dir: "./models"            # 模型缓存目录
  torch_dtype: "bfloat16"          # 数据类型
```

#### LoRA配置
```yaml
lora:
  r: 64                    # LoRA rank
  lora_alpha: 128         # LoRA alpha
  target_modules: [...]   # 目标模块
  lora_dropout: 0.1       # Dropout率
```

#### 课程学习配置
```yaml
curriculum_learning:
  enabled: true
  stages:
    - name: "basic_scoring"
      epochs: 1
      tasks: ["competitiveness", "innovation"]
      data_ratio: 0.3
    - name: "intermediate_scoring"
      epochs: 1
      tasks: ["competitiveness", "innovation", "diversity"]
      data_ratio: 0.6
    - name: "full_scoring"
      epochs: 1
      tasks: ["competitiveness", "innovation", "diversity", "sales_performance", "market_position", "financial_health"]
      data_ratio: 1.0
```

## 📈 训练策略

### 课程学习 (Curriculum Learning)
采用渐进式训练策略：
1. **基础阶段**: 先训练核心维度
2. **中级阶段**: 增加更多维度，扩大数据规模
3. **高级阶段**: 使用全部维度和完整数据

### 多任务学习 (Multi-task Learning)
- 同时在多个评估维度上训练
- 共享底层特征表示
- 提高模型的泛化能力

### 数据配比实验
- 测试不同数据量对模型性能的影响
- 找到最优的数据配比
- 支持成本效益分析

## 🔍 评估指标

### 评分准确性
- **MAE (平均绝对误差)**: 预测评分与真实评分的平均绝对差值
- **RMSE (均方根误差)**: 预测误差的均方根
- **皮尔逊相关系数**: 预测值与真实值的线性相关性
- **分箱准确性**: 按评分区间的分类准确性

### 推理质量
- **逻辑连贯性**: 推理过程的逻辑性
- **专业术语使用**: 专业词汇的使用情况
- **论证充分性**: 推理内容的完整性
- **具体性**: 是否包含具体数据和例子

### 一致性指标
- **维度间相关性**: 不同维度评分的相关性
- **评分方差**: 评分的稳定性
- **评分分布**: 评分的多样性和分布情况

## 📁 项目结构

```
lora-sft/
├── config/
│   └── training_config.yaml    # 训练配置
├── src/
│   ├── data_processor.py       # 数据处理模块
│   ├── model_framework.py      # 模型框架
│   └── evaluation.py           # 评估模块
├── data/                       # 数据目录
├── outputs/                    # 训练输出
├── train.py                    # 主训练脚本
├── requirements.txt            # 依赖列表
└── README.md                   # 项目说明
```

## 💡 高级用法

### 自定义维度
在 `src/data_processor.py` 中添加新的评估维度：

```python
self.all_dimensions = {
    "competitiveness": "竞争力",
    "innovation": "创新性",
    "your_custom_dimension": "自定义维度",  # 添加这里
    # ...
}
```

### 模型推理
```python
from src.model_framework import MultiTaskLoRAModel
from src.data_processor import RiskProfilingDataProcessor

# 加载模型
model = MultiTaskLoRAModel(config)
model.load_model("path/to/trained/model")

# 创建prompt
processor = RiskProfilingDataProcessor(model.tokenizer)
prompt = processor.create_multitask_prompt(
    company_info="信息...", 
    dimensions=["competitiveness", "innovation"],
    mode="inference"
)

# 生成评估
response = model.generate_response(prompt)
print(response)
```

## 🐛 故障排除

### 常见问题

**1. 显存不足**
- 减少 `per_device_train_batch_size`
- 启用 `gradient_checkpointing`
- 使用 `load_in_8bit` 量化

**2. 训练速度慢**
- 增加 `gradient_accumulation_steps`
- 使用更大的batch size
- 启用flash attention

## 📄 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。
