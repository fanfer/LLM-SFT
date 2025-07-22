#!/bin/bash

# 风控画像大模型实验脚本
# 用于批量运行不同配置的训练和评估实验

set -e

# 配置变量
CONFIG_FILE="config/training_config.yaml"
OUTPUT_BASE="./outputs"
EXPERIMENT_NAME="risk_profiling_$(date +%Y%m%d_%H%M%S)"

# 创建实验目录
EXPERIMENT_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}"
mkdir -p "${EXPERIMENT_DIR}"

echo "=========================================="
echo "开始风控画像模型实验"
echo "实验目录: ${EXPERIMENT_DIR}"
echo "时间: $(date)"
echo "=========================================="

# 函数：记录日志
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${EXPERIMENT_DIR}/experiment.log"
}

# 函数：运行课程学习训练
run_curriculum_learning() {
    log "开始课程学习训练..."
    
    python train.py \
        --mode curriculum \
        --config "${CONFIG_FILE}" \
        2>&1 | tee "${EXPERIMENT_DIR}/curriculum_training.log"
    
    if [ $? -eq 0 ]; then
        log "课程学习训练完成"
    else
        log "课程学习训练失败"
        exit 1
    fi
}

# 函数：运行数据配比实验
run_ratio_experiments() {
    log "开始数据配比实验..."
    
    # 定义不同的数据配比
    RATIOS=(0.2 0.4 0.6 0.8 1.0)
    
    python train.py \
        --mode ratio_exp \
        --config "${CONFIG_FILE}" \
        --ratios "${RATIOS[@]}" \
        2>&1 | tee "${EXPERIMENT_DIR}/ratio_experiments.log"
    
    if [ $? -eq 0 ]; then
        log "数据配比实验完成"
    else
        log "数据配比实验失败"
        exit 1
    fi
}

# 函数：运行模型评估
run_model_evaluation() {
    local model_path=$1
    local eval_name=$2
    
    log "开始模型评估: ${eval_name}"
    
    if [ ! -d "${model_path}" ]; then
        log "模型路径不存在: ${model_path}"
        return 1
    fi
    
    python train.py \
        --mode eval \
        --config "${CONFIG_FILE}" \
        --model_path "${model_path}" \
        2>&1 | tee "${EXPERIMENT_DIR}/evaluation_${eval_name}.log"
    
    if [ $? -eq 0 ]; then
        log "模型评估完成: ${eval_name}"
    else
        log "模型评估失败: ${eval_name}"
        return 1
    fi
}

# 函数：生成实验报告
generate_experiment_report() {
    log "生成实验报告..."
    
    REPORT_FILE="${EXPERIMENT_DIR}/experiment_report.md"
    
    cat > "${REPORT_FILE}" << EOF
# 风控画像模型实验报告

## 实验信息
- 实验名称: ${EXPERIMENT_NAME}
- 开始时间: $(date)
- 配置文件: ${CONFIG_FILE}
- 输出目录: ${EXPERIMENT_DIR}

## 实验内容

### 1. 课程学习训练
$([ -f "${EXPERIMENT_DIR}/curriculum_training.log" ] && echo "✅ 已完成" || echo "❌ 未完成")

### 2. 数据配比实验
$([ -f "${EXPERIMENT_DIR}/ratio_experiments.log" ] && echo "✅ 已完成" || echo "❌ 未完成")

### 3. 模型评估
$(find "${EXPERIMENT_DIR}" -name "evaluation_*.log" | wc -l) 个评估任务完成

## 实验结果

### 训练损失变化
请查看 wandb 监控页面或tensorboard日志

### 评估指标
请查看各个模型目录下的 evaluation_report.json

## 文件说明
- experiment.log: 总体实验日志
- curriculum_training.log: 课程学习训练日志
- ratio_experiments.log: 数据配比实验日志
- evaluation_*.log: 各模型评估日志

EOF

    log "实验报告已生成: ${REPORT_FILE}"
}

# 函数：清理临时文件
cleanup() {
    log "清理临时文件..."
    # 这里可以添加清理逻辑
}

# 主实验流程
main() {
    log "检查环境..."
    
    # 检查Python环境
    if ! command -v python &> /dev/null; then
        log "错误: 未找到Python环境"
        exit 1
    fi
    
    # 检查配置文件
    if [ ! -f "${CONFIG_FILE}" ]; then
        log "错误: 配置文件不存在: ${CONFIG_FILE}"
        exit 1
    fi
    
    # 检查GPU
    if command -v nvidia-smi &> /dev/null; then
        log "GPU信息:"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | tee -a "${EXPERIMENT_DIR}/experiment.log"
    else
        log "警告: 未检测到GPU"
    fi
    
    # 解析命令行参数
    CURRICULUM_ONLY=false
    RATIO_ONLY=false
    EVAL_ONLY=false
    SKIP_TRAINING=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --curriculum-only)
                CURRICULUM_ONLY=true
                shift
                ;;
            --ratio-only)
                RATIO_ONLY=true
                shift
                ;;
            --eval-only)
                EVAL_ONLY=true
                shift
                ;;
            --skip-training)
                SKIP_TRAINING=true
                shift
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -h|--help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --curriculum-only    仅运行课程学习训练"
                echo "  --ratio-only         仅运行数据配比实验"
                echo "  --eval-only          仅运行模型评估"
                echo "  --skip-training      跳过训练，仅运行评估"
                echo "  --config FILE        指定配置文件"
                echo "  -h, --help           显示帮助信息"
                exit 0
                ;;
            *)
                log "未知参数: $1"
                exit 1
                ;;
        esac
    done
    
    # 开始实验
    if [ "$EVAL_ONLY" = true ]; then
        log "仅运行评估模式"
        
        # 查找已训练的模型
        MODEL_DIRS=$(find "${OUTPUT_BASE}" -name "stage_*" -type d 2>/dev/null | head -5)
        
        if [ -z "$MODEL_DIRS" ]; then
            log "未找到已训练的模型"
            exit 1
        fi
        
        for model_dir in $MODEL_DIRS; do
            model_name=$(basename "$model_dir")
            run_model_evaluation "$model_dir" "$model_name"
        done
        
    elif [ "$SKIP_TRAINING" = false ]; then
        # 运行训练
        if [ "$RATIO_ONLY" = true ]; then
            run_ratio_experiments
        elif [ "$CURRICULUM_ONLY" = true ]; then
            run_curriculum_learning
        else
            # 运行完整实验
            run_curriculum_learning
            run_ratio_experiments
        fi
        
        # 运行评估
        log "查找训练完成的模型进行评估..."
        TRAINED_MODELS=$(find "${OUTPUT_BASE}" -name "stage_*" -type d -newer "${EXPERIMENT_DIR}" 2>/dev/null)
        
        for model_dir in $TRAINED_MODELS; do
            model_name=$(basename "$model_dir")
            run_model_evaluation "$model_dir" "$model_name"
        done
    fi
    
    # 生成报告
    generate_experiment_report
    
    log "实验完成！"
    log "实验目录: ${EXPERIMENT_DIR}"
}

# 设置错误处理
trap cleanup EXIT

# 运行主函数
main "$@" 