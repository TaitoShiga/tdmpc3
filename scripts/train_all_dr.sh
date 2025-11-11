#!/bin/bash
# Domain Randomization タスク連続学習スクリプト

set -e  # エラーで停止

echo "========================================"
echo "Domain Randomization タスク連続学習開始"
echo "========================================"

# ベースディレクトリ
cd "$(dirname "$0")/.."

# 共通パラメータ
STEPS=100000  # 10万ステップ
SEED=0

# タスクリスト
TASKS=(
    "pendulum-swingup-randomized"
    "cup-catch-randomized"
    "reacher-three-easy-randomized"
    "hopper-stand-randomized"
)

# 各タスクを連続学習
for task in "${TASKS[@]}"; do
    # exp_name: task名に応じて設定
    if [[ $task == "pendulum-swingup-randomized" ]]; then
        exp_name="pendulum_dr"
    elif [[ $task == "cup-catch-randomized" ]]; then
        exp_name="ball_in_cup_dr"
    elif [[ $task == "reacher-three-easy-randomized" ]]; then
        exp_name="reacher_dr"
    elif [[ $task == "hopper-stand-randomized" ]]; then
        exp_name="hopper_dr"
    else
        # フォールバック: 最初のハイフンまで
        domain=$(echo $task | cut -d'-' -f1)
        exp_name="${domain}_dr"
    fi
    
    echo ""
    echo "========================================"
    echo "Training: $task"
    echo "Exp name: $exp_name"
    echo "========================================"
    
    python tdmpc2/train.py \
        task=$task \
        exp_name=$exp_name \
        steps=$STEPS \
        seed=$SEED \
        save_video=true \
        enable_wandb=false \
        compile=false
    
    echo "✅ Completed: $task"
done

echo ""
echo "========================================"
echo "🎉 全タスクの学習が完了しました！"
echo "========================================"
echo ""
echo "学習済みモデル:"
echo "  - logs/pendulum-swingup-randomized/${SEED}/pendulum_dr/"
echo "  - logs/cup-catch-randomized/${SEED}/ball_in_cup_dr/"
echo "  - logs/reacher-three-easy-randomized/${SEED}/reacher_dr/"
echo "  - logs/hopper-stand-randomized/${SEED}/hopper_dr/"

