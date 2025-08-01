# config.py
"""
AgreeMate baseline system の Central configuration
このファイルは, 構成可能な全てのパラメーターを定義し, 構成が完全かつ一貫していることを
確認するための検証ユーティリティを提供する
"""
from typing import Dict, List
from dataclasses import dataclass

from .strategies import STRATEGIES


@dataclass
class ModelConfig:
    """specific model のための Configuration"""
    name: str # 使用するAIモデルの名前(HuggingFace model name)
    max_tokens: int # モデルが一度に生成できる文章の最大長(トークン数)
    temperature: float # 0に近いほど固定的, 1に近いほどランダム的な答えを生成するようになる
    prompt_template: str # プロンプト

@dataclass
class ExperimentConfig:
    """一回の実験実行のための Configuration"""
    num_scenarios: int # 交渉のシミュレーション数(一回の実験で何回交渉するか)
    max_turns: int # 1回の交渉における最大ターン数(これを超えると交渉失敗)
    turn_timeout: float # 1回の応答生成における最大待ち時間(秒)(これを超えるとタイムアウトエラー)
    models: List[str] # 実験で使用するAIモデルの名前をリストで指定
    strategies: List[str] # 実験で使用する交渉戦略をリストで指定


# core model configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "llama3.1": ModelConfig(
        name="ollama/llama3.1",
        max_tokens=4096,
        temperature=0.7,
        prompt_template=(
            "You are a {role} negotiating for {item}.\n"
            "Your strategy is: {strategy}\n\n"
            "Current conversation:\n{history}\n\n"
            "Your target price is: ${target_price}\n"
            "Respond as {role}:"
        ),
    ),
    "llama-3.1-70b": ModelConfig(
        name="meta-llama/Llama-3.1-70B-Instruct",
        max_tokens=128000,
        temperature=0.7,
        prompt_template=(
            "You are a {role} negotiating for {item}.\n"
            "Your strategy is: {strategy}\n\n"
            "Current conversation:\n{history}\n\n"
            "Your target price is: ${target_price}\n"
            "Respond as {role}:"
        ),
    )
}

# あらかじめ定義された実験の configurations
EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "baseline": ExperimentConfig(
        #num_scenarios=100,
        num_scenarios=1,
        max_turns=20,
        turn_timeout=30.0,
        models=["llama3.1"],
        #strategies=["cooperative", "fair", "aggressive"]
        strategies=["cooperative"]
    ),
    "model_comparison": ExperimentConfig(
        num_scenarios=200,
        max_turns=20,
        turn_timeout=30.0,
        models=["llama3.1", "llama-3.1-70b"],
        strategies=["cooperative", "fair"]
    ),
    "strategy_analysis": ExperimentConfig(
        num_scenarios=150,
        max_turns=25,
        turn_timeout=30.0,
        models=["llama-3.1-70b"],
        strategies=["cooperative", "fair", "aggressive"]
    )
}

# 分析の configuration
ANALYSIS_CONFIG = {
    # 実験の評価指標のリスト
    "metrics": [
        "deal_rate", # 合意率
        "avg_utility", # 効用値の平均
        "turns_to_completion", # 交渉成立までの平均ターン数
        "strategy_adherence" # どれくらい戦略に忠実に従ったか
    ],
    # 結果の可視化方法のリスト
    "visualizations": [
        "outcome_dashboard", # 交渉結果の全体像を示すダッシュボード
        "process_visualization", # 交渉過程の可視化
        "behavioral_analysis" # 交渉中のAIの振る舞いを分析したもの
    ]
}


def validate_config(config: ExperimentConfig) -> bool:
    """実験 configuration の完全性と一貫性を検証する"""
    for model in config.models: # モデルが存在するかどうか確認
        if model not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model}")

    for strategy in config.strategies: # 戦略が存在するかどうか確認
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")

    # 数値パラメーターを検証する
    if config.num_scenarios < 1:
        raise ValueError("num_scenarios must be positive")
    if config.max_turns < 1:
        raise ValueError("max_turns must be positive")
    if config.turn_timeout <= 0:
        raise ValueError("turn_timeout must be positive")

    return True