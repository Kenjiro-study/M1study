# metrics_collector.py
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .negotiation_runner import NegotiationMetrics
from .strategies import STRATEGIES

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class StrategyMetrics:
    """戦略分析のための Metrics"""
    strategy_name: str
    adherence_scores: List[float] = field(default_factory=list)
    success_rate: float = 0.0
    avg_turns: float = 0.0
    avg_utility: float = 0.0
    language_metrics: Dict[str, float] = field(default_factory=dict)

    def update(self, success: bool, turns: int, utility: float, adherence: float):
        """新しい交渉結果で metrics を更新"""
        self.adherence_scores.append(adherence)
        n = len(self.adherence_scores)

        # running averages を更新する
        self.success_rate = ((n-1) * self.success_rate + float(success)) / n
        self.avg_turns = ((n-1) * self.avg_turns + turns) / n
        self.avg_utility = ((n-1) * self.avg_utility + utility) / n


@dataclass
class NegotiationAnalysis:
    """交渉セッションの完全な分析"""
    # core identifiers
    scenario_id: str
    buyer_model: str
    seller_model: str

    # basic metrics
    duration: float
    turns_taken: int
    final_price: Optional[float]

    # strategy analysis
    buyer_strategy: str
    seller_strategy: str
    buyer_adherence: float
    seller_adherence: float

    # price trajectory
    initial_price: float
    target_prices: Dict[str, float] # buyer/seller targets
    price_history: List[float]

    # interaction analysis
    message_lengths: List[int]
    response_times: List[float]

    def compute_metrics(self) -> Dict[str, float]:
        """派生 metrics を計算する"""
        metrics = {
            'success': self.final_price is not None,
            'efficiency': self._compute_efficiency(),
            'fairness': self._compute_fairness(),
            'avg_response_time': np.mean(self.response_times),
            'price_convergence': self._compute_convergence()
        }
        return metrics

    def _compute_efficiency(self) -> float:
        """交渉の efficiency スコア (0-1) を計算"""
        if not self.final_price:
            return 0.0

        # 交渉ターン, 時間, 価格変動を考慮する
        max_expected_turns = 20 # from config
        turn_score = 1 - (self.turns_taken / max_expected_turns)

        time_per_turn = self.duration / self.turns_taken
        time_score = np.exp(-time_per_turn / 30) # 30 sec baseline

        prices = [item['price'] for item in self.price_history] # 追加 2025/7/4
        price_movement = np.diff(prices) # 追加 2025/7/4
        # price_movement = np.diff(self.price_history) # 辞書型の引き算はダメと言われたので上の2行に変更した

        directness = 1 - (np.abs(price_movement).sum() / 
                         abs(self.price_history[-1]["price"] - self.price_history[0]["price"])) # ["price"]追加 2025/7/4

        return np.mean([turn_score, time_score, directness])

    def _compute_fairness(self) -> float:
        """交渉の fairness スコア (0-1) を計算"""
        if not self.final_price:
            return 0.0

        # midpoint からの距離
        fair_price = (self.target_prices['buyer'] + 
                     self.target_prices['seller']) / 2
        price_fairness = 1 - (abs(self.final_price - fair_price) /
                             abs(self.target_prices['buyer'] - 
                                 self.target_prices['seller']))

        # 譲歩の balance
        buyer_movement = abs(self.final_price - self.price_history[0]["price"]) # ["price"]追加 2025/7/4
        seller_movement = abs(self.final_price - self.price_history[1]["price"]) # ["price"]追加 2025/7/4
        concession_balance = 1 - abs(buyer_movement - seller_movement) / \
                               (buyer_movement + seller_movement)

        return np.mean([price_fairness, concession_balance])

    def _compute_convergence(self) -> float:
        """C価格収束スコア (0-1) を計算"""
        if len(self.price_history) < 3:
            return 0.0

        # 価格が最終合意に向かってどれだけ直接的に動いたかを測定する
        prices = [item['price'] for item in self.price_history] # 追加 2025/7/4
        price_diffs = np.diff(prices) # 追加 2025/7/4
        # price_diffs = np.diff(self.price_history) # 辞書型の引き算はダメと言われたので上の2行に変更した
        ideal_path = abs(self.price_history[-1]["price"] - self.price_history[0]["price"]) # ["price"]追加 2025/7/4
        actual_path = np.abs(price_diffs).sum()

        return ideal_path / actual_path if actual_path > 0 else 0.0


class MetricsCollector:
    """
    交渉実験から metrics を収集・分析する
    リアルタイム tracking と 実験後の分析の両方を提供する
    """

    def __init__(self):
        """metrics collector の初期化"""
        self.strategy_metrics: Dict[str, StrategyMetrics] = {
            name: StrategyMetrics(strategy_name=name)
            for name in STRATEGIES.keys()
        }

        self.model_pairs: Dict[str, List[NegotiationAnalysis]] = {}
        self.negotiations: Dict[str, NegotiationAnalysis] = {}

    def analyze_negotiation(
        self,
        metrics: NegotiationMetrics,
        buyer_model: str,
        seller_model: str,
        buyer_strategy: str,
        seller_strategy: str,
        scenario_id: str,
        initial_price: float,
        target_prices: Dict[str, float]
    ) -> NegotiationAnalysis:
        """完了した交渉対話を分析する"""

        analysis = NegotiationAnalysis(
            scenario_id=scenario_id,
            buyer_model=buyer_model,
            seller_model=seller_model,
            duration=metrics.compute_duration(),
            turns_taken=metrics.turns_taken,
            final_price=metrics.final_price,
            buyer_strategy=buyer_strategy,
            seller_strategy=seller_strategy,
            buyer_adherence=metrics.strategy_adherence['buyer'],
            seller_adherence=metrics.strategy_adherence['seller'],
            initial_price=initial_price,
            target_prices=target_prices,
            price_history=[p for p in metrics.messages if 'price' in p],
            message_lengths=[len(m['content']) for m in metrics.messages],
            response_times=[m.get('response_time', 0) for m in metrics.messages]
        )

        # strategy metrics の更新
        computed = analysis.compute_metrics()
        self.strategy_metrics[buyer_strategy].update(
            success=computed['success'],
            turns=metrics.turns_taken,
            utility=metrics.buyer_utility or 0.0,
            adherence=metrics.strategy_adherence['buyer']
        )
        self.strategy_metrics[seller_strategy].update(
            success=computed['success'],
            turns=metrics.turns_taken,
            utility=metrics.seller_utility or 0.0,
            adherence=metrics.strategy_adherence['seller']
        )

        # 分析結果を保存
        self.negotiations[scenario_id] = analysis
        pair_key = f"{buyer_model}_{seller_model}"
        if pair_key not in self.model_pairs:
            self.model_pairs[pair_key] = []
        self.model_pairs[pair_key].append(analysis)

        return analysis

    def get_strategy_summary(self) -> pd.DataFrame:
        """各戦略の summary statistics を取得する"""
        records = []
        for strategy_name, metrics in self.strategy_metrics.items():
            # adherence_scoresが空でない場合のみ計算し、空の場合は0.0とする 2025/7/17追加
            adherence_mean = np.mean(metrics.adherence_scores) if metrics.adherence_scores else 0.0
            adherence_std = np.std(metrics.adherence_scores) if metrics.adherence_scores else 0.0
            records.append({
                'strategy': strategy_name,
                'success_rate': metrics.success_rate,
                'avg_turns': metrics.avg_turns,
                'avg_utility': metrics.avg_utility,
                'adherence_mean': adherence_mean,
                'adherence_std': adherence_std
            })
        return pd.DataFrame.from_records(records)

    def get_model_pair_summary(self) -> pd.DataFrame:
        """各モデルペアの組み合わせの summary statistics を取得する"""
        records = []
        for pair_key, analyses in self.model_pairs.items():
            buyer_model, seller_model = pair_key.split('_')

            # aggregate metrics の計算
            success_rate = np.mean([
                bool(a.final_price) for a in analyses
            ])
            avg_efficiency = np.mean([
                a.compute_metrics()['efficiency'] for a in analyses
            ])
            avg_fairness = np.mean([
                a.compute_metrics()['fairness'] for a in analyses
            ])

            records.append({
                'buyer_model': buyer_model,
                'seller_model': seller_model,
                'num_negotiations': len(analyses),
                'success_rate': success_rate,
                'avg_efficiency': avg_efficiency,
                'avg_fairness': avg_fairness
            })
        return pd.DataFrame.from_records(records)

    def export_analysis(self) -> Dict:
        """完全な分析結果をエクスポートする"""
        return {
            'strategy_summary': self.get_strategy_summary().to_dict('records'),
            'model_summary': self.get_model_pair_summary().to_dict('records'),
            'negotiations': {
                sid: analysis.compute_metrics()
                for sid, analysis in self.negotiations.items()
            }
        }


def test_metrics_collector():
    """metrics collector 機能をテストする"""
    from datetime import datetime, timedelta

    collector = MetricsCollector()

    # test metrics を作成
    metrics = NegotiationMetrics(
        start_time=datetime.now() - timedelta(minutes=5),
        end_time=datetime.now(),
        turns_taken=10,
        final_price=150.0,
        buyer_utility=0.8,
        seller_utility=0.7,
        strategy_adherence={'buyer': 0.9, 'seller': 0.85},
        messages=[
            {'role': 'buyer', 'content': 'Offer: $100', 'price': 100},
            {'role': 'seller', 'content': 'Counter: $200', 'price': 200},
            {'role': 'buyer', 'content': 'Accept: $150', 'price': 150}
        ]
    )

    # 分析をテストする
    analysis = collector.analyze_negotiation(
        metrics=metrics,
        buyer_model='llama-3.1-8b',
        seller_model='llama-3.1-8b',
        buyer_strategy='cooperative',
        seller_strategy='fair',
        scenario_id='test_1',
        initial_price=200.0,
        target_prices={'buyer': 100.0, 'seller': 180.0}
    )
    assert analysis.turns_taken == 10
    assert analysis.final_price == 150.0

    # サマリーをテストする
    strategy_summary = collector.get_strategy_summary()
    assert len(strategy_summary) == len(STRATEGIES)
    print("strategy_summary: ", strategy_summary)

    model_summary = collector.get_model_pair_summary()
    assert len(model_summary) > 0
    print("model_summary: ", model_summary)

    print("✓ All metrics collector tests passed")
    return collector

if __name__ == "__main__":
    collector = test_metrics_collector()