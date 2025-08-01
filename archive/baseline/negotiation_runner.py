# negotiation_runner.py
import logging, asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field # 2025/7/18 field追加
from datetime import datetime

from .agents.buyer import BuyerAgent
from .agents.seller import SellerAgent
from .scenario_manager import NegotiationScenario
from .dspy_manager import DSPyManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class NegotiationConfig:
    """単一の交渉の Configuration"""
    scenario: NegotiationScenario
    buyer_model: str
    seller_model: str
    buyer_strategy: str
    seller_strategy: str
    max_turns: int
    turn_timeout: float

@dataclass
class NegotiationMetrics:
    """交渉中に収集された Metrics"""
    start_time: datetime
    end_time: Optional[datetime] = None
    turns_taken: int = 0
    final_price: Optional[float] = None
    buyer_utility: Optional[float] = None
    seller_utility: Optional[float] = None
    strategy_adherence: Dict[str, float] = None
    messages: List[Dict] = field(default_factory=list) # 2025/7/18 Noneからfieldに変更

    def compute_duration(self) -> float:
        """交渉時間を秒単位で計算する"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class NegotiationRunner:
    """
    AgreeMate システムの個々の交渉セッションの実行を管理する
    agent の初期化, 交渉ターンの管理, 及び metrics の収集を処理する
    """

    def __init__(
        self,
        dspy_manager: DSPyManager,
        max_concurrent: int = 4
    ):
        """
        negotiation runner を初期化する

        Args:
            dspy_manager: DSPy LM マネージャーインスタンス
            max_concurrent: 同時交渉の最大数
        """
        self.dspy_manager = dspy_manager
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_negotiations: Dict[str, NegotiationMetrics] = {}

    def _initialize_agents(
        self,
        config: NegotiationConfig
    ) -> Tuple[BuyerAgent, SellerAgent]:
        """交渉のために買い手エージェントと売り手エージェントを初期化する"""
        # 戦略固有の構成をもつ DSPy LMs を取得する
        buyer_lm, seller_lm = self.dspy_manager.configure_negotiation(
            config.buyer_model,
            config.seller_model,
            config.buyer_strategy,
            config.seller_strategy
        )

        # シナリオのコンテキストでエージェントを作成する
        buyer = BuyerAgent(
            strategy_name=config.buyer_strategy,
            target_price=config.scenario.buyer_target,
            category=config.scenario.category,
            max_price=config.scenario.list_price,
            lm=buyer_lm
        )

        seller = SellerAgent(
            strategy_name=config.seller_strategy,
            target_price=config.scenario.seller_target,
            category=config.scenario.category,
            initial_price=config.scenario.list_price,
            min_price=config.scenario.seller_target * 0.9, # 10% below target
            lm=seller_lm
        )

        return buyer, seller


    def _validate_price_movement(
        self,
        agent_role: str,
        new_price: float,
        metrics: NegotiationMetrics
    ) -> bool:
        """価格の変動が交渉ルールに従っているかどうかを検証する"""
        if not metrics.messages:
            return True # 最初のオファーは常に有効

        last_price = next(
            (m['price'] for m in reversed(metrics.messages) 
            if m['price'] is not None),
            None
        )

        if last_price is None:
            return True

        # 価格は交渉内でお互いに譲歩していく必要があるので, 買い手はより高い価格を提示すべきであり, 売り手はより低い価格を提示すべきである
        if agent_role == 'buyer':
            return new_price >= last_price
        else:
            return new_price <= last_price

    async def _run_negotiation_turn(
        self,
        buyer: BuyerAgent,
        seller: SellerAgent,
        metrics: NegotiationMetrics,
        timeout: float
    ) -> bool:
        """
        交渉を1ターン実行する

        Returns:
            bool: 交渉を継続する場合は True
        """
        try:
            # buyerとsellerを交互に行う
            current_agent = buyer if metrics.turns_taken % 2 == 0 else seller

            # タイムアウトでターンを実行する
            async with asyncio.timeout(timeout):
                response = current_agent.step() # 2025/7/18 await current_agent.step()のawait削除

                # 2025/7/18 交渉の流れを見るためのprint追加
                if current_agent.is_buyer == True:
                    print("buyer: ", response)
                else:
                    print("seller: ", response)


                # message の構造を検証する
                if not isinstance(response, dict) or 'role' not in response:
                    raise ValueError("Invalid message format")

                # 必須となる fields を確認する
                response.setdefault('price', None)
                response.setdefault('status', 'counter')

                # 価格の変動を検証する
                if response['price'] is not None:
                    valid = self._validate_price_movement(
                        agent_role=response['role'],
                        new_price=response['price'],
                        metrics=metrics
                    )
                    if not valid:
                        response['status'] = 'reject'
                        logger.warning(f"Invalid price movement: {response}")

            # metrics を更新する
            metrics.turns_taken += 1
            metrics.messages.append(response)

            # handle completion
            if response['status'] in ['accept', 'reject']:
                metrics.end_time = datetime.now()
                metrics.final_price = (
                    response['price'] if response['status'] == 'accept' 
                    else None
                )
                return False

            return True

        except asyncio.TimeoutError:
            logger.warning(f"Turn timeout after {timeout}s")
            metrics.end_time = datetime.now()
            return False
        except Exception as e:
            logger.error(f"Turn error: {str(e)}")
            metrics.end_time = datetime.now()
            return False

    async def run_negotiation(
        self,
        config: NegotiationConfig
    ) -> NegotiationMetrics:
        """
        完全な交渉セッションを実行する

        Args:
            config: Negotiation configuration

        Returns:
            Completed negotiation metrics
        """
        async with self.semaphore:
            # metrics のトラッキングを初期化する
            metrics = NegotiationMetrics(
                start_time=datetime.now(),
                strategy_adherence={
                    'buyer': 1.0, # 交渉中に更新される
                    'seller': 1.0
                }
            )

            try:
                # エージェントを初期化
                buyer, seller = self._initialize_agents(config)

                # active な交渉をトラッキングする
                self.active_negotiations[config.scenario.scenario_id] = metrics

                # 交渉ターンを実行する
                continue_negotiation = True
                
                while (continue_negotiation and metrics.turns_taken < config.max_turns):
                    continue_negotiation = await self._run_negotiation_turn(buyer, seller, metrics, config.turn_timeout)

                # ▼▼▼▼▼ 2025/7/18 このブロックをここに追加 ▼▼▼▼▼
                # 最大ターン数に達して交渉が終了した場合の処理
                if metrics.end_time is None:
                    logger.info(f"Negotiation reached max turns ({config.max_turns}) without an agreement.")
                    metrics.end_time = datetime.now()
                # ▲▲▲▲▲ ここまで ▲▲▲▲▲

                # 最終的な metrics を計算する
                self._compute_final_metrics(metrics, buyer, seller)

                return metrics

            except Exception as e:
                logger.error(f"Negotiation failed: {str(e)}")
                metrics.end_time = datetime.now()
                return metrics
            finally: # active な交渉から除外する
                self.active_negotiations.pop(config.scenario.scenario_id, None)


    def _compute_final_metrics(
        self,
        metrics: NegotiationMetrics,
        buyer: BuyerAgent,
        seller: SellerAgent
    ):
        """最終的な交渉 metrics の計算"""
        if metrics.final_price:
            # utilities の計算
            metrics.buyer_utility = (
                buyer.compute_utility(metrics.final_price)
                if hasattr(buyer, 'compute_utility') else None
            )
            metrics.seller_utility = (
                seller.compute_utility(metrics.final_price)
                if hasattr(seller, 'compute_utility') else None
            )

            # strategy adherence の更新
            if hasattr(buyer, 'get_strategy_adherence'):
                metrics.strategy_adherence['buyer'] = buyer.get_strategy_adherence()
            if hasattr(seller, 'get_strategy_adherence'):
                metrics.strategy_adherence['seller'] = seller.get_strategy_adherence()

    async def run_batch(
        self,
        configs: List[NegotiationConfig]
    ) -> Dict[str, NegotiationMetrics]:
        """バッチの交渉を並列して実行する"""
        tasks = []
        for config in configs:
            task = asyncio.create_task(self.run_negotiation(config))
            tasks.append((config.scenario.scenario_id, task))

        results = {}
        for scenario_id, task in tasks:
            try:
                metrics = await task
                results[scenario_id] = metrics
            except Exception as e:
                logger.error(f"Batch task failed: {str(e)}")

        return results


def test_negotiation_runner():
    """negotiation runner 機能をテストする"""
    from .scenario_manager import ScenarioManager
    from .utils.data_loader import DataLoader
    import logging
    logging.getLogger('dspy.adapters.json_adapter').setLevel(logging.ERROR) # JSONになるWARNINGを消す処理


    # components の初期化
    data_loader = DataLoader()
    scenario_manager = ScenarioManager(data_loader)
    dspy_manager = DSPyManager()
    runner = NegotiationRunner(dspy_manager)

    # テストシナリオの作成
    scenarios = scenario_manager.create_evaluation_batch(
        split='test',
        size=1
    )

    # テスト用の configuration の作成
    config = NegotiationConfig(
        scenario=scenarios[0],
        buyer_model="llama3.1",
        seller_model="llama3.1",
        buyer_strategy="cooperative",
        seller_strategy="fair",
        max_turns=5,
        turn_timeout=30.0
    )
    # テスト交渉を実行
    async def run_test():
        metrics = await runner.run_negotiation(config)
        assert metrics.turns_taken > 0
        assert metrics.end_time is not None
        print("✓ Negotiation completed successfully")
        return metrics
    
    import asyncio
    metrics = asyncio.run(run_test())

    print("✓ All negotiation runner tests passed")
    return runner, metrics

if __name__ == "__main__":
    runner, metrics = test_negotiation_runner()