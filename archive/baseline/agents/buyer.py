# buyer.py
from typing import Dict, Optional
import dspy

from .base_agent import BaseAgent

# buyer の視点から交渉状況を分析する
class BuyerStateAnalysis(dspy.Signature):
    """Analyzes negotiation state from buyer's perspective."""
    current_price: Optional[float] = dspy.InputField()
    target_price: float = dspy.InputField()
    strategy_name: str = dspy.InputField()
    category: str = dspy.InputField()
    num_turns: int = dspy.InputField()

    price_sentiment: str = dspy.OutputField(desc="how good/bad current price is relative to target")
    bargaining_power: str = dspy.OutputField(desc="current negotiating position strength")
    recommended_flexibility: float = dspy.OutputField(desc="how much to deviate from target (0-1)")


class BuyerAgent(BaseAgent):
    """
    AgreeMate baseline negotiation system の Buyer agent
    buyer-specific の交渉行動と戦略解釈を実装する
    """

    def __init__(
        self,
        strategy_name: str,
        target_price: float,
        category: str,
        max_price: Optional[float] = None,
        lm: dspy.LM = None
    ):
        """
        buyer agent を初期化する

        Args:
            strategy_name: STRATEGIES の戦略名
            target_price: Buyer の目標購入金額
            category: 商品のカテゴリー
            max_price: 最大許容価格 (デフォルト値は target より 10%高い価格)
            lm: 応答生成用の DSPy 言語モデル
        """
        super().__init__(
            strategy_name=strategy_name,
            target_price=target_price,
            category=category,
            is_buyer=True,
            lm=lm
        )

        self.max_price = max_price or (target_price * 1.1)
        self.state_analyzer = dspy.ChainOfThought(BuyerStateAnalysis)
        self.best_offer_seen = float('inf') # 最低価格のオファーをトラッキング

        # 交渉状況をトラッキング
        self.total_concessions = 0
        self.moves_since_concession = 0

    def _analyze_state(self) -> Dict:
        """buyer の視点から現在の交渉状況を分析する"""
        if self.current_price is None:
            return {
                'price_sentiment': 'unknown',
                'bargaining_power': 'strong',
                'recommended_flexibility': 0.1 # start conservative
            }

        analysis = self.state_analyzer(
            current_price=self.current_price,
            target_price=self.target_price,
            strategy_name=self.strategy["name"],
            category=self.category,
            num_turns=self.num_turns
        )

        return {
            'price_sentiment': analysis.price_sentiment,
            'bargaining_power': analysis.bargaining_power,
            'recommended_flexibility': analysis.recommended_flexibility
        }

    def update_state(self, message: Dict[str, str]) -> Dict:
        """buyer-specific tracking で状態を更新"""
        super().update_state(message) # 基本状態の更新
                                        # (conversation, price history, actions, etc)

        # 相手（seller）からのオファーの場合のみ、best_offer_seenを更新する 2025/7/15変更
        if message['role'] == 'seller' and self.current_price is not None:
            self.best_offer_seen = min(self.best_offer_seen, self.current_price)

        # 譲歩をトラッキング
        if len(self.price_history) >= 2:
            latest_change = self.price_history[-1] - self.price_history[-2]
            if latest_change > 0: # 価格が上がった場合 (買い手が譲歩した場合)
                self.total_concessions += latest_change
                self.moves_since_concession = 0
            else:
                self.moves_since_concession += 1

        return message

    def predict_action_maneger(self) -> Dict:
        """オーバーライドして buyer-specific の戦略上考慮すべき事項を追加する"""
        prediction = super().predict_action_maneger() # base prediction を取得する

        # buyer context を追加
        analysis = self._analyze_state()
        prediction['state_analysis'] = analysis

        # 必要に応じて最高価格に基づいた調整を行う
        if self.current_price and self.current_price > self.max_price:
            if prediction['action'] == 'accept':
                prediction['action'] = 'reject'
                prediction['rationale'] += f"\nHowever, price (${self.current_price}) exceeds maximum (${self.max_price})"
                prediction['counter_price'] = self.max_price * 0.95 # slightly below max

        return prediction
    
    # 2025/7/15 変更
    def prepare_response_generation(self, action: str, price: Optional[float] = None) -> Dict:
        context = super().prepare_response_generation(action, price) # base prediction を取得する

        analysis = self._analyze_state()

        # 必要な情報をすべて渡してpredictorを呼び出す
        context.update({
            # buyer-specific context を追加する
            "price_sentiment": analysis['price_sentiment'],
            "bargaining_power": analysis['bargaining_power']
        })

        return context

def test_buyer_agent():
    """Test buyer agent の機能をテストする"""
    import os

    # test LM のセットアップ
    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    agreemate_dir = os.path.dirname(baseline_dir)
    pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")

    test_lm = dspy.LM(
        model="ollama/llama3.1",
        provider="ollama",
        cache_dir=pretrained_dir,
    )

    # buyer agent の作成
    buyer = BuyerAgent(
        strategy_name="cooperative",
        target_price=100.0,
        category="electronics",
        max_price=120.0,
        lm=test_lm
    )

    # 初期化のテスト
    assert buyer.role == "buyer"
    assert buyer.max_price == 120.0
    assert buyer.best_offer_seen == float('inf')

    # オファー処理のテスト
    message = {
        "role": "seller",
        "content": "I can offer it for $150"
    }
    buyer.update_state(message)
    assert buyer.current_price == 150.0
    assert buyer.best_offer_seen == 150.0

    # counter-offer 生成のテスト
    response = buyer.step()
    assert response["role"] == "buyer"
    assert "content" in response

    print("✓ All buyer agent tests passed")
    return buyer

if __name__ == "__main__":
    buyer = test_buyer_agent()