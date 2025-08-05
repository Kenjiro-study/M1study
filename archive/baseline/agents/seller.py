# seller.py
from typing import Dict, Optional
import dspy

from .base_agent import BaseAgent

# seller の視点から交渉状況を分析する
class SellerStateAnalysis(dspy.Signature):
    """Analyzes negotiation state from seller's perspective."""
    current_price: Optional[float] = dspy.InputField()
    target_price: float = dspy.InputField()
    strategy_name: str = dspy.InputField()
    category: str = dspy.InputField()
    num_turns: int = dspy.InputField()
    initial_price: Optional[float] = dspy.InputField()

    price_sentiment: str = dspy.OutputField(desc="how good/bad current offer is relative to target")
    market_position: str = dspy.OutputField(desc="strength of current market position")
    recommended_flexibility: float = dspy.OutputField(desc="how much to deviate from target (0-1)")


class SellerAgent(BaseAgent):
    """
    AgreeMate baseline negotiation system の seller agent
    seller-specific の交渉行動と戦略の解釈を実装する
    """

    def __init__(
        self,
        strategy_name: str,
        target_price: float,
        category: str,
        min_price: Optional[float] = None,
        initial_price: Optional[float] = None,
        lm: dspy.LM = None
    ):
        """
        seller agent を初期化する
        
        Args:
            strategy_name: STRATEGIES の戦略名
            target_price: seller の目標販売価格
            category: 商品のカテゴリー
            min_price: 最低許容価格 (デフォルト値は target より 10%低い価格)
            initial_price: 最初の出品価格 (デフォルト値は target より20%高い価格)
            lm: 応答生成用の DSPy 言語モデル
        """
        super().__init__(
            strategy_name=strategy_name,
            target_price=target_price,
            category=category,
            is_buyer=False,
            lm=lm
        )

        self.min_price = min_price or (target_price * 0.9)
        self.initial_price = initial_price or (target_price * 1.2)
        self.state_analyzer = dspy.ChainOfThought(SellerStateAnalysis)
        self.best_offer_seen = 0 # 最高額のオファーをトラッキング

        # 交渉の進行状況をトラッキング
        self.total_discounts = 0
        self.moves_since_discount = 0
        self.initial_offer_made = False

    def _analyze_state(self) -> Dict:
        """Seller の視点から現在の交渉状況を分析する"""
        if self.current_price is None:
            return {
                'price_sentiment': 'initial',
                'market_position': 'strong',
                'recommended_flexibility': 0.1 # 保守的に始める
            }

        analysis = self.state_analyzer(
            current_price=self.current_price,
            target_price=self.target_price,
            strategy_name=self.strategy["name"],
            category=self.category,
            num_turns=self.num_turns,
            initial_price=self.initial_price
        )

        return {
            'price_sentiment': analysis.price_sentiment,
            'market_position': analysis.market_position,
            'recommended_flexibility': analysis.recommended_flexibility
        }

    def update_state(self, message: Dict[str, str]) -> Dict:
        """Update state with seller-specific tracking."""
        super().update_state(message) # 基本状態を更新する
                                        # (conversation, price history, actions, etc)

        # 相手（buyer）からのオファーの場合のみ、best_offer_seenを更新する 2025/7/15変更
        if message['role'] == 'buyer' and self.current_price is not None:
            self.best_offer_seen = max(self.best_offer_seen, self.current_price)

        # 値引きをトラッキング
        if len(self.price_history) >= 2:
            latest_change = self.price_history[-1] - self.price_history[-2]
            if latest_change < 0: # 価格が下がった場合 (seller が値引きした場合)
                self.total_discounts += abs(latest_change)
                self.moves_since_discount = 0
            else:
                self.moves_since_discount += 1

        return message

    def predict_action_maneger(self) -> Dict:
        """オーバーライドして seller-specific の戦略上考慮すべき事項を追加する"""
        # 最初のオファーを処理する
        if not self.initial_offer_made and not self.conversation_history:
            return {
                'action': 'offer',
                'counter_price': self.initial_price,
                'rationale': f"Making initial offer at ${self.initial_price}",
                'state_analysis': self._analyze_state()
            }

        # base prediction を取得する
        prediction = super().predict_action_maneger()

        # seller context を追加する
        analysis = self._analyze_state()
        prediction['state_analysis'] = analysis

        # 必要に応じて最低価格に基づいた調整を行う #!(GUARDRAIL: 最低価格を下回る価格は決して受け入れない！)
        if self.current_price and self.current_price < self.min_price:
            if prediction['action'] == 'accept':
                prediction['action'] = 'reject'
                prediction['rationale'] += f"\nHowever, offer (${self.current_price}) below minimum (${self.min_price})"
                prediction['counter_price'] = self.min_price * 1.05 # 最低価格を僅かに上回る

        # 最初のオファーが行われたことを示すフラグ
        self.initial_offer_made = True

        return prediction

    # 2025/7/15 変更
    def prepare_response_generation(self, action: str, price: Optional[float] = None) -> Dict:
        context = super().prepare_response_generation(action, price) # base prediction を取得する

        analysis = self._analyze_state()

        # 必要な情報をすべて渡してpredictorを呼び出す
        context.update({
            # seller-specific context を追加する
            "price_sentiment": analysis['price_sentiment'],
            "market_position": analysis['market_position']
        })

        return context

def test_seller_agent():
    """seller agent の機能をテストする"""
    import os

    # test LM のセットアップ
    baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agreemate_dir = os.path.dirname(baseline_dir)
    pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")

    test_lm = dspy.LM(
        model="ollama/llama3.1",
        provider="ollama",
        cache_dir=pretrained_dir,
    )

    # seller agent の作成
    seller = SellerAgent(
        strategy_name="cooperative",
        target_price=100.0,
        category="electronics",
        min_price=80.0,
        initial_price=120.0,
        lm=test_lm
    )

    # 初期化のテスト
    assert seller.role == "seller"
    assert seller.min_price == 80.0
    assert seller.initial_price == 120.0
    assert seller.best_offer_seen == 0

    # 最初のオファーのテスト
    response = seller.step()
    assert response["role"] == "seller"
    #assert "120" in response["content"] # should include initial price

    # オファー処理のテスト
    message = {
        "role": "buyer",
        "content": "I can offer $90"
    }
    seller.update_state(message)
    assert seller.current_price == 90.0
    assert seller.best_offer_seen == 90.0

    # counter-offer 生成のテスト
    response = seller.step()
    assert response["role"] == "seller"
    assert "content" in response

    print("✓ All seller agent tests passed")
    return seller

if __name__ == "__main__":
    seller = test_seller_agent()