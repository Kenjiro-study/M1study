# buyer.py
from typing import Dict, Optional
import dspy, random, math

from .base_agent import BaseAgent
from .extractor import PriceExtractor

class BuyerAgent(BaseAgent):
    """
    AgreeMate baseline negotiation system の Buyer agent
    buyer-specific の交渉行動と戦略解釈を実装する
    """

    def __init__(
        self,
        strategy_name: str,
        target_price: float,
        list_price: float, 
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
            list_price=list_price,
            category=category,
            is_buyer=True,
            lm=lm
        )

        self.max_price = max_price or (target_price * 1.1)
        self.best_offer_seen = float('inf') # 最低価格のオファーをトラッキング


    def max_price_select(self) -> float:
        """性格ごとの最高価格の設定"""
        if self.strategy_name == "fair":
            max_price = self.target_price + ((self.list_price - self.target_price) * random.uniform(1.0, 0.7))
        elif self.strategy_name == "utility":
            max_price = self.target_price + ((self.list_price - self.target_price) * random.uniform(1.0, 0.3))
        elif self.strategy_name == "length":
            max_price = self.target_price + ((self.list_price - self.target_price) * random.uniform(1.0, 0.5))
        else:
            raise ValueError("Invalid strategy name")

        return round(max_price, 0)

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
    
    def get_manager_context(self) -> Dict:
        """予測の context を取得する"""
        context = super().get_manager_context()
        context.update({
            "agent_strategy": self.strategy['buyer_manager_style'],
        })
        return context
    
    def fair_manager(self) -> Dict:
        if self.partner_data['price'] != None:
            if (self.target_price >= self.partner_data['price']) or ((len(self.partner_price_history) >= 2) and self.price_history and ((0.4 * self.price_history[-1] + 0.6 * self.partner_price_history[-2]) >= self.partner_data['price'])):
                return{
                    "intent": "agree",
                    "price": None
                }
            elif self.pertner_intent_history.count("counter-price") + self.pertner_intent_history.count("insist") == 4:
                return{
                    "intent": "disagree",
                    "price": None
                }
            elif self.price_history[-1] == self.max_price:
                return{
                    "intent": "insist",
                    "price": self.max_price
                }
        
        prediction = self.intent_predictor(**self.get_manager_context())
        intent = prediction.next_intent

        # init-priceと予測されたが, すでに価格提案がある場合はcounter-priceに変更
        if (intent == "init-price") and (self.price_history) and (self.partner_price_history):
            intent = "counter-price"
        # counter-priceやinsistと予測されたが, まだ価格提案がない場合はinit-priceに変更
        if ((intent == "counter-price") or (intent == "insist")) and (not self.price_history) and (not self.partner_price_history):
            intent = "init-price"

        # 価格の設定
        price =  None  
        if intent == "init-price":
            price = self.target_price
        elif intent == "counter-price":
            if not self.price_history:
                price = self.target_price
            elif len(self.partner_price_history) < 2:
                price = 0.7 * self.price_history[-1] + 0.3 * self.list_price
            else:
                price = 0.7 * self.price_history[-1] + 0.3 * self.partner_price_history[-1]

            # 最高価格を超えていたら最高価格に設定
            if price >= self.max_price:
                price = self.max_price

        elif intent == "insist":
            if not self.price_history:
                price = self.target_price
            else:
                price = self.price_history[-1]

        return {
            "intent": intent,
            "price": price
        }
    
    def utility_manager(self) -> Dict:
        if self.partner_data['price'] != None:
            if (self.target_price >= self.partner_data['price']) or ((len(self.partner_price_history) >= 2) and self.price_history and ((0.7 * self.price_history[-1] + 0.3 * self.partner_price_history[-2]) >= self.partner_data['price'])):
                return{
                    "intent": "agree",
                    "price": None
                }
            elif self.pertner_intent_history.count("counter-price") + self.pertner_intent_history.count("insist") == 3:
                return{
                    "intent": "disagree",
                    "price": None
                }
            elif self.price_history[-1] == self.max_price:
                return{
                    "intent": "insist",
                    "price": self.max_price
                }
        
        prediction = self.intent_predictor(**self.get_manager_context())
        intent = prediction.next_intent

        # init-priceと予測されたが, すでに価格提案がある場合はcounter-priceに変更
        if (intent == "init-price") and (self.price_history) and (self.partner_price_history):
            intent = "counter-price"
        # counter-priceやinsistと予測されたが, まだ価格提案がない場合はinit-priceに変更
        if ((intent == "counter-price") or (intent == "insist")) and (not self.price_history) and (not self.partner_price_history):
            intent = "init-price"

        # 価格の設定
        price =  None  
        if intent == "init-price":
            price = self.target_price
        elif intent == "counter-price":
            if not self.price_history:
                price = self.target_price
            elif len(self.partner_price_history) < 2:
                price = 0.9 * self.price_history[-1] + 0.1 * self.list_price
            else:
                price = 0.9 * self.price_history[-1] + 0.1 * self.partner_price_history[-1]

            # 最高価格を超えていたら最高価格に設定
            if price >= self.max_price:
                price = self.max_price

        elif intent == "insist":
            if not self.price_history:
                price = self.target_price
            else:
                price = self.price_history[-1]

        return {
            "intent": intent,
            "price": price
        }
    
    def length_manager(self) -> Dict:
        if self.partner_data['price'] != None:
            if (self.target_price >= self.partner_data['price']) or ((len(self.partner_price_history) >= 2) and self.price_history and ((0.6 * self.price_history[-1] + 0.4 * self.partner_price_history[-2]) >= self.partner_data['price'])):
                return{
                    "intent": "agree",
                    "price": None
                }
            elif self.pertner_intent_history.count("counter-price") + self.pertner_intent_history.count("insist") == 5:
                return{
                    "intent": "disagree",
                    "price": None
                }
            elif self.price_history[-1] == self.max_price:
                return{
                    "intent": "insist",
                    "price": self.max_price
                }
        
        prediction = self.intent_predictor(**self.get_manager_context())
        intent = prediction.next_intent

        # init-priceと予測されたが, すでに価格提案がある場合はcounter-priceに変更
        if (intent == "init-price") and (self.price_history) and (self.partner_price_history):
            intent = "counter-price"
        # counter-priceやinsistと予測されたが, まだ価格提案がない場合はinit-priceに変更
        if ((intent == "counter-price") or (intent == "insist")) and (not self.price_history) and (not self.partner_price_history):
            intent = "init-price"

        # 価格の設定
        price =  None  
        if intent == "init-price":
            price = self.target_price
        elif intent == "counter-price":
            if not self.price_history:
                price = self.target_price
            elif len(self.partner_price_history) < 2:
                price = 0.8 * self.price_history[-1] + 0.2 * self.list_price
            else:
                price = 0.8 * self.price_history[-1] + 0.2 * self.partner_price_history[-1]
            
            # 最高価格を超えていたら最高価格に設定
            if price >= self.max_price:
                price = self.max_price

        elif intent == "insist":
            if not self.price_history:
                price = self.target_price
            else:
                price = self.price_history[-1]

        return {
            "intent": intent,
            "price": price
        }


    def predict_action_manager(self) -> Dict:
        if self.strategy_name == "fair":
            prediction = self.fair_manager()
        elif self.strategy_name == "utility":
            prediction = self.utility_manager()
        elif self.strategy_name == "length":
            prediction = self.length_manager()
        else:
            raise ValueError("Invalid strategy name")

        return prediction

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
        strategy_name="length",
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
    assert buyer.best_offer_seen == 150.0

    # counter-offer 生成のテスト
    response = buyer.step()
    assert response["role"] == "buyer"
    assert "content" in response

    print("✓ All buyer agent tests passed")
    return buyer

if __name__ == "__main__":
    buyer = test_buyer_agent()