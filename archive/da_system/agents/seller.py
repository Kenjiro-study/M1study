# seller.py
from typing import Dict, Optional
import dspy, random, math

from .base_agent import BaseAgent
from .extractor import PriceExtractor

class SellerAgent(BaseAgent):
    """
    AgreeMate baseline negotiation system の seller agent
    seller-specific の交渉行動と戦略の解釈を実装する
    """

    def __init__(
        self,
        strategy_name: str,
        target_price: float,
        list_price: float,
        category: str,
        min_price: Optional[float] = None,
        lm: dspy.LM = None
    ):
        """
        seller agent を初期化する
        
        Args:
            strategy_name: STRATEGIES の戦略名
            target_price: seller の目標販売価格
            category: 商品のカテゴリー
            min_price: 最低許容価格 (デフォルト値は target より 10%低い価格)
            lm: 応答生成用の DSPy 言語モデル
        """
        super().__init__(
            strategy_name=strategy_name,
            target_price=target_price,
            list_price=list_price,
            category=category,
            is_buyer=False,
            lm=lm
        )

        self.min_price = min_price or (target_price * 0.9)
        self.best_offer_seen = 0 # 最高額のオファーをトラッキング

    def min_price_select(self) -> float:
        """性格ごとの最低価格の設定"""
        if self.strategy_name == "fair":
            min_price = self.list_price * random.uniform(0.8, 0.5)
        elif self.strategy_name == "utility":
            min_price = self.list_price * random.uniform(0.9, 0.8)
        elif self.strategy_name == "length":
            min_price = self.list_price * random.uniform(0.9, 0.7)
        else:
            raise ValueError("Invalid strategy name")

        return round(min_price, 0)
    
    def get_manager_context(self) -> Dict:
        """予測の context を取得する"""
        context = super().get_manager_context()
        context.update({
            "agent_strategy": self.strategy['seller_manager_style'],
        })
        return context
    
    def fair_manager(self) -> Dict:
        if self.partner_data['price'] != None:
            if (self.partner_data['price'] >= self.target_price) or ((len(self.partner_price_history) >= 2) and self.price_history and (self.partner_data['price'] >= (0.4 * self.price_history[-1] + 0.6 * self.partner_price_history[-2]))):
                return{
                    "intent": "agree",
                    "price": None
                }
            elif self.pertner_intent_history.count("counter-price") + self.pertner_intent_history.count("insist") == 4:
                return{
                    "intent": "disagree",
                    "price": None
                }
            elif self.price_history[-1] == self.min_price:
                return{
                    "intent": "insist",
                    "price": self.min_price
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
            elif not self.partner_price_history:
                price = 0.9 * self.price_history[-1]
            else:
                price = 0.7 * self.price_history[-1] + 0.3 * self.partner_price_history[-1]

            # 最低価格を下回っていたら最低価格に設定
            if self.min_price >= price:
                price = self.min_price

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
            if (self.partner_data['price'] >= self.target_price) or ((len(self.partner_price_history) >= 2) and self.price_history and (self.partner_data['price'] >= (0.7 * self.price_history[-1] + 0.3 * self.partner_price_history[-2]))):
                return{
                    "intent": "agree",
                    "price": None
                }
            elif self.pertner_intent_history.count("counter-price") + self.pertner_intent_history.count("insist") == 3:
                return{
                    "intent": "disagree",
                    "price": None
                }
            elif self.price_history[-1] == self.min_price:
                return{
                    "intent": "insist",
                    "price": self.min_price
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
            elif not self.partner_price_history:
                price = 0.95 * self.price_history[-1]
            else:
                price = 0.9 * self.price_history[-1] + 0.1 * self.partner_price_history[-1]

            # 最低価格を下回っていたら最低価格に設定
            if self.min_price >= price:
                price = self.min_price

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
            if (self.partner_data['price'] >= self.target_price) or ((len(self.partner_price_history) >= 2) and self.price_history and (self.partner_data['price'] >= (0.6 * self.price_history[-1] + 0.4 * self.partner_price_history[-2]))):
                return{
                    "intent": "agree",
                    "price": None
                }
            elif self.pertner_intent_history.count("counter-price") + self.pertner_intent_history.count("insist") == 5:
                return{
                    "intent": "disagree",
                    "price": None
                }
            elif self.price_history[-1] == self.min_price:
                return{
                    "intent": "insist",
                    "price": self.min_price
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
            elif not self.partner_price_history:
                price = 0.95 * self.price_history[-1]
            else:
                price = 0.8 * self.price_history[-1] + 0.2 * self.partner_price_history[-1]

            # 最低価格を下回っていたら最低価格に設定
            if self.min_price >= price:
                price = self.min_price

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
        strategy_name="length",
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
    assert seller.best_offer_seen == 90.0

    # counter-offer 生成のテスト
    response = seller.step()
    assert response["role"] == "seller"
    assert "content" in response

    print("✓ All seller agent tests passed")
    return seller

if __name__ == "__main__":
    seller = test_seller_agent()