# seller.py
import dspy, random
from typing import Optional

from .base_agent import BaseAgent
from .base_agent import NegotiationManager
from .extractor import PriceExtractor

# 交渉中に自然言語の応答を生成する
class NegotiationResponse(dspy.Signature):
    """You are a seller. Please negotiate the price of the product described in the item_information. Review the conversation_history to understand the flow of the conversation so far, and then generate a response to the partner_utterance. Your response should follow the given strategy, and if  offer_price is provided, be sure to include that price in your response. Please limit your responses to the main points and keep them as short as possible."""
    item_information: str = dspy.InputField(desc="Product name, category, list price, and detailed description for negotiation")
    conversation_history: str = dspy.InputField(desc="Previous chat history")
    partner_utterance: dict = dspy.InputField(desc="The partner's statement to which we should respond. This includes information on price, role, intended meaning of the statement, and the content of the statement.")
    strategy: str = dspy.InputField(desc="Response strategy. Please generate a response based on this information.")
    offer_price: Optional[float] = dspy.InputField(desc="Your proposed price. If it's not None, please be sure to include this price in your response.")

    response: str = dspy.OutputField(desc="natural language response following strategy guidance")

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
        item_info: dict[str, any],
        min_price: float | None = None,
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
            item_info = item_info, # 2025/9/18 追加
            lm=lm
        )

        self.strategy_name = strategy_name 
        self.min_price = min_price or (target_price * 0.9)

        # predictor modules のセットアップ
        self.response_predictor = dspy.ChainOfThought(NegotiationResponse)
        self.intent_predictor = dspy.ChainOfThought(NegotiationManager)

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
    
    def compute_utility(self, final_price: float, partner_target_price: float) -> float:
        if final_price >= self.target_price:
            return 1.0
        elif final_price <= partner_target_price:
            return (-1.0)

        score_value = 2 * final_price - (self.target_price + partner_target_price)
        target_diff = abs(self.target_price - partner_target_price)
        utility = score_value / target_diff
        return utility
    
    def get_manager_context(self) -> dict:
        """予測の context を取得する"""
        context = super().get_manager_context()
        context.update({
            "agent_strategy": self.strategy['seller_manager_style'],
        })
        return context
    
    def fair_manager(self) -> dict:
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
            elif len(self.price_history) >= 1 and self.price_history[-1] == self.min_price:
                return{
                    "intent": "insist",
                    "price": self.min_price
                }
        
        prediction = self.intent_predictor(**self.get_manager_context())
        intent = (prediction.next_intent).split('\n')[0].strip()

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
    
    def utility_manager(self) -> dict:
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
            elif len(self.price_history) >= 1 and self.price_history[-1] == self.min_price:
                return{
                    "intent": "insist",
                    "price": self.min_price
                }
        
        prediction = self.intent_predictor(**self.get_manager_context())
        intent = (prediction.next_intent).split('\n')[0].strip()

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
    
    def length_manager(self) -> dict:
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
            elif len(self.price_history) >= 1 and self.price_history[-1] == self.min_price:
                return{
                    "intent": "insist",
                    "price": self.min_price
                }
        
        prediction = self.intent_predictor(**self.get_manager_context())
        intent = (prediction.next_intent).split('\n')[0].strip()

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

    def predict_action_manager(self) -> dict:
        if self.last_action == "agree":
            return{
                "intent": "accept",
                "price": None
            }
        elif self.last_action == "disagree":
            return{
                "intent": "reject",
                "price": None
            }
        
        if self.strategy_name == "fair":
            prediction = self.fair_manager()
        elif self.strategy_name == "utility":
            prediction = self.utility_manager()
        elif self.strategy_name == "length":
            prediction = self.length_manager()
        else:
            raise ValueError("Invalid strategy name")

        return prediction
    
    def response_generation(self, intent: str, price: float | None = None) -> dict:
        context = super().response_generation(intent, price)
        response_prediction = self.response_predictor(**context)

        return response_prediction

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

    # counter-offer 生成のテスト
    response = seller.step()
    assert response["role"] == "seller"
    assert "content" in response

    print("✓ All seller agent tests passed")
    return seller

if __name__ == "__main__":
    seller = test_seller_agent()