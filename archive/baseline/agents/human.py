# human.py
import os, dspy
from typing import Dict, List, Optional

from ..strategies import STRATEGIES, CATEGORY_CONTEXT

# 交渉メッセージから構造化された状態情報を抽出する
class PriceExtractor(dspy.Signature):
    """Extracts price information from negotiation messages."""
    message_content: str = dspy.InputField()
    is_buyer: bool = dspy.InputField(desc="True if you are the buyer in the negotiation, False if you are the seller")

    extracted_price: Optional[float] = dspy.OutputField(desc="The price proposed in the message, if any")
    reasoning: str = dspy.OutputField(desc="explanation of extraction")

def _create_train_examples():
    # 学習データを返すヘルパー関数
    return[     
        dspy.Example(message_content="hey ! i can offer $150 for the credenza .", is_buyer = "True", extracted_price="150", reasoning="The buyer states that he can buy the item for $150.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="i do ! but since i'm a bargain shower i'd like to offer you thirty dollors.", is_buyer = "True", extracted_price="30", reasoning="The buyer states in English, not in numbers, that he can buy the item for $30.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="i see you are asking 1100 but i am really on a budget , would you take 650 ?", is_buyer = "True", extracted_price="650", reasoning="The buyer confirms that the seller has offered $1,100 and then offers $650.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="$300 is too expensive. It's old and damaged, how about $250?", is_buyer = "True", extracted_price="250", reasoning="The buyer rejects the seller's offer of $300 as too high and states his own asking price of $250.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="how many miles are on this 2008 ? have they been in town or on the highway ?", is_buyer = "True", extracted_price="None", reasoning="No price was mentioned for the 2008 model year car.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="How many years ago did you purchase this product?", is_buyer = "True", extracted_price="None", reasoning="The question is about the condition of the product, and there is no mention of price.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="I can pick it up tomorrow at 12 noon.", is_buyer = "True", extracted_price="None", reasoning="This is a statement about receiving the product, and does not mention the price.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="i'd be willing to meet you in the middle for $35 ?", is_buyer = "False", extracted_price="35", reasoning="The seller is willing to compromise on $35 to complete the deal.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="was hoping for something around 800 dollars but it's really no use to me so i'd be willing to go lower", is_buyer = "False", extracted_price="800", reasoning="The seller offers $800, but says there's room for negotiation and is willing to go lower.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="i would offer you fifty dollars", is_buyer = "False", extracted_price="50", reasoning="The seller states in English, not in numbers, that he can sell the item for $50.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="It certainly won't sell for $100. Even considering the condition, the maximum I can compromise on is $150.", is_buyer = "False", extracted_price="150", reasoning="The seller rejects the buyer's offer of $100 and proposes $150 as his minimum negotiable price.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="$1600 would be too low . montclaire is a very up and comming neighborhood . utilities are cheaper there due to the prevailing winds .", is_buyer = "False", extracted_price="None", reasoning="The seller rejects the buyer's proposed offer of $1,600 as too low, but has not offered the price he is seeking.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="well there are a lot of aspects to this car ! it is running in great condition and is basically new . it has 10 , 000 miles on the baby !", is_buyer = "False", extracted_price="None", reasoning="The seller mentions the condition and mileage of the car they are selling, but not the price.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="This item was purchased three years ago. It is brand new, unopened, and undamaged.", is_buyer = "False", extracted_price="None", reasoning="The seller mentions the condition of the item and when it was purchased, but not the price.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="This shabby chic desk is made of solid wood in a Parisian style. Its dimensions are 17cm deep x 30cm wide x 30cm high.", is_buyer = "False", extracted_price="None", reasoning="The seller only describes the size of the item and does not mention the price.").with_inputs("message_content", "is_buyer"),
        dspy.Example(message_content="i could give a 100 discount", is_buyer = "False", extracted_price="None", reasoning="The seller only states the discount amount, but does not mention the actual price.").with_inputs("message_content", "is_buyer")
    ]

class HumanAgent:
    """
    AgreeMate baseline negotiation system の Human Agent
    人間が買い手側と売り手側のどちらかのエージェントの役割を果たす場合の機能と抽象メソッドを定義します。
    """
    _compiled_extractor = None

    @classmethod
    def _get_compiled_extractor(cls):
        if cls._compiled_extractor is None:
            from dspy.teleprompt import BootstrapFewShot

            extractor = dspy.ChainOfThought(PriceExtractor)
            train_examples = _create_train_examples()
            
            optimizer = BootstrapFewShot()
            # コンパイルを実行
            cls._compiled_extractor = optimizer.compile(student=extractor, trainset=train_examples)
        return cls._compiled_extractor

    def __init__(
        self,
        strategy_name: str,
        target_price: float,
        category: str,
        is_buyer: bool,
        lm: dspy.LM,
    ):
        """
        negotiation agent を初期化する

        Args:
            strategy_name: STRATEGIES の戦略名
            target_price: このエージェントの目標価格
            category: 商品のカテゴリー (electronics, vehicles, etc)
            is_buyer: buyer (True) であるか seller (False) であるか
            lm: 応答生成のための DSPy 言語モデル 
        """
        if strategy_name not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        self.strategy = STRATEGIES[strategy_name]
        self.category_context = CATEGORY_CONTEXT[category]
        self.target_price = target_price
        self.category = category
        self.is_buyer = is_buyer
        self.role = "buyer" if is_buyer else "seller"
        self.lm = lm # 2025/7/15 追加

        # 状態のトラッキング
        self.conversation_history = []
        self.price_history = []
        self.roles_sequence = []
        self.last_action = None
        self.current_price = None
        self.num_turns = 0

        # すべてのモジュールで提供された言語モデルを使用するように DSPy を構成する
        dspy.settings.configure(lm=lm)

        # predictor modules のセットアップ
        self.price_extractor = dspy.ChainOfThought(PriceExtractor)
        self.compiled_extractor =  self._get_compiled_extractor()

    def update_state(self, message: Dict[str, str]) -> Dict:
        """
        LLM extraction を使用して交渉状態を更新する
        StateExtractor を使用して, メッセージから構造化された情報を取得する

        Args:
            message: Dict containing 'role' and 'content' of message
        """
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise ValueError("Invalid message format")

        # LLM を使用して構造化された情報を抽出する
        extraction = self.compiled_extractor(
            message_content=message['content'],
            is_buyer=self.is_buyer
        )

        # 会話状態を更新する
        self.conversation_history.append(message)
        self.roles_sequence.append(message['role'])
        self.num_turns += 1

        # 新しい価格が検出されたら, 価格の状態を更新する
        if extraction.extracted_price is not None:
            self.current_price = extraction.extracted_price
            self.price_history.append(extraction.extracted_price)
        self.lm.inspect_history(n=1) ###############################

        # action 状態を更新する
        # 人間が交渉を受け入れたり断ったりする時はacceptかrejectと入力する
        if message["content"] == "accept":
            self.last_action = "accept"
        elif message["content"] == "reject":
            self.last_action = "reject"
        else:
            self.last_action = "none" #とりあえずnoneにしておく, 後でDLベースパーサーによって判定してもらうよう変更する

        # 必要に応じてデバッグ情報に抽出理由を追加する
        if hasattr(self, 'extraction_history'):
            self.extraction_history.append(extraction.reasoning)

        message.update({
            "price": extraction.extracted_price,
            "status": self.last_action,
        })

        return message

    def step(self) -> Dict[str, str]:
        """
        交渉ステップを実行する: つまり行動を予測し, 応答を生成する

        Returns:
            応答メッセージのコンテンツと役割を含む辞書
        """

        # 自然言語の応答を生成する

        user_response = input(f"Your turn! Please your message as a {self.role}: ")

        # メッセージを作成する
        message = {
            "role": self.role,
            "content": user_response
        }

        # 自分自身の状態を更新する
        message = self.update_state(message)

        return message


def test_base_agent():
    """BaseAgent の機能をテストする"""
    baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agreemate_dir = os.path.dirname(baseline_dir)
    pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")
    
    test_lm = dspy.LM(
        model="ollama/llama3.1",
        provider="ollama",
        cache_dir=pretrained_dir,
    )

    agent = HumanAgent(
        strategy_name="free",
        target_price=100.0,
        category="electronics",
        is_buyer=True,
        lm=test_lm,
    )
    assert agent.role == "buyer"
    assert agent.strategy["name"] == "free"

    # 状態更新のテスト
    message = {
        "role": "seller",
        "content": "I can offer it for $150"
    }
    agent.update_state(message)
    assert agent.current_price == 150.0
    assert len(agent.conversation_history) == 1
    assert agent.num_turns == 1

    # step のテスト
    response = agent.step()
    assert "role" in response
    assert "content" in response
    assert response["role"] == "buyer"

    print("✓ All base agent tests passed")
    return agent

def test_extractor():
    """PriceExtractor の機能をテストする"""
    baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agreemate_dir = os.path.dirname(baseline_dir)
    pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")
    
    test_lm = dspy.LM(
        model="ollama/llama3.1",
        provider="ollama",
        cache_dir=pretrained_dir,
    )

    agent = HumanAgent(
        strategy_name="free",
        target_price=100.0,
        category="electronics",
        is_buyer=True,
        lm=test_lm,
    )

    while True:
        user_input = input("入力(exitで終了): ")

        if user_input == "exit":
            break
        else:
            compiled_extraction = agent.compiled_extractor(
                message_content=user_input,
                is_buyer=True
                #is_buyer=False
            )
            extraction = agent.price_extractor(
                message_content=user_input,
                is_buyer=True
                #is_buyer=False
            )
            print(f"compiled → price: {compiled_extraction.extracted_price}, reason: {compiled_extraction.reasoning}")
            print(f"normal → price: {extraction.extracted_price}, reason: {extraction.reasoning}")

if __name__ == "__main__":
    #agent = test_base_agent()
    agent = test_extractor()