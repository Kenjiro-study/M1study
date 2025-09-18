# base_agent.py
import os, dspy
from typing import Dict, List, Optional

from ..strategies import STRATEGIES, CATEGORY_CONTEXT, INTENT_CONTEXT
from .extractor import PriceExtractor
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

class NegotiationManager(dspy.Signature):
    """The agent in a price negotiation dialogue determines the next action to be taken by taking into account the other party's statements and their intentions.
From the presented options, output only one intent label that is strategically most appropriate.
    # classification criteria (top priority)
    - intro: Greetings or product introductions to start negotiations. Select at the beginning of the dialogue.
    - inquire: Question about the product. 
    - inform: Response to the question. Must be selected only if partner_role is inquire.
    - init-price: Initial price proposal in negotiations. Select this to propose a price when the dialogu_history and partner_intent do not contain an init-price, counter-price, or insist.
    - vague-price: Negotiating without mentioning the price. Select to indirectly convey one's wishes.
    - counter-price: Counter price proposal.
    - insist: Same price claim. Select this if you want to prioritize your margin.
    - supplemental: Supplementary product description. Select this when partner_role is not inquire but you want to provide information about the product.
    - thanks: Word of thanks. Select this to conclude the negotiation if partner_role is agree or thanks"""
    
    # --- 入力フィールド ---
    dialogue_history = dspy.InputField(desc="Dialogue history and its intention labels")
    partner_utterance = dspy.InputField(desc="The previous statement to which you should respond")
    partner_intent = dspy.InputField(desc="Label of the intention of the previous statement to which you should respond")
    partner_role = dspy.InputField(desc="The role of the speaker of the previous statement to which the response should be made")
    agent_role = dspy.InputField(desc="Your role: Buyer or Seller")
    agent_strategy = dspy.InputField(desc="Your strategy for choosing intent. This is merely a selection guideline, and classification criteria take precedence over this.")

    # --- 出力フィールド ---
    next_intent = dspy.OutputField(
        desc="The label of the intention of the next action the agent should take. Select one of the following 9 types: "
             "intro, inquire, inform, init-price, vague-price, counter-price, insist, supplemental, thanks"
    )

# 交渉中に自然言語の応答を生成する
class NegotiationResponse(dspy.Signature):
    """Generates a natural language response during negotiation."""
    complete_prompt: str = dspy.InputField(desc="Full formatted prompt with strategy & context")
    conversation_history: List[Dict] = dspy.InputField()
    action: str = dspy.InputField()
    price: Optional[float] = dspy.InputField("Your proposed price. Be sure to include this price in your response.")
    strategy_name: str = dspy.InputField()
    category: str = dspy.InputField()
    is_buyer: bool = dspy.InputField()

    response: str = dspy.OutputField(desc="natural language response following strategy guidance")

class BaseAgent:
    """
    AgreeMate baseline negotiation system の Base Agent
    買い手側と売り手側の両方の子エージェントが実装するコア機能と抽象メソッドを定義します。
    """

    def __init__(
        self,
        strategy_name: str,
        target_price: float,
        list_price: float,
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
        self.list_price = list_price,
        self.category = category
        self.is_buyer = is_buyer
        self.role = "buyer" if is_buyer else "seller"
        self.lm = lm # 2025/7/15 追加

        # 状態のトラッキング
        self.conversation_history = []
        self.price_history = [] # 自分の価格の履歴
        self.partner_price_history = [] # 相手の価格の履歴
        self.pertner_intent_history = [] # 相手のインテントの履歴
        self.last_action = None
        self.partner_data = None # 2025/9/17 追加
        self.num_turns = 0
        

        # パーサー用
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint = "archive/da_system/agents/parser/model/roberta_fold_1/checkpoint-82304"
        self.parser = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=12)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        # predictor modules のセットアップ
        self.intent_predictor = dspy.ChainOfThought(NegotiationManager)
        self.response_predictor = dspy.ChainOfThought(NegotiationResponse)

        # すべてのモジュールで提供された言語モデルを使用するように DSPy を構成する
        dspy.settings.configure(lm=lm)

    def update_state(self, message: Dict[str, str]) -> Dict:
        """
        LLM extraction を使用して交渉状態を更新する
        StateExtractor を使用して, メッセージから構造化された情報を取得する

        Args:
            message: Dict containing 'role' and 'content' of message
        """
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise ValueError("Invalid message format")

        # 会話状態を更新する
        self.conversation_history.append(message)
        self.num_turns += 1

        # 新しい価格が検出されたら, 価格の状態を更新する
        if message['price'] is not None:
            self.price_history.append(message['price'])
        #self.lm.inspect_history(n=1) ###############################

        # action 状態を更新する
        self.last_action = message['intent']

        return message
    
    def parse_partner_dialogue(self):
        """パートナーの発言を分析する"""
        parser = self.parser.to(self.device)
        with torch.no_grad():
            parser.eval()
            pre_text = (self.conversation_history[-1])['content'] if self.conversation_history else "[PAD]"

            inputs = self.tokenizer(pre_text, self.partner_data['context'], max_length=512, truncation=True, return_tensors="pt")
            inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
            outputs = parser(**inputs)

            logits = outputs.logits # ロジットの取得
            probabilities = softmax(logits, dim=1) # ロジットをソフトマックス関数で確率に変換
            predicted_class = torch.argmax(probabilities, dim=1).item() # 確率が最も高いものを推定ラベルとして決定
            predicted_class = parser.config.id2label[predicted_class] # ラベル番号をダイアログアクトに変換

        self.num_turns += 1 # ターンを一つ進める

        return predicted_class


    def _get_generator_context(self) -> Dict:
        """予測の context を取得する"""
        return {
            "conversation_history": self.conversation_history,
            "target_price": self.target_price,
            "strategy_name": self.strategy['name'],
            "category": self.category,
            "is_buyer": self.is_buyer,
            "num_turns": self.num_turns
        }
    
    def get_manager_context(self) -> Dict:
        """予測の context を取得する"""
        return {
            "dialogue_history": self.conversation_history,
            "partner_utterance": self.partner_data['content'],
            "partner_intent": self.partner_data['intent'],
            "partner_role": self.partner_data['role'],
            "agent_role": self.partner_data,
        }

    def predict_action_manager(self) -> Dict:
        """交渉における次の intent を予測する"""
        prediction = self.intent_predictor(**self.get_manager_context())
        #self.lm.inspect_history(n=1) ###############################
        return {
            "rationale": prediction.rationale,
            "next_intent": prediction.next_intent,
        }

    def prepare_response_generation(self, action: str, price: Optional[float] = None) -> Dict:
        """自然言語の応答を生成する"""
        from ..utils.model_loader import MODEL_CONFIGS

        context = self._get_generator_context()

        # プロンプト用に会話履歴をフォーマットする
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history
        ])
        
        # get prompt template を取得して入力する
        model_name = self.lm.model.split('/')[-1] # 2025/7/15 model_name → model に変更
        template = MODEL_CONFIGS[model_name].prompt_template
        prompt = template.format(
            role=self.role,
            strategy=self.strategy['description'],
            history=history_text,
            target_price=self.target_price,
            item=context.get('item', {'title': 'the item'})['title'] ##### contextからitemの要素をとってくるらしいが, そもそもcontextにはitemというキーは存在していない！どうにかせねば
        )
        
        # strategy-specific guidance を追加する
        prompt += f"\nYour negotiation approach: {self.strategy['initial_approach']}"
        prompt += f"\nCommunication style: {self.strategy['communication_style']}"
        prompt += f"\nCategory context: {self.category_context['market_dynamics']}"
        
        context.update({
            "action": action,
            "price": price,
            "complete_prompt": prompt,
        })

        return context

    def step(self, partner_data, extractor) -> Dict[str, str]:
        """
        交渉ステップを実行する: つまり行動を予測し, 応答を生成する

        Returns:
            応答メッセージのコンテンツと役割を含む辞書
        """
        # パートナーのデータを更新
        self.partner_data = partner_data

        # パーサー
        # まずはここで相手の発言のインテントを予測し, 必要であれば価格を抽出する
        if self.partner_data is not None:
            self.partner_data['intent'] = self.parse_partner_dialogue()

            # 相手のインテントが価格交渉に関するものの場合, 価格を抽出
            if self.partner_data['intent'] in ["init-price", "counter-price", "insist"]:
                self.partner_data['price'] = extractor.compiled_extractor(
                    message_content=self.partner_data['content']
                )
            # パートナー情報の更新
            self.conversation_history.append(self.partner_data)
            self.pertner_intent_history.append(self.partner_data['intent'])
            if self.partner_data['price'] != None:
                self.predict_price_history.append(self.partner_data['price'])


        # マネージャー
        # 次に自分の応答のインテントを考え, 戦略を決定する
        prediction = self.predict_action_manager()

        # ジェネレーター
        # 自然言語の応答を生成する
        context = self.prepare_response_generation(
            prediction["intent"], 
            prediction["price"]
        )
        response_prediction = self.response_predictor(**context)
        #self.lm.inspect_history(n=1) ###############################

        # メッセージを作成する
        message = {
            "role": self.role,
            "content": response_prediction.response,
            "price": prediction.price,
            "intent": prediction.intent
        }

        # 自分自身の状態を更新する
        print("partner_data: ", partner_data)
        message = self.update_state(message, partner_data)

        print(f"target_price: {self.target_price}")
        print(f"is_buyer: {self.is_buyer}")
        print(f"role: {self.role}")
        print(f"convesation_history: {self.conversation_history}")
        print(f"price_history: {self.price_history}")
        print(f"last_action: {self.last_action}")
        print(f"num_turns: {self.num_turns}")


        return message


def test_base_agent():
    """BaseAgent の機能をテストする"""
    baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agreemate_dir = os.path.dirname(baseline_dir)
    pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")
    
    #test_lm = dspy.LM(
        #model="openai/llama3.1", # llama3.1という名前だが一応llama-3.1-8Bらしい
        #api_base="http://localhost:11434/v1",
        #api_key="",
        #cache_dir=pretrained_dir
    #)
    test_lm = dspy.LM(
        model="ollama/llama3.1",
        provider="ollama",
        cache_dir=pretrained_dir,
    )

    agent = BaseAgent(
        strategy_name="length",
        target_price=100.0,
        category="electronics",
        is_buyer=True,
        lm=test_lm,
    )
    assert agent.role == "buyer"
    assert agent.strategy["name"] == "length"

    # 状態更新のテスト
    message = {
        "role": "seller",
        "content": "I can offer it for $150"
    }
    agent.update_state(message)
    assert len(agent.conversation_history) == 1
    assert agent.num_turns == 1

    # step のテスト
    response = agent.step()
    assert "role" in response
    assert "content" in response
    assert response["role"] == "buyer"

    print("✓ All base agent tests passed")
    return agent

if __name__ == "__main__":
    agent = test_base_agent()