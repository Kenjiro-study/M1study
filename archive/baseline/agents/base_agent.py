# base_agent.py
import os, dspy
from typing import Dict, List, Optional

from ..strategies import STRATEGIES, CATEGORY_CONTEXT

# 交渉メッセージから構造化された状態情報を抽出する
class StateExtractor(dspy.Signature):
    """Extracts structured state information from negotiation messages."""
    message_content: str = dspy.InputField()
    is_buyer: bool = dspy.InputField()

    extracted_price: Optional[float] = dspy.OutputField(desc="price mentioned in message, if any")
    detected_action: str = dspy.OutputField(desc="detected action: offer/counter/accept/reject/none")
    reasoning: str = dspy.OutputField(desc="explanation of extraction")

# 交渉会話の状態をトラッキングする, 現在の状態に基づいて交渉における次のアクションを予測するために使用される
class NegotiationState(dspy.Signature):
    """
    Tracks the state of a negotiation conversation.
    Used to predict next action in negotiation based on current state.
    """
    conversation_history: List[Dict] = dspy.InputField()
    target_price: float = dspy.InputField()
    current_price: float = dspy.InputField()
    strategy_name: str = dspy.InputField()
    category: str = dspy.InputField()
    is_buyer: bool = dspy.InputField()
    num_turns: int = dspy.InputField()

    rationale: str = dspy.OutputField(desc="reasoning about next action")
    action: str = dspy.OutputField(desc="next action: accept/reject/counter")
    counter_price: Optional[float] = dspy.OutputField(desc="if action is counter, the counter-offer price")

# 交渉中に自然言語の応答を生成する
class NegotiationResponse(dspy.Signature):
    """Generates a natural language response during negotiation."""
    complete_prompt: str = dspy.InputField(desc="Full formatted prompt with strategy & context")
    conversation_history: List[Dict] = dspy.InputField()
    action: str = dspy.InputField()
    price: Optional[float] = dspy.InputField()
    strategy_name: str = dspy.InputField()
    category: str = dspy.InputField()
    is_buyer: bool = dspy.InputField()

    response: str = dspy.OutputField(desc="natural language response following strategy guidance")

# 発話が割り当てられられた交渉戦略にどの程度準拠しているかを分析する, この signature はエージェントが一貫した戦略実行を維持できるようにするために役立つ
class StrategyAnalysis(dspy.Signature):
    """
    Analyzes how well an utterance adheres to assigned negotiation strategy.
    This signature helps ensure agents maintain consistent strategy execution.
    """
    message: str = dspy.InputField(desc="message to analyze")
    assigned_strategy: str = dspy.InputField(desc="strategy name from STRATEGIES")
    role: str = dspy.InputField(desc="buyer or seller")
    context: Optional[Dict] = dspy.InputField(desc="additional context", default=None)

    adherence_score: float = dspy.OutputField(desc="strategy adherence score (0-1)")
    analysis: str = dspy.OutputField(desc="explanation of scoring")
    detected_tactics: List[str] = dspy.OutputField(desc="identified negotiation tactics")

# メッセージ内で使用されている特定の交渉戦術を検出する, 交渉行動のトラッキングと分析に役立つ
class TacticDetection(dspy.Signature):
    """
    Detects specific negotiation tactics used in messages.
    Helps track and analyze negotiation behaviors.
    """
    message: str = dspy.InputField(desc="message to analyze")
    strategy_context: Optional[str] = dspy.InputField(desc="assigned strategy", default=None)
    
    tactics: List[str] = dspy.OutputField(desc="detected negotiation tactics")
    confidence: List[float] = dspy.OutputField(desc="confidence scores for detections")
    explanation: str = dspy.OutputField(desc="reasoning for detected tactics")

# 交渉メッセージの言語特性を分析する, sophistication(洗練度), emotional content(感情的内容), 及び persuasion attempts(説得の試み)をトラッキングする
class LanguageAnalysis(dspy.Signature):
    """
    Analyzes language characteristics of negotiation messages.
    Tracks sophistication, emotional content, and persuasion attempts.
    """
    text: str = dspy.InputField(desc="text to analyze")
    context: Dict = dspy.InputField(desc="relevant context info")
    
    complexity_score: float = dspy.OutputField(desc="language complexity score (0-1)")
    emotional_content: Dict[str, float] = dspy.OutputField(desc="emotion scores")
    persuasion_techniques: List[str] = dspy.OutputField(desc="identified techniques")
    coherence_score: float = dspy.OutputField(desc="response coherence score (0-1)")

class BaseAgent:
    """
    AgreeMate baseline negotiation system の Base Agent
    買い手側と売り手側の両方の子エージェントが実装するコア機能と抽象メソッドを定義します。
    """

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

        # predictor modules のセットアップ
        self.state_predictor = dspy.ChainOfThought(NegotiationState)
        self.response_predictor = dspy.ChainOfThought(NegotiationResponse)
        self.state_extractor = dspy.ChainOfThought(StateExtractor)

        # すべてのモジュールで提供された言語モデルを使用するように DSPy を構成する
        dspy.settings.configure(lm=lm)

    def update_state(self, message: Dict[str, str]):
        """
        LLM extraction を使用して交渉状態を更新する
        StateExtractor を使用して, メッセージから構造化された情報を取得する

        Args:
            message: Dict containing 'role' and 'content' of message
        """
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise ValueError("Invalid message format")

        # LLM を使用して構造化された情報を抽出する
        extraction = self.state_extractor(
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

        # action 状態を更新する
        self.last_action = extraction.detected_action

        # 必要に応じてデバッグ情報に抽出理由を追加する
        if hasattr(self, 'extraction_history'):
            self.extraction_history.append(extraction.reasoning)

    def _get_prediction_context(self) -> Dict:
        """予測の context を取得する"""
        return {
            "conversation_history": self.conversation_history,
            "target_price": self.target_price,
            "current_price": self.current_price,
            "strategy_name": self.strategy["name"],
            "category": self.category,
            "is_buyer": self.is_buyer,
            "num_turns": self.num_turns
        }

    def predict_action(self) -> Dict:
        """
        交渉における次の action を予測する

        Returns:
            action の予測とその根拠を含む辞書
        """
        prediction = self.state_predictor(**self._get_prediction_context())
        return {
            "rationale": prediction.rationale,
            "action": prediction.action,
            "counter_price": prediction.counter_price
        }

    def analyze_strategy_adherence(self, message: Dict[str, str]) -> Dict:
        """
        メッセージが割り当てられた戦略にどの程度準拠しているかを分析する

        Args:
            message: 役割とコンテンツを含むメッセージの辞書

        Returns:
            adherence analysis を含む辞書
        """
        analysis = self.state_analyzer(
            message=message['content'],
            assigned_strategy=self.strategy['name'],
            role=self.role,
            context={
                'category': self.category,
                'target_price': self.target_price,
                'current_price': self.current_price
            }
        )

        return {
            'adherence_score': analysis.adherence_score,
            'analysis': analysis.analysis,
            'tactics': analysis.detected_tactics
        }

    def analyze_language(self, message: str) -> Dict:
        """
        メッセージの言語特性を分析する

        Args:
            message: 分析するメッセージテキスト

        Returns:
            言語分析結果を含む辞書
        """
        analysis = self.language_analyzer(
            text=message,
            context={
                'category': self.category,
                'strategy': self.strategy['name'],
                'role': self.role
            }
        )

        return {
            'complexity': analysis.complexity_score,
            'emotions': analysis.emotional_content,
            'techniques': analysis.persuasion_techniques,
            'coherence': analysis.coherence_score
        }

    def generate_response(self, action: str, price: Optional[float] = None) -> str:
        """自然言語の応答を生成する"""
        from ..utils.model_loader import MODEL_CONFIGS

        context = self._get_prediction_context()

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
            item=context.get('item', {'title': 'the item'})['title']
        )
        
        # strategy-specific guidance を追加する
        prompt += f"\n\nYour negotiation approach: {self.strategy['initial_approach']}"
        prompt += f"\nCommunication style: {self.strategy['communication_style']}"
        prompt += f"\nCategory context: {self.category_context['market_dynamics']}"
        
        context.update({
            "complete_prompt": prompt,
            "action": action,
            "price": price
        })
        
        prediction = self.response_predictor(**context)
        #self.lm.inspect_history(n=1) # LLMに与えられるプロンプトを知りたい場合はコメントアウトを外す！
        
        return prediction.response

    def step(self) -> Dict[str, str]:
        """
        交渉ステップを実行する: つまり行動を予測し, 応答を生成する

        Returns:
            応答メッセージのコンテンツと役割を含む辞書
        """

        # 次の action を予測する
        prediction = self.predict_action()
        print("prediction: ", prediction)

        # 自然言語の応答を生成する
        response = self.generate_response(
            prediction["action"], 
            prediction["counter_price"]
        )

        # メッセージを作成する
        message = {
            "role": self.role,
            "content": response
        }

        # 自分自身の状態を更新する
        self.update_state(message)

        return message


def test_base_agent():
    """BaseAgent の機能をテストする"""
    baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agreemate_dir = os.path.dirname(baseline_dir)
    pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")
    
    test_lm = dspy.LM(
        model="openai/llama3.1", # llama3.1という名前だが一応llama-3.1-8Bらしい
        api_base="http://localhost:11434/v1",
        api_key="",
        cache_dir=pretrained_dir
    )
    #test_lm = dspy.LM(
        #model="ollama/llama3.1",
        #provider="ollama",
        #cache_dir=pretrained_dir,
    #)

    agent = BaseAgent(
        strategy_name="cooperative",
        target_price=100.0,
        category="electronics",
        is_buyer=True,
        lm=test_lm,
    )
    assert agent.role == "buyer"
    assert agent.strategy["name"] == "cooperative"

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

if __name__ == "__main__":
    agent = test_base_agent()