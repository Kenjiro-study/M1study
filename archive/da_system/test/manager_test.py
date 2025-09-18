# manager_test.py
import os, dspy, json
from typing import Dict, List, Optional
from dspy.evaluate import Evaluate

from ..strategies import STRATEGIES, CATEGORY_CONTEXT
from ..agents.buyer import BuyerAgent

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

def load_examples_from_json(filepath):
    """JSONファイルを読み込み、dspy.Exampleのリストを返す"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        # JSONの各オブジェクトをdspy.Exampleに変換
        example = dspy.Example(
            dialogue_history=item['dialogue_history'],
            partner_utterance=item['partner_utterance'],
            partner_intent=item['partner_intent'],
            partner_role=item['partner_role'],
            agent_role=item['agent_role'],
            next_intent=item.get('next_intent')
        ).with_inputs("dialogue_history", "partner_utterance", "partner_intent", "partner_role", "agent_role")
        
        examples.append(example)
        
    return examples

def test_manager():

    lm = dspy.LM(
        model="ollama/llama3.1",
        provider="ollama",
    )

    dspy.settings.configure(lm=lm)
    manager = dspy.ChainOfThought(NegotiationManager)

    filepath = "archive/da_system/test/manager_val_data.json"
    val_data = load_examples_from_json(filepath)
    #strategy = STRATEGIES["fair"]
    #strategy = STRATEGIES["utility"]
    strategy = STRATEGIES["length"]

    count = 0

    for i, example in enumerate(val_data):
        if example.agent_role == "buyer":
            prediction = manager(
                dialogue_history = example.dialogue_history,
                partner_utterance = example.partner_utterance,
                partner_intent = example.partner_intent,
                partner_role = example.partner_role,
                agent_role = example.agent_role,
                agent_strategy = strategy['buyer_manager_style']
            )
        else:
            prediction = manager(
                dialogue_history = example.dialogue_history,
                partner_utterance = example.partner_utterance,
                partner_intent = example.partner_intent,
                partner_role = example.partner_role,
                agent_role = example.agent_role,
                agent_strategy = strategy['seller_manager_style']
            )
        
        if example.next_intent == prediction.next_intent:
            count += 1
        print(f"--- データ {i+1} ---")
        print("prediction: ", prediction)
        print("pertner_utterance: ", example.partner_utterance)
        print("true_intent: ", example.next_intent)

    print(f"正解率： {count}/50")

    #dialogue_history = [{"role":"buyer", "content":"i'm interested in this item , but i had some questions", "intent":"intro"},{"role":"seller", "content":"geat , ask away.", "intent":"intro"},{"role":"buyer", "content":"do i have to remove it myself?", "intent":"inquire"}, {"role":"seller", "content":"i am renting out the appartment , you \"don't remove anything\"", "intent":"inform"}]
    #partner_utterance = "nice. is it fully furnished?"
    #partner_intent = "inquire"
    #partner_role = "buyer"
    #prediction = manager(dialogue_history=dialogue_history, partner_utterance=partner_utterance, partner_intent=partner_intent, partner_role=partner_role)

    #lm.inspect_history(n=1)

if __name__ == "__main__":
    agent = test_manager()