# strategies.py

#  LLMの行動を導く高レベルのstrategyの定義
STRATEGIES = {
    "fair": {
        "name": "fair",
        "description": """
            You are a balanced negotiator who:
            - Aims for mutually beneficial outcomes
            - Makes reasonable initial offers
            - Is willing to compromise
            - Values finding a middle ground
            - Maintains professional and friendly tone
            - Considers market value and category norms
            - Explains rationale for offers clearly
        """,
        "initial_approach": "Start with a reasonable offer based on market value",
        "counter_offer_style": "Make measured moves toward middle ground",
        "communication_style": "Clear, professional, and solution-focused",
        "price_buyer_style":"""You are a fair negotiator who: 
- Use balanced intent to negotiate fairly
""",
        "price_seller_style":"""You are a fair negotiator who: 
- Use balanced intent to negotiate fairly
""",
        "info_buyer_style":"""You are a fair negotiator who: 
- Use balanced intent to negotiate fairly
""",
        "info_seller_style":"""You are a fair negotiator who: 
- Use balanced intent to negotiate fairly
""",
        "risk_tolerance": "moderate",
        "patience": "moderate"
    },

    "utility": {
        "name": "utility", 
        "description": """
            You are a tough negotiator who:
            - Prioritizes maximizing your own value
            - Makes assertive initial offers
            - Concedes ground slowly and carefully
            - Emphasizes your position's strengths
            - Maintains firm but professional tone
            - Leverages market knowledge strategically
            - May walk away if target not met
        """,
        "initial_approach": "Start with an ambitious offer favoring your position",
        "counter_offer_style": "Make minimal concessions, hold ground firmly",
        "communication_style": "Direct, confident, and firm",
        "price_buyer_style":"""You are a tough negotiator who: 
- you will actively use a variety of tactics (like counter-price and insist) to secure profits.
""",
        "price_seller_style":"""You are a tough negotiator who: 
- you will actively use a variety of tactics (like counter-price and insist) to secure profits.
""",
        "info_buyer_style":"""You are a tough negotiator who: 
- Be proactive in making your init-price proposal and expressing your expectations.
""",
        "info_seller_style":"""You are a tough negotiator who: 
- Be proactive in making your init-price proposal and expressing your expectations.
""",
        "risk_tolerance": "high",
        "patience": "high"
    },

    "length": {
        "name": "length",
        "description": """
            You are a collaborative negotiator who:
            - Prioritizes reaching an agreement
            - Makes welcoming initial offers
            - Readily offers meaningful concessions
            - Focuses on shared benefits
            - Maintains warm and friendly tone
            - Emphasizes relationship building
            - Works actively toward consensus
        """,
        "initial_approach": "Start with an inviting, relationship-building offer",
        "counter_offer_style": "Make generous moves toward agreement",
        "communication_style": "Warm, friendly, and collaborative", 
        "price_buyer_style":"""You are a clever negotiator who: 
- When negotiating prices, we will negotiate tenaciously using not only the init-price and counter-price but also the vague-price and supplemental.
""",
        "price_seller_style":"""You are a clever negotiator who: 
- When negotiating prices, we will negotiate tenaciousl using not only the init-price and counter-price but also the vague-price and supplemental.
""",
        "info_buyer_style":"""You are a clever negotiator who: 
- Use inquire proactively to seek room for negotiation
""",
        "info_seller_style":"""You are a clever negotiator who: 
- Even if the other person doesn't ask any questions, actively use supplemental words to explain the merits of the product.
""",
        "risk_tolerance": "low",
        "patience": "low"
    },

    "free": {
        "name": "free",
        "description": """
            You are a free negotiator who:
            - This strategy is for human negotiators
        """,
        "initial_approach": "free",
        "counter_offer_style": "free",
        "communication_style": "free",
        "manager_style":"free",
        "risk_tolerance": "free",
        "patience": "free"
    },
}

# 交渉をさらに進めるための category-specific なコンテキスト
CATEGORY_CONTEXT = {
    "electronics": {
        "market_dynamics": """
            - Highly competitive market
            - Regular price changes and sales
            - Strong price comparison shopping
            - Technical specifications matter
            - Warranties often negotiable
        """,
        "negotiation_norms": "Common and expected, but margins typically tight"
    },

    "vehicles": {
        "market_dynamics": """
            - High-value items with negotiation expected
            - Condition and mileage crucial
            - Seasonal price variations
            - Multiple components to negotiate
            - Trade-ins often part of deal
        """,
        "negotiation_norms": "Standard practice with significant room for discussion"
    },

    "furniture": {
        "market_dynamics": """
            - Condition and style important
            - Delivery costs factor in
            - Some seasonal variation
            - Quick turnover desired
            - Display items negotiable
        """,
        "negotiation_norms": "Common on non-retail items, moderate flexibility"
    },

    "housing": {
        "market_dynamics": """
            - Location heavily impacts value
            - Market conditions crucial
            - Long-term implications
            - Multiple terms to negotiate
            - Timing often important
        """,
        "negotiation_norms": "Complex negotiations with many factors to consider"
    }
}

# intentごとの説明
BUYER_INTENT_CONTEXT = {
    # --- 交渉の開始と情報収集 ---
    "intro": "Greet the seller briefly or express your interest in their product briefly.",
    "inquire": "Briefly ask the seller specific questions about the item (e.g., condition, usage, accessories, shipping).",
    "inform": "Answer a question concisely from the seller. Provide the requested information clearly.",
    "supplemental": "Briefly provide supplementary information (e.g., your reason for wanting to buy, your budget, etc) to support your price or request. This is for justification, not a direct offer.",
    # --- 価格交渉（offer_price が必須） ---
    "init-price": "Concisely Make the *first* price proposal. Your response *must* include the `offer_price`.",
    "counter-price": "Concisely Make a counter-offer in response to the seller. Your response *must* include the `offer_price`.",
    "insist": "Re-state your previous `offer_price`. Hold your ground.",
    # --- 価格交渉（offer_price を使わない） ---
    "vague-price": "Negotiate the price concisely *without* making a specific offer. (e.g., 'Can you lower the price?', 'What's your best offer?'). Do *not* include an `offer_price`.",
    # --- 交渉中の応答 ---
    "disagree": "Reject the seller's *current* offer or proposal, *but continue* the negotiation. (e.g., 'That price is still too high.').",
    "agree": "Explicitly accept the seller's *current* offer or price. This signals the price negotiation is over, but does not end the chat.",
    "thanks": "A simple, polite expression of thanks during the negotiation. (e.g., 'Thank you.').",
    # --- 交渉の終了 ---
    "accept": "Formally accept the agreed-upon deal. Express gratitude and finalize the negotiation. (e.g., 'Great, I'll take it for $X. Thank you!').",
    "reject": "Formally *end the negotiation without a deal*. Politely inform the seller that you are walking away. (e.g., 'I understand, but I will pass this time. Thank you.')."
}

SELLER_INTENT_CONTEXT = {
    # --- 交渉の開始と情報収集 ---
    "intro": "Greet the buyer briefly.",
    "inquire": "Briefly ask the buyer if they have any specific questions about the product.",
    "inform": "Answer a question concisely from the buyer. Provide the requested information clearly.",
    "supplemental": "Briefly provide supplementary information (e.g., market price, item flaws, etc) to support your price or request. This is for justification, not a direct offer.",
    # --- 価格交渉（offer_price が必須） ---
    "init-price": "Concisely make the *first* price proposal. Your response *must* include the `offer_price`.",
    "counter-price": "Concisely make a counter-offer in response to the buyer. Your response *must* include the `offer_price`.", #and provide a brief reason.
    "insist": "Re-state your previous `offer_price`. Hold your ground.",
    # --- 価格交渉（offer_price を使わない） ---
    "vague-price": "Negotiate the price concisely *without* making a specific offer. (e.g., 'Can you lower the price?', 'What's your best offer?'). Do *not* include an `offer_price`.",
    # --- 交渉中の応答 ---
    "disagree": "Reject the buyer's *current* offer or proposal, *but continue* the negotiation. (e.g., 'That price is still too low.').",
    "agree": "Explicitly accept the buyer's *current* offer or price. This signals the price negotiation is over, but does not end the chat.",
    "thanks": "A simple, polite expression of thanks during the negotiation. (e.g., 'Thank you.').",
    # --- 交渉の終了 ---
    "accept": "Formally accept the agreed-upon deal. Express gratitude and finalize the negotiation. (e.g., 'Great, I'll take it for $X. Thank you!').",
    "reject": "Formally *end the negotiation without a deal*. Politely inform the buyer that you are walking away. (e.g., 'I understand, but I will pass this time. Thank you.')."
}

BUYER_LANGUAGE_SKILLS = {
    "Emphasis": "Highlight the cost value, quality or highest price of the product to show the rationality of the pricing.",
    "Emotional Strategy": "Use humor, expressions, complaints, and identity recognition to resonate with the other party.",
    "Compare the Market": "Compare the product with other products on the market to justify your proposed price.",
    "Transaction Guarantee": "Promise to ensure transaction security and reliability by not returning or cancelling or by sharing our past transaction history.",
    "Create Urgency": "Create urgency by reiterating the possibility of a more advanced version of the product or a price may drop soon.",
    "Chat": "Do not use techniques and simply reply to the other party."
}

SELLER_LANGUAGE_SKILLS = {
    "Emphasis": "Highlight the cost value, quality or bottom price of the product to show the rationality of the pricing.",
    "Added Value": "Provide additional value beyond the product, such as gifts, free shipping, etc.",
    "Emotional Strategy": "Use humor, expressions, complaints, and identity recognition to resonate with the other party.",
    "Compare the Market": "Compare the product with other products on the market to highlight the advantages of its own products.",
    "Transaction Guarantee": "Promise to ensure transaction security and reliability by offering good after-sales service.",
    "Create Urgency": "Create urgency by reminding that the product may sell out soon or prices may rise shortly.",
    "Chat": "Do not use techniques and simply reply to the other party."
}


def test_strategies():
    """戦略の定義が完了していることを確認するための簡単なテスト"""
    required_fields = [
        "name", "description", "initial_approach", 
        "counter_offer_style", "communication_style",
        "risk_tolerance", "patience"
    ]

    for strategy_name, strategy in STRATEGIES.items():
        print(f"\nTesting {strategy_name} strategy:")
        for field in required_fields:
            assert field in strategy, f"Missing {field} in {strategy_name}"
            print(f"✓ Has {field}")

    print("\nTesting category contexts:")
    for category, context in CATEGORY_CONTEXT.items():
        assert "market_dynamics" in context, f"Missing market_dynamics in {category}"
        assert "negotiation_norms" in context, f"Missing negotiation_norms in {category}"
        print(f"✓ {category} context complete")

if __name__ == "__main__":
    test_strategies()