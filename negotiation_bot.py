# %% [markdown]
# # Llama3 交渉ボット (Dockerコンテナ & VS Codeインタラクティブ実行用)
#
# このスクリプトは、`#%%` で区切られたセルをVS Codeの「インタラクティブウィンドウ」で
# 一つずつ実行することを想定しています。

# %%
# =============================================
# ステップ1: ライブラリのインポート
# =============================================
import torch # 機械学習ライブラリ(GPU計算に必要)
import transformers # Hugging faceのAIモデルライブラリ
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
import os # アクセストークンの取得に使用
from getpass import getpass # アクセストークンの取得に使用

warnings.filterwarnings("ignore") # 警告表示を無くす(※ デバッグするときは外してちゃんと警告を見るように！)
print("✅ step1: ライブラリのインポートが完了しました。")


# %%
# =============================================
# ステップ2: Hugging Faceへのログイン
# =============================================
if 'HUGGING_FACE_HUB_TOKEN' not in os.environ:
    print("Hugging Faceのアクセストークンを入力してください:")
    hf_token = getpass() # getpassで機密情報の入力要求
    os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token # Hugging Faceのアクセストークンを格納する環境変数HUGGING_FACE_HUB_TOKENに入力されたアクセストークンを格納
print("🔑 step2: トークンの設定が完了しました。")


# %%
# =============================================
# ステップ3: 交渉ボットクラスの定義
# =============================================
class NegotiationBot:
    # コンストラクタ(botが作られる時に最初に実行される部分)
    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_id = model_id # モデルの名前
        self.pipeline = None # パイプライン
        self.messages = [] # 対話履歴を保存するリスト
        print(f"ボットの設計図（クラス）を準備しました。モデルID: {self.model_id}")

    def load_model(self):
        if self.pipeline:
            print("モデルは既にロードされています。") # もしすでにパイプラインが存在していればモデルがすでにロードされているのでスキップ
            return
            
        print(f"モデルのロードを開始します: {self.model_id}...")
        print("（これには数分かかる場合があります）")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id) # 指定モデルのトークナイザーをロード
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16, # モデルの重みをbfloat16というGPUメモリを節約しつつ, 推論速度を恒常化させる最適化手法のものに指定
                device_map="auto", # モデルがどのデバイス(GPU or CPU)にアクセスするか自動で判断してくれる
            )
            # テキスト生成パイプラインの生成
            self.pipeline = pipeline(
                "text-generation", model=model, tokenizer=tokenizer
            )
            print("✅ モデルのロードが成功しました。")
            # GPUの利用確認
            if torch.cuda.is_available():
                print(f"✅ GPUが利用可能です: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ GPUが利用できません。CPUで実行します（非常に遅くなります）。")
        # エラー処理
        except Exception as e:
            print(f"❌ モデルのロード中にエラーが発生しました: {e}")
            raise

    def set_scenario(self, item_description: str, seller_price: int, seller_target_price: int):
        # システムプロンプト
        # ここでbotがどのように振る舞うか, どのように交渉を進めるかを指示する
        system_prompt = f"""
あなたは価格交渉を行うチャットボットです。役割は「売り手」です。
これから「買い手」と商品の価格について交渉します。プロフェッショナルかつ、しかし友好的な態度で交渉を進めてください。

**交渉シナリオ:**
- **商品:** {item_description}
- **あなたの提示価格:** ${seller_price}
- **あなたの目標:** できるだけ ${seller_target_price} に近い価格で売ること。

あなたは買い手の提案を検討し、カウンターオファーを提示したり、商品の価値を説明したりして、有利な価格での合意を目指してください。
返答は売り手として、自然な会話形式で簡潔に生成してください。
"""
        self.messages = [{"role": "system", "content": system_prompt.strip()}] # メッセージリストの初期化
        print("\n--- シナリオ設定完了 ---")
        print(f"商品: {item_description}")
        print(f"売り手の提示価格: ${seller_price}")
        print("---------------------\n")

    def talk(self, user_input: str) -> str:
        # モデルのロード確認
        if not self.pipeline:
            return "モデルがロードされていません。 `load_model()` を実行してください。"
        
        # user_inputとしてユーザーからのメッセージを保管
        self.messages.append({"role": "user", "content": user_input})

        # これまでの対話履歴を元にモデルが理解できる形式の入力プロンプトを生成
        # これによりシステムプロンプト, ユーザーの質問, botの応答などが結合されて, モデルに渡されるプロンプトが作成される
        prompt = self.pipeline.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        # モデルがテキスト生成をどこで止めるかを指示する終了トークン(terminators)の設定
        terminators = [
            self.pipeline.tokenizer.eos_token_id, # 一般的なモデルで使用される「end-of-sequence」（系列の終わり）トークンのID
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>") # Llama3で使用される「end of turn」（発話の終わり）トークンのIDをトークナイザーで変換して取得
        ]

        """テキストの生成
        prompt: 入力プロンプト
        max_new_tokens: モデルが生成できる最大トークン数
        eos_token_id: 終了トークン
        do_sample: 生成時にランダムサンプリングを導入する
        temperature: ランダム性の制御, 値が高いほどランダムになる(0.6はバランスが取れている)
        top_p: 確率累積が指定の値になるまでのトークンの中から次のトークンをサンプリングする(非常に低い確率のトークンが選ばれることを防ぐ)
        """
        outputs = self.pipeline(
            prompt, max_new_tokens=256, eos_token_id=terminators,
            do_sample=True, temperature=0.6, top_p=0.9,
        )

        # outputには元のプロンプトなどの情報も含まれるので生成された応答のみを抽出
        generated_text = outputs[0]['generated_text']
        bot_response = generated_text[len(prompt):].strip()


        # botの応答を対話履歴に追加
        self.messages.append({"role": "assistant", "content": bot_response})
        return bot_response

print("✅ step3: 交渉ボットクラスの定義が完了しました。")

# %%
# =============================================
# ステップ4: ボットの作成とモデルのロード
# =============================================
bot = NegotiationBot() # 上記で定義したクラスからインスタンスを生成
bot.load_model() # load_modelの実行(Hugging Face Hubから指定のモデルをロード)
print("✅ step4: ボットの作成とモデルのロードが完了しました。")

# %%
# =============================================
# ステップ5: 交渉シナリオの設定
# =============================================
item = "ほとんど未使用のビンテージ革製ソファ。状態は非常に良いです。"
seller_starting_price = 600
seller_goal = 550
bot.set_scenario(item, seller_starting_price, seller_goal)
print("✅ ステップ5: 交渉シナリオの設定が完了しました。")

# %%
# =============================================
# ステップ6: 交渉の開始（最初の対話）
# =============================================
user_message = f"こんにちは、このソファに興味があります。価格は${seller_starting_price}ですね。"
print(f"あなた (買い手): {user_message}\n")
bot_response = bot.talk(user_message)
print(f"売り手 (Llama3): {bot_response}")

# %%
# =============================================
# ステップ7: 交渉を続ける（対話ループ）
# =============================================
your_next_offer = "とても素敵なソファですね！もしよろしければ、$450でお譲りいただくことは可能でしょうか？"
print(f"あなた (買い手): {your_next_offer}\n")
bot_response = bot.talk(your_next_offer)
print(f"売り手 (Llama3): {bot_response}")
# %%
