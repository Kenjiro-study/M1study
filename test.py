import dspy
import os
from dspy.teleprompt import BootstrapFewShot

lm = dspy.LM(
    model="ollama/llama3.1",
    provider="ollama",
)

dspy.settings.configure(lm=lm)

# --- ステップ2: シグネチャの定義 ---
class BasicQA(dspy.Signature):
    """Answer questions in a humorous way with short factual answers."""
    
    question = dspy.InputField(desc="the user's question")
    answer = dspy.OutputField(desc="often a single word or phrase")

# --- ステップ3: モジュールの作成 ---
# シグネチャを使ってPredictモジュールを作成
qa_predictor = dspy.Predict(BasicQA)

# --- ステップ4: 実行と結果の確認 ---
# 作成したモジュールを実行します。引数名はシグネチャの入力フィールド名に合わせます。
my_question = "What is the capital of Japan?"
prediction = qa_predictor(question=my_question)

# 結果の表示
print(f"質問: {my_question}")
print(f"回答: {prediction.answer}")

# 別の質問も試してみましょう
my_question_2 = "Who wrote the novel 'I Am a Cat'?"
prediction_2 = qa_predictor(question=my_question_2)
print(f"\n質問: {my_question_2}")
print(f"回答: {prediction_2.answer}")

# --- ステップ6: お手本例の準備 ---
train_examples = [
    dspy.Example(question="What city is the Eiffel Tower in?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is the highest mountain in the world?", answer="Mount Everest").with_inputs("question"),
    dspy.Example(question="Who is the current Prime Minister of Japan?", answer="Shigeru Ishiba").with_inputs("question")
]

# --- ステップ7: コンパイル ---
# 1. コンパイラの設定
config = dict(max_bootstrapped_demos=2)
teleprompter = BootstrapFewShot(metric=None, **config)

# 2. コンパイルの実行
# studentには最適化したいプログラム、trainsetにはお手本データを渡す
compiled_qa = teleprompter.compile(student=dspy.Predict(BasicQA), trainset=train_examples)


# --- ステップ8: 比較 ---
# 少し意地悪な質問
test_question = "Who created the programming language Python and in what country was he born?"

# 【比較1】コンパイル前のプログラム（ゼロショット）
uncompiled_qa = dspy.Predict(BasicQA)
prediction_before = uncompiled_qa(question=test_question)
print(f"--- コンパイル前 (ゼロショット) ---")
print(f"質問: {test_question}")
print(f"回答: {prediction_before.answer}\n")


# 【比較2】コンパイル後のプログラム（Few-shot）
prediction_after = compiled_qa(question=test_question)
print(f"--- コンパイル後 (最適化済み) ---")
print(f"質問: {test_question}")
print(f"回答: {prediction_after.answer}\n")

lm.inspect_history(n=1)