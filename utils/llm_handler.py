import time
import random
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

from config import (
    MOCK_RESPONSE_DELAY, 
    MODEL_SPEED_CATEGORY, 
    MOCK_RESPONSES,
    OPENAI_MODELS,
    GEMINI_MODELS
)
from utils.pricing import estimate_tokens


def call_llm(model_name, provider, prompt, temperature, max_tokens, api_key):
    """
    各LLMモデルに対してプロンプトを送信し、結果とメタデータを返す
    
    Args:
        model_name (str): モデル名
        provider (str): "OpenAI" または "Google"
        prompt (str): プロンプト内容
        temperature (float): 生成のランダム性
        max_tokens (int): 最大トークン数
        api_key (str): APIキー
    
    Returns:
        tuple: (出力テキスト, 実行時間, 入力トークン数, 出力トークン数)
    """
    start_time = time.time()
    
    try:
        if provider == "OpenAI":
            chat_model = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=api_key
            )
            
            response = chat_model.invoke(prompt)
            result_text = response.content
            
            # トークン使用量の取得
            usage = response.response_metadata.get("usage", {})
            input_tokens = usage.get("prompt_tokens", estimate_tokens(prompt))
            output_tokens = usage.get("completion_tokens", estimate_tokens(result_text, True))
            
        elif provider == "Google":
            chat_model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=api_key
            )
            
            messages = [HumanMessage(content=prompt)]
            response = chat_model.invoke(messages)
            result_text = response.content
            
            # Gemini APIはトークン使用量の取得が難しいため推定値を使用
            input_tokens = estimate_tokens(prompt)
            output_tokens = estimate_tokens(result_text, True)
    
    except Exception as e:
        result_text = f"エラー: {str(e)}"
        input_tokens = estimate_tokens(prompt)
        output_tokens = 0
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return result_text, execution_time, input_tokens, output_tokens


class MockLLMHandler:
    """モックLLMのレスポンスを生成するハンドラークラス"""
    
    def call_llm(self, model_name, provider, prompt, temperature, max_tokens):
        """
        モックのLLM呼び出し - ランダムな待機時間と応答を生成
        
        Args:
            model_name (str): モデル名
            provider (str): "OpenAI" または "Google"
            prompt (str): プロンプト内容
            temperature (float): 生成のランダム性
            max_tokens (int): 最大トークン数
        
        Returns:
            tuple: (出力テキスト, 実行時間, 入力トークン数, 出力トークン数)
        """
        # モデルの速度カテゴリを取得
        speed_category = MODEL_SPEED_CATEGORY.get(model_name, "medium")
        
        # 速度カテゴリに基づいて実行時間を生成
        min_time, max_time = MOCK_RESPONSE_DELAY[speed_category]
        exec_time = random.uniform(min_time, max_time)
        
        # 実際の待機時間はデモ用に短くする
        time.sleep(min(exec_time / 3, 1.0))
        
        # レスポンスの生成
        response_text = self._generate_mock_response(model_name, provider, prompt)
        
        # トークン数の計算
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(response_text, True)
        
        # max_tokensに基づいて出力トークン数を調整
        output_tokens = min(output_tokens, int(max_tokens * random.uniform(0.7, 0.95)))
        
        return response_text, exec_time, input_tokens, output_tokens
    
    def _generate_mock_response(self, model_name, provider, prompt):
        """プロンプトに基づいてモックレスポンスを生成"""
        # 数学問題の場合
        if "東京から大阪" in prompt:
            response_category = "math_problem"
        else:
            response_category = "generic"
        
        try:
            return MOCK_RESPONSES[response_category][provider][model_name]
        except KeyError:
            # モデルが見つからない場合はジェネリックなレスポンス
            return f"{model_name}による回答: プロンプト「{prompt[:50]}...」に対する結果です。"
