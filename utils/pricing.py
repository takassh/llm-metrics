from config import GEMINI_PRICING, OPENAI_PRICING


def calculate_price(provider, model, input_tokens, output_tokens):
    """
    プロバイダーとモデル名、トークン数から料金を計算する

    Args:
        provider (str): "OpenAI" または "Google"
        model (str): モデル名
        input_tokens (int): 入力トークン数
        output_tokens (int): 出力トークン数

    Returns:
        dict: 料金情報を含む辞書
    """
    if provider == "OpenAI":
        pricing_table = OPENAI_PRICING
    elif provider == "Google":
        pricing_table = GEMINI_PRICING
    else:
        return {
            "input_cost": 0,
            "output_cost": 0,
            "total": 0,
            "formatted_total": "$0.00",
        }

    # モデルが料金表に存在するか確認
    if model not in pricing_table:
        return {
            "input_cost": 0,
            "output_cost": 0,
            "total": 0,
            "formatted_total": "$0.00",
        }

    # 料金計算
    input_cost = (input_tokens / 1000) * pricing_table[model]["input"]
    output_cost = (output_tokens / 1000) * pricing_table[model]["output"]
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total": total_cost,
        "formatted_total": f"${total_cost:.6f}",
    }


def estimate_tokens(text, is_english=False):
    """
    テキストからおおよそのトークン数を推定する簡易関数

    Args:
        text (str): トークン数を推定するテキスト
        is_english (bool): 英語かどうか

    Returns:
        int: 推定トークン数
    """
    if is_english:
        # 英語テキストの場合: 単語数 × 1.3
        return int(len(text.split()) * 1.3)
    else:
        # 日本語テキストの場合: 文字数 ÷ 1.5
        return int(len(text) / 1.5)
