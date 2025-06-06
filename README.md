# 🔍 LLM Metrics

![CI](https://img.shields.io/badge/CI-passed-green)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/streamlit-1.44.1-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)

**LLM Metrics** は、複数の大規模言語モデル（LLM）を横断的に比較・評価するためのツールです。  
Streamlit ベースのインターフェースを通じて、LLM の出力を定量・定性の両面からメトリクス評価できます。

## 🌐 デモアプリ

🔗 https://llm-metrics.streamlit.app/

## ✨ 特徴

- **複数の LLM を簡単比較**  
  LangChain を利用して、様々な LLM（例：OpenAI、Anthropic、Mistral など）を統一的に扱えます。
- **柔軟なメトリクス設定**  
  応答の一貫性、関連性、創造性など、評価基準を自由に定義可能。
- **Web UI による可視化**  
  Streamlit による直感的な操作と結果の可視化。
- **Python 3.12 対応**

## 🚀 使い方

### 環境変数の設定

API キーを設定するには、以下の2つの方法があります：

1. **ローカル環境での開発時**: `.env` ファイルに API キーを設定
   ```
   # .env
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

2. **デプロイ環境**: アプリ内の入力フォームに直接 API キーを入力

### アプリの起動

Streamlit アプリを起動することで、ブラウザ上で操作できます：

```bash
uv run streamlit run app.py
```

## 🛠 使用ライブラリ

本プロジェクトでは以下のライブラリを使用しています：

- LangChain
- Streamlit

## 💻 対応環境

- Python >= 3.12

## 📄 ライセンス

このプロジェクトは MIT License のもとで公開されています。
