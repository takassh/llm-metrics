# 🔍 LLM Metrics

![CI](https://img.shields.io/badge/CI-passed-green)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/streamlit-1.44.1-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)


**LLM Metrics** は、複数の大規模言語モデル（LLM）を横断的に比較・評価するためのツールです。  
Streamlit ベースのインターフェースを通じて、LLMの出力を定量・定性の両面からメトリクス評価できます。

## ✨ 特徴

- **複数のLLMを簡単比較**  
  LangChainを利用して、様々なLLM（例：OpenAI、Anthropic、Mistralなど）を統一的に扱えます。
- **柔軟なメトリクス設定**  
  応答の一貫性、関連性、創造性など、評価基準を自由に定義可能。
- **Web UIによる可視化**  
  Streamlitによる直感的な操作と結果の可視化。
- **Python 3.12 対応**

## 🚀 使い方

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