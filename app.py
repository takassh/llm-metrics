import streamlit as st
import time
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from utils.pricing import calculate_price
from utils.llm_handler import call_llm, MockLLMHandler
from config import (
    OPENAI_MODELS, 
    GEMINI_MODELS, 
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT
)


st.set_page_config(page_title="LLMモデル比較アプリ", layout="wide")

st.title("LLMモデル比較アプリ")
st.write("このアプリは、複数のLLMモデルの出力結果と実行速度、コストを比較します。")

# セッション状態の初期化
if 'mock_mode' not in st.session_state:
    st.session_state['mock_mode'] = True

# APIキーの状態確認用
if 'has_openai_key' not in st.session_state:
    st.session_state['has_openai_key'] = False
if 'has_google_key' not in st.session_state:
    st.session_state['has_google_key'] = False

# サイドバーにモード選択と設定を配置
with st.sidebar:
    # モック/本番モード切り替え
    st.header("モード設定")
    mock_mode = st.checkbox("モックモード（APIを使用しない）", value=True)
    st.session_state['mock_mode'] = mock_mode
    
    if not mock_mode:
        # 本番モード時のAPIキー設定
        st.header("API Keys")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        google_api_key = st.text_input("Google API Key", type="password")
        
        # APIキーの状態確認
        st.session_state['has_openai_key'] = bool(openai_api_key)
        st.session_state['has_google_key'] = bool(google_api_key)
        
        if not openai_api_key:
            st.warning("OpenAIモデルを使用するにはAPI Keyを入力してください")
        if not google_api_key:
            st.warning("Geminiモデルを使用するにはAPI Keyを入力してください")
    else:
        # モックモード時はAPIキー不要
        openai_api_key = "mock_key"
        google_api_key = "mock_key"
        st.session_state['has_openai_key'] = True
        st.session_state['has_google_key'] = True
        st.info("モックモードではAPIキーは不要です")
    
    # モデル選択
    st.header("モデル選択")
    
    # OpenAIモデル選択
    openai_models = st.multiselect(
        "OpenAIモデル",
        OPENAI_MODELS,
        default=[OPENAI_MODELS[0]]
    )
    
    # Geminiモデル選択
    gemini_models = st.multiselect(
        "Geminiモデル",
        GEMINI_MODELS,
        default=[GEMINI_MODELS[0]]
    )
    
    # 実行設定
    st.header("実行設定")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.1)
    max_tokens = st.slider("最大トークン数", min_value=50, max_value=1000, value=DEFAULT_MAX_TOKENS, step=50)

# メイン画面設定
st.header("プロンプト入力")
user_prompt = st.text_area("プロンプトを入力してください", value=DEFAULT_PROMPT, height=150)

# モックハンドラの初期化
mock_handler = MockLLMHandler()

# 実行ボタン
if st.button("実行", type="primary"):
    if not user_prompt:
        st.error("プロンプトを入力してください")
    elif not (openai_models or gemini_models):
        st.error("少なくとも1つのモデルを選択してください")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 結果を保存するリスト
        results = []
        
        # 実行するモデルの総数を計算
        total_models = len(openai_models) + len(gemini_models)
        current_model = 0
        
        # OpenAIモデルの実行
        if openai_models and st.session_state['has_openai_key']:
            for model in openai_models:
                current_model += 1
                progress = int(current_model / total_models * 100)
                progress_bar.progress(progress)
                status_text.text(f"実行中: {model} ({progress}%)")
                
                if st.session_state['mock_mode']:
                    # モックモードでの実行
                    output, exec_time, input_tokens, output_tokens = mock_handler.call_llm(
                        model, "OpenAI", user_prompt, temperature, max_tokens
                    )
                else:
                    # 実際のAPIでの実行
                    try:
                        output, exec_time, input_tokens, output_tokens = call_llm(
                            model, "OpenAI", user_prompt, temperature, max_tokens, openai_api_key
                        )
                    except Exception as e:
                        st.error(f"OpenAI APIエラー: {str(e)}")
                        continue
                
                # 料金計算
                pricing = calculate_price("OpenAI", model, input_tokens, output_tokens)
                
                results.append({
                    "モデル": model,
                    "プロバイダー": "OpenAI",
                    "実行時間(秒)": round(exec_time, 2),
                    "入力トークン数": input_tokens,
                    "出力トークン数": output_tokens,
                    "総トークン数": input_tokens + output_tokens,
                    "API利用料金": pricing["formatted_total"],
                    "API利用料金(数値)": pricing["total"],
                    "入力コスト": pricing["input_cost"],
                    "出力コスト": pricing["output_cost"],
                    "出力": output
                })
        
        # Geminiモデルの実行
        if gemini_models and st.session_state['has_google_key']:
            for model in gemini_models:
                current_model += 1
                progress = int(current_model / total_models * 100)
                progress_bar.progress(progress)
                status_text.text(f"実行中: {model} ({progress}%)")
                
                if st.session_state['mock_mode']:
                    # モックモードでの実行
                    output, exec_time, input_tokens, output_tokens = mock_handler.call_llm(
                        model, "Google", user_prompt, temperature, max_tokens
                    )
                else:
                    # 実際のAPIでの実行
                    try:
                        output, exec_time, input_tokens, output_tokens = call_llm(
                            model, "Google", user_prompt, temperature, max_tokens, google_api_key
                        )
                    except Exception as e:
                        st.error(f"Google Gemini APIエラー: {str(e)}")
                        continue
                
                # 料金計算
                pricing = calculate_price("Google", model, input_tokens, output_tokens)
                
                results.append({
                    "モデル": model,
                    "プロバイダー": "Google",
                    "実行時間(秒)": round(exec_time, 2),
                    "入力トークン数": input_tokens,
                    "出力トークン数": output_tokens,
                    "総トークン数": input_tokens + output_tokens,
                    "API利用料金": pricing["formatted_total"],
                    "API利用料金(数値)": pricing["total"],
                    "入力コスト": pricing["input_cost"],
                    "出力コスト": pricing["output_cost"],
                    "出力": output
                })
        
        # 進捗表示をクリア
        progress_bar.empty()
        status_text.empty()
        
        # 結果の表示
        if results:
            # DataFrameに変換
            df = pd.DataFrame(results)
            
            # 2カラムレイアウト
            col1, col2 = st.columns(2)
            
            with col1:
                # 実行時間グラフの表示
                st.header("実行時間比較")
                
                # カラーマップの設定
                color_map = {"OpenAI": "#00A67E", "Google": "#4285F4"}
                
                # Plotlyでバーチャートを作成
                fig = px.bar(
                    df, 
                    x="モデル", 
                    y="実行時間(秒)", 
                    color="プロバイダー",
                    color_discrete_map=color_map,
                    text=df["実行時間(秒)"].apply(lambda x: f"{x:.2f}s"),
                    height=400,
                    title="各モデルの実行時間比較"
                )
                
                # グラフのレイアウト調整
                fig.update_layout(
                    xaxis_title="モデル",
                    yaxis_title="実行時間 (秒)",
                    legend_title="プロバイダー",
                    font=dict(size=14),
                    xaxis={'categoryorder':'total descending'},
                    hovermode="x unified"
                )
                
                # テキストの位置調整
                fig.update_traces(
                    textposition='outside',
                    textfont=dict(size=14)
                )
                
                # Tooltipのカスタマイズ
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>実行時間: %{y:.2f}秒<br>プロバイダー: %{marker.color}"
                )
                
                # プロットの表示
                st.plotly_chart(fig, use_container_width=True)
                
                # コスト比較グラフ
                st.header("API利用料金比較")
                
                fig = px.bar(
                    df, 
                    x="モデル", 
                    y="API利用料金(数値)", 
                    color="プロバイダー",
                    color_discrete_map=color_map,
                    text=df["API利用料金"],
                    height=400,
                    title="各モデルのAPI利用料金比較"
                )
                
                # グラフのレイアウト調整
                fig.update_layout(
                    xaxis_title="モデル",
                    yaxis_title="コスト (USD)",
                    legend_title="プロバイダー",
                    font=dict(size=14),
                    xaxis={'categoryorder':'total descending'},
                    hovermode="x unified"
                )
                
                # テキストの位置調整
                fig.update_traces(
                    textposition='outside',
                    textfont=dict(size=14)
                )
                
                # Tooltipのカスタマイズ
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>コスト: %{text}<br>プロバイダー: %{marker.color}"
                )
                
                # プロットの表示
                st.plotly_chart(fig, use_container_width=True)
                
                # 使用統計（Plotlyのテーブルで表示）
                st.subheader("使用統計")
                
                # テーブル用のデータフレーム作成
                stats_df = df[["モデル", "プロバイダー", "実行時間(秒)", 
                             "入力トークン数", "出力トークン数", "総トークン数", "API利用料金"]].copy()
                
                # カラー定義
                fill_colors = []
                for provider in stats_df["プロバイダー"]:
                    if provider == "OpenAI":
                        fill_colors.append(['#E5F6F0', '#E5F6F0', '#E5F6F0', '#E5F6F0', '#E5F6F0', '#E5F6F0', '#E5F6F0'])
                    else:
                        fill_colors.append(['#E8F0FE', '#E8F0FE', '#E8F0FE', '#E8F0FE', '#E8F0FE', '#E8F0FE', '#E8F0FE'])
                
                # Plotlyのテーブル作成
                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=list(stats_df.columns),
                        fill_color='#F0F2F6',
                        align='left',
                        font=dict(size=14)
                    ),
                    cells=dict(
                        values=[stats_df[col] for col in stats_df.columns],
                        fill_color=fill_colors,
                        align='left',
                        font=dict(size=13)
                    )
                )])
                
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=250
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # メタデータ
                st.subheader("実行情報")
                
                # メタデータを用意
                metadata = {
                    "実行日時": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "使用プロンプト文字数": len(user_prompt),
                    "Temperature設定": temperature,
                    "最大トークン数": max_tokens,
                    "モード": "モックモード" if st.session_state['mock_mode'] else "本番モード"
                }
                
                # メタデータを視覚的に表示
                col1_meta, col2_meta = st.columns(2)
                
                with col1_meta:
                    st.metric("プロンプト文字数", len(user_prompt))
                    st.metric("最大トークン数", max_tokens)
                
                with col2_meta:
                    st.metric("Temperature設定", f"{temperature:.1f}")
                    st.metric("モード", "モック" if st.session_state['mock_mode'] else "本番")
                
                # トークン数のドーナツチャート
                token_data = stats_df[["モデル", "総トークン数"]].copy()
                
                fig = px.pie(
                    token_data, 
                    values="総トークン数", 
                    names="モデル", 
                    hole=.4,
                    color="モデル",
                    title="トークン使用量の分布"
                )
                
                fig.update_layout(
                    showlegend=True,
                    height=300,
                    margin=dict(t=30, b=0, l=0, r=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 各モデルの出力結果を表示
                st.header("出力結果")
                
                # タブ形式で結果を表示
                model_tabs = st.tabs([f"{r['モデル']}" for r in results])
                for i, tab in enumerate(model_tabs):
                    with tab:
                        result = results[i]
                        
                        # モデル情報をカラーボックスで表示
                        provider_color = "#00A67E" if result['プロバイダー'] == "OpenAI" else "#4285F4"
                        st.markdown(
                            f"""
                            <div style="background-color:{provider_color}; color:white; padding:10px; border-radius:5px; margin-bottom:10px;">
                                <span style="font-size:1.2em; font-weight:bold;">{result['モデル']}</span> | 
                                {result['プロバイダー']} | 
                                実行時間: {result['実行時間(秒)']}秒 | 
                                コスト: {result['API利用料金']}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # トークン情報表示
                        st.markdown(
                            f"""
                            **トークン使用量**: 
                            入力: {result['入力トークン数']} | 
                            出力: {result['出力トークン数']} | 
                            合計: {result['総トークン数']}
                            """
                        )
                        
                        # 出力結果を表示
                        st.text_area(
                            "モデル出力",
                            result["出力"],
                            height=350,
                            key=f"result_{i}"
                        )
                        
                        # シンプルな評価ボタン
                        col_thumbs = st.columns([1, 1, 5])
                        with col_thumbs[0]:
                            st.button("👍 良い", key=f"good_{i}")
                        with col_thumbs[1]:
                            st.button("👎 悪い", key=f"bad_{i}")
                
                # モデル間の比較ページを追加
                st.subheader("⚖️ モデル比較")
                
                if len(results) >= 2:
                    # 比較するモデルを選択
                    col_comp1, col_comp2 = st.columns(2)
                    with col_comp1:
                        comp_model1 = st.selectbox("モデル1", options=[r["モデル"] for r in results], index=0)
                    with col_comp2:
                        comp_model2 = st.selectbox("モデル2", options=[r["モデル"] for r in results], index=min(1, len(results)-1))
                    
                    # 選択されたモデルのデータを取得
                    result1 = next((r for r in results if r["モデル"] == comp_model1), None)
                    result2 = next((r for r in results if r["モデル"] == comp_model2), None)
                    
                    # 比較ビューを表示
                    if result1 and result2 and result1 != result2:
                        # モデル比較の統計表示
                        comp_stats = pd.DataFrame({
                            "指標": ["実行時間(秒)", "入力トークン数", "出力トークン数", "総トークン数", "API利用料金"],
                            result1['モデル']: [
                                result1['実行時間(秒)'],
                                result1['入力トークン数'],
                                result1['出力トークン数'],
                                result1['総トークン数'],
                                result1['API利用料金']
                            ],
                            result2['モデル']: [
                                result2['実行時間(秒)'],
                                result2['入力トークン数'],
                                result2['出力トークン数'],
                                result2['総トークン数'],
                                result2['API利用料金']
                            ],
                            "差異": [
                                f"{round(result1['実行時間(秒)'] - result2['実行時間(秒)'], 2)}秒",
                                result1['入力トークン数'] - result2['入力トークン数'],
                                result1['出力トークン数'] - result2['出力トークン数'],
                                result1['総トークン数'] - result2['総トークン数'],
                                f"${round(result1['API利用料金(数値)'] - result2['API利用料金(数値)'], 6)}"
                            ]
                        })
                        
                        st.dataframe(comp_stats, use_container_width=True)
                        
                        # 出力比較
                        col_view1, col_view2 = st.columns(2)
                        with col_view1:
                            provider_color1 = "#00A67E" if result1['プロバイダー'] == "OpenAI" else "#4285F4"
                            st.markdown(f"<span style='color:{provider_color1}; font-weight:bold;'>{result1['モデル']}</span>", unsafe_allow_html=True)
                            st.text_area("", result1["出力"], height=250, key="comp_1")
                        with col_view2:
                            provider_color2 = "#00A67E" if result2['プロバイダー'] == "OpenAI" else "#4285F4"
                            st.markdown(f"<span style='color:{provider_color2}; font-weight:bold;'>{result2['モデル']}</span>", unsafe_allow_html=True)
                            st.text_area("", result2["出力"], height=250, key="comp_2")
            
            # 結果のダウンロードボタン
            csv_data = df.drop(columns=["出力"]).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="結果をCSVとしてダウンロード",
                data=csv_data,
                file_name=f'llm_comparison_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
            )
            
            # 詳細結果のJSONダウンロードも提供
            json_results = []
            for r in results:
                json_result = r.copy()
                # 数値の丸めなどの処理
                json_result["実行時間(秒)"] = round(json_result["実行時間(秒)"], 3)
                json_result["API利用料金(数値)"] = round(json_result["API利用料金(数値)"], 6)
                json_results.append(json_result)
                
            json_data = json.dumps(json_results, ensure_ascii=False, indent=2).encode('utf-8')
            st.download_button(
                label="詳細結果をJSONとしてダウンロード",
                data=json_data,
                file_name=f'llm_comparison_detail_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json',
            )
