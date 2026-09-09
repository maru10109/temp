import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="서울 기온 예측기", 
    page_icon="🌡️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .big-font {
        font-size:3rem !important;
        font-weight: 700;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-text {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 20px;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 열 이름 추출 (특수문자 포함 방지)
    date_col = [c for c in df.columns if '날짜' in c][0]
    temp_col = [c for c in df.columns if '평균기온' in c][0]
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['연도'] = df[date_col].dt.year
    
    # 연도별 그룹화 및 조건 필터링 (2025년 이하, 관측일 300일 이상)
    df_valid = df.dropna(subset=[temp_col])
    yearly_data = df_valid.groupby('연도').agg(
        평균기온=(temp_col, 'mean'),
        관측일수=(temp_col, 'count')
    ).reset_index()
    
    yearly_data = yearly_data[(yearly_data['연도'] <= 2025) & (yearly_data['관측일수'] >= 300)]
    return yearly_data

st.title("🌡️ 서울 기온 예측 및 트렌드 분석")
st.markdown("<div class='sub-text'>과거 데이터를 기반으로 서울의 연평균 기온 상승폭을 확인하고 미래를 예측해 봅니다.</div>", unsafe_allow_html=True)

df = load_and_preprocess_data()

if df.empty:
    st.error("조건에 맞는 데이터가 존재하지 않습니다.")
else:
    # --- 전체 기간 회귀 분석 ---
    x_all = df['연도'].values
    y_all = df['평균기온'].values
    m_all, c_all = np.polyfit(x_all, y_all, 1)
    
    # 1년 기울기에 100을 곱해 '100년당 상승폭' 계산
    rise_100_all = m_all * 100 
    corr_all = np.corrcoef(x_all, y_all)[0, 1]
    
    start_year = x_all.min()
    end_year = x_all.max()
    total_years = len(x_all)
    
    # --- 최근 20년 회귀 분석 ---
    recent_df = df[df['연도'] > end_year - 20]
    x_rec = recent_df['연도'].values
    y_rec = recent_df['평균기온'].values
    m_rec, c_rec = np.polyfit(x_rec, y_rec, 1)
    
    # 최근 20년 기준 '100년당 상승폭' 계산
    rise_100_rec = m_rec * 100

    st.markdown("### 🔥 지구 온난화 가속도: 100년당 기온 상승 폭")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label=f"📊 전체 기간 ({start_year}~{end_year}) 기준", 
            value=f"+ {rise_100_all:.2f} ℃ / 100년",
            help="전체 분석 기간의 회귀 직선 기울기를 100년 단위로 환산한 값입니다."
        )
    with col2:
        st.metric(
            label=f"🚨 최근 20년 ({end_year-19}~{end_year}) 기준", 
            value=f"+ {rise_100_rec:.2f} ℃ / 100년",
            delta=f"전체 기간 대비 {rise_100_rec - rise_100_all:.2f} ℃ 더 빠름",
            delta_color="inverse", # 기온 상승이 빠른 것을 경고(빨간색)로 표시
            help="최근 20년간의 기온 상승 속도를 보여줍니다."
        )

    st.markdown("---")
    st.markdown("### 📈 기초 데이터 분석 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("데이터 시작 연도", f"{start_year}년")
    c2.metric("데이터 마지막 연도", f"{end_year}년")
    c3.metric("직선을 만든 해의 개수", f"{total_years}개")
    c4.metric("연도-기온 상관계수", f"{corr_all:.4f}")

    st.markdown("---")
    st.markdown("### 🔮 연도별 예상 기온 시뮬레이터")
    st.write("슬라이더를 움직여 특정 연도의 예상 연평균 기온을 확인해 보세요. **전체 기간의 완만한 추세**와 **최근 20년의 가팔라진 추세**를 비교해 볼 수 있습니다.")
    
    selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2050, step=1)
    
    # 두 가지 기준에 대한 예측값 계산
    pred_all = m_all * selected_year + c_all
    pred_rec = m_rec * selected_year + c_rec
    
    # 큼직하게 예측 결과 나란히 비교 표시
    st.markdown(f"""
        <div style='display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;'>
            <div style='flex: 1; min-width: 250px; text-align: center; padding: 20px; background-color: #e0f7fa; border-radius: 15px; border: 2px solid #006d77;'>
                <span style='font-size: 1.2rem; color:#006d77; font-weight: bold;'>📉 전체 기간 추세 반영 시</span><br>
                <span style='font-size: 1rem; color:#555;'>({start_year}년 ~ {end_year}년 기준)</span><br>
                <span style='font-size: 2.8rem; font-weight: 900; color: #006d77;'>{pred_all:.2f} ℃</span>
            </div>
            <div style='flex: 1; min-width: 250px; text-align: center; padding: 20px; background-color: #ffe5d9; border-radius: 15px; border: 2px solid #e26d5c;'>
                <span style='font-size: 1.2rem; color:#e26d5c; font-weight: bold;'>📈 최근 20년 가속화 추세 반영 시</span><br>
                <span style='font-size: 1rem; color:#555;'>({end_year-19}년 ~ {end_year}년 기준)</span><br>
                <span style='font-size: 2.8rem; font-weight: 900; color: #e26d5c;'>{pred_rec:.2f} ℃</span><br>
                <span style='font-size: 1.1rem; color: #c1121f; font-weight: bold;'>(전체 추세 대비 {(pred_rec - pred_all):+.2f} ℃)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()

    # 전체 기간 산점도 (회색/투명)
    fig.add_trace(go.Scatter(
        x=x_all, y=y_all, 
        mode='markers', 
        name='실제 관측치',
        marker=dict(color='#83c5be', size=8, opacity=0.6)
    ))

    # 최근 20년 산점도 강조 (빨간색)
    fig.add_trace(go.Scatter(
        x=x_rec, y=y_rec, 
        mode='markers', 
        name='최근 20년 관측치',
        marker=dict(color='#e29578', size=10, opacity=0.9, symbol='diamond')
    ))

    # 전체 기간 회귀선 (파란색)
    line_x = np.array([1900, 2100])
    line_y_all = m_all * line_x + c_all
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y_all, 
        mode='lines', 
        name=f'전체 추세선 (+{rise_100_all:.2f}℃/100년)',
        line=dict(color='#006d77', width=3)
    ))

    # 최근 20년 회귀선 (주황색/점선) - 기울기가 얼마나 가팔라졌는지 시각적으로 보여줌
    line_y_rec = m_rec * line_x + c_rec
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y_rec, 
        mode='lines', 
        name=f'최근 20년 추세선 (+{rise_100_rec:.2f}℃/100년)',
        line=dict(color='#e26d5c', width=3, dash='dashdot')
    ))

    # 예측값 마커 1: 전체 추세선 기준
    fig.add_trace(go.Scatter(
        x=[selected_year], y=[pred_all],
        mode='markers+text',
        name=f'{selected_year}년 예측 (전체 추세)',
        text=[f"전체: {pred_all:.2f}℃"],
        textposition="top center",
        marker=dict(color='#006d77', size=18, symbol='star', line=dict(color='black', width=1))
    ))

    # 예측값 마커 2: 최근 20년 추세선 기준
    fig.add_trace(go.Scatter(
        x=[selected_year], y=[pred_rec],
        mode='markers+text',
        name=f'{selected_year}년 예측 (최근 20년 추세)',
        text=[f"최근: {pred_rec:.2f}℃"],
        textposition="bottom center",
        marker=dict(color='#e26d5c', size=18, symbol='star', line=dict(color='black', width=1))
    ))

    # 그래프 레이아웃 설정
    fig.update_layout(
        title=dict(text="서울 연평균 기온 변화 시계열 및 예측 비교", font=dict(size=22)),
        xaxis_title="연도 (년)",
        yaxis_title="연평균 기온 (℃)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=100, b=40)
    )
    
    # 스트림릿에 그래프 출력
    st.plotly_chart(fig, use_container_width=True)
