import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기", layout="centered")

@st.cache_data
def load_and_preprocess_data():
    # 데이터 불러오기 (UTF-8 인코딩)
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 열 이름에 특수문자(℃)가 포함되어 있을 수 있으므로 키워드로 열 추출
    date_col = [c for c in df.columns if '날짜' in c][0]
    temp_col = [c for c in df.columns if '평균기온' in c][0]
    
    # 날짜 데이터를 datetime 형태로 변환 후 연도 추출
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['연도'] = df[date_col].dt.year
    
    # 결측치 제거 후 연도별 그룹화 (관측일수 및 평균기온 계산)
    df_valid = df.dropna(subset=[temp_col])
    yearly_data = df_valid.groupby('연도').agg(
        평균기온=(temp_col, 'mean'),
        관측일수=(temp_col, 'count')
    ).reset_index()
    
    # 조건 필터링: 2025년 이하 & 관측일 300일 이상
    yearly_data = yearly_data[(yearly_data['연도'] <= 2025) & (yearly_data['관측일수'] >= 300)]
    return yearly_data

# 메인 UI
st.title("📈 서울 기온 예측기")

df = load_and_preprocess_data()

if df.empty:
    st.error("조건에 맞는 데이터가 없습니다.")
else:
    # 1. 선형 회귀 및 상관계수 계산
    x = df['연도'].values
    y = df['평균기온'].values
    
    # 상관계수 계산
    correlation = np.corrcoef(x, y)[0, 1]
    
    # 1차 다항식(직선) 적합: y = mx + c
    m, c = np.polyfit(x, y, 1)
    
    # 기초 통계량
    start_year = x.min()
    end_year = x.max()
    year_count = len(x)
    
    # 2. 데이터 요약 표시
    st.markdown("### 📊 분석 데이터 요약")
    st.write(f"- **분석 기간**: {start_year}년 ~ {end_year}년 (총 **{year_count}**개 해)")
    st.write(f"- **연도와 평균기온 상관계수**: **{correlation:.4f}**")
    
    st.divider()

    # 3. 사용자 입력 및 예측 표시
    st.markdown("### 🔮 연도별 예상 기온")
    selected_year = st.slider("연도를 선택하세요:", min_value=1900, max_value=2100, value=2030, step=1)
    
    predicted_temp = m * selected_year + c
    st.metric(label=f"{selected_year}년의 예상 연평균 기온", value=f"{predicted_temp:.2f} ℃")

    # 4. Plotly 산점도 및 회귀선 시각화
    fig = go.Figure()

    # 실제 데이터 산점도
    fig.add_trace(go.Scatter(
        x=x, y=y, 
        mode='markers', 
        name='실제 연평균 기온',
        marker=dict(color='royalblue', size=8, opacity=0.7)
    ))

    # 회귀선 (선택한 연도까지 이어지도록 범위 확장)
    line_x = np.array([1900, 2100])
    line_y = m * line_x + c
    
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, 
        mode='lines', 
        name='추세선 (회귀 직선)',
        line=dict(color='firebrick', width=3, dash='dash')
    ))

    # 예측값 포인트 표시
    fig.add_trace(go.Scatter(
        x=[selected_year], y=[predicted_temp],
        mode='markers+text',
        name=f'{selected_year}년 예측값',
        text=[f"{predicted_temp:.2f}℃"],
        textposition="top center",
        marker=dict(color='gold', size=15, symbol='star', line=dict(color='black', width=1))
    ))

    fig.update_layout(
        title="서울 연평균 기온 변화 및 회귀 분석",
        xaxis_title="연도",
        yaxis_title="평균기온 (℃)",
        hovermode="x unified",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
