import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------
# 1. 페이지 및 시각적 설정 (와이드 레이아웃 적용)
# ---------------------------------------------------------
st.set_page_config(
    page_title="서울 기온 예측기", 
    page_icon="🌡️", 
    layout="wide"
)

st.markdown("""
    <style>
    /* 메트릭(수치) 박스 디자인 개선 */
    div[data-testid="metric-container"] {
        background-color: #f7f9fa;
        border: 1px solid #e1e4e8;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    /* 결과 박스 디자인 */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        flex: 1;
        min-width: 250px;
    }
    .result-all { background: linear-gradient(135deg, #006d77, #83c5be); }
    .result-rec { background: linear-gradient(135deg, #e26d5c, #e29578); }
    .result-val { font-size: 2.5rem; font-weight: 800; margin: 10px 0; }
    .result-title { font-size: 1.1rem; font-weight: 600; opacity: 0.9; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 데이터 불러오기 및 전처리
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 열 이름 추출 (특수문자 방어)
    date_col = [c for c in df.columns if '날짜' in c][0]
    temp_col = [c for c in df.columns if '평균기온' in c][0]
    
    # datetime 변환 및 연도 추출
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['연도'] = df[date_col].dt.year
    
    # 2025년 이하 데이터만, 그리고 결측치 제거 후 연도별 그룹화
    df_valid = df.dropna(subset=[temp_col])
    yearly_data = df_valid.groupby('연도').agg(
        평균기온=(temp_col, 'mean'),
        관측일수=(temp_col, 'count')
    ).reset_index()
    
    # 관측일 300일 이상 조건 필터링
    yearly_data = yearly_data[(yearly_data['연도'] <= 2025) & (yearly_data['관측일수'] >= 300)]
    return yearly_data

df = load_and_preprocess_data()

if df.empty:
    st.error("조건에 맞는 데이터가 없습니다.")
    st.stop()

# 전체 데이터
x_all = df['연도'].values
y_all = df['평균기온'].values

# 최근 20년 데이터 (데이터가 정렬되어 있다고 가정 시 마지막 20개)
df_recent = df.tail(20)
x_rec = df_recent['연도'].values
y_rec = df_recent['평균기온'].values

# 전체 기간 선형 회귀 기울기 (1년당) -> 100년당으로 변환
m_all, c_all = np.polyfit(x_all, y_all, 1)
trend_100_all = m_all * 100

# 최근 20년 선형 회귀 기울기 -> 100년당으로 변환
m_rec, c_rec = np.polyfit(x_rec, y_rec, 1)
trend_100_rec = m_rec * 100

# 기초 통계량
start_year, end_year = x_all.min(), x_all.max()
year_count = len(x_all)
correlation_all = np.corrcoef(x_all, y_all)[0, 1]

# ---------------------------------------------------------
# 3. 메인 화면 UI 구성
# ---------------------------------------------------------
st.title("📈 서울 기온 예측 시뮬레이터")
st.markdown(f"**{start_year}년**부터 **{end_year}년**까지 총 **{year_count}개 연도**의 데이터를 바탕으로 서울의 기온 변화 추세를 분석합니다. (관측일수 300일 미만 연도 제외)")

st.markdown("### 📊 데이터 분석 요약 (온난화 가속도 비교)")

# 100년당 기온 상승폭 비교 메트릭 표시
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="분석 대상 연도", value=f"{year_count} 년", delta=f"상관계수: {correlation_all:.2f}", delta_color="off")
with col2:
    st.metric(label="전체 기간 (100년당 기온 상승폭)", value=f"+ {trend_100_all:.2f} ℃")
with col3:
    # 최근 20년이 전체기간보다 얼마나 더 빠른지 delta로 표시
    diff_trend = trend_100_rec - trend_100_all
    st.metric(label="최근 20년 (100년당 기온 상승폭)", value=f"+ {trend_100_rec:.2f} ℃", delta=f"전체 평균 대비 {diff_trend:.2f}℃ 가속됨", delta_color="inverse")

st.markdown("---")

st.markdown("### 🔮 연도별 예상 기온 시뮬레이터")
st.write("슬라이더를 움직여 예상 연평균 기온을 확인하세요. **추세선 모델을 변경**하여 알고리즘별 미래 예측 특성을 비교할 수 있습니다.")

col_model, col_slider = st.columns([1, 2])
with col_model:
    model_type = st.selectbox(
        "📈 추세선 모델 선택",
        ["선형 (직선)", "곡선 (2차 다항식)", "계단형 (의사결정나무)"],
        help="데이터를 가장 잘 설명하는 모델을 선택해 보세요."
    )
with col_slider:
    selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2050, step=1)
    
if model_type == "곡선 (2차 다항식)":
    st.info("💡 **곡선 모델 특징**: 현재의 가속 추세를 잘 반영하지만, 먼 미래를 예측할 경우 값이 기하급수적으로 폭증할 수 있습니다 (과적합).")
elif model_type == "계단형 (의사결정나무)":
    st.warning("⚠️ **계단형 모델 특징**: 과거 구간별 평균을 보여주기 좋습니다. 하지만 트리 모델 특성상 학습된 과거 연도(2025년) 밖의 미래는 더 이상 예측하지 못하고 '마지막 계단 값'으로 똑같이 고정해버리는 한계가 있습니다.")

# 예측 모델 적용 함수
def predict_temp(x_train, y_train, target_x, m_type):
    if m_type == "선형 (직선)":
        coeffs = np.polyfit(x_train, y_train, 1)
        return np.polyval(coeffs, target_x)
    elif m_type == "곡선 (2차 다항식)":
        coeffs = np.polyfit(x_train, y_train, 2)
        return np.polyval(coeffs, target_x)
    elif m_type == "계단형 (의사결정나무)":
        # 시각적 계단 효과를 위해 max_depth 제한
        tree = DecisionTreeRegressor(max_depth=5, random_state=42)
        tree.fit(x_train.reshape(-1, 1), y_train)
        return tree.predict(np.array(target_x).reshape(-1, 1))

# 두 가지 기준에 대한 단일 연도 예측값 계산
pred_all = predict_temp(x_all, y_all, [selected_year], model_type)[0]
pred_rec = predict_temp(x_rec, y_rec, [selected_year], model_type)[0]

# 큼직하게 예측 결과 나란히 비교 표시
st.markdown(f"""
    <div style='display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;'>
        <div class='result-box result-all'>
            <div class='result-title'>전체 기간 추세선 기준 ({selected_year}년)</div>
            <div class='result-val'>{pred_all:.2f} ℃</div>
        </div>
        <div class='result-box result-rec'>
            <div class='result-title'>최근 20년 가속 추세선 기준 ({selected_year}년)</div>
            <div class='result-val'>{pred_rec:.2f} ℃</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. Plotly 차트 시각화
# ---------------------------------------------------------
fig = go.Figure()

# 전체 산점도 (파란색 계열)
fig.add_trace(go.Scatter(
    x=x_all, y=y_all, 
    mode='markers', 
    name='전체 관측치',
    marker=dict(color='#83c5be', size=7, opacity=0.6)
))

# 최근 20년 산점도 강조 (주황색 다이아몬드)
fig.add_trace(go.Scatter(
    x=x_rec, y=y_rec, 
    mode='markers', 
    name='최근 20년 관측치',
    marker=dict(color='#e29578', size=10, opacity=0.9, symbol='diamond')
))

# 모델에 따른 시각화용 연속된 X값 생성 (곡선 및 계단형 표현을 위해 촘촘하게 배열)
line_x = np.linspace(1900, 2100, 400)
line_y_all = predict_temp(x_all, y_all, line_x, model_type)
line_y_rec = predict_temp(x_rec, y_rec, line_x, model_type)

# 전체 기간 회귀선 (짙은 청록색)
fig.add_trace(go.Scatter(
    x=line_x, y=line_y_all, 
    mode='lines', 
    name=f'전체 추세선 ({model_type})',
    line=dict(color='#006d77', width=3)
))

# 최근 20년 회귀선 (주황색/점선)
fig.add_trace(go.Scatter(
    x=line_x, y=line_y_rec, 
    mode='lines', 
    name=f'최근 20년 추세선 ({model_type})',
    line=dict(color='#e26d5c', width=3, dash='dashdot')
))

# 예측값 마커 1: 전체 추세선 기준
fig.add_trace(go.Scatter(
    x=[selected_year], y=[pred_all],
    mode='markers+text',
    name=f'{selected_year}년 예측 (전체 기준)',
    text=[f"전체: {pred_all:.1f}℃"],
    textposition="top center",
    marker=dict(color='#006d77', size=18, symbol='star', line=dict(color='white', width=1))
))

# 예측값 마커 2: 최근 20년 추세선 기준
fig.add_trace(go.Scatter(
    x=[selected_year], y=[pred_rec],
    mode='markers+text',
    name=f'{selected_year}년 예측 (최근 20년 기준)',
    text=[f"최근: {pred_rec:.1f}℃"],
    textposition="bottom center",
    marker=dict(color='#e26d5c', size=18, symbol='star', line=dict(color='white', width=1))
))

# 차트 레이아웃 설정
fig.update_layout(
    title=f"서울 연평균 기온 변화 시뮬레이션 ({model_type})",
    title_font_size=20,
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    template="plotly_white",
    height=600,
    legend=dict(
        yanchor="top", y=0.99,
        xanchor="left", x=0.01,
        bgcolor="rgba(255, 255, 255, 0.8)"
    ),
    margin=dict(l=20, r=20, t=60, b=20)
)

# 세로선(사용자 선택 연도) 추가
fig.add_vline(x=selected_year, line_width=1, line_dash="solid", line_color="rgba(0,0,0,0.3)")

st.plotly_chart(fig, use_container_width=True)
