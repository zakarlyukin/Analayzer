app_code = """
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

st.set_page_config(page_title="Socioeconomic Analyzer", layout="wide")
st.title("📊 Socioeconomic Phenomena Analyzer")

def sigmoid(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

def exponential(t, x0, r):
    return x0 * np.exp(r * t)

def gompertz(t, K, b, c):
    return K * np.exp(-b * np.exp(-c * t))

st.sidebar.header("Анализ")

analysis_type = st.sidebar.radio("Выберите анализ:",
    ["Советник", "Данные", "Симулятор", "Рекомендации"])

if analysis_type == "Советник":
    st.header("Советник по выбору моделей")
    saturation = st.radio("Явление насыщается?", ["Да", "Нет"])
    if st.button("Получить рекомендации"):
        if saturation == "Да":
            st.info("S-КРИВАЯ - адопция, распространение инноваций")
        else:
            st.info("ЭКСПОНЕНЦИАЛЬНЫЙ РОСТ - компаундирование")
            
elif analysis_type == "Данные":
    st.header("Анализ данных")
    time_input = st.text_area("Время (через запятую):", "0,1,2,3,4,5,6,7,8,9,10")
    value_input = st.text_area("Значения (через запятую):", "10,15,25,40,60,80,95,105,110,112,113")
    
    try:
        t_data = np.array([float(x.strip()) for x in time_input.split(',')])
        y_data = np.array([float(x.strip()) for x in value_input.split(',')])
        
        if st.button("Применить S-кривую"):
            popt, _ = curve_fit(sigmoid, t_data, y_data, p0=[100, 0.5, 5], maxfev=10000)
            y_pred = sigmoid(t_data, *popt)
            r2 = r2_score(y_data, y_pred)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_data, y=y_data, mode='markers', name='Данные', marker=dict(color='red', size=8)))
            fig.add_trace(go.Scatter(x=t_data, y=y_pred, mode='lines', name='Модель', line=dict(color='blue', width=2)))
            fig.update_layout(title=f"S-кривая: K={popt[0]:.2f}, r={popt[1]:.3f}, t0={popt[2]:.2f}", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("R2 Score", f"{r2:.4f}")
            col2.metric("RMSE", f"{np.sqrt(np.mean((y_data - y_pred)**2)):.4f}")
            col3.metric("Качество", "Отлично" if r2 > 0.8 else "Среднее")
    except Exception as e:
        st.error(f"Ошибка: {e}")
            
elif analysis_type == "Симулятор":
    st.header("Интерактивный симулятор S-кривой")
    K = st.slider("Емкость (K)", 10, 500, 100)
    r = st.slider("Скорость роста (r)", 0.1, 2.0, 0.5, step=0.1)
    t0 = st.slider("Середина (t0)", 0, 10, 5)
    
    t = np.linspace(0, 20, 200)
    y = sigmoid(t, K, r, t0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, fill='tozeroy', name='S-кривая'))
    fig.update_layout(title=f"S-кривая: K={K}, r={r}, t0={t0}", 
                     xaxis_title="Время", yaxis_title="Значение", height=500)
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Рекомендации":
    st.header("Рекомендации по стратегии")
    position = st.radio("Где находится явление?",
        ["На ранней стадии", 
         "В фазе ускорения (ЗОЛОТОЕ ОКНО)",
         "На насыщении",
         "В фазе упадка"])
    
    if st.button("Получить рекомендацию"):
        if position == "На ранней стадии":
            st.info("Инвестируйте в экспериментирование. KPI: скорость обучения")
        elif position == "В фазе ускорения (ЗОЛОТОЕ ОКНО)":
            st.success("МОМЕНТ ИСТИНЫ! Агрессивный рост. KPI: захват рынка")
        elif position == "На насыщении":
            st.warning("Защита маржи, новые ниши. KPI: прибыльность")
        elif position == "В фазе упадка":
            st.error("Трансформация или выход. KPI: cash generation")

st.divider()
st.markdown("Socioeconomic Phenomena Analyzer v0.1")
"""

with open('app.py', 'w') as f:
    f.write(app_code)
    
print("OK app.py создан!")
