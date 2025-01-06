import datetime
import streamlit as st
import pandas as pd

from utils_func import (
    plot_temperature_time_series,
    plot_seasonal_profile,
    plot_annual_temperature_cycle,
    analyze_city_temperature,
    get_current_temperature
)

st.title("Анализ температурных данных")

uploaded_file = st.file_uploader("Загрузите файл с историческими данными (CSV):", type=["csv"])

# Если файл загружен
if uploaded_file:

    data = pd.read_csv(uploaded_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    st.success("Данные успешно загружены!")

    # Выбор города
    cities = data['city'].unique()
    selected_city = st.selectbox("Выберите город:", cities)

    city_df = data[data['city'] == selected_city].copy()

    # Анализ данных для выбранного города
    city_result = analyze_city_temperature(city_df, selected_city)

    # Описательная статистика
    st.subheader("Описательная статистика")
    st.write(f"Средняя температура: {city_result['average_temperature']:.2f}°C")
    st.write(f"Минимальная температура: {city_result['min_temperature']:.2f}°C")
    st.write(f"Максимальная температура: {city_result['max_temperature']:.2f}°C")

    if city_result['trend'] > 0:
        st.write("Тренд температуры: положительный")
    elif city_result['trend'] > 0:
        st.write("Коэффициент тренда нулевой")
    else:
        st.write("Тренд температуры: отрицательный")

    st.write("Сезонные профили:")
    st.write(city_result['season_profile'])

    # График временного ряда температур с аномальными точками и границами
    st.subheader("Временной ряд температур")

    fig = plot_temperature_time_series(
        data=city_df,
        anomalies=city_result['anomalies'],
        rolling_mean=city_result['rolling_mean'],
        rolling_std=city_result['rolling_std'],
        city_name=selected_city
    )

    st.plotly_chart(fig, use_container_width=True)

    # График сезонных профилей
    st.subheader("Сезонные профили")
    season_profile = city_result['season_profile']
    fig_profile = plot_seasonal_profile(season_profile)
    st.plotly_chart(fig_profile, use_container_width=True)

    # Доп. график распределения средних по месяцам (с границами)
    st.subheader("Дополнительный график")
    st.plotly_chart(plot_annual_temperature_cycle(city_df), use_container_width=True)

    api_key = st.text_input("Введите API-ключ OpenWeatherMap:")

    # Проверка текущей температуры
    if api_key:
        current_temp = get_current_temperature(selected_city, api_key)
        if isinstance(current_temp, dict):
            st.error(current_temp)
        else:
            st.subheader("Текущая температура")
            st.write(f"Текущая температура в городе {selected_city}: {current_temp}°C")

            season_map = {12: "winter", 1: "winter", 2: "winter",
                          3: "spring", 4: "spring", 5: "spring",
                          6: "summer", 7: "summer", 8: "summer",
                          9: "autumn", 10: "autumn", 11: "autumn"}

            current_season = season_map[datetime.datetime.now().month]
            current_season_profile = city_result['season_profile'].loc[current_season]

            mean_temp = current_season_profile['mean']
            std_temp = current_season_profile['std']
            lower_bound = mean_temp - 2 * std_temp
            upper_bound = mean_temp + 2 * std_temp
            is_normal = lower_bound <= current_temp <= upper_bound

            st.write(f"Средняя температура для сезона: {mean_temp:.2f}°C")
            st.write(f"Стандартное отклонение: {std_temp:.2f}°C")
            st.write(f"Нормальный диапазон: [{lower_bound:.2f}°C, {upper_bound:.2f}°C]")

            if is_normal:
                st.success("Текущая температура находится в нормальном диапазоне.")
            else:
                st.warning("Текущая температура выходит за рамки нормального диапазона.")
