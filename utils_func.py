from typing import Dict, Union
import pandas as pd
import plotly.graph_objects as go
import requests
from sklearn.linear_model import LinearRegression

def get_current_temperature(city: str, api_key_current: str) -> Union[float, dict]:
    """
    Получение текущей температуры для города через API OpenWeatherMap.

    Args:
        city (str): Название города.
        api_key_current (str): API ключ для доступа к OpenWeatherMap.

    Returns:
        float: Температура в градусах Цельсия, если запрос успешен.
        dict: Словарь с ключом "error", если произошла ошибка.
    """

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key_current,
        "units": "metric"
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 401:
            return {
                "cod": "401",
                "message":
                    "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."
            }

        if response.status_code == 200:
            return data['main']['temp']

    except Exception as e:
        return {"error": str(e)}

    return {"error": "Unexpected error occurred."}


def analyze_city_temperature(city_df: pd.DataFrame, selected_city: str) -> Dict:
    """
    Анализ временного ряда температурных данных для выбранного города.

    Args:
        city_df (pd.DataFrame): DataFrame с температурными данными.
            Ожидаемые столбцы:
                - "timestamp" (datetime): Метка времени.
                - "temperature" (float): Значения температуры.
                - "season" (str): Сезон для каждого временного промежутка.
        selected_city (str): Название города для анализа.

    Returns:
        dict: Словарь с результатами анализа:
            - 'city': Название города.
            - 'average_temperature': Средняя температура.
            - 'min_temperature': Минимальная температура.
            - 'max_temperature': Максимальная температура.
            - 'season_profile': Профиль сезона (средние и стандартные отклонения температуры).
            - 'trend': Значение коэффициента регрессии (+/-)
            - 'rolling_mean': Скользящее среднее температуры.
            - 'rolling_std': Скользящее стандартное отклонение температуры.
            - 'anomalies': Список аномальных точек (время и температура).
    """

    # Скользящие среднее и стандартное отклонение температуры с окном в 30 дней
    city_df['rolling_mean'] = city_df.groupby('season')['temperature'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    city_df['rolling_std'] = city_df.groupby('season')['temperature'].transform(
        lambda x: x.rolling(window=30, min_periods=1).std()
    )

    # Определение аномалий: температура выходит за пределы среднее ± 2𝜎
    city_df['anomaly'] = city_df.apply(
        lambda x: (x["temperature"] > x['rolling_mean'] + 2 * x['rolling_std']) or
                  (x["temperature"] < x['rolling_mean'] - 2 * x['rolling_std']),
        axis=1
    )

    # Профиль сезона
    season_profile = city_df.groupby('season')['temperature'].agg(['mean', 'std'])

    # Описательная статистика: средняя, минимальная, максимальная температура
    avg_temp = city_df['temperature'].mean()
    min_temp = city_df['temperature'].min()
    max_temp = city_df['temperature'].max()

    # Аномальные точки: временная метка и температура
    anomalies = city_df[city_df['anomaly']]

    # Тренд
    model = LinearRegression()
    model.fit(city_df['timestamp'].values.reshape(-1, 1), city_df["temperature"].values)
    trend = model.coef_[0]

    # Формирование результата
    result = {
        'city': selected_city,
        'average_temperature': avg_temp,
        'min_temperature': min_temp,
        'max_temperature': max_temp,
        'season_profile': season_profile,
        'trend': trend,
        'rolling_mean': city_df[['timestamp', 'rolling_mean']],
        'rolling_std': city_df[['timestamp', 'rolling_std']],
        'anomalies': anomalies[['timestamp', 'temperature']]
    }

    return result

def plot_temperature_time_series(data: pd.DataFrame, anomalies: pd.DataFrame, rolling_mean: pd.DataFrame, rolling_std: pd.DataFrame, city_name: str) -> go.Figure:
    """
    Строит график временного ряда температур с выделением аномалий и скользящих метрик.

    Args:
        data (pd.DataFrame): Данные временного ряда с температурой.
            Ожидаемые столбцы:
                - "timestamp" (datetime): Метка времени.
                - "temperature" (float): Температура.
        anomalies (pd.DataFrame): Данные об аномалиях.
            Ожидаемые столбцы:
                - "timestamp" (datetime): Метка времени аномалии.
                - "temperature" (float): Температура аномалии.
        rolling_mean (pd.DataFrame): Данные о скользящем среднем.
            Ожидаемые столбцы:
                - "timestamp" (datetime): Метка времени.
                - "rolling_mean" (float): Скользящее среднее.
        rolling_std (pd.DataFrame): Данные о скользящем стандартном отклонении.
            Ожидаемые столбцы:
                - "timestamp" (datetime): Метка времени.
                - "rolling_std" (float): Скользящее стандартное отклонение.
        city_name (str): Название города.

    Returns:
        go.Figure: Объект графика Plotly Figure.
    """

    fig = go.Figure()

    # Временной ряд
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['temperature'],
        mode='lines',
        name='Температура',
        line=dict(color='#8f8fe6', width=1),
        hovertemplate='Дата: %{x}<br>Температура: %{y}°C'
    ))

    # Аномальные точки
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies['timestamp'],
            y=anomalies['temperature'],
            mode='markers',
            name='Аномалии',
            marker=dict(color='#e68f8f', size=6, symbol='circle'),
            hovertemplate='Дата: %{x}<br>Температура: %{y}°C'
        ))

    # Скользящее среднее
    if not rolling_mean.empty:
        fig.add_trace(go.Scatter(
            x=rolling_mean['timestamp'],
            y=rolling_mean['rolling_mean'],
            mode='lines',
            name='Скользящее среднее',
            line=dict(color='rgba(143, 230, 143, 0.7)', width=2),
            hovertemplate='Дата: %{x}<br>Скользящее среднее: %{y}°C'
        ))

    # Верхняя и нижняя границы
    if not rolling_std.empty:
        upper_bound = rolling_mean['rolling_mean'] + 2 * rolling_std['rolling_std']
        lower_bound = rolling_mean['rolling_mean'] - 2 * rolling_std['rolling_std']

        fig.add_trace(go.Scatter(
            x=rolling_mean['timestamp'],
            y=upper_bound,
            mode='lines',
            name='Верхняя граница (rolling_mean + 2σ)',
            line=dict(color='rgba(214, 97, 97, 0.7)', dash='dot', width=1.5),
            hovertemplate='Дата: %{x}<br>Граница: %{y}°C'
        ))

        fig.add_trace(go.Scatter(
            x=rolling_mean['timestamp'],
            y=lower_bound,
            mode='lines',
            name='Нижняя граница (rolling_mean - 2σ)',
            line=dict(color='rgba(230, 187, 143, 0.7)', dash='dot', width=1.5),
            hovertemplate='Дата: %{x}<br>Граница: %{y}°C'
        ))

    # Настройка графика
    fig.update_layout(
        title=f"{city_name}",
        xaxis_title="Дата",
        yaxis_title="Температура, °C",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


def plot_seasonal_profile(season_profile: pd.DataFrame) -> go.Figure:
    """
    Строит график сезонных профилей температуры, включая среднюю температуру и область стандартного отклонения.

    Args:
        season_profile (pd.DataFrame): Данные о сезонных профилях.
        Ожидаемые столбцы:
            - 'mean' (float): Средняя температура для сезона.
            - 'std' (float): Стандартное отклонение для сезона.

    Returns:
        go.Figure: Объект Plotly Figure с графиком сезонных профилей.
    """

    seasons = season_profile.index.tolist()
    means = season_profile['mean'].tolist()
    stds = season_profile['std'].tolist()

    # Создание фигуры
    fig_profile = go.Figure()

    # Линия средней температуры
    fig_profile.add_trace(go.Scatter(
        x=seasons,
        y=means,
        mode='lines+markers',
        name='Средняя температура',
        line=dict(color='#8fe68f', width=2),
        hovertemplate='Сезон: %{x}<br>Средняя температура: %{y}°C'
    ))

    # Область стандартного отклонения
    fig_profile.add_trace(go.Scatter(
        x=seasons + seasons[::-1],
        y=[m + s for m, s in zip(means, stds)] + [m - s for m, s in zip(means, stds)][::-1],
        fill='toself',
        fillcolor='rgba(230, 187, 143, 0.2)',  # Прозрачный #e6bb8f
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        name='Стандартное отклонение'
    ))

    # Настройки графика
    fig_profile.update_layout(
        title="Сезонные профили температуры",
        xaxis_title="Сезон",
        yaxis_title="Температура, °C",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig_profile



def plot_annual_temperature_cycle(city_data: pd.DataFrame) -> go.Figure:
    """
    Строит график годового цикла температуры, включая среднее значение, стандартное отклонение и область отклонений.

    Args:
        city_data (pd.DataFrame): Данные о температуре. Ожидаемые столбцы:
            - 'timestamp' (datetime): Временная метка.
            - 'temperature' (float): Температура в градусах Цельсия.

    Returns:
        go.Figure: Объект Plotly Figure с графиком годового цикла температуры.
    """

    city_data['month'] = city_data['timestamp'].dt.month

    # Группировка по месяцам для расчета средней температуры и стандартного отклонения
    monthly_avg = city_data.groupby('month')['temperature'].mean().reset_index()
    monthly_std = city_data.groupby('month')['temperature'].std().reset_index()

    fig = go.Figure()

    # Средняя температура по месяцам
    fig.add_trace(go.Scatter(
        x=monthly_avg['month'],
        y=monthly_avg['temperature'],
        mode='lines+markers',
        name='Средняя температура',
        line=dict(color='#8f8fe6', width=2),
        hovertemplate='Месяц: %{x}<br>Температура: %{y}°C'
    ))

    # Верхняя граница (средняя температура + 2 стандартное отклонение)
    fig.add_trace(go.Scatter(
        x=monthly_avg['month'],
        y=monthly_avg['temperature'] + 2 * monthly_std['temperature'],
        mode='lines',
        name='Верхняя граница (средняя + σ)',
        line=dict(color='#8f8fe6', dash='dot'),
        showlegend=False,
        hovertemplate='Месяц: %{x}<br>Граница: %{y}°C'
    ))

    # Нижняя граница (средняя температура - 2 стандартное отклонение)
    fig.add_trace(go.Scatter(
        x=monthly_avg['month'],
        y=monthly_avg['temperature'] - 2 * monthly_std['temperature'],
        mode='lines',
        name='Нижняя граница (средняя - σ)',
        line=dict(color='#8f8fe6', dash='dot'),
        fill='tonexty',
        fillcolor='rgba(143, 143, 230, 0.2)',
        showlegend=False,
        hovertemplate='Месяц: %{x}<br>Граница: %{y}°C'
    ))

    # Настройки графика
    fig.update_layout(
        title="Средняя температура по месяцам",
        xaxis_title="Месяц",
        yaxis_title="Температура (°C)",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
        ),
        template="plotly_white",
        hovermode="x unified"
    )

    return fig
