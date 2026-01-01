import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from streamlit.runtime.uploaded_file_manager import UploadedFile

st.set_page_config(page_title="Car price prediction", page_icon="🏎️")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.pkl"
MEDIANS_PATH = ARTIFACTS_DIR / "medians.pkl"

CARS_TRAIN_URL = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
CARS_TEST_URL = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv"


def strip_single_scale(val: str) -> float:
    if pd.isna(val):
        return np.nan

    try:
        return float(val.split()[0].strip())
    except ValueError:
        return np.nan


@st.cache_resource
def load_model() -> tuple[Pipeline, pd.Series | float]:
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)
    with open(MEDIANS_PATH, "rb") as f:
        medians = pickle.load(f)
    return pipeline, medians


@st.cache_resource
def prepare_dataset(file: str | UploadedFile, test: bool = False) -> pd.DataFrame:
    df = pd.read_csv(file)

    # Повторяем преобразования из ноутбука
    df.drop(columns=["torque", "name"], inplace=True)
    df["mileage"] = df["mileage"].apply(strip_single_scale)
    df["engine"] = df["engine"].apply(strip_single_scale)
    df["max_power"] = df["max_power"].apply(strip_single_scale)

    # Используем загруженные медианы посчитанные по трейну
    df = df.fillna(MEDIANS)

    if not test:
        df = df[~df.drop(columns=["selling_price"]).duplicated()].reset_index(drop=True)

    fcols = df.select_dtypes("float").columns

    ficols = ["engine", "seats"]
    df[ficols] = df[ficols].apply(pd.to_numeric, downcast="integer")

    icols = df.select_dtypes("integer").columns
    cat_cols = ["fuel", "seller_type", "transmission", "owner"]

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
    df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")
    df[cat_cols] = df[cat_cols].astype("category")
    return df


try:
    PIPELINE, MEDIANS = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

st.title("Предсказание цены автомобиля")


def eda() -> None:
    st.title("EDA")

    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"], key=0)
    if st.button("Загрузить тренировочный датасет", use_container_width=True):
        uploaded_file = CARS_TRAIN_URL

    if uploaded_file is None:
        st.info("Загрузите CSV файл для начала работы")
        return

    # Про st.spinner спросил у LLM
    # noinspection PyTypeChecker
    with st.spinner("Обработка данных..."):
        df = prepare_dataset(uploaded_file)

    st.title("Обработанный датасет")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Количество строк", df.shape[0])
    with col2:
        st.metric("Количество признаков", df.shape[1])

    st.title("Основные статистики")

    st.write("Численные признаки:")
    st.dataframe(df.describe(include="number"))

    st.write("Категориальные признаки:")
    st.dataframe(df.describe(include="category"))

    # Спрашивал у LLM параметры plotly для подписей
    st.title("Визуализации")

    st.write("Распределение целевой переменной:")
    fig = px.histogram(
        df, x="selling_price", labels={"selling_price": "Цена", "count": "Количество"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Корреляционная матрица:")
    fig = px.imshow(
        df.corr(numeric_only=True),
        zmin=-1,
    )
    st.plotly_chart(fig, use_container_width=True)


def model_visualization() -> None:
    st.title("Модель Ridge")

    st.write("Значимость признаков:")
    feature_names = PIPELINE.named_steps["preprocessor"].get_feature_names_out()
    coefficients = PIPELINE.named_steps["model"].coef_

    sorted_idx = coefficients.argsort()
    sorted_features = feature_names[sorted_idx]
    sorted_coefs = coefficients[sorted_idx]

    fig = go.Figure(go.Bar(x=sorted_coefs, y=sorted_features, orientation="h"))
    fig.update_layout(title="")
    st.plotly_chart(fig, use_container_width=True)

    st.title("Запуск модели")
    file: str | UploadedFile
    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"], key=1)
    if st.button("Загрузить тестовый датасет", use_container_width=True):
        uploaded_file = CARS_TEST_URL

    if uploaded_file is None:
        st.info("Загрузите CSV файл для запуска модели")
        return

    # Про st.spinner спросил у LLM
    # noinspection PyTypeChecker
    with st.spinner("Обработка данных..."):
        df = prepare_dataset(uploaded_file)

    x_test = df.drop(columns=["selling_price"])
    y_test = df["selling_price"]
    y_pred = PIPELINE.predict(x_test)
    st.metric("r2_score", r2_score(y_test, y_pred))


def prediction() -> None:
    st.title("Прогноз цены автомобиля с помощью Ridge")
    st.write("Введите данные автомобиля:")

    col1, col2 = st.columns(2)

    # Тут использовал autocomplete для заполнения параметров
    with col1:
        year = st.number_input(
            "Год выпуска", min_value=1990, max_value=2024, value=2015
        )
        km_driven = st.number_input(
            "Пробег (км)", min_value=0, max_value=1000000, value=50000
        )
        mileage = st.number_input(
            "Расход (km/l)", min_value=0.0, max_value=50.0, value=18.0
        )
        engine = st.number_input(
            "Объём двигателя (CC)", min_value=500, max_value=5000, value=1200
        )
        max_power = st.number_input(
            "Мощность (bhp)", min_value=30.0, max_value=500.0, value=80.0
        )

    with col2:
        fuel = st.selectbox("Тип топлива", ["Petrol", "Diesel", "CNG", "LPG"])
        seller_type = st.selectbox(
            "Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"]
        )
        transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
        owner = st.selectbox(
            "Владелец",
            ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"],
        )
        seats = st.number_input("Количество мест", min_value=2, max_value=14, value=5)

    if st.button("Предсказать цену", use_container_width=True):
        input_data = pd.DataFrame(
            [
                {
                    "year": year,
                    "km_driven": km_driven,
                    "fuel": fuel,
                    "seller_type": seller_type,
                    "transmission": transmission,
                    "owner": owner,
                    "mileage": mileage,
                    "engine": engine,
                    "max_power": max_power,
                    "seats": seats,
                }
            ]
        )

        pred = PIPELINE.predict(input_data)[0]
        st.metric("Прогноз цены", round(pred, 2))


# Про tabs спросил у LLM
tab1, tab2, tab3 = st.tabs(["EDA", "Модель", "Интерактивный прогноз"])

with tab1:
    eda()

with tab2:
    model_visualization()

with tab3:
    prediction()
