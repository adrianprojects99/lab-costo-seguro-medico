import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings

# Suprimir advertencias (útil para el despliegue)
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Predicción de Costos de Seguros",
    page_icon="🏥",
    layout="wide"
)

# --- RUTAS Y CARGA DE MODELO/TRANSFORMADORES ---
MODEL_PATH = 'models/best_insurance_model.pkl'
SCALER_PATH = 'models/scaler_insurance.pkl'
ENCODERS_PATH = 'models/label_encoders_insurance.pkl'
FEATURE_NAMES_PATH = 'models/feature_names_insurance.pkl'

@st.cache_resource
def load_assets():
    """Carga el modelo, el scaler y los encoders guardados."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(ENCODERS_PATH, 'rb') as f:
            label_encoders = pickle.load(f)
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, label_encoders, feature_names
    except FileNotFoundError:
        st.error(f"Error: No se encontraron los archivos del modelo o transformadores en la carpeta 'models/'.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar los recursos: {e}")
        st.stop()

model, scaler, label_encoders, feature_names = load_assets()
BEST_MODEL_NAME = "Random Forest Regressor" # Basado en la salida del script
# ---------------------------------------------

##  Título y Descripción de la Aplicación
st.title(" Predictor de Costos de Seguros Médicos")
st.markdown("""
Esta aplicación utiliza el modelo de **Regresión Random Forest** para predecir el **Costo Anual de Seguros Médicos ('charges')**
en base a las características demográficas y de salud del asegurado.
""")

st.markdown("---")

##  Sidebar para Introducción de Datos
with st.sidebar:
    st.header(" Características del Asegurado")
    st.markdown("Introduce los datos para la predicción:")

    # 1. Age (Edad)
    age = st.slider("Edad", min_value=18, max_value=64, value=30)

    # 2. Sex (Sexo)
    sex_map = {'Masculino': 'male', 'Femenino': 'female'}
    sex_input = st.selectbox("Sexo", options=list(sex_map.keys()))
    sex = sex_map[sex_input]

    # 3. BMI (Índice de Masa Corporal)
    bmi = st.number_input("BMI (Índice de Masa Corporal)", min_value=15.0, max_value=55.0, value=25.0, step=0.1)

    # 4. Children (Hijos)
    children = st.slider("Número de Hijos / Dependientes", min_value=0, max_value=5, value=0)

    # 5. Smoker (Fumador)
    smoker_map = {'Sí': 'yes', 'No': 'no'}
    smoker_input = st.selectbox("Fumador", options=list(smoker_map.keys()))
    smoker = smoker_map[smoker_input]

    # 6. Region (Región)
    region_options = ['southwest', 'southeast', 'northwest', 'northeast']
    region = st.selectbox("Región", options=region_options)

    # 7. Botón de Predicción
    st.markdown("---")
    predict_button = st.button("Calcular Costo Estimado", type="primary")

##  Lógica de Predicción
if predict_button:
    # 1. Crear el DataFrame de entrada
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    st.subheader("Datos de Entrada para el Modelo")
    st.dataframe(input_data, hide_index=True)

    # 2. Preprocesamiento (Codificación de Etiquetas)
    data_processed = input_data.copy()
    for col, le in label_encoders.items():
        data_processed[col] = le.transform(data_processed[col])

    # 3. Escalado de Características
    data_scaled = scaler.transform(data_processed)

    # 4. Predicción
    try:
        prediction = model.predict(data_scaled)[0]
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        st.stop()

    # 5. Mostrar Resultados
    st.markdown("## 💰 Resultado de la Predicción")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(
            label="Costo Anual Estimado", 
            value=f"${prediction:,.2f}"
        )
        st.caption(f"Modelo utilizado: {BEST_MODEL_NAME}")
        
    with col2:
        st.success("""
        **Nota de Interpretación:** Este valor representa el costo anual estimado.
        Recuerda que el modelo de Random Forest tiene un **R² de ~0.85** en la prueba,
        lo que significa que explica el 85% de la variabilidad. El error promedio (MAE)
        es de aproximadamente **$2,300**.
        """)
    
    st.markdown("---")
    
    # 6. Interpretación de Impacto
    st.subheader(" Análisis de Factores de Impacto")
    
    # Simulación de un cambio clave: Fumador vs No Fumador
    if smoker == 'yes':
        # Hacer predicción si no fuera fumador
        no_smoker_data = input_data.copy()
        no_smoker_data['smoker'] = 'no'
        
        # Preprocesar
        no_smoker_processed = no_smoker_data.copy()
        for col, le in label_encoders.items():
            no_smoker_processed[col] = le.transform(no_smoker_processed[col])
        
        # Escalar y predecir
        no_smoker_scaled = scaler.transform(no_smoker_processed)
        no_smoker_prediction = model.predict(no_smoker_scaled)[0]
        
        # Mostrar diferencia
        diff = prediction - no_smoker_prediction
        st.info(f"""
        Si la persona tuviera exactamente las mismas características pero **NO FUERA FUMADORA**,
        el costo estimado sería de **${no_smoker_prediction:,.2f}**.
        
        Esto implica que el factor **FUMADOR** incrementa el costo en **${diff:,.2f}** para este caso.
        """)
    
    elif smoker == 'no':
        # Hacer predicción si fuera fumador
        smoker_data = input_data.copy()
        smoker_data['smoker'] = 'yes'
        
        # Preprocesar
        smoker_processed = smoker_data.copy()
        for col, le in label_encoders.items():
            smoker_processed[col] = le.transform(smoker_processed[col])
        
        # Escalar y predecir
        smoker_scaled = scaler.transform(smoker_processed)
        smoker_prediction = model.predict(smoker_scaled)[0]
        
        # Mostrar diferencia
        diff = smoker_prediction - prediction
        st.info(f"""
        Si la persona tuviera exactamente las mismas características pero **FUERA FUMADORA**,
        el costo estimado sería de **${smoker_prediction:,.2f}**.
        
        Esto implica una diferencia potencial de **${diff:,.2f}** debido al factor **FUMADOR**.
        """)

else:
    st.info(" Por favor, introduce los datos del asegurado en la barra lateral y presiona 'Calcular Costo Estimado'.")