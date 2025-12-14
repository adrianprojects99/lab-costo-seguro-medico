# lab-costo-seguro-medico

#  Laboratorio 3: Predicción de Costos de Seguros Médicos

##  Descripción del Proyecto

Este repositorio contiene la solución para el Laboratorio 3 del curso de Inteligencia Artificial, enfocado en la **predicción de costos de seguros médicos (cargos)** utilizando técnicas de Regresión.

El proyecto abarca desde el **Modelado de Datos** inicial hasta el entrenamiento de **múltiples modelos de regresión**, su evaluación comparativa y, finalmente, la implementación de una **Interfaz de Usuario (UI) interactiva** para el consumo del mejor modelo.

**[https://github.com/adrianprojects99/lab-costo-seguro-medico/tree/main]((https://lab-costo-seguro-medico-fnrcyiafdfmra2hygi9z8j.streamlit.app/))**

## 🚀 Tecnologías Utilizadas

* **Lenguaje:** Python 3.x
* **Análisis y Datos:** `pandas`, `numpy`
* **Modelado ML:** `scikit-learn` (para modelos de regresión y métricas)
* **Visualización:** `matplotlib`, `seaborn` (opcional)
* **Interfaz de Usuario (UI):** Gradio / Streamlit (<Elegir uno>)

## 📋 Estructura del Repositorio

* `data/`: Contiene el conjunto de datos original (`train.csv`).
* `notebooks/`: Archivos `.ipynb` con el proceso de EDA, modelado, entrenamiento y análisis de error.
* `model/`: Carpeta para almacenar el modelo entrenado y serializado (e.g., `best_model.pkl`).
* `app/`: Contiene el código de la interfaz de usuario web (`app.py`).
* `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.

## 🛠️ Instalación y Configuración

Sigue estos pasos para configurar y ejecutar el proyecto localmente.

### 1. Clonar el Repositorio

```bash

# Crear el entorno (ejemplo con venv)
python -m venv venv
source venv/bin/activate  # En Linux/macOS
# venv\Scripts\activate   # En Windows

# Instalar las bibliotecas requeridas
pip install -r requirements.txt
