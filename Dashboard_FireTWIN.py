
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="FlameTWIN:Combustion Analytics",
    page_icon="🔥",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

RUTA_CSV = (
    BASE_DIR / "NASA spaceChallenge"
    / "CFI" / "Clean data" / "psi159_limpio.csv"
)

RUTA_MODELO = (
    BASE_DIR / "models" / "modelo_ignicion.pkl"
)

VARIABLES = [
    "Flame Type",
    "Fuel",
    "XO2 before test",
    "approx p (bar)"
]

NUMERICAS = [
    "XO2 before test",
    "approx p (bar)",
    "Peak burner Temp (C)",
    "burn time (sec)"
]

# ==========================================
# 2. CARGAR DATOS Y MODELO
# ==========================================

@st.cache_data
def cargar_datos(ruta, modificacion):
    return pd.read_csv(ruta, encoding="utf-8-sig")

@st.cache_resource
def cargar_modelo(ruta, modificacion):
    return joblib.load(ruta)


if not RUTA_CSV.exists():
    st.error(f"CSV not found: {RUTA_CSV}")
    st.stop()

df = cargar_datos(
    str(RUTA_CSV),
    RUTA_CSV.stat().st_mtime
)

df.columns = df.columns.str.strip()

for columna in NUMERICAS:
    if columna in df.columns:
        df[columna] = pd.to_numeric(
            df[columna].astype("string").str.replace(
                ",", ".", regex=False
            ),
            errors="coerce"
        )

if "Hot ignition" in df.columns:
    df["Hot ignition"] = (
        df["Hot ignition"].astype("string")
        .str.strip().str.lower()
        .replace({"1": "yes", "0": "no"})
    )

modelo = None

if RUTA_MODELO.exists():
    modelo = cargar_modelo(
        str(RUTA_MODELO),
        RUTA_MODELO.stat().st_mtime
    )

# ==========================================
# 3. BARRA LATERAL
# ==========================================

st.sidebar.title("FlameTWIN:Combustion Analytics")

pagina = st.sidebar.radio(
    "Select a section",
    [
        "Summary",
        "Experimental exploration",
        "Ignition prediction",
        "Experimental data",
        "Train new data",
        "Model validation"
    ]
)

st.sidebar.divider()

# Filtros globales
df_filtrado = df.copy()

for columna in ["Fuel", "Flame Type"]:

    if columna in df.columns:

        opciones = sorted(
            df[columna].dropna().astype(str).unique()
        )

        seleccion = st.sidebar.multiselect(
            columna,
            options=opciones,
            default=opciones
        )

        df_filtrado = df_filtrado[
            df_filtrado[columna].astype(str).isin(seleccion)
        ]

# ==========================================
# 4. RESUMEN
# ==========================================

if pagina == "Summary":

    st.title("Combustion experimental analysis")

    st.write(
        "Experimental report exploration"
        " of PSI-159 data collection"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Experiments",
        len(df_filtrado)
    )

    if "Fuel" in df_filtrado.columns:
        col2.metric(
            "Fuels",
            df_filtrado["Fuel"].nunique()
        )

    if "Hot ignition" in df_filtrado.columns:
        validos = df_filtrado[
            df_filtrado["Hot ignition"].isin(["yes", "no"])
        ]

        porcentaje = (
            100 * (validos["Hot ignition"] == "yes").mean()
            if len(validos) else 0
        )

        col3.metric(
            "Hot ignition",
            f"{porcentaje:.1f}%"
        )

        if not validos.empty:
            conteos = (
                validos["Hot ignition"]
                .value_counts()
                .reset_index()
            )
            conteos.columns = ["Ignition", "Quantity"]

            figura = px.bar(
                conteos,
                x="Ignition",
                y="Quantity",
                color="Ignition",
                title="Result distribution"
            )

            st.plotly_chart(figura, use_container_width=True)

# ==========================================
# 5. EXPLORACIÓN EXPERIMENTAL
# ==========================================

elif pagina == "Experimental exploration":

    st.title("Experimental exploration")

    opciones_numericas = [
        col for col in NUMERICAS
        if col in df_filtrado.columns
        and df_filtrado[col].notna().any()
    ]

    if len(opciones_numericas) >= 2:

        col1, col2 = st.columns(2)

        with col1:
            eje_x = st.selectbox(
                "X axis variable",
                opciones_numericas,
                index=0
            )

        with col2:
            eje_y = st.selectbox(
                "Y axis variable",
                opciones_numericas,
                index=1
            )

        color = (
            "Hot ignition"
            if "Hot ignition" in df_filtrado.columns
            else None
        )

        figura = px.scatter(
            df_filtrado,
            x=eje_x,
            y=eje_y,
            color=color,
            hover_data=[
                col for col in ["Fuel", "Flame Type"]
                if col in df_filtrado.columns
            ],
            title=f"{eje_y} frente a {eje_x}"
        )

        st.plotly_chart(
            figura,
            use_container_width=True
        )

        st.subheader("Variable distribution")

        variable = st.selectbox(
            "Select a variable",
            opciones_numericas
        )

        histograma = px.histogram(
            df_filtrado,
            x=variable,
            color=color,
            title=f"{variable} distribution"
        )

        st.plotly_chart(
            histograma,
            use_container_width=True
        )

    else:
        st.warning(
            "There are not enough numeric columns"
            "to creat graphics"
        )

# ==========================================
# 6. PREDICCIÓN DE IGNICIÓN
# ==========================================

elif pagina == "Ignition prediction":

    st.title("Random forest prediction")

    if modelo is None:
        st.error(
            "modelo_ignicion.pkl not found "
            "train and save the model first."
        )
        st.stop()

    faltantes = [
        col for col in VARIABLES
        if col not in df.columns
    ]

    if faltantes:
        st.error(
            f"There are missing columns for the prediction:{faltantes}"
        )
        st.stop()

    st.write(
        "Introduce experimental conditions to"
        " obtain a model prediction."
    )

    def opciones_categoria(columna):
        return sorted(
            df[columna].dropna().astype(str).unique()
        )

    combustibles = opciones_categoria("Fuel")
    tipos = opciones_categoria("Flame Type")

    if not combustibles or not tipos:
        st.error("No categories available.")
        st.stop()

    with st.form("formulario_prediccion"):

        combustible = st.selectbox(
            "Fuel",
            combustibles
        )

        tipo_llama = st.selectbox(
            "Flame kind",
            tipos
        )

        oxigeno = st.number_input(
            "XO2 before test",
            value=0.21,
            format="%.4f"
        )

        presion = st.number_input(
            "Approximate pressure(bar)",
            min_value=0.0,
            value=1.0,
            format="%.3f"
        )

        calcular = st.form_submit_button(
            "Make prediction"
        )

    if calcular:

        entrada = pd.DataFrame([{
            "Flame Type": tipo_llama,
            "Fuel": combustible,
            "XO2 before test": oxigeno,
            "approx p (bar)": presion
        }])

        resultado = modelo.predict(entrada)[0]

        st.subheader("Prediction result")

        if resultado == 1:
            st.warning("Prediction: Hot ignition")
        else:
            st.info("Prediction: without hot ignition")

        if hasattr(modelo, "predict_proba"):

            clases = list(modelo.classes_)

            if 1 in clases:
                indice = clases.index(1)

                probabilidad = modelo.predict_proba(
                    entrada
                )[0, indice]

                st.metric(
                    "Estimated probability of ignition",
                    f"{probabilidad * 100:.1f}%"
                )

                st.progress(
                    float(probabilidad)
                )

        for columna, valor in [
            ("XO2 before test", oxigeno),
            ("approx p (bar)", presion)
        ]:
            minimo = df[columna].min()
            maximo = df[columna].max()

            if pd.notna(minimo) and pd.notna(maximo):
                if not minimo <= valor <= maximo:
                    st.warning(
                        f"{columna} is out of the range "
                        "observed in the data set."
                    )

        st.caption(
            "Exploratory statistical result. "
            "It does not represent a physical guarantee, nor a"
            "necessarily calibrated probability."
        )

# ==========================================
# 7. TABLA DE DATOS
# ==========================================

elif pagina == "Experimental data":

    st.title("Experimental data")

    st.write(
        f"Selected records: {len(df_filtrado)}"
    )

    st.dataframe(
        df_filtrado,
        use_container_width=True
    )

    csv = df_filtrado.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Export filtered data",
        data=csv,
        file_name="experimentos_filtrados.csv",
        mime="text/csv"
    )


# ==========================================
# 8. ENTRENAMIENTO INTERACTIVO
# ==========================================

elif pagina == "Train new data":

    st.title("Interactive training")

    st.write(
        "Add experimental data and train"
        " a new version of the Random Forest."
        'Please use the same form and elements'
        ' of the original experimental data'
    )

    # --------------------------------------
    # DATOS DEL USUARIO
    # --------------------------------------

    metodo = st.radio(
        "Entry method",
        ["Upload CSV", "Introduce manually"]
    )

    nuevos_datos = pd.DataFrame()

    if metodo == "Upload CSV":

        archivos_usuario = st.file_uploader(
            "Select CSV files",
            type=["csv"],
            accept_multiple_files=True
        )

        tablas = []

        for archivo in archivos_usuario:

            try:
                tabla = pd.read_csv(
                    archivo,
                    sep=None,
                    engine="python",
                    encoding="utf-8-sig"
                )

                tabla.columns = tabla.columns.str.strip()

                tablas.append(tabla)

            except Exception as error:
                st.error(
                    f"Could not be read {archivo.name}: {error}"
                )

        if tablas:
            nuevos_datos = pd.concat(
                tablas,
                ignore_index=True
            )

    else:

        plantilla = pd.DataFrame({
            "Flame Type": [""],
            "Fuel": [""],
            "XO2 before test": [np.nan],
            "approx p (bar)": [np.nan],
            "Hot ignition": [""]
        })

        nuevos_datos = st.data_editor(
            plantilla,
            num_rows="dynamic",
            use_container_width=True
        )

    # --------------------------------------
    # VALIDAR COLUMNAS
    # --------------------------------------

    columnas_requeridas = VARIABLES + [
        "Hot ignition"
    ]

    if not nuevos_datos.empty:

        st.subheader("Preview")

        st.dataframe(
            nuevos_datos.head(20),
            use_container_width=True
        )

        faltantes = [
            col for col in columnas_requeridas
            if col not in nuevos_datos.columns
        ]

        if faltantes:
            st.error(
                f"Columns missing: {faltantes}"
            )

        else:

            nuevos_datos = nuevos_datos[
                columnas_requeridas
            ].copy()

            # Convertir datos numéricos
            for col in [
                "XO2 before test",
                "approx p (bar)"
            ]:
                nuevos_datos[col] = pd.to_numeric(
                    nuevos_datos[col].astype("string")
                    .str.replace(
                        ",", ".", regex=False
                    ),
                    errors="coerce"
                )

            # Convertir variables categóricas
            for col in ["Flame Type", "Fuel"]:
                nuevos_datos[col] = (
                    nuevos_datos[col]
                    .astype("string")
                    .str.strip()
                )

            # Normalizar resultados
            nuevos_datos["Hot ignition"] = (
                nuevos_datos["Hot ignition"]
                .astype("string")
                .str.strip()
                .str.lower()
                .replace({
                    "1": "yes",
                    "0": "no"
                })
            )

            # ----------------------------------
            # OPCIONES DE ENTRENAMIENTO
            # ----------------------------------

            incluir_originales = st.checkbox(
                "Combinar with original data",
                value=True
            )

            arboles = st.slider(
                "Number of trees",
                min_value=50,
                max_value=500,
                value=100,
                step=50
            )

            proporcion_prueba = st.slider(
                "Test proportion",
                min_value=0.20,
                max_value=0.40,
                value=0.25,
                step=0.05
            )

            if st.button("Train new model"):

                if modelo is None:
                    st.error(
                        "First, you need to have the"
                        "original template to use "
                        "it's configuration"
                    )
                    st.stop()

                if incluir_originales:

                    datos_originales = df[
                        columnas_requeridas
                    ].copy()

                    entrenamiento = pd.concat(
                        [datos_originales, nuevos_datos],
                        ignore_index=True
                    )

                else:
                    entrenamiento = nuevos_datos.copy()

                # Limpiar objetivo
                entrenamiento["Hot ignition"] = (
                    entrenamiento["Hot ignition"]
                    .astype("string")
                    .str.strip()
                    .str.lower()
                    .replace({
                        "1": "yes",
                        "0": "no"
                    })
                )

                entrenamiento = entrenamiento[
                    entrenamiento["Hot ignition"]
                    .isin(["yes", "no"])
                ].copy()

                # Convertir numéricas
                for col in [
                    "XO2 before test",
                    "approx p (bar)"
                ]:
                    entrenamiento[col] = pd.to_numeric(
                        entrenamiento[col]
                        .astype("string")
                        .str.replace(
                            ",", ".", regex=False
                        ),
                        errors="coerce"
                    )

                # Limpiar categorías
                for col in ["Flame Type", "Fuel"]:
                    entrenamiento[col] = (
                        entrenamiento[col]
                        .astype("string")
                        .str.strip()
                        .replace("", np.nan)
                        .astype(object)
                    )
                    entrenamiento[col] = (
                        entrenamiento[col]
                        .where(
                            entrenamiento[col].notna(),
                            np.nan
                        )
                    )

                entrenamiento = entrenamiento.drop_duplicates(
                    subset=columnas_requeridas
                )

                # Eliminar filas sin entradas
                entrenamiento = entrenamiento.dropna(
                    subset=VARIABLES,
                    how="all"
                )

                y = entrenamiento[
                    "Hot ignition"
                ].map({
                    "yes": 1,
                    "no": 0
                }).astype(int)

                X = entrenamiento[
                    VARIABLES
                ].copy()

                conteo = y.value_counts()

                if (
                    len(conteo) < 2
                    or conteo.min() < 2
                    or int(np.ceil(
                        len(y) * proporcion_prueba
                    )) < 2
                    or int(np.floor(
                        len(y) * (1 - proporcion_prueba)
                    )) < 2
                ):
                    st.error(
                        "There's not enough data  "
                        "to train both classes "
                        "and check the model."
                    )
                    st.stop()

                if X.isna().all().any():
                    st.error(
                        "Some variable is "
                        "completely empty"
                    )
                    st.stop()

                # División de los datos
                X_train, X_test, y_train, y_test = (
                    train_test_split(
                        X,
                        y,
                        test_size=proporcion_prueba,
                        random_state=42,
                        stratify=y
                    )
                )

                # Crear modelo nuevo sin alterar el original
                nuevo_modelo = clone(modelo)

                nuevo_modelo.set_params(
                    clasificador__n_estimators=arboles
                )

                # Entrenar
                nuevo_modelo.fit(
                    X_train,
                    y_train
                )

                predicciones = nuevo_modelo.predict(
                    X_test
                )

                # Guardar resultados en la sesión
                st.session_state["new model"] = nuevo_modelo
                st.session_state["y_test"] = y_test
                st.session_state["predictions"] = predicciones
                st.session_state["X_test"] = X_test

                # Exportar modelo como bytes
                buffer = io.BytesIO()

                joblib.dump(
                    nuevo_modelo,
                    buffer
                )

                st.session_state["model_bytes"] = (
                    buffer.getvalue()
                )

                st.success(
                    "New model trained succesfully"
                )

                st.metric(
                    "Accuracy",
                    f"{accuracy_score(y_test, predicciones):.2%}"
                )

    # --------------------------------------
    # DESCARGAR NUEVO MODELO
    # --------------------------------------

    if "model_bytes" in st.session_state:

        st.download_button(
            "Download trained model",
            data=st.session_state["model_bytes"],
            file_name="modelo_ignicion_usuario.pkl",
            mime="application/octet-stream"
        )


# ==========================================
# 9. VALIDACIÓN DEL MODELO
# ==========================================

elif pagina == "Model validation":

    st.title("Model validation")

    if "new model" not in st.session_state:

        st.info(
            "First train the model in "
            "the section Model training."
        )

    else:

        y_test = st.session_state["y_test"]
        predicciones = st.session_state["predictions"]
        X_test = st.session_state["X_test"]

        # ----------------------------------
        # MÉTRICAS GENERALES
        # ----------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{accuracy_score(y_test, predicciones):.2%}"
        )

        col2.metric(
            "Precision",
            f"{precision_score(y_test, predicciones, zero_division=0):.2%}"
        )

        col3.metric(
            "Recall",
            f"{recall_score(y_test, predicciones, zero_division=0):.2%}"
        )

        col4.metric(
            "F1-score",
            f"{f1_score(y_test, predicciones, zero_division=0):.2%}"
        )

        # ----------------------------------
        # MATRIZ DE CONFUSIÓN
        # ----------------------------------

        matriz = confusion_matrix(
            y_test,
            predicciones,
            labels=[0, 1]
        )

        figura = px.imshow(
            matriz,
            x=["No ignition", "Ignition"],
            y=["No ignitionn", "Ignition"],
            text_auto=True,
            labels={
                "x": "Prediction",
                "y": "Real Result",
                "color": "Casos"
            },
            title="Confusion matrix"
        )

        st.plotly_chart(
            figura,
            use_container_width=True
        )

        # ----------------------------------
        # RESULTADOS POR TIPO DE LLAMA
        # ----------------------------------

        resultados = X_test.copy()

        resultados["Real"] = y_test.to_numpy()
        resultados["Prediction"] = predicciones

        st.subheader("Results per type of flame")

        metricas = []

        for tipo, grupo in resultados.groupby(
            "Flame Type",
            dropna=False
        ):

            metricas.append({
                "Flame Type": str(tipo),
                "Experiments": len(grupo),
                "Accuracy": accuracy_score(
                    grupo["Real"],
                    grupo["Prediction"]
                ),
                "F1-score": f1_score(
                    grupo["Real"],
                    grupo["Prediction"],
                    zero_division=0
                )
            })

        tabla_metricas = pd.DataFrame(metricas)

        st.dataframe(
            tabla_metricas,
            use_container_width=True
        )

        figura_tipos = px.bar(
            tabla_metricas,
            x="Flame Type",
            y="Accuracy",
            title="Accuracy per type of flame",
            range_y=[0, 1]
        )

        st.plotly_chart(
            figura_tipos,
            use_container_width=True
        )

        st.caption(
            "Metrics by group should be interpreted"
            "with caution when there are few experiments."
        )
