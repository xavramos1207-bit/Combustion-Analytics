
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ==========================================
# 1. DEFINIR DIRECTORIOS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

CARPETA_DATOS = BASE_DIR / "NASA spaceChallenge" / "CFI"
CARPETA_LIMPIOS = CARPETA_DATOS / "Clean data"
CARPETA_MODELOS = BASE_DIR / "models"

CARPETA_LIMPIOS.mkdir(parents=True, exist_ok=True)
CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. DEFINIR VARIABLES
# ==========================================

variables = [
    "Flame Type",
    "Fuel",
    "XO2 before test",
    "approx p (bar)"
]

objetivo = "Hot ignition"

categoricas = [
    "Flame Type",
    "Fuel"
]

numericas = [
    "XO2 before test",
    "approx p (bar)"
]

columnas_necesarias = variables + [objetivo]


# ==========================================
# 3. BUSCAR ARCHIVOS CSV
# ==========================================

archivos = sorted(CARPETA_DATOS.glob("*.csv"))

if not archivos:
    raise FileNotFoundError(
        f"No se encontraron archivos CSV en: {CARPETA_DATOS}"
    )

print("\nARCHIVOS ENCONTRADOS:")

for archivo in archivos:
    print("-", archivo.name)


# ==========================================
# 4. LEER Y LIMPIAR ARCHIVOS
# ==========================================

datos = []

for archivo in archivos:

    print(f"\nLeyendo: {archivo.name}")

    df_temporal = pd.read_csv(
        archivo,
        sep=",",
        decimal=".",
        encoding="utf-8-sig"
    )

    # Limpiar nombres de columnas
    df_temporal.columns = (
        df_temporal.columns.str.strip()
    )

    # Normalizar nombres de columnas
    df_temporal = df_temporal.rename(columns={
        "XO2": "XO2 before test",
        "O2 before test": "XO2 before test",
        "X O2 before test": "XO2 before test"
    })

    # Comprobar que el separador sea correcto
    if len(df_temporal.columns) == 1:
        raise ValueError(
            f"El archivo {archivo.name} solo tiene una columna. "
            "Revisa el separador del CSV."
        )

    # Verificar columnas necesarias
    faltantes = [
        columna
        for columna in columnas_necesarias
        if columna not in df_temporal.columns
    ]

    if faltantes:
        print("Columnas encontradas:")
        print(df_temporal.columns.tolist())

        raise ValueError(
            f"Faltan columnas en {archivo.name}: {faltantes}"
        )

    # Conservar columnas necesarias
    df_temporal = df_temporal[
        columnas_necesarias
    ].copy()

    # Reemplazar celdas vacías
    df_temporal = df_temporal.replace(
        r"^\s*$",
        pd.NA,
        regex=True
    )

    # Reemplazar celdas que contienen solo comas
    df_temporal = df_temporal.replace(
        r"^\s*,+\s*$",
        pd.NA,
        regex=True
    )

    # Eliminar filas completamente vacías
    df_temporal = df_temporal.dropna(how="all")

    # Registrar procedencia
    df_temporal["Archivo_origen"] = archivo.name

    print("Registros encontrados:", len(df_temporal))

    datos.append(df_temporal)


# ==========================================
# 5. COMBINAR ARCHIVOS
# ==========================================

df = pd.concat(
    datos,
    ignore_index=True
)

df = df.reset_index(drop=True)

print("\nTOTAL DE REGISTROS:", len(df))

print("\nVALORES ORIGINALES DE HOT IGNITION:")
print(df[objetivo].value_counts(dropna=False))


# ==========================================
# 6. LIMPIAR VARIABLES NUMÉRICAS
# ==========================================

for columna in numericas:

    df[columna] = pd.to_numeric(
        df[columna]
        .astype("string")
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )


# ==========================================
# 7. LIMPIAR VARIABLES CATEGÓRICAS
# ==========================================

for columna in categoricas:

    df[columna] = (
        df[columna]
        .astype("string")
        .str.strip()
    )

    df[columna] = df[columna].replace(
        "",
        pd.NA
    )


# ==========================================
# 8. LIMPIAR VARIABLE OBJETIVO
# ==========================================

df[objetivo] = (
    df[objetivo]
    .astype("string")
    .str.strip()
    .str.lower()
)

print("\nRESULTADOS DESPUÉS DE NORMALIZAR:")
print(df[objetivo].value_counts(dropna=False))

# Conservar solamente respuestas yes/no
df = df[
    df[objetivo].isin(["yes", "no"])
].copy()

# Convertir a binario
df[objetivo] = df[objetivo].map({
    "yes": 1,
    "no": 0
}).astype(int)

print("\nREGISTROS VÁLIDOS:", len(df))

print("\nDISTRIBUCIÓN DE CLASES:")
print(df[objetivo].value_counts())


# ==========================================
# 9. GUARDAR CSV LIMPIO
# ==========================================

salida = CARPETA_LIMPIOS / "psi159_limpio.csv"

df.to_csv(
    salida,
    index=False,
    encoding="utf-8-sig"
)

print("\nCSV LIMPIO GUARDADO EN:")
print(salida)


# ==========================================
# 10. SEPARAR VARIABLES Y RESULTADO
# ==========================================

X = df[variables].copy()
y = df[objetivo].copy()

print("\nVALORES FALTANTES:")
print(X.isna().sum())


# ==========================================
# 11. VERIFICAR DATOS SUFICIENTES
# ==========================================

conteo = y.value_counts()

if len(conteo) < 2:
    raise ValueError(
        "Solo se encontró una clase de Hot ignition. "
        "Revisa los valores originales mostrados anteriormente. "
        "Se necesitan ejemplos de yes y no."
    )

if conteo.min() < 2:
    raise ValueError(
        "Alguna clase tiene menos de 2 registros. "
        "No es posible realizar la división estratificada."
    )

if len(y) * 0.25 < 2:
    raise ValueError(
        "No hay suficientes registros para reservar "
        "al menos un ejemplo por clase en la prueba."
    )

# Verificar columnas completamente vacías
columnas_vacias = [
    col for col in variables
    if X[col].isna().all()
]

if columnas_vacias:
    raise ValueError(
        f"Estas variables no contienen datos válidos: "
        f"{columnas_vacias}"
    )


# ==========================================
# 12. PREPROCESAMIENTO
# ==========================================

procesamiento = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        numericas
    ),
    (
        "cat",
        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]),
        categoricas
    )
])


# ==========================================
# 13. CREAR RANDOM FOREST
# ==========================================

modelo = Pipeline([
    (
        "procesamiento",
        procesamiento
    ),
    (
        "clasificador",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )
    )
])


# ==========================================
# 14. DIVIDIR DATOS
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nDATOS DE ENTRENAMIENTO:", len(X_train))
print("DATOS DE PRUEBA:", len(X_test))


# ==========================================
# 15. ENTRENAR MODELO
# ==========================================

modelo.fit(X_train, y_train)

print("\nENTRENAMIENTO COMPLETADO")


# ==========================================
# 16. EVALUAR MODELO
# ==========================================

predicciones = modelo.predict(X_test)

exactitud = accuracy_score(
    y_test,
    predicciones
)

print("\nEXACTITUD:", exactitud)

print("\nREPORTE DE CLASIFICACIÓN:")
print(
    classification_report(
        y_test,
        predicciones,
        zero_division=0
    )
)

print("\nMATRIZ DE CONFUSIÓN:")
print(
    confusion_matrix(
        y_test,
        predicciones,
        labels=[0, 1]
    )
)


# ==========================================
# 17. GUARDAR MODELO
# ==========================================

ruta_modelo = (
    CARPETA_MODELOS / "modelo_ignicion.pkl"
)

joblib.dump(
    modelo,
    ruta_modelo
)

print("\nMODELO GUARDADO CORRECTAMENTE")
print("Ubicación:", ruta_modelo)
