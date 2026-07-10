import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import ( StandardScaler, OneHotEncoder )
from sklearn.impute import ( SimpleImputer )
from imblearn.over_sampling import SMOTE

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv( BASE_DIR / "datos" / "edufutura_desercion.csv" )

# VARIABLE OBJETIVO

df["deserto"] = df["deserto"].map( { "Si":1, "No":0 } )

X = df.drop( "deserto", axis=1 )
y = df["deserto"]

# DIVISIÓN TRAIN TEST

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# DEFINIR VARIABLES

numericas = [
    "nota_admision",
    "nota_primer_parcial",
    "asistencia",
    "materias_reprobadas",
    "distancia_km",
    "dependientes",
    "horas_trabajo_semana"
]


categoricas = [
    "beca"
]

# PIPELINE NUMÉRICO

pipeline_num = Pipeline([
    (
        "imputacion",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "escalamiento",
        StandardScaler()
    )

])

# PIPELINE CATEGÓRICO

pipeline_cat = Pipeline([
    (
        "imputacion",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )

])

# PREPROCESADOR

preprocesador = ColumnTransformer([
    (
        "numericas",
        pipeline_num,
        numericas
    ),
    (
        "categoricas",
        pipeline_cat,
        categoricas
    )
])

# TRANSFORMACIÓN DE DATOS

X_train_transformado = preprocesador.fit_transform(
    X_train
)

X_test_transformado = preprocesador.transform(
    X_test
)

# APLICAR SMOTE
# SOLO TRAIN

smote = SMOTE( random_state=42 )

X_train_balanceado, y_train_balanceado = smote.fit_resample(
    X_train_transformado,
    y_train
)

# RESULTADOS

print( "Distribución original:" )
print( y_train.value_counts() )
print( "\nDistribución después de SMOTE:")

print( y_train_balanceado.value_counts() )



print(

"\nAntes de SMOTE:",

X_train.shape

)



print(

"Después de SMOTE:",

X_train_balanceado.shape

)
