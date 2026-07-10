import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import ( StandardScaler, OneHotEncoder )
from sklearn.impute import SimpleImputer

BASE_DIR=Path(__file__).resolve().parent

df=pd.read_csv( BASE_DIR / "datos" / "edufutura_desercion.csv" )
df["deserto"]=df["deserto"].map( { "Si":1, "No":0 } )

X=df.drop( "deserto", axis=1 )
y=df["deserto"]

X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=y
)

numericas=[
"nota_admision",
"nota_primer_parcial",
"asistencia",
"materias_reprobadas",
"distancia_km",
"dependientes",
"horas_trabajo_semana"
]
categoricas=[
"beca"
]

num=Pipeline([
("imputacion", SimpleImputer(strategy="median")),
("escalamiento", StandardScaler())
])

cat=Pipeline([
("imputacion", SimpleImputer(strategy="most_frequent")),
("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocesador=ColumnTransformer([
("num",num,numericas),
("cat",cat,categoricas)
])

print( "Pipeline creado" )
