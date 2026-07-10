import pandas as pd
import numpy as np

np.random.seed(42)
n = 8500

datos = {
"nota_admision": np.random.normal(75,12,n),
"nota_primer_parcial": np.random.normal(72,15,n),
"asistencia": np.random.normal(85,10,n),
"materias_reprobadas": np.random.poisson(1.5,n),
"distancia_km": np.random.gamma(2,5,n),
"beca":
np.random.choice(
    [
        "ninguna",
        "parcial",
        "total"
    ],
    n,
    p=[
        0.55,
        0.30,
        0.15
    ]
),
"dependientes": np.random.poisson(1,n),
"horas_trabajo_semana":np.random.normal(20,15,n)
}

df = pd.DataFrame(datos)

# Limitar valores reales

df["nota_admision"] = np.clip( df["nota_admision"],40,100 )
df["nota_primer_parcial"] = np.clip( df["nota_primer_parcial"],0,100 )
df["asistencia"] = np.clip( df["asistencia"],0,100 )
df["horas_trabajo_semana"] = np.clip( df["horas_trabajo_semana"],0,80 )

# Modelo de riesgo
riesgo = (
-0.04 *
df["nota_primer_parcial"]
-0.03 *
df["asistencia"]
+0.25 *
df["materias_reprobadas"]
+0.04 *
df["horas_trabajo_semana"]
+0.02 *
df["distancia_km"]
)

probabilidad = ( 1 / (1+np.exp(-riesgo)) )
umbral = np.quantile( probabilidad, 0.72 )
df["deserto"] = np.where( probabilidad > umbral, "Si", "No" )

# Crear valores faltantes

faltantes = np.random.choice( n, 250, replace=False )

df.loc[ faltantes,"beca"]=np.nan
df.to_csv( "datos/edufutura_desercion.csv", index=False )

print(df.head())
print( df.shape )
