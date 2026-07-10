import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv( "datos/edufutura_desercion.csv" )

df["deserto_bin"] = df[ "deserto"].map({"Si":1,"No":0})

correlacion = df.corr( numeric_only=True )

print( correlacion[ "deserto_bin" ].sort_values( ascending=False))

plt.figure( figsize=(16,12) )

#sns.heatmap( correlacion, annot=True )

sns.heatmap(
    correlacion,
    annot=True,                # muestra los números
    annot_kws={"size":14},     # tamaño de los números
    #cmap="coolwarm",          # paleta de colores opcional
    fmt=".4f"                  # formato de los números (2 decimales)
)


plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2, left=0.2) 

plt.title( "Matriz de correlación" )

plt.savefig( "resultados/graficos/matriz_correlacion.png" )
