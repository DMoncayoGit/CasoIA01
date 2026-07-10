import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv( "datos/edufutura_desercion.csv" )

df["deserto_bin"] = df[ "deserto"].map({"Si":1,"No":0})

correlacion = df.corr( numeric_only=True )

print( correlacion[ "deserto_bin" ].sort_values( ascending=False))

plt.figure( figsize=(10,7) )

sns.heatmap( correlacion, annot=True )

plt.title( "Matriz de correlación" )

plt.savefig( "resultados/graficos/matriz_correlacion.png" )
