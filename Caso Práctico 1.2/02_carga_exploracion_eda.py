import pandas as pd

df = pd.read_csv( "datos/edufutura_desercion.csv" )

print( "Dimensiones:" )
print( df.shape )

print( "\nTipos de datos:" )
print( df.info() )

print( "\nEstadística descriptiva:" )
print( df.describe() )

print( "\nValores faltantes:" )
print( df.isnull().sum() )
