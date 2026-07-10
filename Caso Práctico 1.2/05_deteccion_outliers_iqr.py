import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR=Path(__file__).resolve().parent

df=pd.read_csv( BASE_DIR / "datos" / "edufutura_desercion.csv" )

Q1=df[ "horas_trabajo_semana" ].quantile(0.25)
Q3=df[ "horas_trabajo_semana" ].quantile(0.75)

IQR=Q3-Q1

limite=Q3+1.5*IQR

outliers=df[ df["horas_trabajo_semana"]>limite ]

print( "Outliers:", len(outliers) )

plt.boxplot( df["horas_trabajo_semana"] )
plt.title( "Outliers horas trabajo" )
plt.show()
