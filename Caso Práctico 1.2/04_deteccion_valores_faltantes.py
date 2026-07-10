import pandas as pd
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent

df=pd.read_csv( BASE_DIR / "datos" / "edufutura_desercion.csv" )

print( df.isnull().sum() )
print( "PORCENTAJE" )
print( df.isnull().mean()*100 )
