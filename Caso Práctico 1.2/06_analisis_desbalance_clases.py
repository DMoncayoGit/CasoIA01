import pandas as pd
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent

df=pd.read_csv( BASE_DIR / "datos" / "edufutura_desercion.csv" )

resultado=df["deserto"].value_counts( normalize=True )

print(resultado)
