import pandas as pd
import streamlit as st

st.header('Welcome!')

df = pd.DataFrame({'col_a':[1,2,3], 'col_b':['a', 'b', 'c']})
st.dataframe(df)