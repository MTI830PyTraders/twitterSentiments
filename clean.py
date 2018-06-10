import pandas as pd
import numpy as np
import re

df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='cp1252', verbose="true", keep_default_na=False, na_values=[], delimiter=',', dtype=[('target', np.uint8),('ids', np.uint8), ('date', str), ('flag', str), ('user', str), ('text', str)], names=['target', 'ids', 'date', 'flag', 'user', 'text'] )

def func(columns):
    # @[a-z0-9_]+) : remove twitter user beginning with prefix '@'
    # (https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+) : remove urls starting with 'http(s)'
    combined_re = re.compile(r"http\S+")

    # Lower cases
    columns_lowerd = str(columns).lower()
    columns_cleaned_1rstPass = re.sub(combined_re, " ", columns_lowerd )

    # Remove repeated >=3 letter to one copy.
    repeat_re = re.compile(r'(\w)(\1{2,})')
    repl = r'\1'
    columns_cleaned_2ndPass = re.sub(repeat_re, repl , columns_cleaned_1rstPass )
    return(columns_cleaned_2ndPass)

df['text'] = df['text'].apply(func)

df.to_csv("training.1600000.processed.noemoticon.MT830.csv", sep=',',  encoding='cp1252', index=False, header=False)
