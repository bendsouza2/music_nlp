import pandas as pd
import numpy as np
from langdetect import detect

pd.options.display.max_columns = 4

# Reading in the data
first = pd.read_csv('full_lyric_data')
first.drop('Unnamed: 0', axis=1, inplace=True)

# Checking for nulls
empty = np.where(pd.isnull(first['lyrics']))
print(empty)

# Checking language consistency
first['lang_code'] = first['lyrics'].apply(detect)
languages = (first['lang_code'].unique())

eng_only = first[first['lang_code'] == 'en']


