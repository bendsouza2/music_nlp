import pandas as pd
import numpy as np
from langdetect import detect
import re

pd.options.display.max_columns = 5

# Reading in the data
first = pd.read_csv('full_lyric_data')
first.drop('Unnamed: 0', axis=1, inplace=True)

# Checking for nulls
empty = np.where(pd.isnull(first['lyrics']))
print(empty)

# Checking language consistency
first['lang_code'] = first['lyrics'].apply(detect)
languages = (first['lang_code'].unique())

eng_only = first.drop(first[first['lang_code'] != 'en'].index)
eng_only.drop('lang_code', axis=1, inplace=True)

# Further cleaning
# Removing all characters up until # the first ']' - typically includes translation options, 'Lyrics' and song title
eng_only['lyrics'] = eng_only['lyrics'].apply(lambda x: re.sub(r'^.*?]', '', x))
# Removing all characters within [] - typically includes [Verse x] or [Intro]/[Outro]
eng_only['lyrics'] = eng_only['lyrics'].apply(lambda x: re.sub('[\(\[].*?[\)\]]', '', x))
