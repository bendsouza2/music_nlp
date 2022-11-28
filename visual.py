import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

df = pd.read_csv('ld_clean')
hh = df.loc[df['genre'] == 'rap']
hh_lyrics = " ".join(word for word in hh['lyrics'])
remove = ['ya', 'yo', 'go', 'u']

mask = np.array(Image.open('hh.webp'))

cloud = WordCloud(background_color='white', max_font_size=70, random_state=1, mask=mask)
cloud.generate(hh_lyrics)

plt.figure(figsize=(18, 10))
ic = ImageColorGenerator(mask)
plt.imshow(cloud.recolor(color_func=ic), interpolation='bilinear')
plt.axis('off')
plt.show()
