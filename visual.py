import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import os

df = pd.read_csv('ld_clean')

# Hip-hop word cloud
hh = df.loc[df['genre'] == 'rap']
hh_lyrics = " ".join(word for word in hh['lyrics'])
hh_remove = ['ya', 'yo', 'go', 'u']

mask_hh = np.array(Image.open('hh.webp'))

cloud = WordCloud(background_color='white', max_font_size=70, random_state=1, mask=mask_hh)
cloud.generate(hh_lyrics)

plt.figure(figsize=(18, 10))
ic = ImageColorGenerator(mask_hh)
plt.imshow(cloud.recolor(color_func=ic), interpolation='bilinear')
plt.axis('off')
# plt.show()


# Rock word cloud
rock = df.loc[df['genre'] == 'rock']
rock_lyrics = " ".join((word for word in rock['lyrics']))
rock_remove = []

mask_rr = np.array(Image.open('rr.jpeg'))
cloud_rr = WordCloud(background_color='white', max_font_size=70, random_state=1, mask=mask_rr)
cloud_rr.generate(rock_lyrics)

plt.figure(figsize=(18, 10))
ic_rr = ImageColorGenerator(mask_rr)
plt.imshow(cloud_rr.recolor(color_func=ic_rr), interpolation='bilinear')
plt.axis('off')
plt.savefig(os.getcwd() + '/Visualisations/Data Overview/rock_wordcloud.jpeg')