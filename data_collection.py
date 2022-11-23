import api_key
from lyricsgenius import Genius
import pandas as pd
import os
import time
from requests.exceptions import HTTPError, Timeout

# Rock songs
long_songs = pd.read_csv('classic-rock-song-list.txt')
long_songs.rename(columns={'Song Clean': 'title', 'ARTIST CLEAN': 'artist'}, inplace=True)
unnecessary = ["First?", "Year?", "PlayCount", "F*G", "COMBINED", 'Release Year']
long_songs.drop(unnecessary, axis=1, inplace=True)
long_songs['genre'] = 'rock'

# Hip-hop songs
rap = pd.read_csv('rap_songs.txt')
rap.rename(columns={' title': 'title', ' artist': 'artist'}, inplace=True)
hh_unnecessary = ['rank', ' gender', ' year', ' critic_name', ' critic_rols', ' critic_country', ' critic_country2']
rap.drop(hh_unnecessary, axis=1, inplace=True)
rap['genre'] = 'rap'

# Full song list
merged = pd.concat([long_songs, rap], ignore_index=False)
merged.reset_index(drop=True, inplace=True)

# API access
token = api_key.client_access_token
genius = Genius(token, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], timeout=45, verbose=False)


# Adding lyrics
lyrics = []
for i in merged.index:
    data = pd.DataFrame()
    song = merged.loc[i]['title']
    artist = merged.loc[i]['artist']
    genre = merged.loc[i]['genre']
    success = False
    retries = 1
    while not success and retries < 5:
        try:
            result = genius.search_song(title=song, artist=artist)
            success = True
        except HTTPError:
            wait = retries * 30
            print(HTTPError.errno)
            print('Error, waiting {} secs before trying again'.format(wait))
            time.sleep(wait)
            retries += 1
        except Timeout:
            wait = retries * 30
            print(Timeout.errno)
            print('Error, waiting {} secs before trying again'.format(wait))
            time.sleep(wait)
            retries += 1
    if result is not None:
        song_lyrics = result.lyrics
        data = {'id': i, 'title': song, 'artist': artist, 'genre': genre, 'lyrics': song_lyrics}
        lyrics.append(data)


data_lyrics = pd.DataFrame(lyrics)  # converting list of dictionaries to df
print(data_lyrics.head())

# Writing to csv file
data_lyrics.to_csv(os.getcwd() + '/lyric_data')

