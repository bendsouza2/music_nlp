import api_key
from lyricsgenius import Genius
import requests
import pandas as pd

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
merged = pd.concat([long_songs, rap])

# API access
token = api_key.client_access_token
genius = Genius(token, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], timeout=60, verbose=False)