Project to practice NLP in Python on music data taken from Genius API

## Libraries Used 
* Pandas
* LyricsGenius
* Numpy
* Sklearn
* Imblearn
* Langdetect
* Requests
* Matplotlib



## Data Sources 
* Lyric data - Genius API 
* [Hip-hop songs list](https://github.com/sjockers/bbc-best-rapmusic/blob/master/data/polls.csv)
* [Rock songs list](https://github.com/fivethirtyeight/data/blob/master/classic-rock/classic-rock-song-list.csv)


## Stage 1
[Data collection](https://github.com/bendsouza2/music_nlp/blob/main/data_collection.py) - querying the genius API to get lyric data for our list of songs

## Stage 2
[Data cleaning](https://github.com/bendsouza2/music_nlp/blob/main/data_cleaning.py) - removing non-English language lyrics, punctuation and non-lyrics from the data.

## Stage 3
[Model building and analysis](https://github.com/bendsouza2/music_nlp/blob/main/model.py) - testing different classifiers with and without stop words to determine the best method of predicting music genre using lyrics

## Stage 4
[Visualising the data](https://github.com/bendsouza2/music_nlp/blob/main/visual.py) - creating visuals to display the most common words for each genre