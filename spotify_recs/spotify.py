"""
playlist.py

Access all audio features of a Spotify playlist in an machine-learning
friendly format.
"""

from typing import Union, Iterable

from dotenv import dotenv_values
import numpy as np
from numpy import random
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn import linear_model, model_selection

# Set up Spotify API client credentials
config = dotenv_values('.env')
client_id = config["SPOTIFY_CLIENT_ID"]
client_secret = config["SPOTIFY_CLIENT_SECRET"]

client_credentials_manager = SpotifyClientCredentials(client_id, 
                                                            client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


class Playlist():
    '''Access all audio features of a Spotify playlist in an machine-learning 
    friendly format.
    
    Attributes:
    ----------
    playlist : dict
        The raw playlist data from Spotify.
    raw_tracks : list
        A list of raw track data from Spotify.
    track_attribute_labels : list
        A list of track attribute labels.
    length : int
        The number of tracks in the playlist.
    track_ids : list
        A list of track IDs.
    track_names : list
        A list of track names.
    raw_track_artists : list
        A list of raw track artist data.
    track_artists : list
        A list of track artists.
    track_genres : list
        A list of track genres.
    unique_genres : list
        A list of unique genres.
    audio_features : list
        A list of audio features.
    data : DataFrame
        A DataFrame of audio features.
    audio_feature_labels : list
        A list of audio feature labels.
    ml_feature_labels : list
        A list of machine-learning friendly audio feature labels.
    ml_data : DataFrame
        A DataFrame of machine-learning friendly audio features.
    ml_likes : DataFrame
        A DataFrame of like labels.
    '''

    def __init__(self, playlist_id, get_genres=True):
        ## Get track metadata
        self.playlist = sp.playlist(playlist_id=playlist_id)
        self.raw_tracks = self.playlist['tracks']['items']
        self.track_attribute_labels = list(self.raw_tracks[0]['track'].keys())
        self.length = len(self.raw_tracks)

        ## All useful information
        self.track_ids = self.get_info('id')
        self.track_names = self.get_info('name')
        self.raw_track_artists = self.get_info('artists')
        self.track_artists = self.raw_track_artists

        # Obtain genres if specified
        if get_genres:
            self.track_genres, self.unique_genres = self.get_genres()
        else:
            self.track_genres, self.unique_genres = (None, None)

        audio_features = sp.audio_features(self.track_ids)
        self.audio_features = [{k.strip():v for k,v in d.items()} 
                               for d in audio_features]

        ## Machine Learning-friendly formatting
        self.data = pd.DataFrame(index=[self.track_ids, self.track_names], 
                                        data=self.audio_features)
        self.data = self.data.sort_index(axis='columns')
        self.data.columns = self.data.columns.str.strip()
        self.data.index.names = ['id', 'name']
        self.data['like'] = np.nan
        self.audio_feature_labels = self.data.columns

        ## ML specifically (e.g. random forest)
        self.ml_feature_labels = list(set(self.audio_feature_labels)-\
                {'type','id','uri','track_href','analysis_url','like'})
        # X
        self.ml_data = self.data[self.ml_feature_labels].sort_index(
                                                        axis='columns')
        # y, or "likes". Initialize to None
        self.ml_likes = pd.DataFrame(index=self.ml_data.index, 
                                     columns=['like'],
                                     data=[np.nan]*self.length)

    def extract_from_tracklist(self, raw_tracks, get_genres:bool=True):
        self.track_attribute_labels = list(self.raw_tracks[0]['track'].keys())
        self.length = len(self.raw_tracks)

        ## All useful information
        self.track_ids = self.get_info('id')
        self.track_names = self.get_info('name')
        self.raw_track_artists = self.get_info('artists')
        self.track_artists = self.raw_track_artists

        # Obtain genres if specified
        if get_genres:
            self.track_genres, self.unique_genres = self.get_genres()
        else:
            self.track_genres, self.unique_genres = (None, None)

        audio_features = sp.audio_features(self.track_ids)
        self.audio_features = [{k.strip():v for k,v in d.items()} 
                               for d in audio_features]

    def format_data(self):
        ## Machine Learning-friendly formatting
        self.data = pd.DataFrame(index=[self.track_ids, self.track_names], 
                                        data=[self.audio_features])
        self.data = self.data.sort_index(axis='columns')
        self.data.columns = self.data.columns.str.strip()
        self.data.index.names = ['id', 'name']
        self.data['like'] = np.nan
        self.audio_feature_labels = self.data.columns

        ## ML specifically (e.g. random forest)
        self.ml_feature_labels = list(set(self.audio_feature_labels)-\
                {'type','id','uri','track_href','analysis_url','like'})
        # X
        self.ml_data = self.data[self.ml_feature_labels].sort_index(
                                                        axis='columns')
        # y, or "likes". Initialize to None
        self.ml_likes = pd.DataFrame(index=self.ml_data.index, 
                                     columns=['like'],
                                     data=[np.nan]*self.length)


    def get_info(self, info_tag:str):
        '''Unnest information from raw_tracks dict.'''

        if any([info_tag in track['track'].keys() 
                                            for track in self.raw_tracks]):
            attributes = [track['track'][info_tag] 
                            if info_tag in track['track'].keys()
                            else None
                            for track in self.raw_tracks]
            return attributes
    
    def get_genres(self):
        '''Search for each song's genres based on album and artists. 
        Will return a list of dictionaries, each with keys "artists" and 
        "album" to indicate whether the genre source, if genres are 
        detected. This method takes the longest because of the calls to 
        spotipy.artist and spotipy.album.'''

        genres = []
        albums_list = []
        artists_list = []
        for t in self.raw_tracks:
            song_genre_dict = {'artists':None, 'album':None}
            album = sp.album(t["track"]["album"]["external_urls"]["spotify"])
            artists = [sp.artist(a["external_urls"]["spotify"]) 
                       for a in t["track"]["artists"]]

            albums_list.append(album)
            artists_list.append(artists)
            if 'genres' in album.keys():
                if len(album['genres']) > 0:
                    song_genre_dict['album'] = album['genres']
                else:
                    artists_genres = [artist['genres'] 
                                      if 'genres' in artist.keys() else None
                                      for artist in artists]
                    # Remove empty lists if the artist doesn't have genres
                    artists_genres = list(filter(
                        lambda x: len(x) > 0, artists_genres))
                                    
                    if any(artists_genres):
                        # Join artist genres and only store unique ones
                        artists_genres = np.unique(
                            np.concatenate(artists_genres))
                        song_genre_dict['artists'] = artists_genres

            genres.append(song_genre_dict)

        # clear out Nones
        clean_album_genres = list(filter(lambda x: x is not None, 
                                [g['album'] for g in genres]))
        clean_artists_genres = list(filter(lambda x: x is not None,
                                [g['artists'] for g in genres]))
        if len(clean_album_genres) > 0:
            clean_album_genres = np.concatenate(clean_album_genres)
        if len(clean_artists_genres) > 0:
            clean_artists_genres = np.concatenate(clean_artists_genres)

        unique_genres = np.unique(np.concatenate((
            [clean_album_genres, clean_artists_genres])))

        return genres, unique_genres

    def set_like_status(self, like_labels:Union[Iterable, int]):
        '''Set the like status of songs'''

        if isinstance(like_labels, Iterable):
            if len(like_labels) == self.data.shape[0] and \
                not (isinstance(like_labels, dict) or 
                     isinstance(like_labels, pd.Series)):
                # A list-like object of ordered like values
                self.ml_likes.loc[:,'like'] = like_labels
            elif type(like_labels) in {dict, pd.Series}:
                # Set the like value of specific songs
                if isinstance(like_labels, dict):
                    keys = like_labels.keys()
                    vals = like_labels.values()
                else:
                    keys = like_labels.index
                    vals = like_labels
                try:
                    self.data.loc[keys, 'like'] = vals
                    self.ml_likes.loc[keys, 'like'] = vals
                except KeyError:
                    print(f"Specified songs don't exist in playlist. "+
                          "Like status not set.")

        elif isinstance(like_labels, int):
            # Set all like values of playlist to one number
            likes = [int(like_labels)]*self.data.shape[0]
            self.data.loc[:,'like'] = likes
            self.ml_likes.loc[:,'like'] = likes


class PlaylistCluster():
    '''A class to make combining playlists easier for machine learning.'''

    def __init__(self, playlists:Iterable):
        # type checking raw input
        self.playlist_list = []
        for pl in playlists:
            if isinstance(pl, Playlist):
                self.playlist_list.append(pl)

        self.refresh_playlists_list()
        
    def refresh_playlists_list(self):
        self.data = pd.concat([pl.data for pl in self.playlist_list])
        self.ml_data = pd.concat([pl.ml_data for pl in self.playlist_list])
        self.ml_likes = pd.concat([pl.ml_likes for pl in self.playlist_list])
        self.nsongs = self.ml_data.shape[0]

    def set_playlist_likes(self, nplaylist:int, likes:Union[int, Iterable]):
        self.playlist_list[nplaylist].set_like_status(likes)
        self.refresh_playlists_list()


class Song():
    '''Access audio features of a Spotify song.'''

    def __init__(self, song_id:str=None, song_name:str=None, song_dict:dict=None) -> None:
        for ikwarg, kwarg in enumerate([song_id, song_name, song_dict]):
            
            self.id = None
            self.name = None
            self.artist = None
            self.attributes = None

            if kwarg is not None:
                self.input = kwarg
                if ikwarg == 0:
                    self.id = self.input
                    self.attributes = sp.track(self.id)
                    self.name = self.attributes['name']
                elif ikwarg == 2:
                    self.id = self.get_info_from_dict(self.input, 'id')
                    self.name = self.get_info_from_dict(self.input, 'name')
                    self.artist = self.get_info_from_dict(self.input, 'artists')

        self.audio_features = sp.audio_features(self.id)
        
        self.data = pd.DataFrame(data=self.audio_features).sort_index(axis='columns')
        # initialize like to none
        self.data['like'] = np.nan
        self.ml_likes = np.nan

        data_multiIndex = pd.MultiIndex.from_frame(pd.DataFrame({'id':[self.id], 'name':[self.name]}))
        self.data.index = data_multiIndex
        
        self.audio_feature_labels = self.data.columns

        ## ML specifically (e.g. random forest)
        self.ml_feature_labels = list(set(self.audio_feature_labels)-\
                {'type', 'id','uri','track_href','analysis_url','like'})
        self.ml_data = self.data.loc[:, self.ml_feature_labels].sort_index(axis='columns')

    def get_info_from_dict(self, track:dict, info_tag:str):
        '''Unnest information from raw_tracks dict.'''

        if info_tag in track['track'].keys():
            attributes = track['track'][info_tag] 
            return attributes

def grab_songs(nsongs=10, params:dict=None):
    sp.recommendations(limit=nsongs)

    return playlist

def grab_a_song():
    """Get a random song from Spotify."""

    # Get a random search term or a random track ID
    search_term = random.choice(['love', 'happy', 'dance', 'rock', 'jazz'])
    results = sp.search(q=search_term, type='track', limit=50)

    # Get a random track from the search results
    track = random.choice(results['tracks']['items'])
    song = Song(track['id'])

    return song