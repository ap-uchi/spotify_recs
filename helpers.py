'''
helpers.py

Includes useful class and method definitions for running the jupyter 
notebooks in this repo.
'''
import os
from typing import Iterable, Union

import numpy as np
import pandas as pd
import spotipy
from dotenv import dotenv_values
from sklearn import linear_model, model_selection
from spotipy.oauth2 import SpotifyClientCredentials

# Set up Spotify API client credentials
config = dotenv_values('.env')
client_id = config["SPOTIFY_CLIENT_ID"]
client_secret = config["SPOTIFY_CLIENT_SECRET"]

client_credentials_manager = SpotifyClientCredentials(client_id, 
                                                            client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

class Playlist():
    '''Access all audio features of a Spotify playlist in an machine-learning 
    friendly format.'''

    def __init__(self, playlist_id):
        ## Get track metadata
        self.playlist = sp.playlist(playlist_id=playlist_id)
        self.raw_tracks = self.playlist['tracks']['items']
        self.track_attribute_labels = self.raw_tracks[0]['track'].keys()
        self.length = len(self.raw_tracks)

        ## All useful information
        self.track_ids = self.get_info('id')
        self.track_names = self.get_info('name')
        self.track_artists = self.get_info('artist')

        audio_features = sp.audio_features(self.track_ids)
        self.audio_features = [{k.strip():v for k,v in d.items()} 
                               for d in audio_features]
        
        ## Machine Learning-friendly formatting
        self.data = pd.DataFrame(index=[self.track_ids, self.track_names], 
                                 data=self.audio_features)
        self.data.columns = self.data.columns.str.strip()
        self.data.index.names = ['id', 'name']
        self.audio_feature_labels = self.data.columns

        ## ML specifically (e.g. random forest)
        # initialize likes to None
        self.data['like'] = np.nan
        self.ml_feature_labels = list(set(self.audio_feature_labels)-\
                {'type', 'id', 'uri', 'track_href', 'analysis_url'})
        self.ml_data = self.data[self.ml_feature_labels + ['like']]

    def get_info(self, info_tag:str):
        '''Unnest information from raw_tracks dict.'''

        if any([info_tag in track['track'].keys() 
                                            for track in self.raw_tracks]):
            attributes = [track['track'][info_tag] 
                            if info_tag in track['track'].keys()
                            else None
                            for track in self.raw_tracks]
            return attributes
        
    def set_like_status(self, like_labels:Union[Iterable, int]):
        '''Set the like status of songs'''

        if isinstance(like_labels, Iterable):
            if len(like_labels) == self.data.shape[0] and \
                not (isinstance(like_labels, dict) or 
                     isinstance(like_labels, pd.Series)):
                # A list-like object of ordered like values
                self.data.loc[:,'like'] = like_labels
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
                    self.ml_data.loc[keys, 'like'] = vals
                except KeyError:
                    print(f"Specified songs don't exist in playlist. "+
                          "Like status not set.")

        elif isinstance(like_labels, int):
            # Set all like values of playlist to one number
            likes = [int(like_labels)]*self.data.shape[0]
            self.data.loc[:,'like'] = likes
            self.ml_data.loc[:,'like'] = likes


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
        self.nsongs = self.ml_data.shape[0]

    def set_playlist_likes(self, nplaylist:int, likes:Union[int, Iterable]):
        self.playlist_list[nplaylist].set_like_status(likes)
        self.refresh_playlists_list()