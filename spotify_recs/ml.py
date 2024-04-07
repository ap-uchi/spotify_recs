"""
ml.py

Use random forests to predict whether you will like a song based on its audio features. 
"""

import pickle as pkl
from typing import Iterable, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , roc_auc_score
from sklearn import tree
import graphviz

# Custom packages
#import helpers as h
import spotify

class SpotipyModel():
    """A model that predicts whether you will like a song based on its audio features.
    
    Attributes:
    -----------
    playlist_ids : list
        A list of Spotify playlist IDs.
    playlists : list
        A list of Playlist objects.
    plc : PlaylistCluster
        A PlaylistCluster object that contains the playlists.
    model : RandomForestClassifier
        A random forest model that predicts whether you will like a song based on its audio features.
    """

    def __init__(self,
                 playlist_ids:Union[str, list],
                 playlist_scores:list=None,
                 test_size=0.3, **kwargs) -> None:
        """A random forest model that generates new playlists based on input playlists.
        
        Arguments:
        ----------
        playlist_ids : str or list
            A list of Spotify playlist IDs.
        n_estimators : int, default=6
            The number of trees in the forest.
        max_depth : int, default=2
            The maximum depth of the tree.
        """
        if isinstance(playlist_ids, str):
            # If playlist_ids is a string, convert it to a list
            playlist_ids = [playlist_ids]

        if playlist_scores is None:
            # If playlist_scores is None, set it to a list of ones
            playlist_scores = [1]*len(playlist_ids)

        self.playlist_ids = playlist_ids
        self.playlist_scores = playlist_scores

        # Featurize the playlists
        self.playlists = [spotify.Playlist(playlist_id) for playlist_id in playlist_ids]
        self.plc = spotify.PlaylistCluster(self.playlists)

        # rate the songs in the playlists
        self.score_column = self.score_songs(self.playlist_scores)

        # Split the data into training and testing sets
        self.X = self.plc.ml_data.loc[:, self.plc.ml_data.columns != self.score_column]
        self.y = self.plc.ml_likes[self.score_column]

        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size)

        # Set by child class
        self.model = self.init_model(**kwargs)

    def init_model(self, **kwargs) -> None:
        """Initialize ML model."""
        return None

    def score_songs(self, playlist_scores) -> str:
        """Score the songs in the playlists based on the playlist ratings."""
        return ""

    def train(self) -> None:
        """Train the model on the training data."""
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def generate_playlist(self, nsongs:int, batch_size:int=None) -> list:
        pass


class SpotipyRFClassifier(SpotipyModel):

    def __init__(self, playlist_ids:Union[str, list], n_estimators=6, max_depth=2, test_size=0.3, **kwargs) -> None:
        """A random forest model that generates new playlists based on input playlists.
        
        Arguments:
        ----------
        playlist_ids : str or list
            A list of Spotify playlist IDs.
        n_estimators : int, default=6
            The number of trees in the forest.
        max_depth : int, default=2
            The maximum depth of the tree.
        """
        super().__init__(playlist_ids, test_size,
                         n_estimators=n_estimators,
                         max_depth=max_depth, **kwargs)

    def init_model(self, n_estimators=6, max_depth=2, **kwargs) -> None:
        """Initialize RandomForestClassifier model."""
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, **kwargs)
        
        return model

    def visualize_model_tree(self, tree_num:int=0) -> None:
        """Visualize the a tree in the random forest using graphviz.

        Arguments:
        ----------
        tree_num : int, default=0
            The tree number to visualize.
        
        Returns:
        --------
        graph : graphviz.Source
            A graphviz object that can be rendered to visualize the tree.
        """
        dot_data = tree.export_graphviz(self.model.estimators_[tree_num],
                                        out_file=None,
                                        feature_names=self.plc.ml_data.columns,
                                        class_names=['dislike', 'like'],
                                        filled=True, rounded=True,
                                        special_characters=True)

        graph = graphviz.Source(dot_data)
        graph.render("tree")

        return graph
    
    def score_songs(self, playlist_scores) -> str:
        """Score the songs in the playlists based on user-defined 
        playlist ratings."""
        
        score_column = 'like' # manually defined

        for i, score in enumerate(playlist_scores):
            self.plc.set_playlist_likes(i, score)

        return score_column

    def print_model_tree(self):
        """Print a human-readable text-based representation of the tree."""
        for dt in self.model.estimators_:
            n_nodes = dt.tree_.node_count
            children_left = dt.tree_.children_left
            children_right = dt.tree_.children_right
            feature = dt.tree_.feature
            threshold = dt.tree_.threshold

            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
            while len(stack) > 0:
                # `pop` ensures each node is only visited once
                node_id, depth = stack.pop()
                node_depth[node_id] = depth

                # If the left and right child of a node is not the same we have a split
                # node
                is_split_node = children_left[node_id] != children_right[node_id]
                # If a split node, append left and right children and depth to `stack`
                # so we can loop through them
                if is_split_node:
                    stack.append((children_left[node_id], depth + 1))
                    stack.append((children_right[node_id], depth + 1))
                else:
                    is_leaves[node_id] = True

            print(
                "The binary tree structure has {n} nodes and has "
                "the following tree structure:\n".format(n=n_nodes)
            )
            for i in range(n_nodes):
                if is_leaves[i]:
                    print(
                        "{space}node={node} is a leaf node.".format(
                            space=node_depth[i] * "\t", node=i
                        )
                    )
                else:
                    print(
                        "{space}node={node} is a split node: "
                        "go to node {left} if X[:, {feature}] <= {threshold} "
                        "else to node {right}.".format(
                            space=node_depth[i] * "\t",
                            node=i,
                            left=children_left[i],
                            feature=feature[i],
                            threshold=threshold[i],
                            right=children_right[i],
                        )
                    )

    def generate_playlist(self, nsongs:int, batch_size:int=None) -> list:
        
        numbered_ml_data = self.plc.ml_data.reset_index()
        liked_songs = numbered_ml_data[self.plc.ml_likes.reset_index()['like'] == 1]

        if batch_size is None:
            batch_size = int(nsongs*0.20)

        # First filter songs by the Gaussian statistics of the audio features 
        # in liked playlists
        feature_mins = liked_songs.mean() - liked_songs.std()
        feature_maxs = liked_songs.mean() + liked_songs.std()

        feature_mins.index = 'min_' + feature_mins.index
        feature_maxs.index = 'max_' + feature_maxs.index
        
        # Get seed artists, songs, and tracks
        seed_songs = liked_songs['id']
        #seed_artists = 

        new_songs = []

        # Query based on these features
        while len(new_songs) < nsongs:
            #sp.recommendations()

            spotify.grab_a_song()
        