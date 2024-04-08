# spotify_recs

The goal of this repo is to demonstrate how to use a Random Forest classifier to generate new Spotify playlists based on old ones. 

Go through the notebooks in the following order:

[`0_read_files.ipynb`](./tutorial/0_read_files.ipynb)

[`1_train_model.ipynb`](./tutorial/1_train_model.ipynb)

[`2_playlist_generator.ipynb`](./tutorial/2_playlist_generator.ipynb)

To run the notebooks, you will first need to install the package dependencies in a virtual environment. The easiest and fastest way to do this is using [`mamba`](https://mamba.readthedocs.io/en/latest/index.html), which is just a better and faster version of [`conda`](https://docs.conda.io/en/latest/).

Follow the [installation instructions for `mamba`](https://mamba.readthedocs.io/en/latest/installation.html).

Then when you're ready, run the following in this directory:

        $ mamba env create -f environment.yml
        $ mamba activate rf-spot

You will then be ready to run the notebooks.

If this demo is not enough to satisfy your appetite, look at hooking up [ChatGPT to Spotify](https://jonathansoma.com/words/spotify-langchain-chatgpt.html)!