A semantic search engine for manuscript on the ArXiv. Semantic search means that the search engine will understand the meaning of the query and return results that are semantically similar to the query. This is in contrast to a keyword search engine which will return results that contain the exact keywords in the query. This search engine uses a language model to generate embeddings for each manuscript which can be clustered for visualization applications and queried for entity searches. The embeddings are generated using tfidf+PCA, SciBERT, and GPT-2. The embeddings are then clustered using an approximate nearest neighbor technique (ANNOY) and queried with FAISS to provide recommendations on similar articles to an input prompt. The language model is trained on ~2.1M abstracts from the ArXiv.

## Dependencies

- Create a new virtual environment (e.g. `conda create -n nlp python=3.9`)
- Activate the environment `conda activate nlp`
- Install [pytorch](https://pytorch.org/get-started/locally/)
    - `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia nmslib`
    - `conda install -c conda-forge spacy`
    - `https://github.com/allenai/scispacy`
    - `pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz`

- `pip install transformers sqlalchemy sqlalchemy_utils pyuser_agent tqdm ipython jupyter datasets ftfy clean-text unidecode`

Make sure sqlalchemy is at a version < 2.0

## Training data

1. ~2.3 million abstracts from the [Arxiv](https://arxiv.org/) that you can download on [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).
2. Additionally, we include a script to query [NASA Astrophysical Database](https://ui.adsabs.harvard.edu/)

These data are converted include a SQL database which can be sorted and queried in a quick manner allowing for the easy export of training datasets

## Set up

1. Download the [Arxiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) from Kaggle
2. `unzip arxiv-metadata-oai-snapshot.json.zip`
3. `python database.py` to create sql database
4. `python json2db.py` to populate the db with json data from arxiv (inspect script to filter out categories)
5. `python clean_db.py` to remove entries based on keywords in the abstract
5. `python db_to_vec.py` create embeddings with tfidf+PCA

## Adding to the database

Use [NASA's Astrophysical Database](https://ui.adsabs.harvard.edu/) (ADS) to add more abstracts to the database based on a keyword search. See `query_ads.py -h` for more details. You will need to sign up for an account on ADS and subscribe for an API key.

1. `python query_ads.py -q "transiting exoplanets"` to add entries

## Embeddings

A language model is used to generate encodings for each manuscript which can be clustered for visualization applications and queried for entity searches. The embeddings are generated from the [SciBERT](https://github.com/allenai/scibert) model. The embeddings are then clustered using an approximate nearest neighbor technique (ANNOY) and queried with FAISS to provide recommendations on similar articles to an input prompt.

1. `python db2embedding.py` to use create a vector for each abstract in db using an embedding from the [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) model.
2. `python db2annoy.py` to create an [approximate nearest neighbor](https://github.com/spotify/annoy) tree
3. `python eval_nearest.py` 

or 

1. `python text_to_vec.py` to create vectors based on TF-IDF and PCA
2. `python eval_tfidf.py`


## RESTful API

Create a webserver to access the generative model for a predictive keyboard and to be able to find similar abstracts in real time
1. check: `api.py`
2. `uvicorn api:app --reload`

## Examples

Text generation and nearest neighbor recommendations in a single app:

`python -m bokeh serve --show bokeh_example.py`

## Upload to iOS

`python gpt2_to_coreml.py`

## Training GPT-2

A deep language model, GPT-2, is trained on scientific manuscripts from [ArXiv](https://arxiv.org/). This pilot study uses abstracts from ~2.1M articles as training data in order to explore correlations in scientific literature from a language modelling perspective. A language models are algorithms used to generate sequences of numers that correspond to tokens or words and can be used to represent sentances. The text samples are fed into the [GPT-2](https://openai.com/blog/better-language-models/) 117M and trained for ~500,000 steps with fine tuning. After training, the language model is used to generate embeddings for each manuscript which can be clustered for visualization applications and queried for entity searches.

- ### [View on Hugging Face API](https://huggingface.co/pearsonkyle/gpt2-arxiv?text=We+can+remotely+sense+an+atmosphere+by+observing+its+reflected%2C+transmitted%2C+or+emitted+light+in+varying+geometries.+This+light+will+contain+information+on+the+planetary+conditions+including)

### Get started fast:

```python
from transformers import pipeline

ai = pipeline('text-generation',model='pearsonkyle/gpt2-arxiv', tokenizer='gpt2', config={'max_length':1600})
machina = lambda text: ai(text)[0]['generated_text']
```

A few generated samples are below: 

- *We can remotely sense an atmosphere by observing its reflected, transmitted, or emitted light in varying geometries. This light will contain information on the planetary conditions including* `temperature, pressure, composition, and cloud optical thickness. One such property that is important is...`
- *The reflectance of Earth's vegetation suggests*
`that large, deciduous forest fires are composed of mostly dry, unprocessed material that is distributed in a nearly patchy fashion. The distributions of these fires are correlated with temperature, and also with vegetation...`
- *Directly imaged exoplanets probe* `key aspects of planet formation and evolution theory, as well as atmospheric and interior physics. These insights have led to numerous direct imaging instruments for exoplanets, many using polarimetry. However, current instruments take`

![](Figures/exoplanet_keywords.png)

Interested in training this model in the cloud? Try this repo on [Google Colab](https://colab.research.google.com/drive/1Pur0rFi5YVdn7axYRacXWFMic4NxRexV?usp=sharing)

1. `python train.py` to train a GPT-2 model, will have to make a script to write the abstracts to a txt file first