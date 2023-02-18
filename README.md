A deep language model, GPT-2, is trained on scientific manuscripts from [ArXiv](https://arxiv.org/). This pilot study uses abstracts from ~2.1M articles as training data in order to explore correlations in scientific literature from a language modelling perspective. A language models are algorithms used to generate sequences of numers that correspond to tokens or words and can be used to represent sentances. The text samples are fed into the [GPT-2](https://openai.com/blog/better-language-models/) 117M and 774M model and trained for ~500,000 steps with fine tuning. After training, the language model is used to generate embeddings for each manuscript which can be clustered for visualization applications and queried for entity searches.

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

For the large model (GPT-2 774M) use: `pearsonkyle/gpt2-arxiv-large` (Coming soon...)



## Dependencies

- Create a new virtual environment (e.g. `conda create -n nlp python=3.9`)
- Activate the environment `conda activate nlp`
- Install [pytorch](https://pytorch.org/get-started/locally/)
    - `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
- `pip install transformers sqlalchemy sqlalchemy_utils pyuser_agent tqdm ipython jupyter datasets ftfy clean-text unidecode`

Make sure sqlalchemy is at a version < 2.0

## Training data

1. ~1.7 million abstracts from the [Arxiv](https://arxiv.org/) that you can download on [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).
2. Additionally, we include a script to query [NASA Astrophysical Database](https://ui.adsabs.harvard.edu/)

These data are converted include a SQL database which can be sorted and queried in a quick manner allowing for the easy export of training datasets

## Pre-processing for text generation/embedding
After downloading the Arxiv json above, run the following to convert it into a sqlite database:

1. `python database.py` to create sql database
2. `python json2db.py` to populate the db with json data from arxiv

## Modify the training data

Use [NASA's Astrophysical Database](https://ui.adsabs.harvard.edu/) (ADS) to add more abstracts to the database based on a keyword search. See `query_ads.py -h` for more details. You will need to sign up for an account on ADS and subscribe for an API key.

1. `python query_ads.py -q "transiting exoplanets"` to add entries
2. `python clean_db.py` to remove entries based on keywords in the abstract

## Train on a custom dataset
Train a language model using the commands below:

1. `python db2txt.py` to create a text file with one abstract per line, this script will also clean up various characters in the abstracts
2. `python train.py` to train a GPT-2 model

Interested in training this model in the cloud? Try this repo on [Google Colab](https://colab.research.google.com/drive/1Pur0rFi5YVdn7axYRacXWFMic4NxRexV?usp=sharing)

## Embeddings

A language model is used to generate encodings for each manuscript which can be clustered for visualization applications and queried for entity searches. The embeddings are generated from the [SciBERT](https://github.com/allenai/scibert) model. The embeddings are then clustered using an approximate nearest neighbor technique (ANNOY) and queried with FAISS to provide recommendations on similar articles to an input prompt.

1. `python db2embedding.py` to use create a vector for each abstract in db using an embedding from the [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) model.
2. `python db2annoy.py` to create an [approximate nearest neighbor](https://github.com/spotify/annoy) tree
3. `python eval_nearest.py` 

or 

1. `python text_to_vec.py` to create vectors based on TF-IDF and PCA


## RESTful API

Create a webserver to access the generative model for a predictive keyboard and to be able to find similar abstracts in real time
1. check: `api.py`
2. `uvicorn api:app --reload`

## Examples

Text generation and nearest neighbor recommendations in a single app:

`python -m bokeh serve --show bokeh_example.py`

## Upload to iOS

`python gpt2_to_coreml.py`



References
- https://huggingface.co/roberta-base 
- https://huggingface.co/docs
- https://huggingface.co/transformers/training.html
- https://huggingface.co/transformers/notebooks.html
- https://colab.research.google.com/drive/1vsCh85T_Od7RBwXfvh1iysV-vTxmWXQO#scrollTo=ljknzOlNoyrv
- http://jalammar.github.io/illustrated-gpt2/
- https://github.com/huggingface/swift-coreml-transformers.git
