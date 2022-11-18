# Exo-Machina
A deep language model, GPT-2, is trained on scientific manuscripts from NASA's Astrophysical Data System pertaining to extrasolar planets and the references therein. This pilot study uses the abstracts of each article as training data in order to explore correlations in scientific literature from a language perspective. A language model is a mathematical representation for an algorithm used to generate sequences in the same way a human would to form sentances. Each word or letter in a sentance is encoded to a numerical value (e.g. using word2vec) and is appended to a list forming sequences that represent up to a paragraph worth of text. The sequences are fed into the [GPT-2](https://openai.com/blog/better-language-models/) 117M model and trained for 500,000 steps with fine tuning. After training, the language model is used to generate new text from scratch and from user input. 

- ### [View on Hugging Face API](https://huggingface.co/pearsonkyle/gpt2-exomachina?text=We+can+remotely+sense+an+atmosphere+by+observing+its+reflected%2C+transmitted%2C+or+emitted+light+in+varying+geometries.+This+light+will+contain+information+on+the+planetary+conditions+including)

### Get started fast:

```python
from transformers import pipeline

exo = pipeline('text-generation',model='pearsonkyle/gpt2-exomachina', tokenizer='gpt2', config={'max_length':1600})
machina = lambda text: exo(text)[0]['generated_text']
```

## Training Samples
~40,000 Abstracts from NASA's Astrophysical data system (ADS) and ArXiv. 

![](Figures/exoplanet_keywords.png)

A few generated samples are below: 

- *We can remotely sense an atmosphere by observing its reflected, transmitted, or emitted light in varying geometries. This light will contain information on the planetary conditions including* `temperature, pressure, composition, and cloud optical thickness. One such property that is important is...`
- *The reflectance of Earth's vegetation suggests*
`that large, deciduous forest fires are composed of mostly dry, unprocessed material that is distributed in a nearly patchy fashion. The distributions of these fires are correlated with temperature, and also with vegetation...`
- *Directly imaged exoplanets probe* `key aspects of planet formation and evolution theory, as well as atmospheric and interior physics. These insights have led to numerous direct imaging instruments for exoplanets, many using polarimetry. However, current instruments take`

The current database contains ~40,000 abstracts from:
```
 5364 - The Astrophysical Journal
 3365 - Astronomy and Astrophysics
 2704 - Monthly Notices of the Royal Astronomical Society
 1355 - The Astronomical Journal
 617 - arXiv e-prints
 498 - Icarus
 388 - Publications of the Astronomical Society of the Pacific
 324 - The Astrophysical Journal Supplement Series
 245 - Nature
 187 - Journal of Geophysical Research
 167 - Science
 145 - Astronomische Nachrichten
 129 - Planetary and Space Science
 114 - Space Science Reviews
 109 - Geophysical Research Letters
```

The number of manuscripts for each year:

![](Figures/exoplanet_histogram.png)

TODO update stats

## Generate Training Data

Scrape [ADS](https://ads.readthedocs.io/en/latest/
) and save entries into a sql database: 

`python query_ads.py -q "transiting exoplanet"`

## Pre-processing for text generation
Extract abstracts from the database and create a new file where each line is a different abstract

`python db_to_text.py`

## Create custom vocab based on the training data

`python custom_vocab.py`

## Train on a custom dataset

`python train.py`

- Train a model on [Google Colab](https://colab.research.google.com/drive/1Pur0rFi5YVdn7axYRacXWFMic4NxRexV?usp=sharing) 
- or set up your python environment: `conda env create -f environment.yml`


## Nearest Neighbor Recommendations

Build a nearest neighbor tree from the training data and use it to recommend similar abstracts

`python text_to_vec.py`

## Local Webserver

`python -m bokeh serve --show bokeh_example.py`

## Convert to Hugging Face API

TODO

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
