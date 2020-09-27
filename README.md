# Exo-Machina
A deep language model, GPT-2, is trained on scientific manuscripts from NASA's Astrophysical Data System pertaining to extrasolar planets and the references therein. This pilot study uses the abstracts of each article as training data in order to explore correlations scientific literature from a language perspective. 

Here is a collection of words from the titles of ~20000 exoplanet-related manuscripts:

![](Figures/exoplanet_wordcloud.png)


A language model is a mathematical representation for an algorithm used to generate sequences in the same way a human would to form sentances. Each word or letter in a sentance is encoded to a numerical value (e.g. using word2vec) and is appended to a list forming sequences that represent up to a paragraph worth of text. The sequences are fed into the [GPT-2](https://openai.com/blog/better-language-models/) 355M model and trained for 10,000 steps with fine tuning. After training, the language model is used to generate new text from scratch and from user input. A few generated samples are below: 

```
We present, as a starting point, a simplified approach to identifying 
exoplanets by measuring their chemical abundances using an empirical 
Bayesian framework for modeling planetary biosignature atmospheres. 
In addition, we present an empirical Bayesian model to predict the 
chemical compositions and chemical abundances for exoplanets from 
gas-driven stellar evolution models. Theoretical models used to construct 
this general model include the stellar evolution (e.g., solar-type 
evolution), stellar evolutions (e.g., chemical evolution, photometric
evolution and stellar age), and chemical abundance variations (e.g.,
formation, transport and/or elimination rate of the organic aerosols
from their atmospheres). The simulated atmospheres of an exoplanet of
a certain physical type, for which we have simulated the chemistry and 
chemical composition by applying our model, are compared with observations 
made before the beginning of the Kepler period. We find that our chemical 
models match well with the observations in some planetary systems, and 
agree well with stellar evolution and stellar age models. A comparison 
of the chemical compositions and abundances found in our models with 
those seen in the solar system indicates that chemical evolution is a 
natural part of the mass-radius relation.
```

```
Context. Atmospheric chemistry and evolution of solar-type stars has been extensively studied from decades of space missions that have provided abundant observations for stellar parameters, the chemical abundances, and chemical abundances of abundances of organic species. For example, the first generation of the James Webb Space Telescope (JWST) instrument has provided data that may help to constrain and calibrate chemical abundance observations for stars in the habitable zone of the Milky Way. 
Aims: We aim to use a large set of high-precision photometric information to confirm and refine the properties of the Earth-like exoplanet Kepler-5b using the best data obtained so far for this system. Methods: We have constructed a database of high-quality high-resolution spectra for seven nearby stars and studied the abundance differences between two samples selected from a set of Kepler parameter space maps. For high-precision measurements of the stars' abundance functions we have used a Bayesian procedure that utilizes Bayesian reasoning and Bayesian phylogenetics. 
Results: We confirm the metallicity and abundance ratios of the Kepler-5b sample by analysing photometric variation in the Kepler-7/K1/K2 data. However, based on our new metallicity, the abundance ratios do not reflect stellar metallicity. Finally, based on the stars' abundance ratios, we find no evidence for high metallicity abundance variations in the sample chosen.
```

The language model is also capable of generating sequences from user input: 
```json
Input: The infrared atmospheres of transiting exoplanets
Output:
```

Explore more outputs here

## Instructions

python dependencies: sqlalchemy, ads, matplotlib, numpy, wordcloud

The articles from ADS are stored in a SQL database on AWS in order to centralize and make the data open access

## Setup a SQL Databse to store training samples
A postegres SQL database is set up on Amazon RDS in order to provide online access to the same data for multiple computers. Follow the instructions below to set up your own database using the Free Tier of services on AWS: 

1. sign in or register: https://aws.amazon.com/
2. Search for a services and go to RDS 

Add your credentials to a new file called `settings.json` like such:
```
{
    "database":{
        "dialect":"postgresql",
        "username":"readonly",
        "password":"readonly",
        "endpoint":"exomachina.c4luhvcn1k1s.us-east-2.rds.amazonaws.com",
        "port":5432,
        "dbname":"exomachina"
    }
}
```

## Scraping NASA ADS

https://ads.readthedocs.io/en/latest/

Scrape ADS and save entries into a sql database: 

`python ads_query.py -s settings.json -q exoplanet`

```
usage: ads_query.py [-h] [-q QUERY] [-s SETTINGS] [-k KEY]

optional arguments:
  -h, --help            show this help message and exit
  -q QUERY, --query QUERY
                        Initial search criteria
  -s SETTINGS, --settings SETTINGS
                        Settings file
  -k KEY, --key KEY     Settings key
```

Letting the scrape run for ~2 hours found articles from these publications in descending order:
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

## Pre-processing
Extract abstracts from the database and create a new file where each line is an new sample. 

`python make_training.py`

## Language Model Optimization

Here is a great resource for learning how to train the GPT-2 model for free using Google Collab: 
https://colab.research.google.com/drive/1AwnFTGzqvGNFxfhwfWo6cVbwEFOiAfrl#scrollTo=H7LoMj4GA4n_

Browse results on our [website]()


## Things to improve
- bigger language model (355M + 10000 steps was used here)
- supervised training on sentances with references
- extract text from manuscript PDF
- more training samples (currently 26569 abstracts)
- twitter bot?

## Other samples

```
A sample for the first study of transiting extrasolar planets is presented. In this study, the transits of both transits and non-transits of the primary companion were obtained for the two stars in order to test the existence of a period of time extending beyond the initial value of the transit time table that is required to be confirmed as transit data from future measurements. The results indicate that the two stars may have similar radii of about 0.5 R☉ while the secondary mass star can be identified as a massless mass. The data imply that there is a period of time, ≳20,000 days, that has elapsed between each transit that can be ascribed to the gravitational interactions of the two stars. This period appears to be consistent with the reported values for the orbital period of the two companion stars. This suggests that the properties of the orbits of both primary stars are similar.
```