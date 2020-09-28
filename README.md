# Exo-Machina
A deep language model, GPT-2, is trained on scientific manuscripts from NASA's Astrophysical Data System pertaining to extrasolar planets and the references therein. This pilot study uses the abstracts of each article as training data in order to explore correlations in scientific literature from a language perspective. 

### [Try it!](https://pearsonkyle.github.io/Exo-Machina/)

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

```json
{
    "samples":[
        {
            "Input":"Phase curves of exoplanets",
            "Ouput":"Phase curves of exoplanets around Sun-like stars using AO-reconstruction techniques. The spectral range is analogous to the AO performance of large-aperture telescopes. AO-reconstruction techniques yield much better spectral capabilities than the existing ground-based AO instruments combined. This paper presents the main results of the study and presents a catalog of the 16 known systems with transit light curves. This catalog is based on eight years of AO-reconstruction follow-up observations, and we model the light curves with a Light-Time Replicator (LTR) algorithm, which is applicable to the whole wavelength range of the transit light curves. We obtained a total of 23 observational light curves for the 16 systems, and we generate synthetic light curves to fit the light curves. For six of the 16 systems, we derive a linear ephemeris for the stellar rotation. We find that the system parameters are generally consistent with the experimental values, both with an accuracy better than ±0.5 per cent. The mean transit depth is 6.2 ± 1.2 ppm, and the orbital period is 3.0452 ± 0.00001 days. The corrected systems are given in Table A.1."
        },
        {
            "Input":"Infrared observations of transiting exoplanets",
            "Ouput":"Infrared observations of transiting exoplanets, the degree of chromospheric emission from the host star is sensitive to the stellar parameters, and for a given planet, the effects of rotation affect the planet-star flux ratio. We have investigated the means by which stellar winds affect the planetary luminosity, and have found that for a given luminosity, the planet's atmospheric mass loss rate depends not only on the stellar wind, but also on the stellar wind mass-loss rate. This implies that if the stellar wind mass-loss rate is not sufficiently high, then the planetary orbit does not transit the stellar disc."
        },
        {
            "Input":"Infrared observations of transiting exoplanets probe the temperature structure",
            "Output":"Infrared observations of transiting exoplanets probe the temperature structure under the influence of stellar irradiation. However, the observational parameters are not well constrained due to the limited time-sampling and sensitivity of the observations. The transmission spectra of hot Jupiter exoplanets revealed a strong thermal inversion layer above the planetary atmosphere. The inversion layer is likely to be dominated by planet-to-star-integrated heat, which is dominant over planetary gravity and thermal energy, and may be observable by ground-based telescopes. Here, we use an inversion model to show how a thermal inversion layer in the planet atmosphere can be inferred from light curves of hot Jupiter exoplanets. The planets in the Kepler-186 system are among the most eccentric (e > 0.3) exoplanets discovered to date, and are the only known cases of high eccentricity planets that orbit a star with a stellar mass comparable to that of the Sun. These planets orbit the star at a separation of 8.0 AU, and are poised to receive large amounts of irradiation for which they are unlikely to be in a gas-giant proto-planetary disk. This means that the planet's orbit and the star's orbit will be moving during transit, and the planet's orbital eccentricity"
        },
        {
           "Input":"",
           "Output":"The planets in a binary system can undergo two formation pathways. In the most common case, the planet leaves the binary system in a close-in orbit and migrates inward through the disk. The evolution is different for different types of binary systems. For an open binary system, the planet can become trapped in a mean motion resonance with the binary orbit. For an enclosed binary system, the planet can enter the inner disk and migrate outward. By studying the evolution of the planets in a closed binary system, we find that the migration is determined by the eccentricity of the binary orbit. When the planet is not in the disk, it does not enter an intermediate disk from which it can migrate inward. It can also enter a second intermediate disk. A detailed study of the evolution of the mass-ratio distribution of the migration is possible."
        }
    ]
}
```

## Dev Instructions

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
- more training samples (currently 20569 abstracts)
- twitter bot?