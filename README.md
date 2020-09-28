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

```json
{
    "samples":[
        {
            "Input":"Phase curves of exoplanets",
            "Ouput":"Phase curves of exoplanets around Sun-like stars using AO-reconstruction techniques. The spectral range is analogous to the AO performance of large-aperture telescopes. AO-reconstruction techniques yield much better spectral capabilities than the existing ground-based AO instruments combined. This paper presents the main results of the study and presents a catalog of the 16 known systems with transit light curves. This catalog is based on eight years of AO-reconstruction follow-up observations, and we model the light curves with a Light-Time Replicator (LTR) algorithm, which is applicable to the whole wavelength range of the transit light curves. We obtained a total of 23 observational light curves for the 16 systems, and we generate synthetic light curves to fit the light curves. For six of the 16 systems, we derive a linear ephemeris for the stellar rotation. We find that the system parameters are generally consistent with the experimental values, both with an accuracy better than ±0.5 per cent. The mean transit depth is 6.2 ± 1.2 ppm, and the orbital period is 3.0452 ± 0.00001 days. The corrected systems are given in Table A.1."
        },
        {
            "Input":"Infrared observations of transiting exoplanets",
            "Ouput":"Infrared observations of transiting exoplanets with an accuracy of 0.2 to 10 μm, and the resulting precision of the spectroscopic transit depth, we find that a combination of the two approaches provides a detection sensitivity of about 10 times better than that of the Doppler technique alone, and a sensitivity about 10 times greater than the combined technique. We also demonstrate that we can reliably detect starspots through a combination of the two techniques. We find that the Doppler technique does not provide a robust detection of spots on the stellar surface that is required to explain the lack of spots detected in the Kepler data. We also discuss the implications of our results for the detection of magnetic spots on the stellar surface, and for the parameters of planetary transits and spot features in the visible and near-infrared."
        },
        {
            "Input":"Infrared observations of transiting exoplanets",
            "Ouput":"Infrared observations of transiting exoplanets, the degree of chromospheric emission from the host star is sensitive to the stellar parameters, and for a given planet, the effects of rotation affect the planet-star flux ratio. We have investigated the means by which stellar winds affect the planetary luminosity, and have found that for a given luminosity, the planet's atmospheric mass loss rate depends not only on the stellar wind, but also on the stellar wind mass-loss rate. This implies that if the stellar wind mass-loss rate is not sufficiently high, then the planetary orbit does not transit the stellar disc."
        },
        {
            "Input":"",
            "Output":"We study the formation of a planetary system around a brown dwarf by means of two-dimensional hydrodynamic simulations. We consider two different initial conditions: a solar-like metallicity and a massive disk (mainly a massive disk with a hydrogen-rich core). In all the simulations, the two disk components evolve towards a common final configuration. The planetary system is initially formed with a mass of 0.06 M☉ and a semimajor axis of 1.86 AU. The initial conditions have different initial mass ratios, but the mass ratio and initial semimajor axis follow the same trend, which indicates that the mass of the disk is not required to form a planet, even in the presence of the accretion disk. The initial configuration of the disk is determined by the initial mass and semimajor axis of the planets, and the disk mass is estimated from its radial surface brightness. The disk has a mass of 0.03 M☉, radius of 0.55 AU, and has a hydrogen-rich core. We perform simulations that are constrained by the observational uncertainties on the mass, radius, and luminosity of the disk. We find that the mass and radius of the disk are large enough to form a planet with a mass of 0.12 M☉, with a semimajor axis of 0.28 AU, and a mass accretion rate of &gt;about  10-10 M☉ yr-1. The planetary system has a semimajor axis of 0.03 AU, and the mass accretion rate is &gt;about  10-10 M☉ yr-1. We find that the planet does not accrete directly through the disk, but depends on the disk for its initial configuration. We also investigate how the mass accretion rate and mass accretion rate depend on the mass ratio for the system. We find that the mass accretion rate increases as the mass ratio decreases, but the mass accretion rate decreases as the mass ratio increases. We also find that, in a solar system like the solar one, the mass accretion rate is dominated by disk accretion for the higher mass disks."
        },
        {
           "Input":"",
           "Output":"We present a detailed spectroscopic analysis of the recently discovered candidate extrasolar planetary systems SDSS 1256+48 and HIP 51634, which both host exoplanets with orbital periods of 1.9 days and transit radii of 4.5 and 19.0 ± 0.6 R_earth , respectively. We have carried out detailed analyses of the spectral and photometric properties of both stars, and have identified a strong correlation between their colours and masses. Our analysis of the SDSS 1256+48 system reveals that the main component of the system is relatively metal poor, while for the other three stars the abundance pattern is consistent with that of a typical solar-type star of the same mass and metallicity. We also confirm the existence of a previous suggestion that SDSS 1256+48 and HIP 51634 host a candidate brown dwarf companion. Our analysis of the HADS data for SDSS 1256+48 reveals that the system is likely to host a substellar companion with a minimum mass of at least 0.03 M☉ .Based on observations obtained at the Canada-France-Hawaii Telescope (CFHT) with ESO telescopes at the Paranal Observatory (IP) and at the ESO Observatory (La Silla, Chile).The reduced spectra as FITS files are only available at the CDS via anonymous ftp to"
        }
    ]
}
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