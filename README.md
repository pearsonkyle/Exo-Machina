# Exo-Machina
A deep language model is trained on scientific manuscripts from NASA's Astrophysical Data System pertaining to planets out side of our Solar System. Fine tuning is done in a supervised manner in order for the AI to recommend references.

![](Figures/exoplanet_keywords.png)

A collection of keywords and titles from a ~20000 manucripts related to exoplanets

## Scraping NASA ADS

https://ads.readthedocs.io/en/latest/

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

`python ads_query.py -s settings.json -q exoplanet`

Top 15 Journals in scrape:
```
 7728 - The Astrophysical Journal
 4835 - Monthly Notices of the Royal Astronomical Society
 4250 - Astronomy and Astrophysics
 1860 - The Astronomical Journal
 758 - arXiv e-prints
 610 - Physical Review D
 535 - The Astrophysical Journal Supplement Series
 505 - Icarus
 456 - Publications of the Astronomical Society of the Pacific
 378 - Journal of Cosmology and Astroparticle Physics
 351 - Nature
 215 - Science
 182 - Journal of Geophysical Research
 177 - Astronomische Nachrichten
 170 - Annual Review of Astronomy and Astrophysics
```

Manuscript Count
![](Figures/exoplanet_histogram.png)


https://colab.research.google.com/drive/1AwnFTGzqvGNFxfhwfWo6cVbwEFOiAfrl#scrollTo=H7LoMj4GA4n_

Some generated texts: 


```
We present a new technique to determine the distance to the Andromeda galaxy using the distance-distance relation. The technique is based on the orbit of the companion star to the lens, as well as the distance from the star to the Milky Way. The distance of the companion star is determined by its intensity, its rotation period, and the resulting homogeneity of the determination. We make the method available to the community and study.
```

```
High-precision photometry of the young stellar population (about 1 Myr) has revolutionized our understanding of the stellar population. The ground-based photometric surveys are now able to measure the properties of the stars and their masses. These methods are also used to detect the radial velocities of exoplanets, and to characterize hot Jupiter host stars. However, radial velocity measurements are difficult to obtain even with current techniques. In this review, we present methods and techniques used to estimate the radial velocity measurements of the young stellar population of the Galaxy using the measurement of radial velocities from the K2-MPS radial velocity surveys. These methods are more sensitive to the photometric properties of the stars and may also provide valuable information about the properties of some exoplanets.
```

```
Abundance of helium and/or hydrogen in the gas phase is expected to be the primary cause of the main flow of hydrogen in the turbulent envelope of a planetary system. However, the observational data present in the literature are not consistent with predictions of these predictions. To assess the impact of the orbital eccentricity and non-ideal conditions on the observed binary-planet system, we compare the orbital characteristics of the binary with the dynamical parameters of the host star. We find that the system is strongly influenced by the degree of eccentricity perturbations due to the non-ideal configuration, but as a result, it is not equally affected by the eccentricity perturbations. We conclude that the main flow of hydrogen is affected by the eccentricity perturbations and the orbital eccentricities of the pre-main sequence, but the non-ideal configuration and non-ideal conditions are also different.
```
[](generated_abstracts.txt)