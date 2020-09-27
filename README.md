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
 4761 - The Astrophysical Journal
 2868 - Astronomy and Astrophysics
 2412 - Monthly Notices of the Royal Astronomical Society
 1244 - The Astronomical Journal
 495 - arXiv e-prints
 364 - Icarus
 361 - Publications of the Astronomical Society of the Pacific
 302 - The Astrophysical Journal Supplement Series
 234 - Nature
 180 - Journal of Geophysical Research
 140 - Science
 133 - Journal of Quantitative Spectroscopy and Radiative Transfer
 124 - Astronomische Nachrichten
 116 - Astrobiology
 110 - Planetary and Space Science
```

Manuscript Count
![](Figures/exoplanet_histogram.png)