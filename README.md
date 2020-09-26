# Exo-Machina
A deep language model is trained on scientific manuscripts from NASA's Astrophysical Data System pertaining to planets out side of our Solar System. Fine tuning is done in a supervised manner in order for the AI to recommend references.

```python
import ads

papers = ads.SearchQuery(
    q="transiting exoplanet", 
    fl=['title', 'citation_count', 'abstract', 'bibtex', 
        'pub', 'year', 'page', 'keyword','identifier'],
    sort="citation_count",max_pages=4
)

for paper in papers:
    print(paper.title, paper.citation_count)

query = ads.SearchQuery(identifier='2011arXiv1111.5621B')

'''
first_paper.
first_paper.abstract              first_paper.build_citation_tree   first_paper.first_author_norm     first_paper.keys                  first_paper.pubdate
first_paper.aff                   first_paper.build_reference_tree  first_paper.id                    first_paper.keyword               first_paper.read_count
first_paper.author                first_paper.citation              first_paper.identifier            first_paper.metrics               first_paper.reference
first_paper.bibcode               first_paper.citation_count        first_paper.issue                 first_paper.page                  first_paper.title
first_paper.bibstem               first_paper.database              first_paper.items                 first_paper.property              first_paper.volume
first_paper.bibtex                first_paper.first_author          first_paper.iteritems             first_paper.pub                   first_paper.year
'''
```
