
import arxiv
# Query for a paper of interest, then download
paper = arxiv.query(id_list=["1707.08567"])[0]
arxiv.download(paper)
# You can skip the query step if you have the paper info!
paper2 = {"pdf_url": "http://arxiv.org/pdf/1707.08567v1",
          "title": "The Paper Title"}
arxiv.download(paper2)

# Download the gzipped tar file
arxiv.download(paper,prefer_source_tarfile=True)

# Returns the object id
def custom_slugify(obj):
    return obj.get('id').split('/')[-1]

# Download with a specified slugifier function
arxiv.download(paper, slugify=custom_slugify)