import re
import os
import glob
import json
import spacy
import torch
import string
import mammoth
import markdown
import argparse
import numpy as np
from tqdm import tqdm
from umap import UMAP
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import torch.nn.functional as F
# Set environment variable before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"  
from transformers import AutoTokenizer, AutoModel
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE

# torch device: cuda, mps, cpu
device = "mps"

# use tokenizer from spacy trained on sci corpus
parser = spacy.load("en_core_web_trf",disable=["ner"])
parser.max_length = 7000000
punctuations = string.punctuation
stopwords = list(STOP_WORDS)

def spacy_tokenizer(sentence):
    """
    Tokenize a sentence using spacy and remove stopwords and punctuation. 
    Only use for embedding, not text generation.

    Parameters
    ----------
    sentence : str
        Sentence to tokenize

    Returns
    -------
    mytokens : str
        Tokenized sentence
    """
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

def markdown_to_text(md_text):
    """
    Convert markdown text to plain text

    Parameters
    ----------
    md_text : str
        Markdown text

    Returns
    -------
    text : str
        Plain text
    
    """
    # remove images from text
    text = re.sub(r'\!\[.*\]\(.*\)', '', md_text)

    # remote links
    text = re.sub(r'\[.*\]\(.*\)', '', text)

    # convert formatting to html
    text = markdown.markdown(text)

    # convert to plain text
    #text = ''.join(BeautifulSoup(text, "html.parser").findAll(text=True))
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # remove new lines
    text = text.replace('\n', ' ')

    # remove extra whitespace
    text = re.sub(' +', ' ', text)

    # clean up other characters
    text = text.replace('\\', '')
    text = text.replace('__', '')

    return text

def docx_to_markdown(file_path, output_path=None):
    """
    Convert a docx file to markdown

    Parameters
    ----------
    file_path : str
        Path to docx file

    output_path : str
        Path to save markdown file

    Returns
    -------
    markdown : str
        Markdown string or path to markdown file
    """
    with open(file_path, "rb") as docx_file:
        result = mammoth.convert_to_markdown(docx_file)
        markdown = result.value

    if output_path:
        with open(output_path, "w") as markdown_file:
            markdown_file.write(markdown)
        return output_path
    else:
        return markdown

def docx_to_text(file_path, output_path=None):
    """
    Convert a docx file to plain text

    Parameters
    ----------
    file_path : str
        Path to docx file

    output_path : str
        Path to save text file

    Returns
    -------
    text : str
        Plain text string or path to text file
    """
    markdown = docx_to_markdown(file_path)
    text = markdown_to_text(markdown)

    if output_path:
        with open(output_path, "w") as text_file:
            text_file.write(text)
        return output_path
    else:
        return text



class Chonker():
    def __init__(self, chunking='sentence'):
        """
        A class for chunking text into sentences, newlines, or none.

        Parameters
        ----------
        chunking : str
            Chunking method, one of 'sentence', 'newline', or 'none'
        """
        if chunking == 'sentence':
            self.chunk_fn = lambda x: [str(s) for s in parser(x).sents]
        elif chunking == 'newline':
            self.chunk_fn = lambda x: x.split('\n')
        elif chunking == 'none':
            self.chunk_fn = lambda x: [x]
        elif isinstance(chunking, int):
            # chunk per number of words
            def word_chunk(x):
                chunks = []
                words = x.split()
                # use 50% overlap
                for i in range(0, len(words), chunking//2):
                    chunks.append(' '.join(words[i:i+chunking]))
                return chunks
            self.chunk_fn = word_chunk
        else:
            # default to sentence
            print('Invalid chunking method. Defaulting to none.')
            self.chunk_fn = lambda x: [x]

    def __call__(self, X):
        return self.chunk_fn(X)

class HuggingFaceEncoder():
    def __init__(self, name='sentence-transformers/all-MiniLM-L6-v2', device='mps'):
        """
        A class for encoding text using hugging face models.

        Parameters
        ----------
        name : str
            Name of hugging face model
            e.g. 
            sentence-transformers/all-mpnet-base-v2
            sentence-transformers/all-MiniLM-L6-v2
            sentence-transformers/multi-qa-MiniLM-L6-cos-v1
            facebook/dragon-plus-query-encoder
            facebook/dragon-plus-context-encoder
        
        device : str
            Device to use for encoding (cuda, mps, cpu)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device)
        self.device = device
        self.name = name

    def fit_transform(self, X, y=None):
        """
        Fit the model with X and apply the dimensionality reduction to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : None
            Ignored

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Transformed array
        """

        if "all-mpnet-base-v2" in self.name or "all-MiniLM-L6-v2" in self.name:
            # Tokenize sentences
            encoded_input = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt').to(self.device)

            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform pooling
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded =  encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

        elif 'multi-qa-MiniLM-L6-cos-v1' in self.name:
            # Tokenize sentences
            encoded_input = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt').to(self.device)

            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform pooling - take average of all tokens
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded =  encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

        elif 'facebook/dragon-plus-query-encoder' in self.name:
            # process query
            query_input = self.tokenizer(X, return_tensors='pt').to(self.device)
            embeddings = self.model(**query_input).last_hidden_state[:, 0, :]

        elif 'facebook/dragon-plus-context-encoder' in self.name:
            # process context
            ctx_input = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt').to(self.device)
            embeddings = self.model(**ctx_input).last_hidden_state[:, 0, :]

        else:
            encoded_input = self.tokenizer(X, return_tensors='pt', padding=True, truncation=True).to(self.device)
            embeddings = self.model(**encoded_input).pooler_output

        return embeddings.cpu().detach().numpy()

    # keep for compatibility with sklearn
    def transform(self, X, y=None):
        return self.fit_transform(X, y)

    def __call__(self, X, y=None):
        return self.fit_transform(X, y)


def parse_args():
    # args for text encoding
    parser = argparse.ArgumentParser(description='Text Encoding Arguments')
    parser.add_argument('--chunking', type=int, default=100,
                        help='Chunk document by number of words')
    parser.add_argument('--text_encoding', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Name of model from hugging face')
    parser.add_argument('--data_dir', type=str, default='markdown',
                        help='Directory of markdown files')
    parser.add_argument('--spacy_tokenizer', action='store_true',
                        help='Use spacy tokenizer') # lemmanizer
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    
    # get all markdown files
    markdown_files = glob.glob(os.path.join(args.data_dir, '*.md'))

    # separate file names into category labels
    # example: cat1-01.md cat1-02.md cat2-01.md cat2-02.md cat3-01.md cat4-01.md
    category = []
    for file in markdown_files:
        category.append(os.path.basename(file).split('-')[0])

    # convert to integer labels, y
    category = np.array(category)
    category_labels = np.unique(category)
    labels = np.zeros(len(category))
    for i in range(len(category_labels)):
        labels[category == category_labels[i]] = i

    # convert to text
    processed_text = []
    for i in range(len(markdown_files)):
        with open(markdown_files[i], 'r') as f:
            processed_text.append(markdown_to_text(f.read()))

    # chunk each document into ~100 word, 200 word, chunks
    chunker = Chonker(chunking=200)
    chunked_text = []
    chunked_labels = []
    for i in range(len(processed_text)):
        # break text into chunks
        chunks = chunker(processed_text[i])
        chunked_text.extend(chunks)
        # repeat labels for each chunk
        chunked_labels.extend([labels[i]] * len(chunks))

    # encode text
    encoder = HuggingFaceEncoder(name='sentence-transformers/all-MiniLM-L6-v2', device='mps')
    embeddings = encoder(chunked_text)

    # normalize embeddings across features with shape (n_samples, n_features)
    #n_embeddings = embeddings - np.mean(embeddings, axis=0)
    #n_embeddings = n_embeddings / np.std(n_embeddings, axis=0) # may lead to nans?

    # compute clustering metrics on chunked text
    scores = {
        'silhouette': silhouette_score(embeddings, chunked_labels), # higher is better
        'davies_bouldin': davies_bouldin_score(embeddings, chunked_labels), # lower is better
        'calinski_harabasz': calinski_harabasz_score(embeddings, chunked_labels) # higher is better
    }

    print(scores)

    # run data through a PCA, LDA, Tsne
    pca = PCA(n_components=2)
    lda = LDA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=10)
    umap = UMAP()

    pca_text = pca.fit_transform(embeddings)
    lda_text = lda.fit_transform(embeddings, chunked_labels)
    tsne_text = tsne.fit_transform(embeddings)
    umap_text = umap.fit_transform(embeddings, y=chunked_labels)

    # plot data
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    im = axs[0,0].scatter(pca_text[:, 0], pca_text[:, 1], c=chunked_labels, cmap='tab20')
    cbar = fig.colorbar(im, ax=axs[0,0])
    # change labels to category names
    cbar.set_ticks(np.arange(len(category_labels)))
    cbar.set_ticklabels(category_labels)
    axs[0,0].set_title('PCA')
    # plot lda
    im = axs[1,0].scatter(lda_text[:, 0], lda_text[:, 1], c=chunked_labels, cmap='tab20')
    cbar = fig.colorbar(im, ax=axs[1,0])
    # change labels to category names
    cbar.set_ticks(np.arange(len(category_labels)))
    cbar.set_ticklabels(category_labels)
    axs[1,0].set_title('LDA')
    # plot tsne
    im = axs[0,1].scatter(tsne_text[:, 0], tsne_text[:, 1], c=chunked_labels, cmap='tab20')
    cbar = fig.colorbar(im, ax=axs[0,1])
    # change labels to category names
    cbar.set_ticks(np.arange(len(category_labels)))
    cbar.set_ticklabels(category_labels)
    axs[0,1].set_title('t-SNE')
    # plot umap
    im = axs[1,1].scatter(umap_text[:, 0], umap_text[:, 1], c=chunked_labels, cmap='tab20')
    cbar = fig.colorbar(im, ax=axs[1,1])
    # change labels to category names
    cbar.set_ticks(np.arange(len(category_labels)))
    cbar.set_ticklabels(category_labels)
    axs[1,1].set_title('UMAP')
    plt.tight_layout()
    plt.show()