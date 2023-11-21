import re
import os
import glob
import json
import types
import spacy
import torch
import string
import hashlib
import mammoth
import markdown
import argparse
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

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
    

# Steps for information retrieval
# 1. Text preprocessing: tokenization, stopword removal, stemming, etc.
# 2. Embedding: tfidf, bag of words, llm
# 3. Vector preprocessing: standardization vs none 
# 4. Dimensionality reduction: PCA, LDA (2 components, 0%, 25%, 50%)

# Evaluation
# Silhouette score (closer to 1 is better)
# Davies-Bouldin score (lower is better)
# Calinski-Harabasz score (higher is better)
# Avg. cosine similarity of categories (higher is better)



# Mean Pooling - Take attention mask into account for correct averaging
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class HuggingFaceEncoder():
    def __init__(self, name='sentence-transformers/all-mpnet-base-v2'):
        """
        A class for encoding text using hugging face models.

        Parameters
        ----------
        name : str
            Name of hugging face model
            e.g. HuggingFaceH4/zephyr-7b-beta
        """
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device)
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

        if "sentence-transformers" in self.name:

            # Tokenize sentences
            encoded_input = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt').to(device)

            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform pooling
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

        else:
            encoded_input = self.tokenizer(X, return_tensors='pt', padding=True, truncation=True).to(device)
            embeddings = self.model(**encoded_input).pooler_output

        return embeddings.cpu().detach().numpy()
        
class NoProcesor():
    def __init__(self):
        """
        A class for no processing while maintaining the same interface as sklearn preprocessing classes.
        """
        pass

    def fit_transform(self, X, y=None):
        return X

    def transform(self, X):
        return X

    def fit(self, X, y=None):
        return self

class TextEncoder():
    def __init__(self, chunking='sentence', text_preprocessing='spacy', 
                 text_encoding='tfidf', encode_size='4K', vector_preprocessing='standard',
                 dim_reduction='PCA', dim_reduction_components=2, **kwargs):
        """
        A class for embedding documents into vectors. The pipeline includes
        text preprocessing, embedding, vector preprocessing, and dimensionality reduction.
        Has a similar interface to sklearn classes.

        Parameters
        ----------
        chunking : str
            Chunking method, one of 'sentence', 'newline', 'none', or int (number of words per chunk)
        text_preprocessing : str
            Text preprocessing method, one of 'spacy' or 'none'/None
        text_encoding : str
            Text encoding method, one of 'tfidf', 'bow', or 'llm'
        encode_size : str
            Vocabulary size for text encoding using tfidf or bag of words (4K, 8K or 16K)
        vector_preprocessing : str
            Vector preprocessing method, one of 'standard', 'normalize' or 'none'/None
        dim_reduction : str
            Dimensionality reduction method, one of 'pca' or 'lda'
        dim_reduction_components : int
            Number of components for dimensionality reduction (2, 3, 4, 5, 10, 25, 50, 100)
        kwargs : dict
            Additional arguments to pass to sklearn classes (data_dir)
        """

        # Chunking
        if chunking == 'sentence':
            self.chunk_preprocessing = lambda x: [str(s) for s in parser(x).sents]
        elif chunking == 'newline':
            self.chunk_preprocessing = lambda x: x.split('\n')
        elif chunking == 'none':
            self.chunk_preprocessing = lambda x: [x]
        elif isinstance(chunking, int):
            # chunk with 50% overlap
            self.chunk_preprocessing = lambda x: [x[i:i+chunking] for i in range(0, len(x), chunking//2)]
        else:
            # default to sentence
            print('Invalid chunking method. Defaulting to none.')
            self.chunk_preprocessing = lambda x: [x]

        # Vector preprocessing
        if text_preprocessing is None:
            self.text_preprocessing = lambda x: x
        elif text_preprocessing == 'spacy':
            self.text_preprocessing = spacy_tokenizer
        elif text_preprocessing == 'none':
            self.text_preprocessing = lambda x: x
        # custom callable
        elif isinstance(text_preprocessing, (types.FunctionType, types.LambdaType)):
            self.text_preprocessing = text_preprocessing
        else:
            # default to spacy
            print('Invalid text preprocessing method. Defaulting to none.')
            self.text_preprocessing = lambda x: x

        # interpret max_features for text encoding
        if encode_size.lower() == '4k':
            max_features = 2**12
        elif encode_size.lower() == '8k':
            max_features = 2**13
        elif encode_size.lower() == '16k':
            max_features = 2**14
        else:
            # default to 4K
            max_features = 2**12

        # Text encoding
        if text_encoding == 'tfidf':
            self.text_encoding = TfidfVectorizer(max_features=max_features)
        elif text_encoding == 'bow':
            self.text_encoding = CountVectorizer(max_features=max_features)
        elif text_encoding == 'llm':
            self.text_encoding = HuggingFaceEncoder()
            encode_size = self.text_encoding.model.config.hidden_size
        elif '/' in text_encoding: # custom hugging face model
            self.text_encoding = HuggingFaceEncoder(name=text_encoding)
            encode_size = self.text_encoding.model.config.hidden_size
        else:
            # default to tfidf
            print('Invalid text encoding method. Defaulting to tfidf.')
            self.text_encoding = TfidfVectorizer(max_features=max_features)

        # Vector preprocessing only for tfidf and bow
        if text_encoding.__class__.__name__ == 'TfidfVectorizer' or \
            text_encoding.__class__.__name__ == 'CountVectorizer':
            if vector_preprocessing is None:
                self.vector_preprocessing = lambda x: x
            elif vector_preprocessing == 'standard': # standard scaler
                self.vector_preprocessing = lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            elif vector_preprocessing == 'normalize': # normalize
                self.vector_preprocessing = lambda x: x / np.linalg.norm(x, axis=0)
            elif vector_preprocessing == 'none':
                self.vector_preprocessing = lambda x: x
            else:
                # default to none
                print('Invalid vector preprocessing method. Defaulting to none.')
                self.vector_preprocessing = lambda x: x
        else:
            # no vector preprocessing for llm
            self.vector_preprocessing = lambda x: x
            vector_preprocessing = 'none'


        # Dimensionality reduction
        if dim_reduction_components < 1: # fraction of max_features
            dim_reduction_components = int(dim_reduction_components * max_features)
        # select algorithm
        if dim_reduction is None:
            self.dim_reduction = NoProcesor()
        elif dim_reduction.lower() == 'pca':
            self.dim_reduction = PCA(n_components=dim_reduction_components)
            print('PCA components:', dim_reduction_components)
        elif dim_reduction.lower() == 'lda':
            self.dim_reduction = LDA(n_components=dim_reduction_components)
            print('LDA components:', dim_reduction_components)
        else:
            # default to none
            print('Invalid dimensionality reduction method. Defaulting to none.')
            self.dim_reduction = NoProcesor()

        self.settings = {
            'chunking': chunking,
            'text_preprocessing': text_preprocessing,
            'text_encoding': text_encoding,
            'encode_size': encode_size,
            'vector_preprocessing': vector_preprocessing,
            'dim_reduction': dim_reduction,
            'dim_reduction_components': dim_reduction_components
        }

        # name is md5 hash of settings
        self.name = hashlib.md5(json.dumps(self.settings, sort_keys=True).encode('utf-8')).hexdigest()


    def fit_transform(self, docs, y=None):
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

        processed_docs = []
        processed_y = []

        print('Chunking text...')
        for i in range(len(docs)):
            chunked = self.chunk_preprocessing(docs[i])
            processed_docs.extend(chunked)
            # repeat labels for each chunk
            processed_y.extend([y[i]] * len(chunked))

        print('Preprocessing text...')
        for i in tqdm(range(len(processed_docs))):
            processed_docs[i] = self.text_preprocessing(processed_docs[i])

        print('Encoding text...')
        X = self.text_encoding.fit_transform(processed_docs)
        print('Encoding shape:', X.shape)

        print('Preprocessing vectors...')
        if self.text_encoding.__class__.__name__ == 'TfidfVectorizer':
            X = self.vector_preprocessing(X.toarray())
        else:
            X = self.vector_preprocessing(X) # llm usually ~(N, 768)
        print('Vector shape:', X.shape)

        print('Reducing dimensionality...')
        if y is not None and self.dim_reduction.__class__.__name__ == 'LDA':
            X = self.dim_reduction.fit_transform(X, y)
        else:
            X = self.dim_reduction.fit_transform(X)

        print('Reduced shape:', X.shape)
        return X, np.array(processed_y)

    def transform(self, X, y=None):
        """
        Convert documents to vectors using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Transformed array
        """
        processed_text = []
        processed_y = []

        # chunk text
        for i in range(len(X)):
            chunked = self.chunk_preprocessing(X[i])
            processed_text.extend(chunked)
            # repeat labels for each chunk
            processed_y.extend([y[i]] * len(chunked))

        # preprocess text
        for i in tqdm(range(len(processed_text))):
            processed_text[i] = self.text_preprocessing(processed_text[i])

        # encode text
        X = self.text_encoding.transform(processed_text)

        # preprocess vectors
        X = self.vector_preprocessing(X.toarray())

        # reduce dimensionality
        if y is not None and self.dim_reduction.__class__.__name__ == 'LDA':
            X = self.dim_reduction.transform(X, y)
        else:
            X = self.dim_reduction.transform(X)
        
        return X, processed_y

    def score(self, X, y, save=False):
        """
        Score the embeddings using various clustering metrics

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,)

        save : bool
            Save scores to json file

        Returns
        -------
        score : dict
            Dictionary of scores
            Silhouette score (closer to 1 is better)
            Davies-Bouldin score (lower is better)
            Calinski-Harabasz score (higher is better)
        """
        scores = {
            'silhouette': silhouette_score(X, y),
            'davies_bouldin': davies_bouldin_score(X, y),
            'calinski_harabasz': calinski_harabasz_score(X, y)
        }

        # cosine similarity of matrix
        #cosine_similarity = np.dot(X, X.T) / np.linalg.norm(X, axis=1) / np.linalg.norm(X, axis=1)[:, np.newaxis]
        
        # average cosine similarity of each category
        #for i in range(len(np.unique(y))):
        #    scores[f'cosine_similarity_{i}'] = np.mean(cosine_similarity[y == i, :][:, y == i])

        # convert each to float for json serialization
        for key in scores.keys():
            scores[key] = float(scores[key])

        if save:
            fname = f"{self.name}.json"
            # add scores to settings
            self.settings['scores'] = scores
            with open(fname, 'w') as f:
                json.dump(self.settings, f)
            print(f'Saved scores to {fname}')

        return scores


def parse_args():
    # args for text encoding
    parser = argparse.ArgumentParser(description='Text Encoding Arguments')
    parser.add_argument('--chunking', type=str, default='sentence',
                        help='Chunking method, one of "sentence", "newline", "none", or int (number of words per chunk)')
    parser.add_argument('--text_preprocessing', type=str, default='spacy',
                        help='Text preprocessing method, one of "spacy" or "none"/None')
    parser.add_argument('--text_encoding', type=str, default='llm',
                        help='Text encoding method, one of "tfidf", "bow", "llm" or hugging face model name')
    parser.add_argument('--encode_size', type=str, default='4K',
                        help='Vocabulary size for text encoding using tfidf or bag of words (4K, 8K or 16K)')
    parser.add_argument('--vector_preprocessing', type=str, default='standard',
                        help='Vector preprocessing method, one of "standard", "normalize" or "none"/None')
    parser.add_argument('--dim_reduction', type=str, default='PCA',
                        help='Dimensionality reduction method, one of "pca" or "lda"')
    parser.add_argument('--dim_reduction_components', type=float, default=32,
                        help='Number of components for dimensionality reduction (if <1 will be fraction of max_features)')
    parser.add_argument('--data_dir', type=str, default='markdown',
                        help='Directory of markdown files')
    args = parser.parse_args()
    return args

# create test in main block of code to read markdown files from directory and convert to text then embed
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

    # set up encoder
    #doc2vec = TextEncoder(text_preprocessing='spacy', text_encoding='tfidf', encode_size='4K', 
    #                      vector_preprocessing='standard', dim_reduction='PCA', dim_reduction_components=0.5)
    doc2vec = TextEncoder(**vars(args))

    # create text embeddings
    X,y = doc2vec.fit_transform(processed_text, labels)
    scores = doc2vec.score(X, y, save=True)
    print('Document scores:')
    print(scores)

    dude()

    # create grid of parameters
    pars ={
        'chunking': ['sentence', 'newline', 'none', 200, 500],
        'text_preprocessing': ['spacy', 'none'],
        'text_encoding': ['tfidf', 'bow', 'llm', 
                          'sentence-transformers/all-mpnet-base-v2',
                          'sentence-transformers/all-MiniLM-L12-v2',
                          'sentence-transformers/msmarco-MiniLM-L12-cos-v5'],
        'encode_size': ['4K', '8K', '16K'],
        'vector_preprocessing': ['standard', 'normalize', 'none'],
        'dim_reduction': ['PCA', 'LDA', 'none'],
        'dim_reduction_components': [2, 64, 128, 512],
    }

    # compute some sort of document-document similarity matrix

    # create all combinations of parameters
    import itertools
    keys, values = zip(*pars.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print('Total combinations:', len(combinations))

    # create a scores for each combination
    for i in range(len(combinations)):
        doc2vec = TextEncoder(**combinations[i])
        X = doc2vec.fit_transform(processed_text, y)
        scores = doc2vec.score(X, y, save=True)
        print('Document scores:')
        print(scores)

    # create some random data to score
    # Xr = np.random.random(X.shape)
    # yr = np.random.randint(0, len(category_labels), X.shape[0])
    # print('Random data scores:')
    # print(doc2vec.score(Xr, yr))

    """
    From https://huggingface.co/sentence-transformers/msmarco-MiniLM-L12-cos-v5

    # Sentences we want sentence embeddings for
    query = "How many people live in London?"
    docs = ["Around 9 Million people live in London", "London is known for its financial district"]

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-MiniLM-L12-cos-v5")
    model = AutoModel.from_pretrained("sentence-transformers/msmarco-MiniLM-L12-cos-v5")

    #Encode query and docs
    query_emb = encode(query)
    doc_emb = encode(docs)

    #Compute dot score between query and all document embeddings
    scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()

    #Combine docs & scores
    doc_score_pairs = list(zip(docs, scores))

    #Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    #Output passages & scores
    for doc, score in doc_score_pairs:
        print(score, doc)
    """