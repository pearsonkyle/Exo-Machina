import re
import os
import glob
import json
import types
import spacy
import string
import mammoth
import markdown
import argparse
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

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

class LlamaEncoder():
    def __init__(self):
        """
        A class for encoding text using the llama2 encoder. TODO
        """
        pass

    def fit_transform(self, X, y=None):
        return X

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
    def __init__(self, text_preprocessing='spacy', text_encoding='tfidf', encode_size='4K', 
                 vector_preprocessing='standard', dim_reduction='PCA', dim_reduction_components=2):
        """
        A class for embedding documents into vectors. The pipeline includes
        text preprocessing, embedding, vector preprocessing, and dimensionality reduction.
        Has a similar interface to sklearn classes.

        Parameters
        ----------
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
        """

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
            # TODO load llama weights
            self.text_encoding = LlamaEncoder()
        else:
            # default to tfidf
            print('Invalid text encoding method. Defaulting to tfidf.')
            self.text_encoding = TfidfVectorizer(max_features=max_features)

        # Vector preprocessing
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

        # Dimensionality reduction
        if dim_reduction_components < 1: # fraction of max_features
            dim_reduction_components = int(dim_reduction_components * max_features)
        # select algorithm
        if dim_reduction is None:
            self.dim_reduction = NoProcesor()
        elif dim_reduction == 'pca':
            self.dim_reduction = PCA(n_components=dim_reduction_components)
        elif dim_reduction == 'lda':
            self.dim_reduction = LDA(n_components=dim_reduction_components)
        else:
            # default to none
            print('Invalid dimensionality reduction method. Defaulting to none.')
            self.dim_reduction = NoProcesor()
    
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

        print('Preprocessing text...')
        processed_docs = []
        for i in tqdm(range(len(docs))):
            processed_docs.append(self.text_preprocessing(docs[i]))

        print('Encoding text...')
        X = self.text_encoding.fit_transform(processed_docs)
        print('Encoding shape:', X.shape)

        print('Preprocessing vectors...')
        X = self.vector_preprocessing(X.toarray())
        print('Vector shape:', X.shape)

        print('Reducing dimensionality...')
        if y is not None:
            X = self.dim_reduction.fit_transform(X, y)
        else:
            X = self.dim_reduction.fit_transform(X)

        print('Reduced shape:', X.shape)
        return X

    def transform(self, X):
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
        if isinstance(X, str):
            X = [X]
        preprocessed_text = self.text_preprocessing(X)
        X = self.text_encoding.transform([preprocessed_text])
        X = self.vector_preprocessing(X.toarray())
        X = self.dim_reduction.transform(X)
        return X

    def __call__(self, X):
        return self.transform(X)
    
    def score(self, X, y):
        """
        Score the embeddings using various clustering metrics

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,)

        Returns
        -------
        score : dict
            Dictionary of scores
            Silhouette score (closer to 1 is better)
            Davies-Bouldin score (lower is better)
            Calinski-Harabasz score (higher is better)
        """
        # TODO add more metrics
        return {
            'silhouette': silhouette_score(X, y),
            'davies_bouldin': davies_bouldin_score(X, y),
            'calinski_harabasz': calinski_harabasz_score(X, y)
        }


def parse_args():
    # args for text encoding
    parser = argparse.ArgumentParser(description='Text Encoding Arguments')
    parser.add_argument('--text_preprocessing', type=str, default='spacy',
                        help='Text preprocessing method, one of "spacy" or "none"/None')
    parser.add_argument('--text_encoding', type=str, default='tfidf',
                        help='Text encoding method, one of "tfidf", "bow", or "llm"')
    parser.add_argument('--encode_size', type=str, default='4K',
                        help='Vocabulary size for text encoding using tfidf or bag of words (4K, 8K or 16K)')
    parser.add_argument('--vector_preprocessing', type=str, default='standard',
                        help='Vector preprocessing method, one of "standard", "normalize" or "none"/None')
    parser.add_argument('--dim_reduction', type=str, default='PCA',
                        help='Dimensionality reduction method, one of "pca" or "lda"')
    parser.add_argument('--dim_reduction_components', type=float, default=0.5,
                        help='Number of components for dimensionality reduction (if <1 will be fraction of max_features)')
    args = parser.parse_args()
    return args

# create test in main block of code to read markdown files from directory and convert to text then embed
if __name__ == '__main__':

    args = parse_args()
    
    # get all markdown files
    markdown_files = glob.glob('markdown/*.md')

    # separate file names into category labels
    # example: cat1-01.md cat1-02.md cat2-01.md cat2-02.md cat3-01.md cat4-01.md
    category = []
    for file in markdown_files:
        category.append(os.path.basename(file).split('-')[0])

    # convert to integer labels, y
    category = np.array(category)
    category_labels = np.unique(category)
    y = np.zeros(len(category))
    for i in range(len(category_labels)):
        y[category == category_labels[i]] = i

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
    X = doc2vec.fit_transform(processed_text)
    scores = doc2vec.score(X, y)

    print('Document scores:')
    print(scores)

    # save scores to json
    fname = f"doc2vec_{args.text_preprocessing}_{args.text_encoding}_{args.encode_size}_{args.vector_preprocessing}_{args.dim_reduction}_{args.dim_reduction_components}.json"
    with open(fname, 'w') as f:
        json.dump(scores, f)

    # create some random data to score
    # Xr = np.random.random(X.shape)
    # yr = np.random.randint(0, len(category_labels), X.shape[0])
    # print('Random data scores:')
    # print(doc2vec.score(Xr, yr))