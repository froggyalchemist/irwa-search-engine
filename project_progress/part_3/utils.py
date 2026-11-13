from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk

def build_terms(text):
    """
    Preprocess text removing stop words, stemming,
    transforming in lowercase and return the tokens of the text.

    :param text: to be preprocessed
    :returns: a list of tokens corresponding to the input text after the preprocessing
    """

    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')             # keep only words & numbers, drop punctuation
    stop_words = set(stopwords.words("english"))
    for w in ['not', 'no']:
        stop_words.remove(w)                        # keep 'no' and 'not' to preserve meaning of sentences like "shirt with no sleeves"
    text= text.lower()                              # transform in lowercase
    text=  tokenizer.tokenize(text)                 # tokenize the text to get a list of terms
    text=[t for t in text if t not in stop_words]   # eliminate the stopwords
    text=[stemmer.stem(t) for t in text]            # perform stemming
    return text
