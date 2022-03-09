
# Text librairies
import re
import contractions
import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from autocorrect import Speller

spell = Speller()
token = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
charac = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
stop_words = set(stopwords.words("english"))


def expand_contractions(text):
    """expand shortened words, e.g. 'don't' to 'do not'"""
    text = contractions.fix(text)
    return text


def clean_text(text):
    text = re.sub(r"\'", "'", text) # match all literal apostrophe pattern then replace them by a single whitespace
    text = re.sub(r"\n", " ", text) # match all literal Line Feed (New line) pattern then replace them by a single whitespace
    text = re.sub(r"\xa0", " ", text) # match all literal non-breakable space pattern then replace them by a single whitespace
    text = re.sub('\s+', ' ', text) # match all one or more whitespace then replace them by a single whitespace
    text = text.strip(' ')
    return text


def remove_single_letter(text):
    """remove single alphabetical character"""
    text = re.sub(r"\b\w{1}\b", "", text) # remove all single letter
    text = re.sub("\s+", " ", text) # remove whitespaces left after the last operation
    text = text.strip(" ")
    return text


def remove_non_alphabetical_character(text):
    """remove all non-alphabetical character"""
    text = re.sub("[^a-z]+", " ", text) # remove all non-alphabetical character
    text = re.sub("\s+", " ", text) # remove whitespaces left after the last operation
    return text


def remove_stopwords(text):
    """remove common words in english by using nltk.corpus's list"""
    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]

    return ' '.join(map(str, filtered)) # Return the text untokenize



def lemmatize_text(text):
    """Lemmatize the text by using tag """

    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))  # tokenize the text then return a list of tuple (token, nltk_tag)
    lemmatized_text = []
    for word, tag in tokens_tagged:
        if tag.startswith('J'):
            lemmatized_text.append(lemmatizer.lemmatize(word,'a')) # Lemmatisze adjectives. Not doing anything since we remove all adjective
        elif tag.startswith('V'):
            lemmatized_text.append(lemmatizer.lemmatize(word,'v')) # Lemmatisze verbs
        elif tag.startswith('N'):
            lemmatized_text.append(lemmatizer.lemmatize(word,'n')) # Lemmatisze nouns
        elif tag.startswith('R'):
            lemmatized_text.append(lemmatizer.lemmatize(word,'r')) # Lemmatisze adverbs
        else:
            lemmatized_text.append(lemmatizer.lemmatize(word)) # If no tags has been found, perform a non specific lemmatization
    return " ".join(lemmatized_text) # Return the text untokenize