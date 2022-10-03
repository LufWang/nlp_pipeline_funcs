import re
import nltk
from nltk.stem import PorterStemmer


#####
# text cleaning
#####
def clean_tokenize_text(text):
    """
    function that cleans and returns tokenized text
    """
    
    ps = PorterStemmer()
    
    text = re.sub(r'[,!?;-]+', '.', text) # replace all termination punctuation with '.'
    tokens = [ ps.stem(ch.lower()) for ch in nltk.word_tokenize(text) # stem
             if ch.isalpha() # only keep alphabetical words
             or ch == '.'   # only keep punctuation that is '.'
           ]
    
    return tokens

def clean_text(text, do_filter_numbers=True):
    """
    clean a piece of text (from akqa)
    """
    
    # Coerce to lowercase and remove leading/lagging whitespace
    text = str(text).lower().strip()
    # reduce 'additional info' to alpha chars and clean up unhelpful text
    if re.sub('[\W_]', '', text) in ('n', 'no', 'none', 'yes', 'notatthistime', 'noneknown', 'notthatiknowof',
                                     'notsure', 'name', 'unknown', 'seeabove', 'unsure', 'nope', 'notthatiamawareof',
                                     'noasfarasiknow', 'noneiamawareof', 'na', 'idontthinkso', 'noneatthistime',
                                     'possibly', 'notthatimawareof', 'nonethatiamawareof'):
        text = ""
    # remove special characters (not yet refactored)
    text = text.lower().replace("'", '').replace('"', '').replace('/', ' ').replace('*', '').replace('&', '')
    text = text.replace('(', ' ').replace(')', '').replace(',', '.').replace('°', '*').replace(';', '.')
    text = text.replace(':', '.').replace('<', ' less than ').replace('>', ' greater than ').replace('#', '')
    text = text.replace('@', ' at ')
    # Remove any words with numbers
    if do_filter_numbers is True:
        text = re.sub('\S*\d+\S*', '', text)
    # Remove all non-alpha symbols
    text = re.sub(r'([^\s\w]|_)+', '', text)
    # Remove multi-spaces and leading/lagging spaces
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text