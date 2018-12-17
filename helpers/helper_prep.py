### K. Ong | Fall 2018
### Helper functions for text preprocessing

import pandas as pd 
import numpy as np
import string
import re 
import nltk
from nltk.stem import WordNetLemmatizer
lemmer = WordNetLemmatizer()
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
en_stopwords = set(stopwords.words('english')) #179 words 

def sub_word(text):
    """
    Corrections to most common truncations
    Note that some of these replacements might be corpus specific
    """
    text = re.sub('-', '', text) # non-sense --> nonsense
    text = re.sub(' / ', '/', text) 
    text = re.sub('e\.g\.', 'eg', text) 
    text = re.sub("n't", ' not', text) #doesn't --> does not 
    text = re.sub("'ve", ' have', text) #i've --> i have 
    text = re.sub("i'm", 'i am', text)
    text = re.sub("im ", 'i am ', text)
    text = re.sub('w/', 'with', text)
    text = re.sub(' w ', ' with ', text)
    return text

def remove_special_char(text):
    """
    remove all special char in text 
    """
    regex = re.compile('[' + re.escape(string.punctuation) + '\\r\\t\\n]')
    text = regex.sub(' ', text)
    return text

def custom_lemmatize(sentence):
    """
    custom_lemmatize() applies lemmatization based on pos tag, see:
    https://stackoverflow.com/questions/32957895/wordnetlemmatizer-not-returning-the-right-lemma-unless-pos-is-explicit-python
    word_tokenize() splits on space
    """     
    lemmed_list = [] 
    for word, tag in pos_tag(word_tokenize(sentence)):
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if not wntag:
            lemma = word
        else:
            lemma = lemmer.lemmatize(word, wntag)
        lemmed_list.append(lemma)
    return lemmed_list

def prep_dataset(df, input_col_name, output_col_name, list_of_lists=False):
    """
    keep rows with non-numeric free text
    performs preprocessing: lowering, substitutions, special char removal, and lemmatization
    drops rows where len(strip(text))<=1 which are when text contains only punctuation
    returns: text df, unnested list of tokens, unnested list of lemmas 
    """
    text = df[pd.isnull(df[input_col_name]) == False]
    text = text[~(text[input_col_name].apply(lambda x: x.isnumeric()))]
    
    text[output_col_name] = text[input_col_name]
    text[output_col_name] = text[output_col_name].apply(lambda x: x.lower())
    text[output_col_name] = text[output_col_name].apply(sub_word)
    text[output_col_name] = text[output_col_name].apply(remove_special_char)
    text = text[text[output_col_name].apply(lambda x: len(x.strip())>1)]
    
    tokens_nested = text[output_col_name].drop_duplicates().apply(lambda x: x.split()).values.tolist()
    lemmas_nested = text[output_col_name].drop_duplicates().apply(custom_lemmatize).values.tolist()
    tokens_unnested = [i for i in tokens_nested for i in i]
    lemmas_unnested = [i for i in lemmas_nested for i in i]
 
    print('Dimension of Text: ', text.shape)
    print('Number of Tokens: ', len(tokens_unnested))
    print('Number of Lemmas: ', len(lemmas_unnested))
    print('')

    if list_of_lists==True:
        return text, tokens_nested, lemmas_nested
    if list_of_lists==False:
        return text, tokens_unnested, lemmas_unnested

### Following two functions taken from Nicha Ruchirawat's collocations notebooks:
### https://medium.com/@nicharuch/collocations-identifying-phrases-that-act-like-individual-words-in-nlp-f58a93a2f84a
def rightTypes(ngram):
    """
    filter right POS-pattern for bigrams
    """
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False

def rightTypesTri(ngram):
    """
    filter right POS-pattern for trigrams
    """
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[2][1] in third_type:
        return True
    else:
        return False

def find_ngrams(word_list, n = None, pct = None, pmi_flag = False):
    """
    find bigrams and trigrams from given word list
    using both parts of speech rules and PMI threshold+frequency filter 
    output frequency tables for each 
    considers PMI approach if pmi_flag = True
    """
    bigrams = nltk.collocations.BigramAssocMeasures()
    bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(word_list)
    bigram_freq = bigramFinder.ngram_fd.items()
    bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)
    
    trigrams = nltk.collocations.TrigramAssocMeasures()
    trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(word_list)
    trigram_freq = trigramFinder.ngram_fd.items()
    trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)
    
    bigrams_pos = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]
    trigrams_pos = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]

    if pmi_flag == True:
 
        bigramFinder.apply_freq_filter(n)
        trigramFinder.apply_freq_filter(n)

        bigrams_pmi = pd.DataFrame(list(bigramFinder.score_ngrams(bigrams.pmi)),\
                                   columns=['bigram','PMI']).sort_values(by='PMI', ascending=False)
        trigrams_pmi = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.pmi)),\
                                    columns=['trigram','PMI']).sort_values(by='PMI', ascending=False)

        bigram_pct = np.percentile(bigrams_pmi.PMI.values, pct)
        trigram_pct = np.percentile(bigrams_pmi.PMI.values, pct) 

        bigrams_pmi_filtered = bigrams_pmi[bigrams_pmi.PMI>bigram_pct]
        trigrams_pmi_filtered = trigrams_pmi[trigrams_pmi.PMI>trigram_pct]

        return bigrams_pos, trigrams_pos, bigrams_pmi_filtered, trigrams_pmi_filtered
    else:
        return bigrams_pos, trigrams_pos

def replace_gram(text, gram_list):
    """
    replaces instances of collocations in a text with underscored versions 
    """
    text = text.lower()
    for i in gram_list:
        raw = str(' '.join(i))
        clean = str('_'.join(i))
        text = text.replace(raw, clean)
    return text

def regram(df, input_col_name, output_col_name, bigram_list, trigram_list):
    """
    replace identified grams in list (first bi then tri to increase term commonality)
    """
    df[output_col_name] = df[input_col_name].apply(lambda x: replace_gram(x, bigram_list))
    df[output_col_name] = df[output_col_name].apply(lambda x: replace_gram(x, trigram_list))   
    return df 

def retokenize(text):  
    """
    retokenize (lemmatize) text once ngrams are replaced and special char removed
    stop words are now removed
    """
    tokens = custom_lemmatize(text) # from helper.py 
    tokens = [i for i in tokens if i not in en_stopwords]
    tokens = [i for i in tokens if i.isnumeric() == False]
    return tokens 
