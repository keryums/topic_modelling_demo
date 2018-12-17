### K. Ong | Fall 2018
### Helper functions for topic modelling 


import string
from helper_prep import retokenize

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import SparsePCA, NMF

from gensim.models import LdaModel, CoherenceModel
from gensim.corpora.dictionary import Dictionary
from corextopic import corextopic as ct


def get_vecs(sentences):
    """
    create both count and tfidf vectorizers 
    output docs + features 
    """
    cv_vectorizer = CountVectorizer(tokenizer = retokenize) # from helper.py
    cv_docs = cv_vectorizer.fit_transform(sentences)
    cv_features = cv_vectorizer.get_feature_names()
    tf_vectorizer = TfidfVectorizer(tokenizer = retokenize) # from helper.py
    tf_docs = tf_vectorizer.fit_transform(sentences)
    tf_features = tf_vectorizer.get_feature_names()
    print('Num. Features (Count Vec): ', len(cv_features))
    print('Num. Features (TFIDF Vec): ', len(tf_features))
    print('')
    print('Dim. Docs (Count Vec): ', cv_docs.shape)
    print('Dim. Docs (TFIDF Vec): ', tf_docs.shape)
    return cv_features, cv_docs, tf_features, tf_docs


def get_nmf_topics(num_topics_list, docs, features):
    for i in num_topics_list:
        nmf = NMF(n_components=i, random_state=10, alpha=.1, l1_ratio=.5, init='nndsvd').fit(docs)
        print('Num. Topics: ', i)
        num_top_words = 20
        for topic_idx, topic in enumerate(nmf.components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([features[i]
                            for i in topic.argsort()[:-num_top_words - 1:-1]]))
        print('')

def get_gensim_topics(num_topics_list, sentences, print_flag = False):
    """
    Gensim by default employs a version of count vectorization
    input: sentences (list of list of words)
    outputs coherence, perplexity, and topics 
    prints topics if print == True 
    """
    texts = sentences.apply(retokenize).tolist() 
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    perplexity_ls = []
    coherence_ls = []
    for i in num_topics_list:
        lda = LdaModel(corpus, num_topics=i, id2word = dictionary, random_state = 10)
        perplexity = lda.log_perplexity(corpus)
        perplexity_ls.append(perplexity) 
        coherence_model_lda = CoherenceModel(model = lda, texts = texts, dictionary = dictionary, coherence = 'c_v')
        coherence = coherence_model_lda.get_coherence()
        coherence_ls.append(coherence)

        if print_flag == True:
            print('Num. Topics: ', i)
            print('')
            for i in (lda.print_topics()):
                words = i[1]
                words_ls = words.split('+')
                words_ls = ([i.split('*')[1] for i in words_ls])
                words_ls = [i.replace('"', '') for i in words_ls]
                print(', '.join(words_ls))
        print('')
    return perplexity_ls, coherence_ls

def get_corex_topics(num_topics_list, docs, features, print_flag = False):
    """
    outputs correlation list for model selection
    """
    
    total_corr = []
    for i in num_topics_list:
        topic_model = ct.Corex(n_hidden=i, seed = 10)
        topic_model.fit(docs, words=features) 
        total_corr.append(topic_model.tc)
        
        if print_flag == True:
            topics = topic_model.get_topics()
            print('Num topics: ', i)
            for topic_n, topic in enumerate(topics):
                words,mis = zip(*topic)
                topic_str = str(topic_n+1)+': '+', '.join(words)
                print(topic_str)
            print('')

    return total_corr
    

def pick_top_n(metric_list, num_topic_list, n):
    """
    picks best num topics w.r.t. measure of interest 
    """
    top_n = []
    for i in sorted(metric_list, reverse = True)[:n]:
        ind = metric_list.index(i)
        top_n.append(num_topic_list[ind])
    print('Best Num. Topics: ', top_n)
    return top_n
