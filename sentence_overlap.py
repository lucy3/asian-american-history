'''
Calculate the number of sentences with overlapping material
'''

from tqdm import tqdm
import spacy
from nltk import ngrams
from collections import defaultdict, Counter
import pandas as pd
import json

ROOT = '/data0/lucy/asian-american-textbooks/'
LOGS = ROOT + 'logs/'

def get_n_gramlist(toks, n=10): 
    nngramlist = []
    # stack overflow said this was fastest
    for s in ngrams(toks,n=n):        
        nngramlist.append(' '.join(s))            
    return nngramlist

def find_overlaps(): 
    nlp = spacy.load('en_core_web_trf', exclude=["transformer", "ner", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    
    dtypes = {'state':str, 'book_id':str, 'book_filename':str, 'subject':str, 
          'sentence_id':str, 'sentence':str, 'aapi':str}
    for i in range(50): 
        dtypes['topic_' + str(i)] = float
    sent_topics_50 = pd.read_csv(LOGS + 'combined_coref_data_topics_50.csv', dtype=dtypes)
    aapi_sent_topics_50 = sent_topics_50[sent_topics_50.aapi.notnull()]
    
    CA_df = aapi_sent_topics_50[aapi_sent_topics_50['state'] == 'CA']
    CA_sents = pd.Series(CA_df.sentence.values,index=CA_df.sentence_id).to_dict()
    
    sent_ngrams = defaultdict(list) # ngram to sent_ID
    for sent_ID in tqdm(CA_sents): 
        sent = CA_sents[sent_ID]
        doc = nlp(sent, disable=["transformer", "ner", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        toks = [doc[i].text.lower() for i in range(len(doc))]
        ngrams = get_n_gramlist(toks)
        for ng in ngrams: 
            sent_ngrams[ng].append(('CA', sent_ID))
    
    TX_df = aapi_sent_topics_50[aapi_sent_topics_50['state'] == 'TX']
    TX_sents = pd.Series(TX_df.sentence.values,index=TX_df.sentence_id).to_dict()
    
    for sent_ID in tqdm(TX_sents): 
        sent = TX_sents[sent_ID]
        doc = nlp(sent, disable=["transformer", "ner", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        toks = [doc[i].text.lower() for i in range(len(doc))]
        ngrams = get_n_gramlist(toks)
        for ng in ngrams: 
            sent_ngrams[ng].append(('TX', sent_ID))
        
    with open('./results/overlapping_ngrams.json', 'w') as outfile: 
        json.dump(sent_ngrams, outfile)
        
def examine_overlaps(): 
    with open('./results/overlapping_ngrams.json', 'r') as infile: 
        overlap_ngrams = json.load(infile)
    dtypes = {'state':str, 'book_id':str, 'book_filename':str, 'subject':str, 
              'sentence_id':str, 'sentence':str, 'aapi':str}
    sent_df = pd.read_csv(LOGS + 'combined_coref_data.csv', dtype=dtypes)
    CA_df = sent_df[sent_df['state'] == 'CA']
    CA_sents = pd.Series(CA_df.sentence.values,index=CA_df.sentence_id).to_dict()
    TX_df = sent_df[sent_df['state'] == 'TX']
    TX_sents = pd.Series(TX_df.sentence.values,index=TX_df.sentence_id).to_dict()
    
    ngram_sent_count = Counter() # number of times ngram appears
    sent_book_count = Counter() # number of times phrases in this sentence appear in other books
    total_overlap_sents = set() 
    for ngram in overlap_ngrams: 
        ngram_sent_count[ngram] = len(overlap_ngrams[ngram])
        states = []
        for tup in overlap_ngrams[ngram]: 
            state = tup[0]
            states.append(state)
            sent_ID = tup[1]
            total_overlap_sents.add(state + '_' + str(sent_ID))
            if state == 'CA': 
                sent = CA_sents[sent_ID]
            elif state == 'TX': 
                sent = TX_sents[sent_ID]
            sent_book_count[sent] = max(sent_book_count[sent], ngram_sent_count[ngram])

    print("TOTAL # OF OVERLAPPING SENTS:", len(total_overlap_sents))
    print()
    print("---- SENTENCES WITH NGRAMS REUSED BETWEEN STATES ----")
    for tup in sent_book_count.most_common(100): 
        print("NUMBER OF OTHER SENTS WITH PHRASES:", tup[1])
        print(tup[0])
        print()

def main(): 
    #find_overlaps()
    examine_overlaps()

if __name__ == '__main__':
    main()