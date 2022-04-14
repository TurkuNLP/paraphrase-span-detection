import json
import numpy as np
import random
import re
import string
import collections
import sklearn.metrics.pairwise as pairwise
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import argparse
import tqdm

from metrics import calculate_exact_match, average_f1_score, compute_f1, normalize_answer

def get_oracle_predictions(gold, contexts):

    assert len(gold) == len(contexts)
    selected = {}
    for i in gold:
        best = None
        gold_segment = normalize_answer(gold[i])
        context_segments = contexts[i]
        for j, candidate in enumerate(context_segments):
            candidate = normalize_answer(candidate)
            if candidate == gold_segment:
                selected[i] = candidate
                break
            f = compute_f1(gold_segment, candidate)
            if best == None or f > best:
                selected[i] = candidate
                best = f
    return selected


def train_vectorizers(data):

    vectorizers=[]

    # words
    #vec = TfidfVectorizer(max_features=300000) # effectively the same as max_features=200000 or 300000 but faster!
    #vec.fit(data)
    #vectorizers.append(vec)

    # char ngrams
    #max_ngram=[4, 5] # max ngram range for char vectorizers
    #min_ngram=[2, 3] # min ngram range for char vectorizers
    #for max in max_ngram:
    #    for min in min_ngram:
    #        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(min,max), max_features=300000)
    #        vec.fit(data)
    #        vectorizers.append(vec)
    
    
    # best setting for final run: ngram_range(2, 4)
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=300000)
    vec.fit(data)
    vectorizers.append(vec)
        
    return vectorizers
    
def tf_idf(questions, contexts, vectorizer):
    preds={}
    for key in tqdm.tqdm(questions):
        q = questions[key]
        candidates = contexts[key]
        mat = vectorizer.transform(candidates) # Transform documents to document-term matrix.
        q_vec = vectorizer.transform([q]) # grab the question and vectorize it
        similarity = pairwise.cosine_similarity(q_vec, mat) 
        max_index = np.argmax(similarity)
        best_segment = candidates[max_index]
        preds[key] = best_segment
    return preds


def main(args):


    # TRAIN TF-IDF VECTORIZER USING TRAINING SET CONTEXTS
    # This data is used to fit the vectorizer
    with open(args.train_data, "rt", encoding="utf-8") as f:
        train_texts = json.load(f) # list of sentences

    print("Training vectorizers...")
    vectorizers = train_vectorizers(train_texts)
    del train_texts # free memory
    print("Done.\n")

    
    ## READ DATA ##
    with open(f'{args.eval_data_dir}/contexts.json', 'r') as json_file:
        contexts = json.load(json_file)
    with open(f'{args.eval_data_dir}/questions.json', 'r') as json_file:
        questions = json.load(json_file)
    assert(len(contexts)==len(questions))
        
    ## TF-IDF ##
    predictions = []
    for i, v in enumerate(vectorizers):
        preds = tf_idf(questions, contexts, v) # predictions is a dict
        predictions.append(preds)
        

    ## EVALUATE ##
    
    with open(f'{args.eval_data_dir}/answers.json', 'r') as json_file:
        answers = json.load(json_file)
    assert len(answers)==len(questions)
    
    for preds in predictions:
        f1 = average_f1_score(answers, preds)
        print("F-score:", f1)
        
        em = calculate_exact_match(answers, preds)
        print("Exact match:", em)
        print()
    
    
        
    ## ORACLE ##
    print("Calculating oracle scores...")
    oracle_predictions = get_oracle_predictions(answers, contexts)

    f1 = average_f1_score(answers, oracle_predictions)
    print("Oracle average F-score:", f1)
    
    em = calculate_exact_match(answers, oracle_predictions)
    print("Oracle exact match:", em)
    print()


    
  
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True) # json file with list of training sentences
    parser.add_argument('--eval_data_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    main(args)

