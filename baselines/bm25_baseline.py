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
from BM25 import BM25

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



    
def run_bm25(questions, contexts, bm25_model):
    preds={}
    for key in tqdm.tqdm(questions):
        q = questions[key]
        candidates = contexts[key]
        scores = []
        for c in candidates:
            score = bm25_model.score(q, c)
            scores.append(score)
        max_index = np.argmax(scores)
        best_segment = candidates[max_index]
        preds[key] = best_segment
    return preds


def main(args):

    ## TRAIN ##
    with open(args.train_data, "rt", encoding="utf-8") as f:
        train_texts = json.load(f) # key: id, value: doc as a list of sentences NOTE: documents are not unique!
        train_docs = []
        for key, val in train_texts.items():
            if val not in train_docs:
                train_docs.append(val)
        train_sentences = [s for d in train_docs for s in d]
        print("Training sentences for the vectorizer:", len(train_sentences))
        del train_texts, train_docs # free memory

    print("Training the BM25 model...")
    bm25=BM25(k=args.k, b=args.b)
    bm25.train(train_sentences)
    del train_sentences # free memory
    print("Done.\n")

    
    ## READ EVAL DATA ##
    with open(f'{args.eval_data_dir}/contexts.json', 'r') as json_file:
        contexts = json.load(json_file)
    with open(f'{args.eval_data_dir}/questions.json', 'r') as json_file:
        questions = json.load(json_file)
    assert(len(contexts)==len(questions))
        
    ## BM25 ##
    preds = run_bm25(questions, contexts, bm25) # predictions is a dict
        

    ## EVALUATE ##
    
    with open(f'{args.eval_data_dir}/answers.json', 'r') as json_file:
        answers = json.load(json_file)
    assert len(answers)==len(questions)
    
    f1 = average_f1_score(answers, preds)
    print("F-score:", f1)
        
    em = calculate_exact_match(answers, preds)
    print("Exact match:", em)
    print()
    
    import sys
    sys.exit()
        
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
    parser.add_argument('--k', type=float, default=1.2)
    parser.add_argument('--b', type=float, default=0.75)
    
    args = parser.parse_args()
    
    main(args)

