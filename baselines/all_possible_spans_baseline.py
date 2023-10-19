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
import torch
from BM25 import BM25
from sentence_transformers import SentenceTransformer
import transformers

from metrics import calculate_exact_match, average_f1_score, compute_f1, normalize_answer




# class wrappers to include:
# -init (prepare model)
# -train (train model if needed)
# -score (return a similarity score for (q,doc) pair)

class TFIDF(object):

    def __init__(self):
        self.model = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=300000)
        
    def train(self, data):
        self.model.fit(data)
        
    def embed(self, texts):
        vec = self.model.transform(texts)
        return vec
        
    def score(self, q_vec, doc_vec):
        similarity = pairwise.cosine_similarity(q_vec, doc_vec)
        return similarity[0,:]
        
class BERT(object):

    def __init__(self):
        print("CUDA:", torch.cuda.is_available())
        self.tokenizer = transformers.BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
        self.model = transformers.BertModel.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1").eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda() 
            
            #self.model = transformers.pipeline("feature-extraction", model="TurkuNLP/bert-base-finnish-cased-v1", device=0)
            #self.model = transformers.pipeline("feature-extraction", model="TurkuNLP/bert-base-finnish-cased-v1")

    def train(self, data):
        pass
        
    def embed(self, texts):
        with torch.no_grad():
            tok = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", return_special_tokens_mask=True)
            if torch.cuda.is_available():
                input_ids, token_type_ids, attention_mask, spec_mask = tok["input_ids"].cuda(), tok["token_type_ids"].cuda(), tok["attention_mask"].cuda(), tok["special_tokens_mask"].cuda()
            else:
                input_ids, token_type_ids, attention_mask, spec_mask = tok["input_ids"], tok["token_type_ids"], tok["attention_mask"], tok["special_tokens_mask"]
            emb = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            last_hidden=emb.last_hidden_state
            attention_mask=attention_mask*(spec_mask*-1+1)
            attention_mask_sum=torch.sum(attention_mask,dim=-1)
            last_hidden_masked=last_hidden.mul(attention_mask.unsqueeze(-1))
            last_hidden_masked_sum=torch.sum(last_hidden_masked,dim=1)
            last_hidden_mean=torch.div(last_hidden_masked_sum,attention_mask_sum.unsqueeze(-1))
            return torch.nn.functional.normalize(last_hidden_mean, dim=1)
        
    def score(self, q_emb, doc_emb):
        similarity = torch.mm(q_emb, doc_emb.T)
        if torch.cuda.is_available():
            similarity = similarity.cpu()
        return similarity.numpy()[0]
        
        
class SBERT(object):

    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        print("CUDA:", torch.cuda.is_available())
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
    def train(self, data):
        pass
        
    def embed(self, texts):
        emb = self.model.encode(texts, convert_to_tensor=True, device=self.model.device, normalize_embeddings=True)
        return emb
        
    def score(self, q_emb, context_emb):
        
        similarity = torch.mm(q_emb, context_emb.T)
        if torch.cuda.is_available():
            similarity = similarity.cpu()
        return similarity.numpy()[0]



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
        #if selected[i] != gold_segment:
        #    print("Oracle error:", selected[i], gold_segment)
    return selected




def prepare_model(args):
    # model options: BM25, TF-IDF, BERT, mSBERT, fiSBERT
    # load training data if needed (bm25, tf-idf)
    if args.model in ["BM25", "TF-IDF"]: 
        with open(args.train_data, "rt", encoding="utf-8") as f:
            train_texts = json.load(f) # key: id, value: doc as a list of sentences NOTE: documents are not unique!
            train_docs = []
            for key, val in train_texts.items():
                if val not in train_docs:
                    train_docs.append(val)
            train_sentences = [s for d in train_docs for s in d]
            print("Training sentences:", len(train_sentences))
            del train_texts, train_docs # free memory
    if args.model == "BM25":
        model = BM25()
        model.train(train_sentences)
        del train_sentences
    elif args.model == "TF-IDF":
        model = TFIDF()
        model.train(train_sentences)
        del train_sentences
    elif args.model == "BERT":
        model = BERT()
    elif args.model == "mSBERT":
        model = SBERT("sentence-transformers/paraphrase-xlm-r-multilingual-v1")
    elif args.model == "fiSBERT":
        model = SBERT("TurkuNLP/sbert-cased-finnish-paraphrase")
    else:
        print("Model not recognized:", args.model)
        assert False
    return model
    
def batch_iter(context_tokens, batch_size):

    batch = []
    for i in range(0, len(context_tokens)): # i is the start of span
        for j in range(i+1, min(i+66, len(context_tokens))): # j is the end of span
            span = " ".join(context_tokens[i:j])
            batch.append(span)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
        
    
# Max number of tokens (train): 181
# Number of tokens (dev): 57
# Number of tokens (test): 66
def predict_all_spans(questions, contexts, model, batch_size=10):

    predictions = {}
    for idx in tqdm.tqdm(questions.keys()):
        q = questions[idx]
        context = " ".join([s for s in contexts[idx]]) # from sentences to document
        tokens = context.split()
        max_sim = 0.0
        max_span = ""
        c = 0
        q_vec=model.embed([q])
        for batch in batch_iter(tokens, batch_size):
            batch_vec = model.embed(batch)
            sims = model.score(q_vec, batch_vec)
            for i, s in enumerate(sims):
                if s > max_sim:
                    max_sim = s
                    max_span = batch[i]
            c+=len(batch)
        #print(max_sim, max_span, q)
        #print("c=",c)
        predictions[idx] = max_span
    return predictions
        
        
def predict_sentences(questions, contexts, model):

    predictions = {}
    for idx in tqdm.tqdm(questions.keys()):
        q = questions[idx]
        context = contexts[idx]
        max_sim = 0.0
        max_target = ""
        for target in context:
            s = model.score(q, target)
            if s > max_sim:
                max_sim = s
                max_target = target
        predictions[idx] = max_target
    return predictions
        


def sample(data, size):
    if size == 0:
        return data
    sample={}
    for key in sorted(data.keys()):
        sample[key]=data[key]
        if len(sample) >= size:
            return sample
    return sample

def main(args):

    ## PREPARE MODEL ##
    model=prepare_model(args)
    print("Preparing models ready.\n")

    
    ## READ EVAL DATA ##
    with open(f'{args.eval_data_dir}/contexts.json', 'r') as json_file:
        contexts = json.load(json_file)
        contexts = sample(contexts, args.sample_size)
    with open(f'{args.eval_data_dir}/questions.json', 'r') as json_file:
        questions = json.load(json_file)
        questions = sample(questions, args.sample_size)
    with open(f'{args.eval_data_dir}/answers.json', 'r') as json_file:
        answers = json.load(json_file)
        answers = sample(answers, args.sample_size)
    assert len(answers)==len(questions)
    assert(len(contexts)==len(questions))
    print("Preparing data ready.\n")
        
    if args.model=="BM25":
        args.batch_size = 1 # overwrite batch_size
        
    ## predict ##
    if args.all_spans:
        print("Comparing against all possible spans.")
        preds = predict_all_spans(questions, contexts, model, batch_size=args.batch_size) # predictions is a dict
    else:
        print("Comparing against original segments.")
        preds = predict_sentences(questions, contexts, model)

        
    ## EVALUATE ##
    
    
    
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
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--all-spans', default=False, action="store_true", help="Compare to all possible spans, default False.")
    parser.add_argument('--batch-size', type=int, default=10, help="Batch size for embedding, default=10.")
    parser.add_argument('--sample-size', type=int, default=0, help="Sample size, default=0 (all).")
    
    args = parser.parse_args()
    
    main(args)

