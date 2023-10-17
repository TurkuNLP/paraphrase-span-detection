
import ufal.udpipe as udpipe
import json
import argparse
import sys
import hashlib
import os


def filter_negatives(examples):
    skip_labels = ["2", "1", "x"]
    filt_examples = []
    for e in examples:
        if e["label"] in skip_labels:
            continue
        filt_examples.append(e)
    return filt_examples


def conllu2sent(conllu_str, tokenize=False):
    sentences = []
    tokens = []
    lines = conllu_str.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            if tokenize==True and tokens:
                sentences.append(" ".join(tokens))
                tokens = []
            continue
        if tokenize==False and line.startswith("# text ="): # this has the sentence without tokenization
            sentences.append(line.replace("# text = ", "").strip())
        if tokenize==True and not line.startswith("#"): # token line
            cols = line.split("\t")
            tokens.append(cols[1])
    if tokenize == True and tokens:
        sentences.append(" ".join(tokens))
    return sentences
    
def tokenize_chunk(chunk, model, tokenize=False):
    # tokenixe a text chunk without splitting it to sentences
    if tokenize==False:
        return chunk
    chunk = chunk.replace("\n\n", "\n")
    conllu = model.process(chunk)
    tokens = []
    lines = conllu.split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        tokens.append(cols[1])
    return " ".join(tokens)
    
    


def main(args):
    with open(args.file, "rt", encoding="utf-8") as json_file:
        data = json.load(json_file)
        print(len(data["data"]), "examples in the original data")
    if args.skip_negatives == True:
        data["data"] = filter_negatives(data["data"])
    print("Using", len(data["data"]), "examples")

    print("Loading udpipe", file=sys.stderr)
    model = udpipe.Model.load(args.udpipe_model)
    pipeline = udpipe.Pipeline(model,"tokenize","none","none","conllu") # return conllu
    print("Done", file=sys.stderr)
    cache = {} # do not segment the same context many times, rather cache the results

    tokenized_questions={}
    tokenized_contexts={}
    tokenized_answers={}
    for i in range(len(data['data'])):
        q = pipeline.process(data['data'][i]['question'])
        c = data['data'][i]['context'].replace("\n\n", "\n") # remove empty lines
        c_hash = hashlib.sha256(c.encode('utf-8')).hexdigest()
        if c_hash in cache:
            docs = cache[c_hash]
        else:
            docs = pipeline.process(c)
            cache[c_hash] = docs
        if len(data['data'][i]['answers']['text']) == 0: # negative example
            a = ''
        else:
            a = data['data'][i]['answers']['text'][0]
        
        
        tokenized_questions[str(i)]=tokenize_chunk(data['data'][i]['question'], pipeline, tokenize=args.tokenize)
        tokenized_contexts[str(i)]=conllu2sent(docs, tokenize=args.tokenize)
        tokenized_answers[str(i)]=tokenize_chunk(a, pipeline, tokenize=args.tokenize)


    # save
    # make dir if not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'contexts.json'), 'w', encoding='utf-8') as f:
        json.dump(tokenized_contexts, f, ensure_ascii=False, indent=4)

    with open(os.path.join(args.output_dir, 'questions.json'), 'w', encoding='utf-8') as f:
        json.dump(tokenized_questions, f, ensure_ascii=False, indent=4)

    with open(os.path.join(args.output_dir, 'answers.json'), 'w', encoding='utf-8') as f:
        json.dump(tokenized_answers, f, ensure_ascii=False, indent=4)
        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--udpipe_model', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--skip_negatives', action="store_true", default=False, help="Skip negative examples (label 2 and below), default False")
    parser.add_argument('--tokenize', action="store_true", default=False, help="Tokenize examples for word-level models (BM25), default False")
    
    args = parser.parse_args()
    
    main(args)
