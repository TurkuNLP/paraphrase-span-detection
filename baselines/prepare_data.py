
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


def conllu2sent(conllu_str):
    sentences = []
    lines = conllu_str.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("# text ="): # this has the sentence without tokenization
            sentences.append(line.replace("# text = ", "").strip())
    return sentences
        


def main(args):
    with open(args.file, "rt", encoding="utf-8") as json_file:
        data = json.load(json_file)
    if args.positives_only == True:
        data["data"] = filter_negatives(data["data"])

    print("Loading udpipe", file=sys.stderr)
    model = udpipe.Model.load(args.model)
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
      
        tokenized_questions[str(i)]=data['data'][i]['question']
        tokenized_contexts[str(i)]=conllu2sent(docs)
        tokenized_answers[str(i)]=a


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
    parser.add_argument('--file', '-f', type=str, required=True)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--positives-only', action="store_true", default=False, help="Include only positive paraphrases, skip negatives (label 2 and below)")
    
    args = parser.parse_args()
    
    main(args)
