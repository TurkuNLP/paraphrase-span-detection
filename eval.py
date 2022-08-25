import json
import argparse
import sys
from baselines.metrics import average_f1_score, calculate_exact_match

def read_gold(fname):

    with open(fname, "rt", encoding="utf-8") as f:
        data = json.load(f)
        data = data["data"]
    gold_segments = {}
    for example in data:
        idx = example["id"]
        seg = example["answers"]["text"][0]
        gold_segments[idx] = seg
    return gold_segments

def read_preds(fname):
    with open(fname, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main(args):

    g = read_gold(args.gold)
    p = read_preds(args.predictions)
    
    assert len(g) == len(p)
    print("Number of examples:", len(g))
    
    f1 = average_f1_score(g, p)
    print("F-score:", f1)

    em = calculate_exact_match(g, p)
    print("Exact match:", em)



if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", "-g", default=None, required=True, help="Gold json file")
    parser.add_argument("--predictions", "-p", default=None, required=True, help="Predictions json file")
    args = parser.parse_args()
    
    main(args)

