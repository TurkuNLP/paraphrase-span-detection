## Running baselines

### create data
`python prepare_data.py --file ../qa_data/train.json --udpipe_model finnish-tdt.udpipe --output-dir baseline-data-dev --skip_negatives`

### TF-IDF
`python tfidf_baseline.py --train_data baseline-data-train/contexts.json --eval_data_dir baseline-data-dev`

### FinBERT
`python bert_baseline.py --eval_data_dir baseline-data-dev --pooling-method AVG --gpu`

### SBERT

TODO
