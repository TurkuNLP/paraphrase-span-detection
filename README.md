# Fine-tuning BERT

All of the original code for SQuAD can be found in [this HuggingFace repository](https://github.com/huggingface/transformers/tree/master/examples/pytorch/question-answering). The code for this project was fetched from the above mentioned repository on 7th of June 2021.

To run the code in CSC Mahti:

Clear the loaded modules and load pytorch

```bash
module purge 
module load pytorch/1.8
```

Important! To run HuggingFace example code install transformers from the source:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
python -m pip install --user . 
```
To install the requirements:
```bash
python -m pip install --user -r requirements.txt
```

**Notes:** 
- This script only works with models that have a fast tokenizer 
- If your dataset contains samples with no possible answers (like SQUAD version 2), you need to pass along the flag `--version_2_with_negative`.

Example code for fine-tuning BERT on the SQuAD1.0 dataset. 

```bash
python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/
```
