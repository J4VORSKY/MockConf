# MockConf: Student Interpretation Dataset

![Dataset-example](./resources/example.png?raw=true)

This repository provides code and experiments that are described in the paper `MockConf: A Student Interpretation Dataset: Analysis, Word- and Span-level Alignment and Baselines`.

# InterAlign

We developed our own annotation tool that is available [here](https://github.com/J4VORSKY/InterAlign).

# Dataset

The dataset itself can be downloaded [here](http://hdl.handle.net/11234/1-5906). Put the `data` folder to the root directory of this repository to make everything properly working.

# Experiments

The code was tested with Python version 3.9.7.

The paper describes 3 models:
1. BERTAlign (BA)
2. Subsegmentation (BA+sub)
3. Labeling (BA+sub+lab)

The outputs of each systems are in the `outputs` folder. In case you want to replicate the experiments, run the following commands.

## BertAlign

```bash
python scripts/bertalign_sentence_alignment.py --input_dir ./data/alignments/one-annotation --output_dir ./outputs/bertalign-sentence-alignment --max_align 10 --top_k 10 --window 10 --skip 0.0 --len_penalty
```

## BertAlign + Subsegments

```bash
python scripts/bertalign_sentence_alignment.py --input_dir ./data/alignments/one-annotation --output_dir ./outputs/bertalign-sentence-alignment-subsegments --max_align 10 --top_k 10 --window 10 --skip 0.0 --len_penalty --subsegments_word_alignment
```

## BertAlign + Subsegments + Labels

Calculating cosine similarity that is used as input feature for label training:

```bash
python scripts/calculate_similarity_scores.py ./data/alignments/one-annotation/ ./tmp/similarity_scores.json --extended
```

Training the label classifier and labeling the `BA+sub` data.

```bash
python scripts/label_prediction.py --data_path ./tmp/similarity_scores.json --hidden_size 100 --lr 0.001 --epochs 200 --batch_size 16 --val_split 0.2 --bertalign_subsegments_path ./outputs/bertalign-sentence-alignment-subsegments/ --bertalign_subsegments_labels_path ./outputs/bertalign-sentence-alignment-subsegments-labels-200/
```

## Word alignment baseline

```bash
python scripts/doc_lvl_word_alignment.py --layers 8 8 --outAdd outputs/word-align-baseline --model "xlm-roberta-base" --window_size 128 --filter_distant 50 --directory data/alignments/one-annotation/
```

# Evaluation

`analysis/evaluation.ipynb` shows the result table that is reported in the paper.

# Credit

```
TBA
```