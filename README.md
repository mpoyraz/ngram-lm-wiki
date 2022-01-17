# ngram-lm-wiki
Train n-gram language models (LM) on Wikipedia articles, mainly intended for Automatic Speech Recognition (ASR) applications.

## Dependencies
The following dependencies should be available on your system:
- [KenLM](https://github.com/kpu/kenlm): KenLM is used to create the n-gram LM. Please see [KenLM docs](https://kheafield.com/code/kenlm/) for more details.
- Scripts are tested with Python 3.7 and required packages are listed in the requirements.txt.

## Process Wikipedia Dump
The script `process_wiki.py` currently support Turkish language in terms of sentence tokenization but can be easily configured for other languages.
1. It extracts Wikipedia (e.g. trwiki-latest-pages-articles.xml.bz2) article dump into individual json files.
2. Then, loads each extracted wiki file and tokenizes & cleans sentences for LM training.
3. Finally, it saves tokenized and cleaned sentences in the output directory.

Example usage:
```bash
python process_wiki.py \
    --wiki_dump trwiki-latest-pages-articles.xml.bz2 \
    --output data \
    --language_id tr \
    --processes 8
```

## Train n-gram LM
KenLM binary `lmplz` is used to create a n-gram LM in arpa format and then the arpa LM is converted to binary format using `build_binary`.

The following usage creates a 4-gram LM with top 200000 most frequent words and pruning.

For ASR applications, `--vocab` option can be used with ASR output vocabulary of characters to filter unigrams.

```bash
python generate_lm.py \
    --input data/sentences.txt \
    --output lm \
    --kenlm_bins kenlm/build/bin \
    --vocab vocab.txt \
    --top_k 200000 \
    --order 4 \
    --prune "0 0 1" \
    --binary_type "trie" \
    -a 255 \
    -q 8 \
```
