import os
import argparse
import subprocess
from multiprocessing import Pool
from collections import Counter

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate LM")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to an input text file, sentence per line"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory to save language model files"
    )
    parser.add_argument(
        "--kenlm_bins", type=str, required=True, help="File path to the KENLM binaries lmplz and build_binary",
    )
    parser.add_argument(
        "--vocab", type=str, help="Vocabulary of allowed characters in unigrams, character per line"
    )
    parser.add_argument(
        "--top_k", type=int, default=200000, help="Use top_k most frequent words in the input text file",
    )
    parser.add_argument(
        "--order", type=int, default=4, help="Order of n-grams for building LM",
    )
    parser.add_argument(
        "--memory", type=str, default="80%", help="Sorting memory to use for building LM",
    )
    parser.add_argument(
        "--prune", type=str, help="Prune n-grams with count less than or equal to the given threshold",
    )
    parser.add_argument(
        "--binary_type", type=str, help="Data structure type in build_binary",
    )
    parser.add_argument(
        "-a", type=int, help="Pointer compression to save memory in build_binary",
    )
    parser.add_argument(
        "-q", type=int, help="Activates quantization and set the number of bits in build_binary",
    )
    args = parser.parse_args()

    # Read input file
    with open(args.input) as fp:
        sentences = [line.strip() for line in fp]
        print("Number of sentences: {}".format(len(sentences)))

    # Read vocab (allowed characters in unigrams)
    if args.vocab:
        with open(args.vocab) as fp:
            vocab = set([line.strip() for line in fp])
    else:
        vocab = None

    # Unigrams with counts
    word_counter = Counter()
    for sent in sentences:
        if vocab:
            word_counter.update(
                [word for word in sent.split() if all([ch in vocab for ch in word])]
            )
        else:
            word_counter.update(sent.split())
    print("Number of unique words: {}".format(len(word_counter)))

    # Top-k common words
    top_k = min(args.top_k, len(word_counter))
    top_words = word_counter.most_common(top_k)
    top_words_sum = sum(count for word, count in top_words)
    all_words_sum = sum(word_counter.values())
    print("Top {} words are {:.2f} % of all words".format(
        top_k, 100*top_words_sum/all_words_sum)
    )

    # Save unigrams for LM training
    fpath_unigrams = os.path.join(args.output, "unigrams.txt")
    with open(fpath_unigrams, 'w') as fp:
        for word, count in top_words:
            fp.write("{}\n".format(word))

    # Create arpa LM
    print("Creating {}-gram arpa LM with {} pruning".format(
        args.order, args.prune if args.prune else "no")
    )
    fpath_arpa = os.path.join(args.output, "lm.arpa")
    subargs_lmplz = [
        os.path.join(args.kenlm_bins, "lmplz"),
        "--text", args.input,
        "--arpa", fpath_arpa,
        "--order", str(args.order),
        "--memory", args.memory,
        "--limit_vocab_file", fpath_unigrams,
    ]
    if args.prune:
        subargs_lmplz += ["--prune", *args.prune.split()]
    subprocess.check_call(subargs_lmplz)

    # Convert arpa LM to binary format
    print("Converting {}-gram arpa LM to binary".format(args.order))
    fpath_binary = os.path.join(args.output, "lm.bin")
    subargs_binary = [
        os.path.join(args.kenlm_bins, "build_binary"),
        "-v",
    ]
    subargs_binary += ["-a", str(args.a)] if args.a else []
    subargs_binary += ["-q", str(args.q)] if args.q else []
    subargs_binary += [args.binary_type] if args.binary_type else []
    subargs_binary += [fpath_arpa, fpath_binary]
    subprocess.check_call(subargs_binary)

    return

if __name__ == "__main__":
    main()