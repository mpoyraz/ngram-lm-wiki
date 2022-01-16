import os
import re
import json
import argparse
import subprocess
from tqdm import tqdm
from multiprocessing import Pool
from unicode_tr import unicode_tr
from trtokenizer.tr_tokenizer import SentenceTokenizer

# Global variables
tokenize_fn = None
lower_fn = None
chars_to_remove_regex = "[#$%&()*+,-./:;<=>?@[\]^_{|}~!\"\\\]"
apostrophes = "[’`´ʹʻʼʽʿˈ]"

def parse_sentences_wiki_json_file(fpath):
    """ Parses & cleans sentences from a wikipedia file in JSON format

    Args:
    fpath (str): Path to a extracted wikipedia file

    Returns:
    sentences (List[str]): List of cleaned sentences
    """
    # Load text from wiki articles
    with open(fpath) as fp:
        texts = [json.loads(line.strip())['text'] for line in fp]
    # Senteces from paragraphs
    sentences = []
    for text in texts:
        for sent in tokenize_fn(text):
            # Lower the sentence
            sent = lower_fn(sent)
            # Remove pre-defined chars
            sent = re.sub(chars_to_remove_regex, "", sent)
            # Unify apostrophes
            sent = re.sub(apostrophes, "'", sent)
            # Remove multiple spaces
            sent = re.sub(r"\s+", " ", sent)
            # Append
            if len(sent) > 0:
                sentences.append(sent)

    return sentences

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Process extracted wikipedia files")
    parser.add_argument(
        "--wiki_dump", type=str, required=True, help="Path to a wikipedia dump"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory to save extracted files and sentences file"
    )
    parser.add_argument(
        "--language_id", type=str, default="tr", help="Language id of the wikipedia dump"
    )
    parser.add_argument(
        "--processes", type=int, default=8, help="Number of processes to use"
    )
    args = parser.parse_args()

    global tokenize_fn
    global lower_fn

    # Sentence tokenizer and string lower functions for the language
    if args.language_id == 'tr':
        sentence_tokenizer = SentenceTokenizer()
        tokenize_fn = sentence_tokenizer.tokenize
        lower_fn = lambda x: unicode_tr(x).lower()
    else:
        raise NotImplementedError("Language id '{}' is not supported!".format(args.language_id))

    # Run extractor on the wiki dump
    extract_dir = os.path.join(args.output, "extract")
    subargs = [
        "wikiextractor", args.wiki_dump,
        "-o", extract_dir,
        "--no-templates",
        "--json",
        "--processes", str(args.processes)
    ]
    subprocess.check_call(subargs)

    # Paths of the extracted wiki files
    filepaths = [os.path.join(root, filename)
                    for root, dirnames, filenames in os.walk(extract_dir)
                    for filename in filenames]

    # Load wiki files and parse sentences
    with Pool(args.processes) as pool:
        sentences = []
        for sentences_pool in tqdm(pool.imap(parse_sentences_wiki_json_file, filepaths), total=len(filepaths)):
            sentences += sentences_pool
        print("Number of extracted sentences: {}".format(len(sentences)))

    # Save all parsed sentences
    with open(os.path.join(args.output, 'sentences.txt'), 'w') as fp:
        for sent in sentences:
            fp.write('{}\n'.format(sent))

    return

if __name__ == "__main__":
    main()