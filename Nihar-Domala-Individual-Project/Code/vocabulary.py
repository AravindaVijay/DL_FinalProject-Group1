import os.path
import pickle
from collections import Counter

import nltk
from pycocotools.coco import COCO


class Vocabulary(object):
    def __init__(
        self,
        vocab_threshold=3,
        vocab_file="./vocab.pkl",
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        pad_word="<pad>",
        annotations_file="../coco_dataset/annotations/captions_train2017.json",
        vocab_exists=False,
    ):
        """Initialize the vocabulary.
        Args:
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          annotations_file: Path for train annotation file.
          vocab_exists: If False, create vocab from scratch and override any existing vocab_file
                           If True, load vocab from existing vocab_file, if it exists
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.annotations_file = annotations_file
        self.vocab_exists = vocab_exists
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        # load an existing vocabulary
        if os.path.exists(self.vocab_file) and self.vocab_exists:
            with open(self.vocab_file, "rb") as f:
                vocab = pickle.load(f)
            self.word2idx = vocab.word2idx
            self.idx2word = vocab.idx2word
            print("Vocabulary successfully loaded from vocab.pkl file!")

        # create a new vocab file
        else:
            print("Building vocabulary...")
            self.build_vocab()
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)
            print("Vocabulary successfully built and saved to vocab.pkl file!")

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        # Initialize the vocabulary
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_word(self.pad_word)

        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        for i, idx in enumerate(ids):
            caption = str(coco.anns[idx]["caption"])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)