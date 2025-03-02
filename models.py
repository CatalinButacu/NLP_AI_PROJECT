# models.py

# models.py

from sentiment_data import *
from utils import *

from collections import Counter
import numpy as np
import random
import math

import spacy
from string import punctuation
from spacy.lang.en import stop_words
from typing import List


SEED = 0
np.random.seed(SEED)
random.seed(SEED)


############################################################################################################
#   Feature Extractors
############################################################################################################

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


### Acc >= 74% ###
class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        for word in sentence:
            word = word.lower()
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(f"UNIGRAM_{word}")
            else:
                idx = self.indexer.index_of(f"UNIGRAM_{word}")
            if idx != -1:
                features[idx] += 1
        return features


### Acc >= 77% ###
class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        for i in range(len(sentence) - 1):
            bigram = f"{sentence[i].lower()}_{sentence[i + 1].lower()}"
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(f"BIGRAM_{bigram}")
            else:
                idx = self.indexer.index_of(f"BIGRAM_{bigram}")
            if idx != -1:
                features[idx] += 1
        return features


### Accuracy: 670 / 872 = 0.768349 ###
class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor -> Unigram++
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'])

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        processed_sent = [word.lower() for word in sentence if word.lower() not in self.stopwords]

        for word in processed_sent:
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(f"UNIGRAM_{word}")
            else:
                idx = self.indexer.index_of(f"UNIGRAM_{word}")
            if idx != -1:
                features[idx] += 1

        sentiment_words = {'great', 'good', 'bad', 'terrible', 'excellent', 'poor', 'amazing', 'awful'}
        for word in processed_sent:
            if word in sentiment_words:
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(f"SENTIMENT_{word}")
                else:
                    idx = self.indexer.index_of(f"SENTIMENT_{word}")
                if idx != -1:
                    features[idx] = 1

        return features


### Accuracy: 671 / 872 = 0.769495 ###
class BetterFeatureExtractor_V2(FeatureExtractor):
    """
    Feature extractor that uses TF-IDF weighting for unigram features.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of'])
        self.doc_freqs = Counter()  # document frequency for each word
        self.num_docs = 0
        self.idf_values = {}
        self.initialized = False

    def get_indexer(self):
        return self.indexer

    def initialize_idf(self, examples: List[SentimentExample]):
        """
        Compute document frequencies and IDF values from a list of examples.
        """
        self.num_docs = len(examples)

        # count frequencies
        for ex in examples:
            processed_sent = [word.lower() for word in ex.words if word.lower() not in self.stopwords]

            word_set = set(processed_sent)
            for word in word_set:
                self.doc_freqs[word] += 1

        # compute IDF
        for word, doc_freq in self.doc_freqs.items():
            self.idf_values[word] = math.log(self.num_docs / (doc_freq + 1)) + 1

        self.initialized = True

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()

        processed_sent = [word.lower() for word in sentence if word.lower() not in self.stopwords]

        term_freqs = Counter(processed_sent)

        for word, tf in term_freqs.items():
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(f"TFIDF_{word}")
                features[idx] = tf

                if self.initialized:
                    idf = self.idf_values.get(word, math.log(self.num_docs) + 1)  # default IDF for unseen words
                    features[idx] *= idf

            else:
                idx = self.indexer.index_of(f"TFIDF_{word}")

                if idx != -1:
                    features[idx] = tf

                    if self.initialized:
                        idf = self.idf_values.get(word, math.log(self.num_docs) + 1)
                        features[idx] *= idf

        sentiment_words = {'great', 'good', 'bad', 'terrible', 'excellent', 'poor', 'amazing', 'awful'}
        for word in processed_sent:
            if word in sentiment_words:
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(f"SENTIMENT_{word}")
                else:
                    idx = self.indexer.index_of(f"SENTIMENT_{word}")
                if idx != -1:
                    features[idx] = 1

        # negation features
        for i, word in enumerate(processed_sent):
            if word in {'not', 'no', 'never', "n't", 'cannot'} and i + 1 < len(processed_sent):
                next_word = processed_sent[i + 1]
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(f"NEG_{next_word}")
                else:
                    idx = self.indexer.index_of(f"NEG_{next_word}")
                if idx != -1:
                    features[idx] = 1

        return features


############################################################################################################
#   Classifiers
############################################################################################################

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """

    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence)
        score = 0.0
        for feat_idx, feat_val in features.items():
            score += self.weights[feat_idx] * feat_val
        return 1 if score >= 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence)
        score = 0.0
        for feat_idx, feat_val in features.items():
            score += self.weights[feat_idx] * feat_val
        return 1 if self._sigmoid(score) >= 0.5 else 0

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))


############################################################################################################
#   Training
############################################################################################################

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    weights = np.zeros(len(feat_extractor.get_indexer()))

    # Train for several epochs
    num_epochs = 30
    for _ in range(num_epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words)
            score = 0.0
            for feat_idx, feat_val in features.items():
                score += weights[feat_idx] * feat_val
            prediction = 1 if score >= 0 else 0

            # Update on mistake
            if prediction != ex.label:
                for feat_idx, feat_val in features.items():
                    weights[feat_idx] += feat_val * (ex.label - prediction)

    return PerceptronClassifier(weights, feat_extractor)


import string


#en_core_web_sm for efficiency
#en_core_web_trf for accuracy
nlp = spacy.load("en_core_web_sm", disable=["tok2vec","tagger", "attribute_ruler", "lemmatizer","ner", "parser", "textcat"])
#nlp.enable_pipe("senter") #senter faster than parser


def process_batch(train_exs: List[SentimentExample]) -> List[SentimentExample]:
    # Process all SentimentExamples at once (batch processing)
    texts = [" ".join(ex.words) for ex in train_exs]
    docs = list(nlp.pipe(texts))

    # Process the documents and update the SentimentExamples
    for ex, doc in zip(train_exs, docs):
        ex.words = [token.text for token in doc if not token.is_punct]

    return train_exs


def keep_only_letters(train_exs: List[SentimentExample]) -> List[SentimentExample]:
    texts = [" ".join(ex.words) for ex in train_exs]
    docs = list(nlp.pipe(texts))

    for ex, doc in zip(train_exs, docs):
        ex.words = [token.text for token in doc if token.is_alpha]

    return train_exs


def remove_prepositions(train_exs: List[SentimentExample]) -> List[SentimentExample]:
    texts = [" ".join(ex.words) for ex in train_exs]
    docs = list(nlp.pipe(texts))

    for ex, doc in zip(train_exs, docs):
        ex.words = [token.text for token in doc if token.pos_ != "ADP"]

    return train_exs


def purge_text(train_exs: List[SentimentExample]) -> List[SentimentExample]:
    train_exs = process_batch(train_exs)  # BEST SOLO Accuracy: 677 / 872 = 0.776376 (UNIGRAM)
    train_exs = keep_only_letters(train_exs)  # BEST SOLO Accuracy: 678 / 872 = 0.777523 (UNIGRAM)
    train_exs = remove_prepositions(train_exs)  # BEST SOLO Accuracy: 683 / 872 = 0.783257 (UNIGRAM)
    return train_exs


def train_logistic_regression(train_exs: List[SentimentExample],
                              feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    train_exs = purge_text(train_exs)  # BEST COMBINED Accuracy: 677 / 872 = 0.776376 (UNIGRAM)

    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    weights = np.zeros(len(feat_extractor.get_indexer()))

    def activation(
            x):  # --model LR --feats UNIGRAM            --model LR --feats BIGRAM             --model LR --feats BETTER
        return 1.0 / (1.0 + math.exp(
            -x))  # sigmoid function                                   ~~~ Accuracy: 679 / 872 = 0.778670    ~~~ Accuracy: 641 / 872 = 0.735092    ~~~ Accuracy: 678 / 872 = 0.777523
        # return math.log(1 + math.exp(x)) # Softplus function                                  ~~~ Accuracy: 627 / 872 = 0.719037    ~~~ Accuracy: 541 / 872 = 0.620413    ~~~ Accuracy: 619 / 872 = 0.709862
        # return max(0, x) # ReLU function                                                      ~~~ Accuracy: 599 / 872 = 0.686927    ~~~ Accuracy: 544 / 872 = 0.623853    ~~~ Accuracy: 594 / 872 = 0.681193
        # return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)) # tanh function    ~~~ Accuracy: 573 / 872 = 0.657110    ~~~ Accuracy: 514 / 872 = 0.589450    ~~~ Accuracy: 594 / 872 = 0.681193

    num_epochs = 30
    learning_rate = 0.05

    for _ in range(num_epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words)
            score = 0.0
            for feat_idx, feat_val in features.items():
                score += weights[feat_idx] * feat_val

            prediction = activation(score)
            error = ex.label - prediction

            for feat_idx, feat_val in features.items():
                weights[feat_idx] += learning_rate * error * feat_val

    return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        if False:
            feat_extractor = BetterFeatureExtractor(Indexer())
        else:
            feat_extractor = BetterFeatureExtractor_V2(Indexer())
            feat_extractor.initialize_idf(train_exs)
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model


