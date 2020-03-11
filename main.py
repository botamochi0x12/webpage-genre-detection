# %%
import dataclasses
import datetime
import enum
import json
import logging as _logging
import pickle
import random
import re
import string
import sys
import typing
from typing import Dict, List
from urllib.parse import splitquery, urljoin

import numpy as np
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.linear_model import SGDClassifier as SVM
from symspellpy.symspellpy import SymSpell, Verbosity

# Define `logger` only once
try:
    logger
except NameError:
    logger: _logging.Logger = _logging.getLogger("webpage_genre_detection")
    handler_1 = _logging.StreamHandler(sys.stdout)
    handler_1.setLevel(_logging.DEBUG)
    logger.addHandler(handler_1)
    handler_2 = _logging.FileHandler("debug.log", encoding="utf-8")
    handler_2.setLevel(_logging.DEBUG)
    logger.addHandler(handler_2)
    logger.setLevel(_logging.DEBUG)
    logger.propagate = False

sym_spell = SymSpell(2, 7)
if not sym_spell.create_dictionary("frequency_dictionary_en_82_765.txt"):
    logger.warning("Symspell isn't loaded!")

uint = typing.NewType("unsigned_int", int)
URL = typing.NewType("URL", str)
Sentence = typing.NewType("Sentence", str)

try:
    NewsCategory = enum.Enum(
        "NewsCategory",
        list(
            np.loadtxt(
                "categories.csv",
                dtype=np.str,
                delimiter=",",
                skiprows=1,
                usecols=0,
                )
            )
        )
except OSError as ex:
    logger.error(ex)
    raise ex

# %%
DELTA = 2
GAMMA = 3
SIGMA = 100

# NOTE:
# * List instances are mutable
# * Once `URL_LIST` is changed, it does never auto-reset.
#     So, please assign it an empty list manually.
URL_LIST: List[URL] = []


def construct_tree_from(
    url: URL,
    *,
    delta_,
    gamma=GAMMA,
):
    if delta_ < 0:
        raise ValueError(
            f"Max. depth must be positive. (delta_ as depth < {delta_})")

    # Check if starting with `http` or `https`
    # NOTE: After below, InvalidSchema and/or MissingSchema mustn't happen?
    if not url.startswith("http"):
        if url.startswith("/"):  # `/` or `//`
            # FIXME: `URL_LIST[-1]` may not be the same host as the current
            base_url = URL_LIST[-1]
            url = urljoin(base_url, url)
        else:
            return {"data": {"url": None, "content": None}, "nodes": None}

    URL_LIST.append(url)

    # Create an empty tree :math:`T`.
    tree = {"data": None, "nodes": None}

    # Starting from :math:`u`, get the web page HTML script.
    # Insert :math:`u` into the tree as root.
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "lxml")
    tree["data"] = {"url": url, "content": soup}

    if delta_ == 0:
        return tree

    tree["nodes"] = scoop(soup, delta_=delta_)
    return tree


def scoop(
    soup: BeautifulSoup,
    *,
    delta_,
    gamma=GAMMA,
):
    # Create new children not exceeding :math:`γ`.
    # From HTML script, get new URLs
    i_href = 0
    for anchor in soup.find_all("a"):
        # Apply until tree reaches depth :math:`δ`.
        if gamma < i_href:
            break

        href = anchor["href"]

        if is_duplicated(href):
            continue

        if not href.startswith('http') and not href.startswith('/'):
            continue

        logger.debug(href)
        try:
            yield construct_tree_from(
                    url=href,
                    delta_=delta_-1, gamma=gamma)
            i_href += 1
        except requests.exceptions.MissingSchema as ex:
            logger.error(ex)
        except requests.exceptions.InvalidSchema as ex:
            logger.error(ex)
        except requests.HTTPError as ex:
            logger.error(ex)


def is_duplicated(url, *, tree=None, url_list=URL_LIST, allowing_query=False):
    def compare_urls(url, url_, *, allowing_query=False):
        # NOTE: Optional URL components
        # (such as query & fragment) affect what page is shown.
        if not allowing_query:
            url_ = splitquery(url_)[0]
        return url == url_

    def with_tree(tree):
        if not tree:
            return False

        if tree["data"] is None:
            return False

        url_ = tree["data"]["url"]
        if compare_urls(url, url_):
            return True

        nodes = tree["nodes"]
        if not nodes:
            return False
        for subtree in nodes:
            if is_duplicated(url, tree=subtree):
                return True
        return False

    def with_url_list(url_list):
        if not url_list:
            return False

        for url_ in url_list:
            if compare_urls(url, url_):
                return True

        return False

    if tree:
        return with_tree(tree)
    elif url_list:
        return with_url_list(url_list)
    else:
        return False


def traversed(tree: dict) -> list:
    if not tree:
        raise ValueError("A tree node must have each value.")
    data, subtrees = tree["data"], tree["nodes"]
    if data:
        yield data
    if subtrees:
        for subtree in subtrees:
            yield from traversed(subtree)


def parsed(soup: BeautifulSoup, *, sigma=SIGMA):
    # FIXME: Fix the problem caused by parsing "Prof." "Dr." and so on.
    regex = re.compile(r"\!|\?|\.")

    paragraphs = soup.find_all("p", limit=sigma)
    for p in paragraphs:
        sentences: List[str] = regex.split(p.text)
        for s in sentences:
            if s.strip():
                yield s.strip()


def flatten(nodes, *, sigma=SIGMA, on_demand=False) -> List[Sentence]:
    if not on_demand:
        # TODO: Wrap a return value of `parsed`
        pass

    # Repeat until all nodes are traversed.
    for node in nodes:
        url, soup = node["url"], node["content"]

        # Now, traverse each node :math:`n` in :math:`T`
        # and derive maximum :math:`σ` sentences.
        # Add each sentence :math:`s` into :math:`\mathcal{X}`.
        sentences = list(parsed(soup, sigma=sigma))
        logger.debug({
            "url": url,
            "content": sentences,
        })
        yield sentences


def proceed_problem1(
        url: URL,
        delta_: uint = DELTA,
        gamma: uint = GAMMA,
        sigma: uint = SIGMA,
) -> List[Sentence]:
    r"""Create a web page tree to parse
    Arguments:
        url {URL} -- the URL of a web page
    Keyword Arguments:
        delta_ {uint} -- Max. depth of each tree (default: {DELTA})
        gamma {uint} -- Max. number of children of each node (default: {GAMMA})
        sigma {uint} -- Max. number of sentence of each web-page
            (default: {SIGMA})
    Returns:
        List[Sentence] -- Array of all derived sentences
    """
    # Create an empty tree :math:`T`.
    tree = construct_tree_from(
        url=url,
        delta_=delta_, gamma=gamma)
    nodes = traversed(tree)

    # Create an empty array :math:`\mathcal{X}`.
    sentences: List[Sentence] = list(
        itm for lst in flatten(nodes, sigma=sigma) for itm in lst)
    return sentences


# %%
@dataclasses.dataclass
class Tense:
    """Tense for English
    """
    present: str
    speech: str
    is_irregular: bool
    past: str
    perfect: str
    id: int


# %%
DICTIONARY_PATHS = [
    "WordNet/index.adj",
    "WordNet/index.adv",
    "WordNet/index.noun",
    "WordNet/index.verb",
    ]
EXCEPTIONAL_DICTIONARY_PATH = "WordNet/exc"
EDIT_DISTANCE_LIMIT = 12

try:
    stopwords.words
except LookupError as ex:
    logger.warning(ex)
    logger.warning("Downloading `stopwords`...")
    import nltk
    nltk.download("stopwords")
    del nltk
finally:
    STOPWORDS = set(stopwords.words('english'))


def load_dictionary() -> Dict[str, Tense]:
    # Load the structure.
    full_lines = []
    files = DICTIONARY_PATHS
    for f in files:
        with open(f, "r") as fi:
            full_lines.extend(fi.readlines())
    words = [w.split()[0].replace("_", " ") for w in full_lines]
    tags = [t.split()[1] for t in full_lines]
    structure = np.array([words, tags]).T.tolist()

    # Load exceptional structures for verbs.
    # NOTE: ["have", "had", "had"] is not contained in the dictionary.
    with open(EXCEPTIONAL_DICTIONARY_PATH, "r") as fi:
        verb_dict: Dict[List[str]] = dict([
            (line.split()[0], line.split())
            for line in fi.readlines() if line
            ])

    dictionary = {}
    for lst in structure:

        tense: Tense
        if lst[1] == 'v':  # Verb
            if lst[0] in verb_dict:
                verb = verb_dict[lst[0]]
                tense = Tense(
                    verb[0],
                    "v",
                    True,
                    verb[1],
                    verb[2],
                    None,
                )
            else:
                tense = Tense(
                    lst[0],
                    "v",
                    False,
                    lst[3] if 2 < len(lst) else "",
                    lst[4] if 2 < len(lst) else "",
                    None,
                )
        else:
            tense = Tense(
                lst[0],
                lst[1],
                False,
                lst[3] if 2 < len(lst) else "",
                lst[4] if 2 < len(lst) else "",
                None,
            )

        assert(tense)
        dictionary[tense.present] = tense

    # Assign each id to the word definitions
    for id_, item in enumerate(dictionary):
        dictionary[item].id = id_

    return dictionary


try:
    ENGLISH_WORD_WITH_PARAMETER_DICTIONARY
except NameError:
    ENGLISH_WORD_WITH_PARAMETER_DICTIONARY = load_dictionary()


def complete_edit_distance(
    word,
    english_dictionary=ENGLISH_WORD_WITH_PARAMETER_DICTIONARY
):
    if word in english_dictionary:
        return english_dictionary[word].present

    similar = sym_spell.lookup(word, Verbosity.ALL)
    if not similar:
        return None
    nearest = similar[0].term

    if nearest in english_dictionary:
        return english_dictionary[nearest].present

    return None


def proceed_problem2(
        sentence: Sentence,
        english_dictionary=ENGLISH_WORD_WITH_PARAMETER_DICTIONARY,
) -> List[int]:
    r"""Creation of a parse vector generated from the input.
    Arguments:
        sentence {Sentence} -- An English sentence from the web page tree
    Keyword Arguments:
        english_dictionary -- Dictionary of English words
            and their parameters, taken from open source libraries.
            (default: {ENGLISH_WORD_WITH_PARAMATER_DICTIONARY})
    Returns:
        List[int] -- Vector :math:`v` as :math:`{0, 1}
    """

    # Create an vector :math:`v` with 0's.
    # TODO: Initiate `vec` as `ndarray`
    vec = [0 for i in range(len(english_dictionary))]

    sentence = re.sub(f"[{string.punctuation}{string.digits}]", ' ', sentence)
    words = sentence.split()
    non_empty_words = (w for w in words if w and w not in STOPWORDS)
    for word in non_empty_words:
        # TODO: Use `word` as a key of the dictionary
        edited = complete_edit_distance(word, english_dictionary)

        if edited:  # not (None or empty string)
            vec[english_dictionary[edited].id] = 1

    # Match the index of tuple :math:`w` in :math:`D`,
    # replace :math:`v[i]` by 1.
    return vec


# %%
PATH_TO_DATASET = 'News_Category_Dataset_v2_new.json'
BATCH_COUNT = 200


def load_dataset(path_to_dataset=PATH_TO_DATASET) -> List[Dict[str, str]]:
    # TODO: Check the type of `dataset`
    with open(path_to_dataset, "r") as file:
        dataset = json.load(file)
    return dataset


try:
    NEWS_CATEGORY_DATASET
except NameError:
    NEWS_CATEGORY_DATASET = load_dataset()


def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath) -> SVM:
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    if isinstance(model, SVM):
        return model
    raise TypeError(
        f"The loaded object is not a {SVM}. type: {type(model)}")


N_EPOCHS = 5
N_GROUPS = 12
K_FOLDS = 3
SAMPLE_RATIO = 10


def cross_validation(
        dataset=NEWS_CATEGORY_DATASET,
        sample_ratio=SAMPLE_RATIO,
        verbose=1,
        n=N_GROUPS,
        k=K_FOLDS,
):

    model: SVM = SVM(tol=0.0001, verbose=verbose, loss='log')

    random.shuffle(dataset)
    dataset = dataset[:(len(dataset) // sample_ratio)]
    part_size = len(dataset) // n
    cluster_size = n // k

    for i in range(k):

        logger.debug(f"Clustering {i} starts.")

        dataset_train = dataset[
            i * part_size: i * part_size + cluster_size * part_size]
        dataset_test = [x for x in dataset if x not in dataset_train]

        logger.debug("The train part starts...")

        for j in range(0, len(dataset_train), BATCH_COUNT):
            # TODO: Initialize `X` and `y` as `ndarray`
            X = []
            y = []

            for c in dataset_train[j:j + BATCH_COUNT]:
                X.append(proceed_problem2(c['headline']))
                y.append(NewsCategory[c['category']].value)

            X = np.asarray(X)
            y = np.asarray(y)

            logger.debug(f"The {j + 1}(st|nd|rd|th) train iteration starts.")
            model.partial_fit(X, y, classes=range(1, len(NewsCategory) + 1))

        logger.debug("The train part ended.")

        logger.debug("The test part starts...")

        scores = []
        for j in range(0, len(dataset_test), BATCH_COUNT):
            X_test = []
            y_test = []
            for c in dataset_test[j: j + BATCH_COUNT]:
                X_test.append(proceed_problem2(c['headline']))
                y_test.append(NewsCategory[c['category']].value)

            X_test = np.asarray(X_test)
            y_test = np.asarray(y_test)

            logger.debug(f"The {j + 1}(st|nd|rd|th) test iteration starts.")
            scores.append(model.score(X_test, y_test))

        ave_score = np.average(scores)
        logger.debug(f"The total score is {ave_score}.")

        logger.debug("The test part ended.")


# TODO: Move `cross_validation` to another file
# cross_validation()


RATIO_OF_TRAINING_SET = 1.0
PERIOD_STORING_MODE = 1000
MODEL_FILE_EXTENSION = "svm.pickle"


def train_svm(
        dataset=NEWS_CATEGORY_DATASET,
        *,
        verbose=1,
        period_storing_model=PERIOD_STORING_MODE,
        checking_accuracy=True,
) -> SVM:

    model: SVM = SVM(tol=0.0001, verbose=verbose, loss='log')
    for i in range(N_EPOCHS):
        logger.debug(f"The training epoch {i + 1} starts.")
        random.shuffle(dataset)

        for j in range(0, len(dataset), BATCH_COUNT):
            # TODO: Initialize `X` and `y` as `ndarray`
            X = []
            y = []

            for c in dataset[j:j + BATCH_COUNT]:
                X.append(proceed_problem2(c['headline']))
                y.append(NewsCategory[c['category']].value)

            X = np.asarray(X)
            y = np.asarray(y)

            logger.debug(f"The training iteration {j + 1} starts.")
            model.partial_fit(X, y, classes=range(1, len(NewsCategory) + 1))
            logger.debug(f"The training iteration {j + 1} ended.")

            if checking_accuracy:
                logger.debug(f"Accuracy on this round: {model.score(X, y)}")

            if (j % period_storing_model) == 0:
                today = datetime.datetime.today().strftime(r"%Y%m%dT%H%M%S")
                save_model(
                    model, f"model_e{i}-i{j}_{today}.{MODEL_FILE_EXTENSION}")
                logger.debug("A model was saved successfully.")

        today = datetime.datetime.today().strftime(r"%Y%m%dT%H%M%S")
        save_model(model, f"model_e{i}_{today}.{MODEL_FILE_EXTENSION}")
        logger.debug("A model was saved successfully.")

        logger.debug(f"The training epoch {i + 1} ended.")

    logger.debug("A model training accomplished")

    return model


try:
    svm = load_model(f"model.{MODEL_FILE_EXTENSION}")
except IOError:
    logger.warning(
            (
                "Start training model "
                f"because of not found model.{MODEL_FILE_EXTENSION}"
            )
        )
    svm = train_svm()


def proceed_problem3(
        V,
        m: SVM = svm,
) -> int:
    r"""Classification of SVM

    Arguments:
        V {2d_array_like} -- Set of every vector :math:` v \in V`

    Keyword Arguments:
        m {SVM} -- A model of Support Vector Machine (default: {svm})

    Returns:
        int -- Category :math:`c`
    """

    # Feed each vector :math:`v \in V` to SVM.
    C = m.predict(V)

    return np.bincount(C).argmax()


# %%
def proceed_problem4(
        C,
) -> str:
    r"""Get the maximum occurrence count of news categories

    Arguments:
        C {1d_array_like} -- List of every category :math:`c \in C`

    Returns:
        str -- Category :math:`c`
    """
    # Count unique :math:`c` and return an array.
    return NewsCategory(np.bincount(C).argmax()).name


# %%
def main():
    N_URLS = 1
    categories = []
    for _ in range(N_URLS):
        vec_list = []
        url = input("Give a URL to me: ") or "https://buzzfeed.com"
        sentences = proceed_problem1(url)
        for sentence in sentences:
            vec_list.append(proceed_problem2(sentence))
        c = proceed_problem3(vec_list)
        categories.append(c)
    representative_category = proceed_problem4(categories)
    logger.debug(representative_category)


if __name__ == "__main__":
    main()
