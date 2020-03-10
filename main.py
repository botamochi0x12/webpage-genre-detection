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
from urllib.parse import splitquery

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
    handler_2 = _logging.FileHandler("debug.log")
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

NewsCategory = enum.Enum("NewsCategory", "Default")
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
    print(ex, file=sys.stderr)

# %%
DELTA = 2
GAMMA = 3
SIGMA = 100

# NOTE:
# * List instances are mutable
# * Once `URL_LIST` is changed, it does never auto-reset.
#     So, please assign it an empty list manually.
URL_LIST = []


def construct_tree_from(
    url,
    *,
    delta_,
    gamma=GAMMA,
):
    if delta_ < 0:
        raise ValueError(
            f"Max. depth must be positive. (delta_ as depth < {delta_})")

    if(url.startswith('/')):
        url = URL_LIST[0] + url

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
            print(ex, file=sys.stderr)
        except requests.exceptions.InvalidSchema as ex:
            print(ex, file=sys.stderr)
        except requests.HTTPError as ex:
            print(ex, file=sys.stderr)


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
    # TODO: Fix the problem caused by parsing "Prof." "Dr." and so on.
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
        List[int] -- Vector :math:`v` as :math:`{-1, 1}
    """

    # Create an vector :math:`v` with 0's.
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


def load_dataset(path_to_dataset=PATH_TO_DATASET):
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


def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


RATIO_OF_TRAINING_SET = 1.0
N_EPOCHS = 5
PERIOD_STORING_MODE = 1000


def train_svm(
        dataset=NEWS_CATEGORY_DATASET,
        *,
        verbose=1,
        period_storing_model=PERIOD_STORING_MODE,
):

    model: SVM = SVM(tol=0.0001, verbose=verbose, loss='log')
    for i in range(N_EPOCHS):
        random.shuffle(dataset)
        for j in range(0, len(dataset), BATCH_COUNT):
            X = []
            y = []
            t = 0

            for c in dataset[j:j + BATCH_COUNT]:
                X.append(proceed_problem2(c['headline']))
                y.append(NewsCategory[c['category']].value)

            X = np.asarray(X)
            y = np.asarray(y)

            logger.debug(f"Iteration {j + 1} starts.")

            # ? can a stochastic gradient descend model train itself
            model.partial_fit(X, y, classes=range(1, len(NewsCategory) + 1))

            for x, y_ in zip(X, y):
                x = np.asarray([x])
                res = model.predict(x)
                if(res == y_):
                    t = t + 1

            logger.debug(f"Accuracy this round: {t / BATCH_COUNT}")

            if (j % period_storing_model) == 0:
                save_model(
                    model,
                    "{}-{}_{}.svm.pickle".format(
                        "model",
                        j,
                        datetime.datetime.today().strftime(r"%Y%m%dT%H%M%S")
                        )
                    )

    return model


try:
    svm = load_model("model.svm.pickle")
except IOError:
    svm = train_svm()


def proceed_problem3(
        V,
        m: SVM = svm,
) -> int:
    r"""Classification of SVM

    Arguments:
        V {List[List[bool]]} -- Set of every vector :math:` v \in V`

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
        C: List[int],
) -> str:
    r"""Get the maximum occurrence count of news categories

    Arguments:
        C {List[int]} -- List of every category :math:`c \in C`

    Returns:
        str -- Category :math:`c`
    """
    # Count unique :math:`c` and return an array.
    return NewsCategory(np.bincount(C).argmax()).name


# %%
def main():
    categories = []
    for _ in range(1):
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
