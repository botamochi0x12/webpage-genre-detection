# %%
import dataclasses
import enum
import json
import re
import string
import sys
import typing
from typing import Dict, List, Set
from urllib.parse import splitquery

import editdistance
import numpy as np
import requests
import sklearn.svm
from bs4 import BeautifulSoup

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
    unique_anchors = (
        a for a in soup.find_all("a") if not is_duplicated(a["href"]))

    # Create new children not exceeding :math:`γ`.
    # From HTML script, get new URLs
    for anchor in unique_anchors:
        # TODO: Fix below since each `tree` is local
        print(anchor["href"])
        try:
            # Apply until tree reaches depth :math:`δ`.
            yield construct_tree_from(
                    url=anchor["href"],
                    delta_=delta_-1, gamma=gamma)
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
        print({
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

    def __getitem__(self, i: int):
        # Deprecated method

        if i == 0:
            return self.present
        if i == 1:
            return self.speech
        if i == 2:
            return self.is_irregular
        if i == 3:
            return self.past
        if i == 4:
            return self.perfect
        if i == 5:
            return self.id


# %%
DICTIONARY_PATHS = [
    "WordNet/index.adj",
    "WordNet/index.adv",
    "WordNet/index.noun",
    "WordNet/index.verb",
    ]
EXCEPTIONAL_DICTIONARY_PATH = "WordNet/exc"
EDIT_DISTANCE_LIMIT = 12


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
    for id_, lst in enumerate(structure):

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
                    id_,
                )
            else:
                tense = Tense(
                    lst[0],
                    "v",
                    False,
                    lst[3] if 2 < len(lst) else "",
                    lst[4] if 2 < len(lst) else "",
                    id_,
                )
        else:
            tense = Tense(
                lst[0],
                lst[1],
                False,
                lst[3] if 2 < len(lst) else "",
                lst[4] if 2 < len(lst) else "",
                id_,
            )

        assert(tense)
        dictionary[tense.present] = tense

    return dictionary


ENGLISH_WORD_WITH_PARAMETER_DICTIONARY = load_dictionary()


def complete_edit_distance(
    word,
    english_dictionary=ENGLISH_WORD_WITH_PARAMETER_DICTIONARY
):
    if word in english_dictionary:
        return word

    min_dist = np.inf
    nearest = None
    for k, v in english_dictionary.items():
        dist = editdistance.eval(word, v[0])
        if(dist < min_dist and dist < EDIT_DISTANCE_LIMIT):
            min_dist = dist
            nearest = k
    if nearest is None:
        return None
    return english_dictionary[nearest][0]


def proceed_problem2(
        sentence: Sentence,
        english_dictionary=ENGLISH_WORD_WITH_PARAMETER_DICTIONARY,
) -> List[bool]:
    r"""Creation of a parse vector generated from the input.

    Arguments:
        sentence {Sentence} -- An English sentence from the web page tree

    Keyword Arguments:
        english_dictionary -- Dictionary of English words
            and their parameters, taken from open source libraries.
            (default: {ENGLISH_WORD_WITH_PARAMATER_DICTIONARY})

    Returns:
        List[bool] -- Vector :math:`v` as :math:`{0, 1}
    """

    # Create an vector :math:`v` with 0's.
    vec = [False for i in range(len(english_dictionary))]

    sentence = re.sub(f"[{string.punctuation}{string.digits}]", ' ', sentence)
    words = sentence.split()
    non_empty_words = (w for w in words if w)
    for word in non_empty_words:
        # TODO: Use `word` as a key of the dictionary
        edited = complete_edit_distance(word, english_dictionary)
        if edited and edited in english_dictionary:
            vec[english_dictionary[edited][5]] = True

    # Match the index of tuple :math:`w` in :math:`D`,
    # replace :math:`v[i]` by 1.
    return vec


# %%
KERNEL_FUNCTION = None
PATH_TO_DATASET = 'News_Category_Dataset_v2_new.json'


def load_dataset(path_to_dataset=PATH_TO_DATASET):
    with open(path_to_dataset, "r") as file:
        dataset = json.load(file)
    return dataset


NEWS_CATEGORY_DATASET = load_dataset()


def train_svm(dataset=NEWS_CATEGORY_DATASET, ratio_of_training_set=1.0):
    arr = [
        [NewsCategory[news['category']].value, news['headline']]
        for news in dataset
        ]

    for i in range(len(arr)):
        arr[i][1] = proceed_problem2(arr[i][1])
        if (i % 100) == 0:
            print(f"Step {i}")

    return arr


def proceed_problem3(
        V: List[List[bool]],
        C: Set[NewsCategory] = set(NewsCategory),
        f=KERNEL_FUNCTION,
) -> int:
    r"""Classification of SVM

    Arguments:
        V {List[List[bool]]} -- Set of every vector :math:` v \in V`

    Keyword Arguments:
        C {Set[NewsCategory]} -- Pre-processed categories
            from News Category Data-set (default: {set(NewsCategory)})
        f -- Kernel function (default: {KERNEL_FUNCTION})

    Returns:
        int -- Category :math:`c`
    """

    # Feed each vector :math:`v \in V` to SVM.
    svm = sklearn.svm.SVC()
    for v in V:
        break
    c: NewsCategory = NewsCategory.Default
    return c.value

# %%


def proceed_problem4(
        C: List[NewsCategory],
) -> str:
    r"""Get the maximum occurrence count of news categories

    Arguments:
        C {List[NewsCategory]} -- List of every category :math:`c \in C`

    Returns:
        str -- Category :math:`c` as string
    """
    # Count unique :math:`c` and return an array.
    occurrence_counts = dict((c, 0) for c in NewsCategory)
    for c in C:
        occurrence_counts[c] += 1

    # Select the maximum of the array.
    c: NewsCategory = max(occurrence_counts, key=occurrence_counts.get)
    return c.name


# %%
vec_list = []
url = input("Give a URL to me: ") or "https://buzzfeed.com"
urls_and_sentences = proceed_problem1(url)
for sentence in urls_and_sentences:
    vec_list.append(proceed_problem2(sentence))

# %%

x = train_svm()
