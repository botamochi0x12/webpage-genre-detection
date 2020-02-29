# %%
import csv
import enum
import re
import sys
import typing
import numpy as np
import editdistance
from typing import List, Set
from urllib.parse import splitquery

import requests
from bs4 import BeautifulSoup

uint = typing.NewType("unsigned_int", int)
URL = typing.NewType("URL", str)
Sentence = typing.NewType("Sentence", str)

NewsCategory = enum.Enum("NewsCategory", "Default")
try:
    with open("categories.csv") as f:
        NewsCategory = enum.Enum(
            "NewsCategory", [row[0] for row in csv.reader(f) if row[0]])
except OSError as ex:
    print(ex, file=sys.stderr)

# %%
DELTA = 2
GAMMA = 4
SIGMA = 100
URL_LIST = []
# NOTE:
# * List instances are mutable
# * Once `URL_LIST` is changed, it does never auto-reset.
#     So, please assign it an empty list manually.



def construct_tree_from(
    url,
    *,
    delta_,
    gamma=GAMMA,
):
    if delta_ < 0:
        raise ValueError(
            f"Max. depth must be positive. (delta_ as depth < {delta_})")

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
    for anchor in soup.find_all("a", limit=gamma):
        # TODO: Fix below since each `tree` is local
        print(anchor["href"])
        if is_duplicated(anchor["href"]):
            print("pass")
            continue
        try:
            # Apply until tree reaches depth :math:`δ`.
            yield construct_tree_from(
                    url=anchor["href"],
                    delta_=delta_-1, gamma=gamma)
        except requests.exceptions.MissingSchema as ex:
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


def flatten(nodes, *, sigma=SIGMA, on_demand=False):
    if not on_demand:
        # TODO: Wrap a return value of `parsed`
        pass

    # Repeat until all nodes are traversed.
    for node in nodes:
        url, soup = node["url"], node["content"]

        # Now, traverse each node :math:`n` in :math:`T`
        # and derive maximum :math:`σ` sentences.
        # Add each sentence :math:`s` into :math:`\mathcal{X}`.
        yield {
            "url": url,
            "content": list(parsed(soup, sigma=sigma)),
        }


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
    sentence_list: List[Sentence] = list(flatten(nodes, sigma=sigma))

    return sentence_list


# %%

def load_dictionary():
    full_lines = []
    # Load the structure.
    files = ["WordNet/index.adj", "WordNet/index.adv", "WordNet/index.noun", "WordNet/index.verb"]
    for f in files:
        fi = open(f, "r")
        for x in fi:
            full_lines.append(x)
        fi.close()
    words = [w.split(" ")[0] for w in full_lines]
    tags = [t.split(" ")[1] for t in full_lines]
    words = [w.replace("_", " ") for w in words]
    structure = np.array([words, tags]).T.tolist()
    
    # Load exceptional structures for verbs.
    verbs = []
    fi = open("WordNet/exc", "r")
    for x in fi:
        verbs.append(x)
    verbs = [w.split(" ") for w in verbs]
    for i in range(len(structure)):
        for j in range(len(verbs)):
            if(structure[i][0] == verbs[j][0] and structure[i][1] == 'v'):
                if(len(structure[i]) < 3):
                    structure[i].append('True')
                    structure[i].append(verbs[j][1])
                    structure[i].append(verbs[j][2])
                else:
                    structure[i][2] = 'True'
                    structure[i][3] = verbs[j][1]
                    structure[i][4] = verbs[j][2]
            else:
                if(len(structure[i]) < 3):
                    structure[i].append('False')
                    structure[i].append("")
                    structure[i].append("")
    return structure
        
ENGLISH_WORD_WITH_PARAMETER_DICTIONARY = load_dictionary()

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
    
    def edit_distance(w, english_dictionary):
        for e in english_dictionary:
            d = editdistance.eval(w, e[0])
            print(d)
        return w
    
    # Create an vector :math:`v` with 0's.

    vec = [False for i in range(len(english_dictionary))]
    for i in range(len(sentence['content'])):
        partition = sentence['content'][i].split(' ')
        for j in range(len(partition)):
            for k in range(len(english_dictionary)):
                if(vec[k] == False):
                    if(partition[j] == english_dictionary[k][0]):
                        vec[k] = True

    # Match the index of tuple :math:`w` in :math:`D`,
    # replace :math:`v[i]` by 1.
    return vec


# %%
KERNEL_FUNCTION = None


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
    svm: typing.Any
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


vecs = []
url = input("Give a URL to me: ")
sentences = proceed_problem1(url)
for sentence in sentences:
    vecs.append(proceed_problem2(sentence))