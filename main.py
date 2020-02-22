# %%
import enum
import re
import sys
import typing
from typing import List, Set

import requests
from bs4 import BeautifulSoup

uint = typing.NewType("unsigned_int", int)
URL = typing.NewType("URL", str)
Sentence = typing.NewType("Sentence", str)
Table = typing.Dict


class NewsCategory(enum.Enum):
    Default = enum.auto()


# %%
DELTA = 3
GAMMA = 3
SIGMA = 3


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
    def construct_tree_from(url, *, delta_, gamma=gamma):
        if delta_ < 0:
            raise ValueError(
                f"Max. depth delta_ must be positive. (delta_ = {delta_})")

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

        nodes = []
        for anchor in soup.find_all("a", limit=gamma):
            try:
                # From HTML script, get new URLs
                # Apply until tree reaches depth :math:`δ`.
                nodes.append(construct_tree_from(
                    url=anchor["href"], delta_=delta_-1, gamma=gamma))
            except requests.exceptions.MissingSchema as ex:
                print(ex, file=sys.stderr)
            except requests.HTTPError as ex:
                print(ex, file=sys.stderr)

        # Create new children not exceeding :math:`γ`.
        tree["nodes"] = nodes
        return tree

    def traversed(tree: dict) -> list:
        if not tree:
            raise ValueError("A tree node must have each value.")
        data, subtrees = tree["data"], tree["nodes"]
        if data:
            yield data
        if subtrees:
            for subtree in subtrees:
                yield from traversed(subtree)

    def parsed(soup: BeautifulSoup, *, sigma=sigma):
        regex = re.compile(r"\!|\?|\.")
        # TODO: Fix the problem caused by parsing "Prof." "Dr." and so on.
        paragraphs = soup.find_all("p", limit=sigma)
        for p in paragraphs:
            sentences: List[str] = regex.split(p.text)
            for s in sentences:
                if s.strip():
                    yield s.strip()

    # Create an empty tree :math:`T`.
    tree = construct_tree_from(url=url, delta_=delta_, gamma=gamma)
    nodes = traversed(tree)

    # Create an empty array :math:`\mathcal{X}`.
    sentence_list: List[Sentence] = list()

    for node in nodes:
        url, soup = node["url"], node["content"]

        # Now, traverse each node :math:`n` in :math:`T`
        # and derive maximum :math:`σ` sentences.
        # Add each sentence :math:`s` into :math:`\mathcal{X}`.
        sentence_list.append({
            "url": url,
            "content": list(parsed(soup, sigma=sigma)),
        })
    # Repeat until all nodes are traversed.

    return sentence_list


# %%
ENGLISH_WORD_WITH_PARAMATER_DICTIONARY = None


def proceed_problem2(
        sentence: Sentence,
        english_dictionary=ENGLISH_WORD_WITH_PARAMATER_DICTIONARY,
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
    vec: List[bool] = [False for i in range(
        len(english_dictionary))]

    # For each :math:`s` in :math:`S`, match :math:`s`
    # with a tuple :math:`w` in :math:`D` or ``null``.
    vec = [sentence == w for w in english_dictionary]

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
        C: Table[NewsCategory, int],
) -> str:
    r"""Get the maximum occurrence count of news categories

    Arguments:
        C {Table[NewsCategory, int]} -- Set of every category :math:`c \in C`

    Returns:
        str -- Category :math:`c` as string
    """
    # Count unique :math:`c` and return an array.

    # Select the maximum of the array.
    c: NewsCategory = NewsCategory.Default
    return c.name
