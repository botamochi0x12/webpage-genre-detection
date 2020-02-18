# %%
import enum
import re
import typing
from typing import List, Set

import requests
from bs4 import BeautifulSoup

uint = typing.NewType("unsigned int", int)
URL = typing.NewType("URL", str)
Sentense = typing.NewType("Sentense", str)
Table = typing.Dict

class NewsCategory(enum.Enum):
    Default = enum.auto()

# %%
DELTA = 1
GAMMA = 1
SIGMA = 3
def proceed_problem1(
    url: URL, 
    delta_: uint = DELTA, 
    gamma: uint = GAMMA, 
    sigma: uint = SIGMA,
    ) -> List[Sentense]:
    """Create a web page tree to parse
    
    Arguments:
        url {URL} -- the URL of a web page
    
    Keyword Arguments:
        delta_ {uint} -- Max. depth of each tree (default: {DELTA})
        gamma {uint} -- Max. number of children of each node (default: {GAMMA})
        sigma {uint} -- Max. number of sentense of each web-page (default: {SIGMA})
    
    Returns:
        List[Sentense] -- Array of all derived sentences
    """    
    def construct_tree_from(url, delta_):
        if delta_ < 0: 
            raise ValueError(
                f"Max. depth delta_ must be positive. (delta_ = {delta_})")

        # Create an empty tree :math:`T`.
        tree = {"data": None, "nodes": None}

        # Starting from :math:`u`, get the web page HTML script.
        # Insert :math:`u` into the tree as root.
        response = requests.get(url)
        if 200 != response.status_code:
            return tree
        soup = BeautifulSoup(response.content, "lxml")
        tree["data"] = {"url": url, "content": soup}

        if 0 == delta_:
            return tree

        nodes = []
        for anchor, __ in zip(soup.find_all("a"), range(gamma)):
            # From HTML script, get new URLs
            nodes.append(construct_tree_from(url=anchor["href"], delta_=delta_-1))
            # Apply until tree reaches depth :math:`δ`.
        # Create new children not exceeding :math:`γ`.
        tree["nodes"] = nodes
        return tree

    def traversed(tree: dict) -> list:
        print(tree)
        if not tree:
            raise ValueError("A tree node must have each value.")
        data, subtree = tree["data"], tree["nodes"]
        if not subtree:
            return data
        return [data] + [traversed(node) for node in subtree]

    def parsed(paragraphs: str):
        regex = re.compile(r"\?|\.")
        # TODO: Fix the problem caused by parsing ones such as "Prof." and "Dr."
        return (regex.split(p.text).strip() for p in paragraphs)
    # Create an empty tree :math:`T`.
    tree = construct_tree_from(url=url, delta_=delta_)
    nodes = traversed(tree)
    print(tree)

    # Create an empty array :math:`\mathcal{X}`.
    sentense_list: List[Sentense] = list()

    for node in nodes:
        url, soup = node["url"], node["content"]
        # response = requests.get(url)
        # soup = BeautifulSoup(response.content, "lxml")
        paragraphs = soup.find_all("p")
        sentense_list.extend(s for s, __ in zip(parsed(paragraphs), range(sigma)))
    # Now, traverse each node :math:`n` in :math:`T`
    # and derive maximum :math:`σ` sentences.
    # Add each sentence :math:`s` into :math:`\mathcal{X}`.
    # Repeat until all nodes are traversed.

    return sentense_list

# %%
ENGLISH_WORD_WITH_PARAMATER_DICTIONARY = None
def proceed_problem2(
    sentense: Sentense,
    english_word_with_paramater_dictionary = ENGLISH_WORD_WITH_PARAMATER_DICTIONARY,
    ) -> List[bool]:
    """Creation of a parse vector generated from the input.
    
    Arguments:
        sentense {Sentense} -- An English sentence from the web page tree
    
    Keyword Arguments:
        english_word_with_paramater_dictionary -- Dictionary of English words 
            and their parameters, taken from open source libraries. 
            (default: {ENGLISH_WORD_WITH_PARAMATER_DICTIONARY})
    
    Returns:
        List[bool] -- Vector :math:`v` as :math:`{0, 1}
    """
    # Create an vector :math:`v` with 0's.
    v: List[bool] = [False for i in range(len(english_word_with_paramater_dictionary))]

    # For each :math:`s` in :math:`S`, match :math:`s`
    # with a tuple :math:`w` in :math:`D` or ``null``.
    v = [sentense == w for w in english_word_with_paramater_dictionary]

    # Match the index of tuple :math:`w` in :math:`D`,
    # replace :math:`v[i]` by 1.
    return v

# %%
KERNEL_FUNCTION = None
def proceed_problem3(
    V: List[List[bool]],
    C: Set[NewsCategory] = set(NewsCategory),
    f = KERNEL_FUNCTION,
    ) -> int:
    """Classification of SVM
    
    Arguments:
        V {List[List[bool]]} -- Set of every vector :math:` v \in V`
    
    Keyword Arguments:
        C {Set[NewsCategory]} -- Pre-processed categories from News Category Data-set (default: {set(NewsCategory)})
        f -- Kernel function (default: {KERNEL_FUNCTION})
    
    Returns:
        int -- Category :math:`c`
    """    
        
    # Feed each vector :math:`v \in V` to SVM.
    svm = None
    for v in V:
        break
    c: NewsCategory = NewsCategory.Default
    return int(c)

# %%
def proceed_problem4(
    C: Table[NewsCategory],
    ) -> str:
    """Get the maximum occurrence count of news categories
    
    Arguments:
        C {Table[NewsCategory]} -- Set of every category :math:`c \in C`
    
    Returns:
        str -- Category :math:`c` as string
    """    
    # Count unique :math:`c` and return an array.

    # Select the maximum of the array.
    c: NewsCategory = NewsCategory.Default
    return c.name
