import itertools
import operator
import pydoc

from ._base import ConflictResolution, Metadata
from common.schemas import InferredSchema

import numpy as np
import pandera.typing as pt
import pandas as pd


class Argumentation(ConflictResolution):
    def forward(
        self,
        static: pt.DataFrame[InferredSchema],
        dynamic: pt.DataFrame[InferredSchema],
        probabilistic: pt.DataFrame[InferredSchema],
        metadata: Metadata,
    ) -> pt.DataFrame[InferredSchema] | None:

        if (df := self._basic_voting(static, dynamic, probabilistic, metadata)) is not None:
            return df

        elif (df := self._discussion(static, dynamic, probabilistic, metadata)) is not None:
            return df

        else:
            return None

    def _basic_voting(
        self,
        static: pt.DataFrame[InferredSchema],
        dynamic: pt.DataFrame[InferredSchema],
        probabilistic: pt.DataFrame[InferredSchema],
        metadata: Metadata,
    ) -> pt.DataFrame[InferredSchema] | None:
        # Basic voting procedure; If more than one approach predicts the same type,
        # then infer this type, and mark accordingly
        combined = pd.concat([static, dynamic, probabilistic], ignore_index=True)

        # Ignores NA
        anno_freqs = combined["anno"].value_counts(ascending=False)
        # print(anno_freqs)

        # More than one approach inferred just the one type
        # Others diverged individually
        if len(no_single_guesses := anno_freqs[anno_freqs > 1]) == 1:
            correct_inferrences = pd.merge(
                left=no_single_guesses,
                right=combined,
                how="inner",
                left_index=True,
                right_on="anno",
            )
            inferrer_naming = correct_inferrences["method"].unique()
            methodology = "+".join(inferrer_naming)

            return pt.DataFrame[InferredSchema](
                {
                    "method": [methodology],
                    "file": [metadata.file],
                    "category": [metadata.category],
                    "qname": [metadata.qname],
                    "anno": [correct_inferrences["anno"].iloc[0]],
                }
            )

        return None

    def _discussion(
        self,
        static: pt.DataFrame[InferredSchema],
        dynamic: pt.DataFrame[InferredSchema],
        probabilistic: pt.DataFrame[InferredSchema],
        metadata: Metadata,
    ) -> pt.DataFrame[InferredSchema] | None:
        """Perform n rounds of discussions, and select discussion with the most approvers"""

        G, profile, predictions = build_discussion_from_predictions(static, dynamic, probabilistic)
        candidates: list[tuple[str, int]] = []

        for target in predictions:
            G_target: nx.DiGraph = G.copy()

            for p in predictions:
                G_target.nodes[p]["label"] = const.IN if _subtyping(p, target) else const.OUT

            # TODO: pick correct aggregation function
            collective_labelling = compute_collective_labelling(G_target, profile, target, CF)
            decision = compute_collective_decision(collective_labelling, target)

            if decision == const.IN:
                candidates.append(
                    (target, sum(map(lambda node: node["label"] == const.IN, G_target.nodes)))
                )

        if not candidates:
            return None

        anno, _ = max(candidates, key=operator.itemgetter(1))

        return pt.DataFrame[InferredSchema](
            {
                "method": ["placeholder"],
                "file": [metadata.file],
                "category": [metadata.category],
                "qname": [metadata.qname],
                "anno": anno,
            }
        )


import networkx as nx


def build_discussion_from_predictions(
    static: pt.DataFrame[InferredSchema],
    dynamic: pt.DataFrame[InferredSchema],
    probabilistic: pt.DataFrame[InferredSchema],
) -> tuple[nx.DiGraph, list[dict[str, str]], list[str]]:
    """Returns discussion Graph, profile, unique predictions"""
    agents = lambda: itertools.chain([static, dynamic, probabilistic])

    combined = pd.concat(list(agents()), ignore_index=True)
    unique_predictions: list[str] = combined["anno"].unique().tolist()

    G = nx.DiGraph()
    G.add_nodes_from(unique_predictions, label=const.UNDEC)

    # if argument A is derived from argument B, then create defending edge from A -> B
    # as A can always be typed as B, but B cannot always be typed as A
    subtypes = list(
        filter(lambda ps: _subtyping(*ps), itertools.permutations(unique_predictions, r=2))
    )
    G.add_edges_from(subtypes, color=const.DEFENCE_COLOUR, label=const.DEFENCE)

    profile: list[dict[str, str]] = [
        {prediction: _agent_opinion(agent, prediction) for prediction in unique_predictions}
        for agent in agents()
    ]

    return G, profile, unique_predictions


## Edges denote defence or attacking between arguments.
## NOTE: arg1 and arg2 cannot be the same as all arguments are unique
## TODO: Use to capture subtyping
def _subtyping(arg1: str, arg2: str) -> bool:
    "Return True if arg1 should support arg2, i.e. arg1 is derived from from arg2"
    t1: type = pydoc.locate(arg1)
    t2: type = pydoc.locate(arg2)

    return t2 in t1.mro()


## Argument labellings are put forward to demonstrate agent opinions
def _agent_opinion(agent: pt.DataFrame[InferredSchema], pred: str) -> str:
    if pred in agent["anno"].values:
        return const.IN
    return const.OUT


### NOTE: All following source code was taken from
### NOTE: https://bitbucket.org/jariiia/argumentation-for-collective-decision-making/src/master/

__author__ = "jar"

import matplotlib.pyplot as plt
import networkx as nx


class const(object):
    __slots__ = ()
    ATTACK = -1
    DEFENCE = 1

    ATTACK_COLOUR = "r"
    DEFENCE_COLOUR = "g"

    IN = "IN"
    OUT = "OUT"
    UNDEC = "UNDEC"
    NULL = ""

    allLabels = [IN, OUT, UNDEC]

    IN_COLOUR = "g"
    OUT_COLOUR = "r"
    UNDEC_COLOUR = "y"

    LABEL_FONT_SIZE = 16
    NODE_SIZE = 600


const = const()


def attacking(G, argument):
    return [
        node
        for node in G
        if node in list(G.predecessors(argument)) and G[node][argument]["label"] == const.ATTACK
    ]


def defenders(G, argument):
    return [
        node
        for node in G
        if node in list(G.predecessors(argument)) and G[node][argument]["label"] == const.DEFENCE
    ]


def count_labels(G, labelling, arguments, label):
    counter = 0
    for a in arguments:
        if labelling[a] == label:
            counter = counter + 1
    return counter


def Pro(G, labelling, argument):
    d = defenders(G, argument)
    a = attacking(G, argument)
    return count_labels(G, labelling, d, const.IN) + count_labels(G, labelling, a, const.OUT)


def Con(G, labelling, argument):
    d = defenders(G, argument)
    a = attacking(G, argument)
    return count_labels(G, labelling, a, const.IN) + count_labels(G, labelling, d, const.OUT)


def direct_support(profile, argument, label):
    support = 0
    for labeling in profile:
        if labeling[argument] == label:
            support = support + 1
    return support


def coherent(G, labelling):
    is_coherent = True
    for argument in G:
        if labelling[argument] == const.IN:
            is_coherent = is_coherent and (
                Pro(G, labelling, argument) >= Con(G, labelling, argument)
            )
        if labelling[argument] == const.OUT:
            is_coherent = is_coherent and (
                Pro(G, labelling, argument) <= Con(G, labelling, argument)
            )
    return is_coherent


def cCoherent(G, labelling, c):
    is_coherent = True
    for argument in G:
        if labelling[argument] == const.IN:
            is_coherent = is_coherent and (
                Pro(G, labelling, argument) > Con(G, labelling, argument) + c
            )
        if labelling[argument] == const.OUT:
            is_coherent = is_coherent and (
                Pro(G, labelling, argument) < Con(G, labelling, argument) + c
            )
        if labelling[argument] == const.UNDEC:
            is_coherent = abs(Pro(G, labelling, argument) - Con(G, labelling, argument)) <= c
    return is_coherent


def getIndirectOpinion(G, labelling, argument):
    pros = Pro(G, labelling, argument)
    cons = Con(G, labelling, argument)
    if pros > cons:
        return 1
    elif pros < cons:
        return -1
    else:
        return 0


def get_direct_opinion(G, profile, argument):
    direct_positive_support = direct_support(profile, argument, const.IN)
    direct_negative_support = direct_support(profile, argument, const.OUT)
    if direct_positive_support > direct_negative_support:
        return 1
    elif direct_positive_support < direct_negative_support:
        return -1
    else:
        return 0


def Majority(G, profile, collective_labelling, argument):
    direct_positive_support = direct_support(profile, argument, const.IN)
    direct_negative_support = direct_support(profile, argument, const.OUT)
    if direct_positive_support > direct_negative_support:
        collective_labelling[argument] = const.IN
    elif direct_positive_support < direct_negative_support:
        collective_labelling[argument] = const.OUT
    else:
        collective_labelling[argument] = const.UNDEC


def OF(G, profile, collective_labelling, argument):
    pros = Pro(G, collective_labelling, argument)
    cons = Con(G, collective_labelling, argument)
    direct_positive_support = direct_support(profile, argument, const.IN)
    direct_negative_support = direct_support(profile, argument, const.OUT)
    if (direct_positive_support > direct_negative_support) or (
        (pros > cons) and (direct_positive_support == direct_negative_support)
    ):
        collective_labelling[argument] = const.IN
    elif (direct_positive_support < direct_negative_support) or (
        (pros < cons) and (direct_positive_support == direct_negative_support)
    ):
        collective_labelling[argument] = const.OUT
    else:
        collective_labelling[argument] = const.UNDEC


def SF(G, profile, collective_labelling, argument):
    pros = Pro(G, collective_labelling, argument)
    cons = Con(G, collective_labelling, argument)
    direct_positive_support = direct_support(profile, argument, const.IN)
    direct_negative_support = direct_support(profile, argument, const.OUT)
    if (pros > cons) or ((pros == cons) and (direct_positive_support > direct_negative_support)):
        collective_labelling[argument] = const.IN
    elif (pros < cons) or ((pros == cons) and (direct_positive_support < direct_negative_support)):
        collective_labelling[argument] = const.OUT
    else:
        collective_labelling[argument] = const.UNDEC


def CF(G, profile, collective_labelling, argument):
    indirect_opinion = getIndirectOpinion(G, collective_labelling, argument)
    direct_opinion = get_direct_opinion(G, profile, argument)
    aggregated_opinion = direct_opinion + indirect_opinion
    if aggregated_opinion > 0:
        collective_labelling[argument] = const.IN
    elif aggregated_opinion < 0:
        collective_labelling[argument] = const.OUT
    else:
        collective_labelling[argument] = const.UNDEC


def compute_collective_labelling(G, profile, target, AF):
    H = G.copy()
    collective_labelling = {}
    toVisit = [n for n in H if not (list(H.predecessors(n)))]
    while toVisit:
        argument = toVisit.pop(0)
        AF(G, profile, collective_labelling, argument)
        successors = list(H.successors(argument))
        for successor in successors:
            H.remove_edge(argument, successor)
            if not (list(H.predecessors(successor))):
                toVisit.append(successor)
    return collective_labelling


def compute_collective_decision(collective_labelling, argument):
    return collective_labelling[argument]


def draw_labelling(G, labelling, target, title):

    pos = nx.spring_layout(G)

    in_nodes = [node for node in G if labelling[node] == const.IN]
    out_nodes = [node for node in G if labelling[node] == const.OUT]
    undec_nodes = [node for node in G if labelling[node] == const.UNDEC]

    nx.draw_networkx_nodes(G, pos, in_nodes, node_size=const.NODE_SIZE, node_color=const.IN_COLOUR)

    nx.draw_networkx_nodes(
        G, pos, out_nodes, node_size=const.NODE_SIZE, node_color=const.OUT_COLOUR
    )

    nx.draw_networkx_nodes(
        G, pos, undec_nodes, node_size=const.NODE_SIZE, node_color=const.UNDEC_COLOUR
    )

    # edges
    nx.draw_networkx_edges(G, pos)

    attack_edges = [(u, v) for (u, v) in G.edges() if G[u][v]["color"] == const.ATTACK_COLOUR]
    nx.draw_networkx_edges(
        G, pos, attack_edges, width=4, alpha=0.25, edge_color=const.ATTACK_COLOUR
    )

    defence_edges = [(u, v) for (u, v) in G.edges() if G[u][v]["color"] == const.DEFENCE_COLOUR]
    nx.draw_networkx_edges(
        G, pos, defence_edges, width=4, alpha=0.25, edge_color=const.DEFENCE_COLOUR
    )

    labels = {}
    no_targets = [node for node in G if not node == target]
    for node in no_targets:
        labels[node] = r"$a" + str(node) + "$"
    labels[target] = r"$\tau$"

    nx.draw_networkx_labels(G, pos, labels, font_size=const.LABEL_FONT_SIZE)

    plt.title("Labeling : " + title)
    plt.axis("off")
    plt.show()


def draw_profile(G, profile, target, titles):

    iter_titles = iter(titles)

    for labelling in profile:

        in_nodes = [node for node in G if labelling[node] == const.IN]
        out_nodes = [node for node in G if labelling[node] == const.OUT]
        undec_nodes = [node for node in G if labelling[node] == const.UNDEC]

        plt.figure()

        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(
            G, pos, in_nodes, node_size=const.NODE_SIZE, node_color=const.IN_COLOUR
        )

        nx.draw_networkx_nodes(
            G, pos, out_nodes, node_size=const.NODE_SIZE, node_color=const.OUT_COLOUR
        )

        nx.draw_networkx_nodes(
            G, pos, undec_nodes, node_size=const.NODE_SIZE, node_color=const.UNDEC_COLOUR
        )

        # edges
        nx.draw_networkx_edges(G, pos)

        attack_edges = [(u, v) for (u, v) in G.edges() if G[u][v]["color"] == const.ATTACK_COLOUR]
        nx.draw_networkx_edges(
            G, pos, attack_edges, width=4, alpha=0.25, edge_color=const.ATTACK_COLOUR
        )

        defence_edges = [(u, v) for (u, v) in G.edges() if G[u][v]["color"] == const.DEFENCE_COLOUR]
        nx.draw_networkx_edges(
            G, pos, defence_edges, width=4, alpha=0.25, edge_color=const.DEFENCE_COLOUR
        )

        labels = {}
        noTargets = [node for node in G if not node == target]
        for node in noTargets:
            labels[node] = r"$a" + str(node) + "$"
        labels[target] = r"$\tau$"

        nx.draw_networkx_labels(G, pos, labels, font_size=const.LABEL_FONT_SIZE)

        plt.title("Labeling : " + next(iter_titles))
        plt.axis("off")
    plt.show()
