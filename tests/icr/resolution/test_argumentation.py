import pprint
from common.schemas import TypeCollectionCategory, InferredSchema
from icr.resolution.argumentation import (
    Majority,
    OF,
    SF,
    CF,
    build_discussion_from_predictions,
    compute_collective_decision,
    compute_collective_labelling,
    draw_profile,
    const,
)

import pandera.typing as pt


class IntSubclass(int):
    ...


def test_discussion_building():
    sc_qualname = IntSubclass.__module__ + "." + IntSubclass.__qualname__

    # static inference infers type int
    static = pt.DataFrame[InferredSchema](
        {
            "method": ["static"],
            "file": ["x.py"],
            "category": [TypeCollectionCategory.VARIABLE],
            "qname": ["x"],
            "anno": ["int"],
        }
    )

    # dynamic inference infers bool
    dynamic = pt.DataFrame[InferredSchema](
        {
            "method": ["dynamic"],
            "file": ["x.py"],
            "category": [TypeCollectionCategory.VARIABLE],
            "qname": ["x"],
            "anno": ["bool"],
        }
    )

    # probabilistic inference infers top-3, including subtype of int in place 2
    prob = pt.DataFrame[InferredSchema](
        {
            "method": ["prob"] * 3,
            "file": ["x.py"] * 3,
            "category": [TypeCollectionCategory.VARIABLE] * 3,
            "qname": ["x"] * 3,
            "anno": ["str", sc_qualname, "float"],
        }
    )

    Gs, profile = build_discussion_from_predictions(
        static=static, dynamic=dynamic, probabilistic=prob
    )

    """  ### static
    static_profile = profile[0].copy()
    # Support own prediction
    assert static_profile.get("int") == const.IN
    # Disregard all other predictions
    static_profile.pop("int")
    assert all(v == const.OUT for v in static_profile.values())

    ### dynamic
    dyn_profile = profile[1].copy()
    # Support own prediction
    assert dyn_profile.get("bool") == const.IN
    # Disregard all other predictions
    dyn_profile.pop("bool")
    assert all(v == const.OUT for v in dyn_profile.values())

    ### probabilistic
    prob_profile = profile[2].copy()
    # Support own prediction
    for pred in prob["anno"].values:
        assert prob_profile.get(pred) == const.IN
        # Disregard all other predictions
        prob_profile.pop(pred)

    assert all(v == const.OUT for v in prob_profile.values()) """

    # Majority only looks at const.INs and const.OUTs

    for agg in (Majority, OF, SF, CF):
        for target, G in Gs.items():
            #if target == sc_qualname:
            collective_labelling = compute_collective_labelling(G, profile, target, agg)
            decision = compute_collective_decision(collective_labelling, target)

            print(f"{target=}, {agg.__name__=}, {decision=}")

            profiles = [collective_labelling]
            pprint.pprint(profiles)

            # all_titles = [
            #     fr"static - $\tau$={target}",
            #     fr"dynamic - $\tau$={target}",
            #     fr"prob - $\tau$={target}",
            #     fr"Collective Decision - $\tau$={target} under {agg.__name__}",
            # ]

            # draw_profile(G, profiles, target, all_titles)
            break