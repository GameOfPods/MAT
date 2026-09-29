#  MAT - Toolkit to analyze media
#  Copyright (c) 2025.  RedRem95
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
"""
One list of characters per book from the PERSON entities of its chapters.

A book calls the same person "Lord Eddard Stark", "Eddard Stark", "Eddard" and "Ned". Titles are dropped, and a name
that is part of exactly one longer name joins it ("Eddard" -> "Eddard Stark"). A name that fits several longer ones
("Stark") stays on its own, guessing would merge different people. Nicknames like "Ned" need coreference or a list
by hand, see the roadmap.
"""
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

TITLES = {
    "mr", "mrs", "ms", "miss", "dr", "sir", "ser", "lord", "lady", "king", "queen", "prince", "princess", "maester",
    "captain", "herr", "frau", "fräulein", "könig", "königin", "prinz", "prinzessin", "fürst", "fürstin", "graf",
    "gräfin", "hauptmann", "meister", "the", "der", "die", "das",
}
_EDGES = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
_POSSESSIVE = re.compile(r"(?:['’]s|['’])$", re.IGNORECASE)


def clean(name: str) -> str:
    """The name as written, without quotes around it, a possessive at the end or doubled spaces."""
    name = " ".join(name.split())
    name = _POSSESSIVE.sub("", _EDGES.sub("", name))
    return _EDGES.sub("", name)


def key(name: str) -> Tuple[str, ...]:
    tokens = [token for token in clean(name).casefold().split() if token]
    without_titles = tuple(token for token in tokens if token.rstrip(".") not in TITLES)
    return without_titles or tuple(tokens)


def character_list(mentions: Sequence[Tuple[str, str]], min_mentions: int = 2) -> List[Dict]:
    """mentions are (chapter, name as written). Returns one entry per character, most mentioned first:
    {"name", "mentions", "variants": {spelling: count}, "chapters": {chapter: count}}."""
    by_key: Dict[Tuple[str, ...], List[Tuple[str, str]]] = {}
    for chapter, raw in mentions:
        name = clean(raw)
        if name:
            by_key.setdefault(key(name), []).append((chapter, name))

    # longest names first, every shorter one joins the single longer name it is part of
    roots: List[Tuple[str, ...]] = []
    parent: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
    for current in sorted(by_key, key=lambda k: (-len(k), k)):
        containers = [root for root in roots if set(current) < set(root)]
        if len(containers) == 1:
            parent[current] = containers[0]
        else:
            roots.append(current)
            parent[current] = current

    characters = []
    for root in roots:
        members = [k for k, p in parent.items() if p == root]
        found = [mention for k in members for mention in by_key[k]]
        if len(found) < min_mentions:
            continue
        # the most used spelling of the full name, not of a short form. On a tie the shorter one, without a title
        full = Counter(name for _, name in by_key[root])
        characters.append({
            "name": max(full, key=lambda name: (full[name], -len(name))),
            "mentions": len(found),
            "variants": dict(Counter(name for _, name in found).most_common()),
            "chapters": dict(Counter(chapter for chapter, _ in found)),
        })
    return sorted(characters, key=lambda c: (-c["mentions"], c["name"]))


__all__ = ["TITLES", "clean", "key", "character_list"]
