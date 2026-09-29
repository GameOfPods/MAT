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

A book calls the same person "Lord Eddard Stark", "Eddard Stark", "Eddard" and "Ned". Without any model: titles are
dropped, and a name that is part of exactly one longer name joins it ("Eddard" -> "Eddard Stark"). A name that fits
several longer ones ("Stark") stays on its own, guessing would merge different people.

On top of that, `candidates()` finds what only understanding the text can decide: two names in one sentence that
say they are the same person ("Davos, den alle den Zwiebelritter nannten"), English nicknames ("Ned" and
"Edward"), and every sentence with an ambiguous short name. A judge (llm-characters) decides them, and `build()`
applies only what it confirmed. There is no list of aliases in MAT: static merging is left to the program that reads
the results, it gets every spelling with its counts.
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TITLES = {
    "mr", "mrs", "ms", "miss", "dr", "sir", "ser", "lord", "lady", "king", "queen", "prince", "princess", "maester",
    "captain", "herr", "frau", "fräulein", "könig", "königin", "prinz", "prinzessin", "fürst", "fürstin", "graf",
    "gräfin", "hauptmann", "meister", "the", "der", "die", "das",
}
_EDGES = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
_POSSESSIVE = re.compile(r"(?:['’]s|['’])$", re.IGNORECASE)

Key = Tuple[str, ...]


def clean(name: str) -> str:
    """The name as written, without quotes around it, a possessive at the end or doubled spaces."""
    name = " ".join(name.split())
    name = _POSSESSIVE.sub("", _EDGES.sub("", name))
    return _EDGES.sub("", name)


def key(name: str) -> Key:
    tokens = [token for token in clean(name).casefold().split() if token]
    without_titles = tuple(token for token in tokens if token.rstrip(".") not in TITLES)
    return without_titles or tuple(tokens)


@dataclass
class Mention:
    chapter: str
    name: str
    # the sentence it was found in, the judge needs it as evidence
    sentence: str = ""


@dataclass
class Clusters:
    """Mentions grouped by the rules: every key has a root, the root is the character."""
    mentions: List[Mention]
    keys: List[Key]  # per mention
    parent: Dict[Key, Key]
    roots: List[Key]
    # short keys that fit several roots ("stark"), with those roots
    ambiguous: Dict[Key, List[Key]] = field(default_factory=dict)

    def display(self, root: Key) -> str:
        """The most used spelling of the full name, not of a short form. On a tie the shorter one."""
        full = Counter(m.name for m, k in zip(self.mentions, self.keys) if k == root)
        if not full:
            full = Counter(m.name for m, k in zip(self.mentions, self.keys) if self.parent[k] == root)
        return max(full, key=lambda name: (full[name], -len(name)))

    def sentences(self, root: Key, limit: int = 5) -> List[str]:
        seen: List[str] = []
        for m, k in zip(self.mentions, self.keys):
            if self.parent[k] == root and m.sentence and m.sentence not in seen:
                seen.append(m.sentence)
                if len(seen) >= limit:
                    break
        return seen


def cluster(mentions: Iterable[Mention]) -> Clusters:
    kept = []
    for m in mentions:
        name = clean(m.name)
        if name:
            kept.append(Mention(chapter=m.chapter, name=name, sentence=m.sentence))
    keys = [key(m.name) for m in kept]
    # longest names first, every shorter one joins the single longer name it is part of
    roots: List[Key] = []
    parent: Dict[Key, Key] = {}
    ambiguous: Dict[Key, List[Key]] = {}
    for current in sorted(set(keys), key=lambda k: (-len(k), k)):
        containers = [root for root in roots if set(current) < set(root)]
        if len(containers) == 1:
            parent[current] = containers[0]
        else:
            roots.append(current)
            parent[current] = current
            if len(containers) > 1:
                ambiguous[current] = containers
    return Clusters(mentions=kept, keys=keys, parent=parent, roots=roots, ambiguous=ambiguous)


# ---------------------------------------------------------------- candidates for the judge

@dataclass
class PairCandidate:
    """Two characters that may be one person."""
    a: Key
    b: Key
    reason: str  # "pattern" or "nickname"
    sentences_a: List[str]
    sentences_b: List[str]
    together: List[str]


@dataclass
class MentionCandidate:
    """One occurrence of an ambiguous short name, and the characters it can belong to."""
    index: int  # into Clusters.mentions
    name: str
    sentence: str
    options: List[Key]


def _pattern(x: str, y: str) -> List[re.Pattern]:
    x, y = re.escape(x), re.escape(y)
    return [re.compile(p, re.IGNORECASE) for p in (
        rf"{x}\s*,\s*(?:den|die|das|dem)\s+(?:alle|man|sie|er|jeder)\s+(?:nur\s+)?(?:den\s+|die\s+)?{y}\s+nann?t",
        rf"{x}\s*,\s*(?:auch\s+)?genannt\s+{y}", rf"{x}\s*,\s*(?:auch\s+)?{y}\s+genannt",
        rf"{x}\s*,\s*(?:also\s+)?(?:called|known\s+as|nicknamed)\s+(?:the\s+)?{y}",
        rf"{x}\s*,\s*(?:whom|who)\s+(?:everyone|they|people|all)\s+called\s+(?:the\s+)?{y}",
        rf"{x}\s*\(\s*(?:the\s+|der\s+|die\s+)?{y}\s*\)",
    )]


def _nickname_pairs(names: Dict[Key, str]) -> Iterable[Tuple[Key, Key]]:
    """English given names and their nicknames (nicknames package), for example Edward and Ned. Only candidates:
    Al can be Albert or Alexander."""
    try:
        from nicknames import NickNamer
    except ImportError:
        return []
    namer = NickNamer()
    first = {root: root[0] for root in names}
    pairs = []
    roots = sorted(names)
    for i, a in enumerate(roots):
        nicks = {n.casefold() for n in namer.nicknames_of(first[a])} | {n.casefold() for n in
                                                                      namer.canonicals_of(first[a])}
        for b in roots[i + 1:]:
            if first[b] in nicks and first[a] != first[b]:
                pairs.append((a, b))
    return pairs


def candidates(clusters: Clusters, language: Optional[str] = None, max_pairs: int = 200,
               max_mentions: int = 300) -> Tuple[List[PairCandidate], List[MentionCandidate]]:
    """What the judge should look at. Nothing here merges anything."""
    names = {root: clusters.display(root) for root in clusters.roots}
    pairs: Dict[Tuple[Key, Key], PairCandidate] = {}

    def add(a: Key, b: Key, reason: str, together: List[str]):
        a, b = sorted((a, b))
        if a == b:
            return
        found = pairs.get((a, b))
        if found is None:
            pairs[(a, b)] = PairCandidate(a, b, reason, clusters.sentences(a), clusters.sentences(b), list(together))
        else:
            found.together.extend(s for s in together if s not in found.together)

    # two names of one sentence that the sentence says are the same person
    by_sentence: Dict[str, List[Tuple[str, Key]]] = {}
    for m, k in zip(clusters.mentions, clusters.keys):
        if m.sentence:
            by_sentence.setdefault(m.sentence, []).append((m.name, clusters.parent[k]))
    for sentence, found in by_sentence.items():
        for x_name, x_root in found:
            for y_name, y_root in found:
                if x_root != y_root and any(p.search(sentence) for p in _pattern(x_name, y_name)):
                    add(x_root, y_root, "pattern", [sentence])
    if (language or "en")[:2] == "en":
        for a, b in _nickname_pairs(names):
            add(a, b, "nickname", [s for s in by_sentence if {a, b} <= {r for _, r in by_sentence[s]}][:5])

    mentions = []
    for index, (m, k) in enumerate(zip(clusters.mentions, clusters.keys)):
        if k in clusters.ambiguous and m.sentence and len(mentions) < max_mentions:
            mentions.append(MentionCandidate(index, m.name, m.sentence, clusters.ambiguous[k]))
    return list(pairs.values())[:max_pairs], mentions


# ---------------------------------------------------------------- the list

def build(clusters: Clusters, min_mentions: int = 2, joins: Sequence[Tuple[Key, Key, str]] = (),
          resolved: Optional[Dict[int, Key]] = None) -> List[Dict]:
    """One entry per character, most mentioned first: {"name", "mentions", "variants", "chapters", "joined"}.
    joins are confirmed pairs (root, root, evidence), resolved maps mention indexes of ambiguous short names to the
    character they belong to."""
    resolved = resolved or {}
    group = {root: root for root in clusters.roots}

    def find(root: Key) -> Key:
        while group[root] != root:
            root = group[root]
        return root

    evidence: Dict[Key, List[Tuple[Key, str]]] = {}
    for a, b, quote in joins:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        group[rb] = ra
        evidence.setdefault(ra, []).append((b, quote))
        evidence[ra].extend(evidence.pop(rb, []))

    members: Dict[Key, List[int]] = {}
    for index, k in enumerate(clusters.keys):
        root = resolved.get(index, clusters.parent[k])
        members.setdefault(find(root), []).append(index)

    characters = []
    for top, indexes in members.items():
        if len(indexes) < min_mentions:
            continue
        # the group is named after its most mentioned character
        counts = Counter(clusters.parent[clusters.keys[i]] for i in indexes if i not in resolved)
        leader = max((r for r in clusters.roots if find(r) == top), key=lambda r: (counts.get(r, 0), len(r)))
        found = [clusters.mentions[i] for i in indexes]
        characters.append({
            "name": clusters.display(leader),
            "mentions": len(found),
            "variants": dict(Counter(m.name for m in found).most_common()),
            "chapters": dict(Counter(m.chapter for m in found)),
            "joined": [{"name": clusters.display(other), "evidence": quote} for other, quote in evidence.get(top, [])],
        })
    return sorted(characters, key=lambda c: (-c["mentions"], c["name"]))


def character_list(mentions: Sequence[Tuple[str, str]], min_mentions: int = 2) -> List[Dict]:
    """mentions are (chapter, name as written). Only the rules, no judge."""
    return build(cluster(Mention(chapter, name) for chapter, name in mentions), min_mentions=min_mentions)


__all__ = ["TITLES", "clean", "key", "Mention", "Clusters", "cluster", "PairCandidate", "MentionCandidate",
           "candidates", "build", "character_list"]
