# Proposal: nicknames, short names and coreference for character counts

Status: proposal, nothing built yet (2026-09-29). Part of the follow-up to stage 8.

## The problem

The character list (`MAT/utils/characters.py`) joins names only when that's safe without any understanding of the text:
titles are dropped ("Lord Eddard Stark" -> "Eddard Stark") and a short name joins the one full name it is part of
("Eddard" -> "Eddard Stark"). Three things are left:

1. **Ambiguous short names**: "Stark" fits Eddard, Arya, Sansa, ... and stays a character of its own.
2. **Nicknames and other names**: "Ned" for Eddard, "the Onion Knight" for Davos, "Reek" for Theon. Nothing in the
   string says they belong together.
3. **References without a name**: "he", "the king", "his father". A character's name is only a part of how often
   they appear.

The same happens in podcast transcripts: hosts say "Ned" and "Eddard", and `entity_counts()` counts them apart.

The rule for all of it: **a wrong join or a wrong reference is worse than a missing one.** Counting Theon's mentions
under Ramsay makes the list wrong, a missing "he" only makes it a bit low. Every step below has to prefer "unsure".

## What's out there (checked 2026-09)

| Tool | Languages | What it does for us | License, dependencies | Verdict |
|---|---|---|---|---|
| [BookNLP](https://github.com/booknlp/booknlp) | English | character name clustering, coreference, quotes, all for books | MIT, torch, spaCy | English only. The multilingual BookNLP project (German among the first languages) has no release yet. Worth a look for English books. |
| [nicknames](https://pypi.org/project/nicknames/) | English | ~1100 given names with their nicknames (Alexander -> Al, Alex) | Apache 2.0, no dependencies | Real-world names only: it can know Ned for Edward, but not for a made-up Eddard, and nothing German. Cheap first layer for English. |
| [nameparser](https://pypi.org/project/nameparser/) | English | splits a name into title, given, middle, family | LGPL | Nice for title handling, our title list does the same for both languages. |
| [coreferee](https://github.com/msg-systems/coreferee) 1.5 | en, de, fr, pl | coreference on top of spaCy | MIT, works with spaCy 3.8 | **Tried on 2026-09-29, too many wrong links** (Stannis = "woman", Stannis = "Frau"). Out. |
| [maverick-coref-de](https://github.com/uhh-lt/maverick-coref-de) (KONVENS 2025) | German | coreference, one model trained on German novels (DROC, 73.75 CoNLL F1, +9 over the previous best), ModernGBERT 1B with 8192 tokens of context | **CC BY-NC-SA 4.0**, pins `transformers==4.49` | Best German option. Non-commercial like DiariZen, and the pin clashes with our transformers, so it would go into its own environment (`envs/`, like DiariZen). Doesn't return confidence per link. Clustering is quadratic in mentions, the authors say a whole book only works when mentions are limited, so per chapter. |
| [maverick-coref](https://github.com/sapienzanlp/maverick-coref) | English | same architecture, 78.0 on LitBank | CC BY-NC-SA 4.0 | English counterpart, same caveats. |
| [fastcoref](https://pypi.org/project/fastcoref/) (LingMess) | English | fast coreference | MIT | English only, last release 2023, weaker than Maverick. |
| [Stanza](https://stanfordnlp.github.io/stanza/coref.html) 1.14 coref | en, de, 11 more | coreference trained on CorefUD, German included | Apache 2.0, Electra-large per language | The only permissive German option. Trained on news (PotsdamCC), not fiction. Needs a test. |
| CorPipe 26 (winner of CRAC 2026) | 19 languages | best multilingual coreference right now | research code, heavy | Too heavy to run next to everything else, not a library. |
| LLM coreference | any | the LLM track of CRAC 2026 was only 3 points behind the best system, with Gemma 3 27B and Qwen 3 14B | depends on the model | Too slow to resolve every pronoun of a book with a local 8B model, fine for checking a few candidates. |

Takeaway: no permissive, German capable, fiction trained coreference model exists. For names, the reliable tools are
rules, a list the user writes, and an LLM that has to show its evidence.

## Proposed design, in three layers

### Layer 1: aliases without a model (build first)

1. **An alias file per show or book series**, like `podcast.vocabulary`: `book.aliases` / `podcast.aliases`, one
   character per line.

   ```text
   # canonical name = other names, comma separated
   Eddard Stark = Ned, Lord Stark, Ned Stark
   Davos Seaworth = Onion Knight, Zwiebelritter
   Theon Graufreud = Reek, Stinker
   ```

   Written once per series and shared between the book pipeline and the podcast entities, it's the only way that
   is always right, and fan wikis have these lists ready. Entries from the file win over everything else.
2. **Pattern evidence in the text**: appositions that name a person twice in one sentence.
   `X, (den|die) (alle|man) Y nannte`, `X, called Y`, `Y, as X was known`, `X (Y)`. Only when X and Y are both PERSON
   entities in the same sentence. Such a pair is a candidate, not a join: it goes into layer 2.
3. **The `nicknames` package** for English given names that aren't story specific. Also only a candidate, because
   "Al" can be Alexander or Albert.

### Layer 2: an LLM decides about candidate pairs (build second)

Only for pairs that layer 1 couldn't decide: ambiguous short names ("Stark" -> which Stark?), pattern candidates and
nickname candidates. One call per batch of up to 20 pairs. The LLM sees the sentences where each name occurs, not the
whole book. Settings in their own section `[llm-characters]`, following `[llm]` until it has a preset of its own,
exactly like `[llm-names]`. Structured output (Ollama `format` with a JSON schema, OpenAI `response_format`) so a
small model can't answer in prose.

Prompt draft (system message):

```text
You decide whether two names in a book refer to the same character.

You are scored only on correct decisions. A wrong "same" is the worst answer: it merges two people in every count
that follows. "unsure" costs nothing. When in doubt, answer "unsure".

Rules:
- Decide only from the quoted sentences. Don't use what you know about the book, its sequels or adaptations. If the
  text doesn't say it, answer "unsure", even if you are sure from memory.
- "same" needs a sentence that shows it: both names for one person in one sentence, or a sentence that says one is
  the other's name or nickname. Quote that sentence.
- Two people with the same family name are not the same person.
- Answer with the JSON object described by the schema, nothing else.
```

User prompt per batch:

```text
Book language: {language}

{for each pair}
Pair {id}: "{name_a}" and "{name_b}"
Sentences with "{name_a}":
{up to 5 sentences}
Sentences with "{name_b}":
{up to 5 sentences}
Sentences with both:
{all, up to 5}
{end}

For every pair: {"pair": id, "answer": "same" | "different" | "unsure", "evidence": "quoted sentence or empty"}
```

Kept only when the answer is `same`, the evidence is quoted verbatim from the given sentences (checked in code, like
`llm-names` does), and a second call with name A and B swapped gives `same` again. The swap costs one more call and
catches answers that depend on order. Why "don't use what you know": "Reek is Theon" is a big spoiler for readers who
are at that chapter, and it's exactly what an LLM knows from training.

For ambiguous short names the pair becomes a choice: "Stark" in this sentence, is it Eddard, Arya, or unsure. Same
rules, per occurrence instead of per name, so that "Stark" in a chapter about Arya can count for Arya without
merging all "Stark" mentions.

Cost: a book has maybe 50 to 300 candidate pairs, so 5 to 30 calls of a few thousand tokens. Fine with qwen3:8b.

### Layer 3: references without a name (experiment only)

Counting "he" and "the king" needs coreference. Plan:

1. **Keep it apart.** Characters get `mentions` (names, as now) and `references` (coreference, new). Never one sum:
   whoever reads the result decides whether to trust the second number.
2. **Two independent systems have to agree**, since none of the models gives a confidence per link:
   - German: maverick-coref-de in its own environment (`envs/maverick-de`, non-commercial). English: maverick-coref.
     Stanza coref as the permissive second opinion in both languages.
   - A pronoun counts for a character only when both systems put it in a cluster whose only named character (after
     layers 1 and 2) is that character, within the same paragraph or the one before.
   - Clusters that contain two different character names are thrown away completely.
   - Gender and number have to agree with how the text refers to the character elsewhere (majority of "he"/"she"
     in that character's clusters).
3. **Measure before shipping.** Hand-check 2 chapters per language (every pronoun that got a character, and a random
   sample of the ones that didn't) and report precision. Merge only above 97 % precision. If it doesn't get there,
   the numbers stay in a report and nothing goes into the result.
4. **LLM as the check for the rest.** Pronouns where the two systems disagree can go to the LLM with the paragraph
   and the candidate characters, same scoring rules as above ("unsure" costs nothing). That's the expensive part:
   a novel has tens of thousands of pronouns, so only the disagreements, and only when the user turns it on.

Pronoun prompt draft:

```text
Below is a paragraph from a book. One pronoun or description is marked like [[this]].
Which character does it refer to? Candidates: {list of character names from this chapter}.

Answer "unsure" unless the paragraph leaves no other reading. A wrong answer is worse than "unsure".
Decide only from the paragraph and the sentence before it, not from what you know about the book.

{paragraph with one marker}

JSON: {"answer": "<one candidate>" | "unsure", "reason": "one short sentence"}
```

## Order of work

1. Alias file for books and podcast entities, pattern candidates, `nicknames` for English. No model, can't be wrong
   beyond what the user writes.
2. `[llm-characters]` with the pair judgement and the swap check. Test on one German and one English book against a
   hand-made alias list.
3. Coreference experiment: `envs/maverick-de`, Stanza, agreement rule, hand-checked precision. Only if that passes,
   `references` in the result format (added field, mat-format 2.x).

## Questions for you

- Is a non-commercial model acceptable for the book pipeline, like DiariZen for diarization? Maverick is the only
  German coreference model trained on novels.
- Should the alias file be one per series that both books and podcast episodes use? That fits Game of Pods, where the
  podcast talks about the same characters as the books.
</content>
</invoke>
