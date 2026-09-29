# Research: NLP tools for MAT, state of 2026-09

Status: research, nothing built yet (2026-09-29). For every text task MAT has or could have: what we use, what
else exists, and whether an LLM is the right tool. The goal is to use small specialized models where they are good
enough and an LLM only where understanding is needed. Speech models are in the roadmap (stage 6), this is about text.

Constraints that shaped the verdicts: German and English, a GTX 1080 Ti (11 GB, no bfloat16), transformers is still
capped below 5 by gliner2, and non-commercial weights are acceptable but go into their own environment when their
dependencies clash (like DiariZen).

## Overview

| Task | Now | Proposal | LLM? |
|---|---|---|---|
| Sentences of a transcript | none (speaker turns) | SaT (wtpsplit) | no |
| Sentences and lemmas of a book | spaCy md | keep; simplemma only if lemmas are needed without spaCy | no |
| Named entities | GLiNER2 large | try GLiNER2 multi for German, keep GLiNER2 | no |
| Character names and nicknames | rules | alias file + rules + LLM for candidates, see [the character proposal](character-aliases-and-coreference.md) | only for candidates |
| Coreference | none | experiment, see the character proposal | only for disagreements |
| Keywords / topics of an episode | none | KeyBERT with a multilingual embedding model, or LLM tags | either, LLM is simpler |
| Topics across a season | none | BERTopic | no |
| Chapters (topic segments with times) of an episode | none | embeddings + TextTiling for boundaries, LLM for titles | titles only |
| Speaker names | LLM | keep, add structured output | yes |
| Summaries | LLM | keep | yes |
| Punctuation of lowercase transcripts | unused dependency | drop `deepmultilingualpunctuation` or use it for whisper output without punctuation | no |

## Sentence segmentation

**SaT, Segment any Text** ([wtpsplit](https://github.com/segment-any-text/wtpsplit), MIT, EMNLP 2024, release 2.2.2
on 2026-09-25) is the tool for transcripts. It was trained with punctuation and casing randomly removed, so it
splits ASR output that has none, and it has LoRA modules for TED-talk style speech in 81 languages, German included.
It beat LLMs on badly formatted text in the paper. Small models run on CPU; it needs `transformers>=4.22`, no clash.

Why we'd want it: NER, the summary and future topic segmentation all work on speaker turns now. A turn can be one word
or three minutes long. Sentences are a better unit, and whisper without a vocabulary sometimes writes everything in
lowercase without punctuation (the 30 s sample does), which spaCy can't split.

Books keep spaCy: the text has punctuation, and we need the lemmas anyway.

## Lemmas

spaCy md is fine. [simplemma](https://pypi.org/project/simplemma/) (MIT, dictionary based, release 2026-08) gives
lemmas for 50 languages without a model or a parser. Only interesting if lemma counts for transcripts come up, where
loading spaCy for that alone would be heavy.

## Named entities

We use GLiNER2 large (`fastino/gliner2-large-v1`). Findings:

- `fastino/gliner2-multi-v1` (Apache 2.0, 205M parameters, 6 languages, made for CPU) is already in the dev machine's
  cache. The large model was mostly trained on English. On German transcripts it called the Seven "PERSON", which is
  right for that show, but a proper comparison on German is missing. Test: 200 hand-checked German entities, large
  against multi.
- GLiNER-Relex (2026) does NER and relations in one model and beats GLiNER2 and GPT-5-mini on document level
  benchmarks. Relations ("Stannis is the brother of Robert") would feed a character graph later. Watch it, not now.
- Flair German NER (`ner-german-large`) and SpanMarker are strong on news with the fixed CoNLL labels (PER, LOC, ORG,
  MISC). We want labels we can describe, GLiNER stays.
- NER with an LLM: not needed. GLiNER2 is much cheaper and good enough; the LLM is better spent on deciding which
  names mean the same person.

## Coreference and character names

See [the character proposal](character-aliases-and-coreference.md). Short version: no permissive, German, fiction
trained model exists. maverick-coref-de (non-commercial, German novels) and Stanza (Apache 2.0, news) are the
candidates for an experiment where two systems have to agree, and an LLM decides only the few cases where rules and
models don't.

## Keywords and topics

- **Per episode**: [KeyBERT](https://github.com/MaartenGr/KeyBERT) (MIT) finds the phrases closest to the whole text
  in embedding space. Quality depends on the embedding model. For German the candidates are
  [BGE-M3](https://huggingface.co/BAAI/bge-m3) (strong German retrieval, 8192 tokens), multilingual-e5-large and
  EmbeddingGemma-300M (small, runs on CPU). An LLM asked for 10 topic tags with a JSON schema is simpler and usually
  better, and fine for a small local model because the answer is short.
- **Across a season**: [BERTopic](https://maartengr.github.io/BERTopic/) clusters episodes or segments into topics
  and tracks them over time. That's the one thing an LLM can't do alone, it needs all episodes at once. Candidate for
  Mosaicast rather than for MAT, because MAT sees one file at a time.

## Chapters of an episode

Chapter markers with times ("0:12:30 news, 0:18:00 chapter review") are something podcast apps show and that MAT
doesn't have yet. Research: TextTiling on sentence embeddings (TT-BERT), TreeSeg (hierarchical), PODTILE (Spotify,
LLM based), and 2026 work on multi-level segmentation.

Proposal: a hybrid. Boundaries from embeddings (SaT sentences, BGE-M3 or EmbeddingGemma vectors, TextTiling depth
scores, minimum chapter length 3 minutes), then one small LLM call per chapter for a title from its first and last
lines. The boundaries don't depend on the LLM, and the titles are a short job a qwen3:8b does well.

LLM-only alternative for comparison: map-reduce over 20 minute windows with the timestamps in the transcript,
asking for topic changes. Needs a lot more tokens and a bigger model to get the times right.

## Speaker names and summaries

Stay LLM jobs, there is no specialized tool for "who is Alex in this conversation" or for summaries.

Improvement for all LLM tasks: **structured output**. Ollama takes a JSON schema in `format` and then only produces
matching JSON; OpenAI takes `response_format` with `json_schema`. DeepSeek only has a JSON mode without a schema,
there we keep parsing and checking the answer as now. langchain has `with_structured_output` for both, so no new
library is needed ([instructor](https://python.useinstructor.com/integrations/ollama/) would add retries on
validation errors, but our checks already do that job). `llm-names` should use it first: its answer is JSON already.

## LLM settings per task

Every LLM job gets its own section that follows `[llm]` until it has a preset of its own, like `[llm-names]`:

| Section | Job | Model size that should work |
|---|---|---|
| `[llm]` | summaries, chapter summaries | 14B or an API (8B failed on long German episodes) |
| `[llm-names]` | speaker names | API or 14B (8B named one of three people wrong) |
| `[llm-characters]` | alias pairs, pronoun checks | 8B, short structured answers with evidence |
| `[llm-topics]` | topic tags, chapter titles | 8B |

On the GTX 1080 Ti, qwen3:14b at Q4 needs about 9 GB with a small context and doesn't fit next to anything else,
qwen3:8b about 6 GB. Since MAT's own models are unloaded before the LLM steps run, a 14B model works as long as
Ollama unloads it again before the next episode (`OLLAMA_KEEP_ALIVE=0`).

## Things to remove

- `deepmultilingualpunctuation` is a dependency of MAT but nothing uses it. Either drop it, or use it to restore
  punctuation when whisper returns a lowercase transcript without any (then it also helps SaT and NER).

## Suggested order

1. Structured output for `llm-names` (small, improves what exists).
2. GLiNER2 multi against large on German (a test, maybe a new default).
3. Alias file and `[llm-characters]` from the character proposal.
4. SaT sentences for transcripts, NER on sentences instead of turns.
5. Episode chapters (embeddings + LLM titles) and topic tags.
6. Coreference experiment.

## Sources

- [Segment any Text (wtpsplit)](https://github.com/segment-any-text/wtpsplit), [paper](https://aclanthology.org/2024.emnlp-main.665/)
- [GLiNER2 paper](https://arxiv.org/pdf/2507.18546), [gliner2-multi-v1](https://huggingface.co/fastino/gliner2-multi-v1), [GLiNER-Relex](https://arxiv.org/html/2605.10108v1)
- [Findings of the CRAC 2026 shared task](https://arxiv.org/html/2605.21369), [CorPipe at CRAC 2026](https://arxiv.org/abs/2605.30133)
- [maverick-coref-de, KONVENS 2025](https://aclanthology.org/2025.konvens-1.12.pdf), [maverick-coref](https://github.com/sapienzanlp/maverick-coref), [Stanza coreference](https://stanfordnlp.github.io/stanza/coref.html), [coreferee](https://github.com/msg-systems/coreferee), [BookNLP](https://github.com/booknlp/booknlp), [BookCoref](https://arxiv.org/html/2507.12075v1)
- [nicknames](https://pypi.org/project/nicknames/), [nameparser](https://pypi.org/project/nameparser/), [simplemma](https://pypi.org/project/simplemma/)
- [KeyBERT](https://github.com/MaartenGr/KeyBERT), [BERTopic](https://maartengr.github.io/BERTopic/), [German embedding models 2026](https://wz-it.com/en/blog/best-embedding-models-german/), [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [PODTILE](https://arxiv.org/pdf/2410.16148), [TreeSeg](https://arxiv.org/pdf/2407.12028), [multi-level transcript segmentation](https://arxiv.org/html/2601.02128v1), [unsupervised topic segmentation with BERT](https://arxiv.org/html/2106.12978v1)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
