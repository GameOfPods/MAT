from MAT.utils.characters import character_list, clean


def test_clean_drops_quotes_possessives_and_spaces():
    assert clean(" “Eddard  Stark’s” ") == "Eddard Stark"
    assert clean("Davos'") == "Davos"
    assert clean("...") == ""


def test_short_names_join_the_one_long_name_they_belong_to():
    mentions = [("1", "Lord Eddard Stark"), ("1", "Eddard"), ("2", "Eddard Stark"), ("2", "Eddard"),
                ("2", "Arya Stark"), ("3", "Arya"), ("3", "Stark"), ("3", "Stark")]
    characters = {c["name"]: c for c in character_list(mentions, min_mentions=1, min_full=1)}
    eddard = characters["Eddard Stark"]
    assert eddard["mentions"] == 4
    assert eddard["variants"] == {"Eddard": 2, "Lord Eddard Stark": 1, "Eddard Stark": 1}
    assert eddard["chapters"] == {"1": 2, "2": 2}
    assert characters["Arya Stark"]["mentions"] == 2
    # Stark fits Eddard and Arya, so it stays on its own
    assert characters["Stark"]["mentions"] == 2


def test_rare_names_are_left_out_and_the_order_is_by_mentions():
    mentions = [("1", "Davos"), ("1", "Davos"), ("2", "Davos Seaworth"), ("2", "Melisandre"), ("3", "Stannis"),
                ("3", "Stannis")]
    characters = character_list(mentions, min_mentions=2, min_full=1)
    assert [(c["name"], c["mentions"]) for c in characters] == [("Davos Seaworth", 3), ("Stannis", 2)]


def _mentions(rows):
    from MAT.utils.characters import Mention

    # tiny books: a full name written once already takes in its short forms (cluster(..., min_full=1))
    return [Mention(chapter=chapter, name=name, sentence=sentence) for chapter, name, sentence in rows]


def test_a_sentence_that_names_one_person_twice_is_a_candidate_not_a_merge():
    from MAT.utils.characters import build, candidates, cluster

    sentence = "Davos, den alle den Zwiebelritter nannten, stand am Strand."
    clusters = cluster(_mentions([("1", "Davos", sentence), ("1", "Zwiebelritter", sentence),
                                  ("2", "Davos", "Davos schwieg."), ("2", "Zwiebelritter", "Der Zwiebelritter lachte.")]))
    pairs, _ = candidates(clusters, language="de")
    assert [(p.reason, clusters.display(p.a), clusters.display(p.b)) for p in pairs] == [
        ("pattern", "Davos", "Zwiebelritter")]
    assert pairs[0].together == [sentence]
    # without a judge both stay characters of their own
    assert sorted(c["name"] for c in build(clusters, min_mentions=1)) == ["Davos", "Zwiebelritter"]
    # a confirmed pair becomes one character, with the evidence
    joined = build(clusters, min_mentions=1, joins=[(pairs[0].a, pairs[0].b, sentence)])
    assert [(c["name"], c["mentions"]) for c in joined] == [("Davos", 4)]
    assert joined[0]["joined"] == [{"name": "Zwiebelritter", "evidence": sentence}]


def test_every_ambiguous_short_name_is_asked_about_on_its_own():
    from MAT.utils.characters import build, candidates, cluster

    clusters = cluster(_mentions([("1", "Arya Stark", "Arya Stark ran."), ("1", "Eddard Stark", "Eddard Stark sat."),
                                  ("2", "Stark", "Stark, said Arya, laughing."), ("2", "Stark", "The Stark lord sat.")]),
                       min_full=1)
    _, mentions = candidates(clusters)
    assert [(m.name, sorted(clusters.display(o) for o in m.options)) for m in mentions] == [
        ("Stark", ["Arya Stark", "Eddard Stark"]), ("Stark", ["Arya Stark", "Eddard Stark"])]
    arya = next(o for o in mentions[0].options if o[0] == "arya")
    result = {c["name"]: c for c in build(clusters, min_mentions=1, resolved={mentions[0].index: arya})}
    assert result["Arya Stark"]["mentions"] == 2 and result["Arya Stark"]["variants"] == {"Arya Stark": 1, "Stark": 1}
    assert result["Stark"]["mentions"] == 1


def test_english_nicknames_are_candidates():
    pytest = __import__("pytest")
    pytest.importorskip("nicknames")
    from MAT.utils.characters import candidates, cluster

    clusters = cluster(_mentions([("1", "Edward", "Edward came."), ("1", "Ned", "Ned left.")]))
    pairs, _ = candidates(clusters, language="en")
    assert [p.reason for p in pairs] == ["nickname"]


def test_pronouns_and_titles_alone_are_not_characters():
    mentions = [("1", "er"), ("1", "Er"), ("1", "Ich singe"), ("1", "Der König"), ("1", "Mylord"), ("1", "Aegon I"),
                ("2", "Aegon I"), ("2", "König Robb"), ("2", "Robb")]
    # König Robb is Robb once the title is gone, written both ways once, the shorter spelling names him
    characters = {c["name"]: c["mentions"] for c in character_list(mentions, min_mentions=1)}
    assert characters == {"Aegon I": 2, "Robb": 2}


def test_words_that_come_with_an_article_are_not_names():
    from MAT.utils.characters import Mention, build, cluster

    # "der Mann" and "sein Vater" nearly always have an article, a name like Cersei almost never. spaCy's proper noun
    # tag decides the rest ("der kleine Paul" is still a name)
    rows = [("Mann", False, True)] * 3 + [("Mann", False, False)] + [("Cersei", False, False)] * 3 + [
        ("Paul", True, True)] * 2
    clusters = cluster([Mention("1", name, proper=proper, article=article) for name, proper, article in rows])
    assert {c["name"]: c["mentions"] for c in build(clusters, min_mentions=1)} == {"Cersei": 3, "Paul": 2}
    assert clusters.not_names == {"Mann": 4}


def test_read_trims_verbs_and_pronouns_off_a_name():
    from MAT.utils.characters import read

    sentence = "Melisandre seufzte leise."
    words = [(0, 10, "PROPN", False, False), (11, 18, "VERB", False, False), (19, 24, "ADV", False, False),
             (24, 25, "PUNCT", False, False)]
    assert read("Melisandre seufzte", 0, 18, words) == ("Melisandre", True, False, False)
    assert read("seufzte leise", 11, 24, words) == ("", None, None, None)
    # a model that doesn't tag leaves the name as it is
    assert read("Melisandre seufzte", 0, 18, None) == ("Melisandre seufzte", None, None, None)
    assert sentence[0:10] == "Melisandre"


def test_groups_and_genitives():
    from MAT.utils.characters import Mention, build, cluster

    # "Robbs Männer zogen los": a group, not a character. "Joffreys" is Joffrey's without an apostrophe
    rows = [("Robbs Männer", True)] * 3 + [("Männer", True)] * 2 + [("Joffrey", False)] * 3 + [("Joffreys", False)]
    clusters = cluster([Mention("1", name, proper=True, article=False, plural=plural) for name, plural in rows])
    assert [(c["name"], c["mentions"], c["variants"]) for c in build(clusters, min_mentions=1)] == [
        ("Joffrey", 4, {"Joffrey": 3, "Joffreys": 1})]
    assert clusters.not_names == {"Robbs Männer": 3, "Männer": 2}


def test_a_rare_long_name_does_not_swallow_short_ones():
    # "fragte Arya Gendry": NER made one span of two people
    mentions = ([("1", "Arya Gendry")] * 2 + [("1", "Gendry")] * 5 + [("2", "Tywin Lennister")] * 3
                + [("2", "Tywin")] * 4)
    characters = {c["name"]: c["mentions"] for c in character_list(mentions, min_mentions=1)}
    assert characters == {"Gendry": 5, "Arya Gendry": 2, "Tywin Lennister": 7}
