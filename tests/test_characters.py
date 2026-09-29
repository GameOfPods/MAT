from MAT.utils.characters import character_list, clean


def test_clean_drops_quotes_possessives_and_spaces():
    assert clean(" “Eddard  Stark’s” ") == "Eddard Stark"
    assert clean("Davos'") == "Davos"
    assert clean("...") == ""


def test_short_names_join_the_one_long_name_they_belong_to():
    mentions = [("1", "Lord Eddard Stark"), ("1", "Eddard"), ("2", "Eddard Stark"), ("2", "Eddard"),
                ("2", "Arya Stark"), ("3", "Arya"), ("3", "Stark"), ("3", "Stark")]
    characters = {c["name"]: c for c in character_list(mentions, min_mentions=1)}
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
    characters = character_list(mentions, min_mentions=2)
    assert [(c["name"], c["mentions"]) for c in characters] == [("Davos Seaworth", 3), ("Stannis", 2)]
