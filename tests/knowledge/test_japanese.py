# Tests for the rule-based conjugator, the romaji converter and the own tools backed by the local data.

import json

import pytest

from sensai.knowledge import japanese
from sensai.knowledge.tools import conjugate_verb, lookup_dictionary, lookup_kanji, registry


@pytest.mark.parametrize("verb, form, expected", [
    ("食べる", "te", "食べて"), ("食べる", "past_negative_polite", "食べませんでした"),
    ("飲む", "te", "飲んで"), ("書く", "ta", "書いた"), ("泳ぐ", "te", "泳いで"), ("話す", "te", "話して"),
    ("待つ", "volitional", "待とう"), ("買う", "nai", "買わない"), ("遊ぶ", "ta", "遊んだ"),
    ("行く", "te", "行って"), ("行く", "tara", "行ったら"), ("帰る", "polite", "帰ります"),
    ("入る", "nai", "入らない"), ("ある", "nai", "ない"), ("ある", "nakatta", "なかった"),
    ("来る", "nai", "来ない"), ("来る", "te", "来て"), ("する", "potential", "できる"),
    ("勉強する", "polite_past", "勉強しました"), ("見る", "potential", "見られる"), ("読む", "ba", "読めば"),
])
def test_conjugations_follow_the_verb_class(verb, form, expected):
    assert conjugate_verb(verb, form)["result"] == expected


def test_a_conjugation_comes_with_its_reading_and_romaji():
    result = conjugate_verb("来る", "nai")
    assert (result["result_reading"], result["romaji"], result["class_source"]) == ("こない", "konai", "dictionary")


def test_romaji_and_kana_spellings_find_the_verb():
    assert conjugate_verb("taberu", "masu")["result"] == "食べます"
    assert conjugate_verb("かえる", "te")["result"] == "帰って"


def test_an_unknown_kana_verb_is_conjugated_from_its_ending_and_says_so():
    result = conjugate_verb("ほめる", "te")
    assert result["result"] == "ほめて" and result["class_source"] == "guessed from the ending"
    assert conjugate_verb("のぼる", "te")["result"] == "のぼって"


@pytest.mark.parametrize("args, message", [
    (("猫", "te"), "not a verb"), (("cat", "te"), "not a verb"), (("食べる", "subjunctive"), "unknown form"),
])
def test_conjugate_reports_what_it_cannot_do(args, message):
    assert message in conjugate_verb(*args)["error"]


@pytest.mark.parametrize("form, normal", [("te-form", "te"), ("masu", "polite"), ("Past Negative", "nakatta"),
                                          ("polite past negative", "past_negative_polite")])
def test_form_names_are_normalised(form, normal):
    assert japanese.normalize_form(form) == normal


def test_conjugate_rejects_bad_input():
    with pytest.raises(ValueError, match="verb class"):
        japanese.conjugate("食べる", "たべる", "weird", "te")
    with pytest.raises(ValueError, match="does not end like"):
        japanese.conjugate("飲む", "のみ", "godan", "te")


@pytest.mark.parametrize("kana, romaji", [
    ("がっこう", "gakkou"), ("しゃしん", "shashin"), ("まっちゃ", "matcha"), ("きって", "kitte"),
    ("せんえん", "sen'en"), ("コーヒー", "koohii"), ("ちょっと", "chotto"), ("じゃあ", "jaa"),
])
def test_romaji(kana, romaji):
    assert japanese.to_romaji(kana) == romaji


@pytest.mark.parametrize("word, reading, verb_class", [
    ("食べる", "たべる", "ichidan"), ("飲む", "のむ", "godan"), ("勉強する", "べんきょうする", "suru"),
    ("来る", "くる", "kuru"), ("とる", "とる", "godan"),
])
def test_guess_class(word, reading, verb_class):
    assert japanese.guess_class(word, reading) == verb_class


def test_dictionary_finds_a_word_by_kanji_reading_romaji_or_meaning():
    assert lookup_dictionary("食べる")["meaning"] == "to eat"
    assert lookup_dictionary("ねこ")["word"] == "猫"
    assert lookup_dictionary("neko")["word"] == "猫"
    assert lookup_dictionary("cat")["word"] == "猫"
    assert lookup_dictionary("to go")["entries"][0]["word"] == "行く"


def test_dictionary_keeps_the_particle_notes_the_corrector_needs():
    assert "never を" in lookup_dictionary("行く")["note"]


def test_an_unknown_word_is_reported_not_invented():
    result = lookup_dictionary("飛行機")
    assert result["found"] is False and "instead of guessing" in result["message"]


def test_a_single_kanji_in_the_dictionary_gives_its_readings_and_words():
    result = lookup_dictionary("行")
    assert result["kanji"]["onyomi"] == "コウ, ギョウ"
    assert [entry["word"] for entry in result["words_with_it"]] == ["行く"]


def test_kanji_gives_readings_and_strokes():
    assert lookup_kanji("食") | {"found": True} == lookup_kanji("食")
    assert lookup_kanji("食")["stroke_count"] == 9
    several = lookup_kanji("学校")
    assert [entry["kanji"] for entry in several["characters"]] == ["学", "校"] and several["unknown"] == []
    assert lookup_kanji("龍")["found"] is False
    assert lookup_kanji("学龍")["unknown"] == ["龍"]


def test_registry_run_says_whether_the_tool_worked():
    assert registry.run("conjugate", {"verb": "食べる", "form": "te"})[1] is True
    text, ok = registry.run("conjugate", {"verb": "猫", "form": "te"})
    assert ok is False and "not a verb" in json.loads(text)["error"]
    assert registry.run("conjugate", {"verb": "食べる"})[1] is False
    assert registry.run("conjugate", ["not", "a", "dict"])[1] is False
    assert registry.run("missing", {}) == ("Error: Tool 'missing' not found.", False)


def test_registry_subset_keeps_only_known_names():
    chosen = registry.subset(["kanji", "nope"])
    assert chosen.names() == ["kanji"] and len(chosen) == 1
