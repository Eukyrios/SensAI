from dataclasses import dataclass

_KANA = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "を": "o", "ん": "n", "ゔ": "vu",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
}
_SMALL_Y = {"ゃ": "a", "ゅ": "u", "ょ": "o"}
_SMALL_VOWELS = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}
# Godan verbs whose masu-stem and imperative end in い instead of り: いらっしゃいます, ください.
HONORIFIC_ARU = ("いらっしゃる", "おっしゃる", "くださる", "下さる", "なさる", "ござる")

# Godan verb endings and the kana they take in each row.
_ROWS = {
    "う": ("わ", "い", "え", "お"), "く": ("か", "き", "け", "こ"), "ぐ": ("が", "ぎ", "げ", "ご"),
    "す": ("さ", "し", "せ", "そ"), "つ": ("た", "ち", "て", "と"), "ぬ": ("な", "に", "ね", "の"),
    "ぶ": ("ば", "び", "べ", "ぼ"), "む": ("ま", "み", "め", "も"), "る": ("ら", "り", "れ", "ろ"),
}
_TE = {"う": "って", "つ": "って", "る": "って", "む": "んで", "ぶ": "んで", "ぬ": "んで",
       "く": "いて", "ぐ": "いで", "す": "して"}
_E_I_ROWS = set("えけげせぜてでねへべぺめれいきぎしじちぢにひびぴみり")

FORMS = ("dictionary", "polite", "polite_negative", "polite_past", "past_negative_polite", "te", "ta",
         "nai", "nakatta", "potential", "volitional", "ba", "tara", "passive", "causative", "imperative")
FORM_ALIASES = {
    "masu": "polite", "masen": "polite_negative", "mashita": "polite_past",
    "masen_deshita": "past_negative_polite", "polite_past_negative": "past_negative_polite",
    "past": "ta", "plain_past": "ta", "negative": "nai", "plain_negative": "nai",
    "past_negative": "nakatta", "plain_past_negative": "nakatta", "can": "potential",
    "lets": "volitional", "conditional": "ba", "if": "tara", "plain": "dictionary",
    "negative_past": "nakatta", "plain_negative_past": "nakatta", "negative_polite": "polite_negative",
    "polite_negative_past": "past_negative_polite", "past_polite_negative": "past_negative_polite",
    "negative_past_polite": "past_negative_polite", "past_polite": "polite_past", "polite_present": "polite",
}
CLASSES = ("ichidan", "godan", "suru", "kuru")

def to_romaji(kana: str) -> str:
    out: list[str] = []
    double_next = False
    for index, char in enumerate(kana):
        char = _to_hiragana(char)
        if char == "っ":
            double_next = True
            continue
        if char in _SMALL_Y and out:
            out[-1] = _combine_small_y(out[-1], _SMALL_Y[char])
            continue
        if char in _SMALL_VOWELS and out and out[-1][-1] in "aiueo" and len(out[-1]) > 1 or \
                char in _SMALL_VOWELS and out and out[-1] == "u":
            out[-1] = _combine_small_vowel(out[-1], _SMALL_VOWELS[char])
            continue
        if char == "ー" and out:
            out[-1] += out[-1][-1]
            continue
        romaji = _KANA.get(char, char)
        if char == "ん" and index + 1 < len(kana) and _KANA.get(_to_hiragana(kana[index + 1]), "x")[0] in "aiueoy":
            romaji = "n'"
        if double_next and romaji[0].isalpha():
            romaji = ("t" if romaji.startswith("ch") else romaji[0]) + romaji
        double_next = False
        out.append(romaji)
    return "".join(out)


def _to_hiragana(char: str) -> str:
    code = ord(char)
    return chr(code - 0x60) if 0x30A1 <= code <= 0x30F6 else char


def _combine_small_vowel(previous: str, vowel: str) -> str:
    if previous == "u":
        return "w" + vowel
    if previous.endswith(("shi", "chi", "ji")):
        return previous[:-1] + vowel
    if previous.endswith(("fu", "tsu")):
        return previous[:-1] + vowel
    return previous[:-1] + vowel


def particle_romaji(kana: str, part_of_speech: str) -> str:
    if part_of_speech == "particle" and kana in ("は", "へ"):
        return {"は": "wa", "へ": "e"}[kana]
    if part_of_speech == "expression" and kana.endswith(("にちは", "ばんは")):
        return to_romaji(kana[:-1]) + "wa"
    return to_romaji(kana)


def _combine_small_y(previous: str, vowel: str) -> str:
    if previous.endswith(("shi", "chi", "ji")):
        return previous[:-1] + vowel
    return previous[:-1] + "y" + vowel


def guess_class(word: str, reading: str) -> str:
    if word == "する" or word.endswith("為る") or (word.endswith("する") and _has_kanji(word[:-2])):
        return "suru"
    if word in ("来る", "くる") or reading == "くる":
        return "kuru"
    if reading.endswith("る") and len(reading) >= 2 and reading[-2] in _E_I_ROWS:
        return "ichidan"
    return "godan"


def _has_kanji(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


@dataclass(frozen=True)
class Conjugation:
    verb: str
    reading: str
    verb_class: str
    form: str
    result: str
    result_reading: str

    def as_dict(self) -> dict:
        return {"verb": self.verb, "reading": self.reading, "class": self.verb_class, "form": self.form,
                "result": self.result, "result_reading": self.result_reading,
                "romaji": to_romaji(self.result_reading)}


def normalize_form(form: str) -> str:
    key = form.strip().lower().replace("-form", "").replace(" form", "").replace("-", "_").replace(" ", "_")
    key = key.removesuffix("_form")
    return FORM_ALIASES.get(key, key)


def conjugate(verb: str, reading: str, verb_class: str, form: str) -> Conjugation:
    form = normalize_form(form)
    if form not in FORMS:
        raise ValueError(f"unknown form '{form}' (known: {', '.join(FORMS)})")
    if verb_class not in CLASSES:
        raise ValueError(f"unknown verb class '{verb_class}'")
    if not reading.endswith(verb[-1]) and verb_class == "godan":
        raise ValueError(f"reading {reading} does not end like {verb}")
    result = _conjugate_surface(verb, reading, verb_class, form)
    return Conjugation(verb, reading, verb_class, form, *result)


def _conjugate_surface(verb: str, reading: str, verb_class: str, form: str) -> tuple[str, str]:
    if verb_class == "suru":
        return _join(verb, reading, 2, _SURU[form])
    if verb_class == "kuru":
        written = _KURU_WRITTEN[form] if verb.startswith("来") else _KURU[form]
        return verb[:-2] + written if len(verb) > 2 else written, reading[:-2] + _KURU[form]
    if verb_class == "ichidan":
        if form == "imperative" and reading.endswith("くれる"):
            return _join(verb, reading, 1, "")
        return _join(verb, reading, 1, _ICHIDAN[form])
    return _godan(verb, reading, form)


def _join(verb: str, reading: str, cut: int, ending: str) -> tuple[str, str]:
    return verb[:-cut] + ending, reading[:-cut] + ending


def _godan(verb: str, reading: str, form: str) -> tuple[str, str]:
    last = verb[-1]
    a_row, i_row, e_row, o_row = _ROWS[last]
    if reading in HONORIFIC_ARU or verb in HONORIFIC_ARU:
        i_row = "い"
        if form == "imperative":
            return _join(verb, reading, 1, "い")
    if verb.endswith(("行く", "いく")) and form in ("te", "ta", "tara"):
        te = "って"
    else:
        te = _TE[last]
    ta = te[:-1] + ("だ" if te.endswith("で") else "た")
    if verb in ("ある", "有る") and form in ("nai", "nakatta"):
        endings = {"nai": "ない", "nakatta": "なかった"}
        return verb[:-2] + endings[form] if len(verb) > 2 else endings[form], endings[form]
    endings = {
        "dictionary": last, "polite": i_row + "ます", "polite_negative": i_row + "ません",
        "polite_past": i_row + "ました", "past_negative_polite": i_row + "ませんでした",
        "te": te, "ta": ta, "nai": a_row + "ない", "nakatta": a_row + "なかった",
        "potential": e_row + "る", "volitional": o_row + "う", "ba": e_row + "ば", "tara": ta + "ら",
        "passive": a_row + "れる", "causative": a_row + "せる", "imperative": e_row,
    }
    return _join(verb, reading, 1, endings[form])


_ICHIDAN = {
    "dictionary": "る", "polite": "ます", "polite_negative": "ません", "polite_past": "ました",
    "past_negative_polite": "ませんでした", "te": "て", "ta": "た", "nai": "ない", "nakatta": "なかった",
    "potential": "られる", "volitional": "よう", "ba": "れば", "tara": "たら", "passive": "られる",
    "causative": "させる", "imperative": "ろ",
}
_SURU = {
    "dictionary": "する", "polite": "します", "polite_negative": "しません", "polite_past": "しました",
    "past_negative_polite": "しませんでした", "te": "して", "ta": "した", "nai": "しない",
    "nakatta": "しなかった", "potential": "できる", "volitional": "しよう", "ba": "すれば", "tara": "したら",
    "passive": "される", "causative": "させる", "imperative": "しろ",
}
_KURU = {
    "dictionary": "くる", "polite": "きます", "polite_negative": "きません", "polite_past": "きました",
    "past_negative_polite": "きませんでした", "te": "きて", "ta": "きた", "nai": "こない",
    "nakatta": "こなかった", "potential": "こられる", "volitional": "こよう", "ba": "くれば", "tara": "きたら",
    "passive": "こられる", "causative": "こさせる", "imperative": "こい",
}
_KURU_WRITTEN = {form: "来" + kana[1:] for form, kana in _KURU.items()}
