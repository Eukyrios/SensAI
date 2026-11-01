# Tests for the feature video builder: the feature list, the terminal screen, the timeline and the
# recorder. Nothing here needs Ollama or ffmpeg; drawing is checked only where the fonts are installed.

import os
import sys
from pathlib import Path

import pytest

from scripts import build_video
from scripts.build_video import (Chunk, Recorder, Screen, Step, clock_label, feature_from, load_features,
                                 rating_label, real_clock, record_shell, screen_at, timeline)

FONTS_INSTALLED = Path(build_video.LATIN_FONT).exists() and Path(build_video.CJK_FONT).exists()
FAKE_CHAT = "import sys\nwhile True:\n    print('\\n> ', end='', flush=True)\n    line = sys.stdin.readline()\n" \
            "    if not line:\n        break\n    print('echo ' + line.strip(), flush=True)\n"


def test_the_shipped_features_follow_the_catalogue():
    features = load_features()
    codes = [feature.code for feature in features]
    assert len(codes) == len(set(codes)) and codes[0] == "B1"
    assert all(0 <= feature.rating <= 3 and feature.steps for feature in features)
    assert all(step.kind in ("chat", "shell") for feature in features for step in feature.steps)


def test_feature_from_reads_chat_and_shell_steps():
    feature = feature_from({"category": "Tools", "code": "T1", "name": "MCP", "rating": 3, "built": "tools",
                            "steps": [{"kind": "chat", "args": ["--agent", "checker"], "inputs": ["hi"]},
                                      {"kind": "shell", "command": "ls", "display": "list"}]})
    assert feature.steps == (Step("chat", ("--agent", "checker"), ("hi",)), Step("shell", command="ls", display="list"))


def test_rating_label():
    assert rating_label(0) == "base feature"
    assert rating_label(3) == "★" * 3


def test_screen_wraps_at_the_width_and_counts_japanese_as_two_cells():
    screen = Screen(columns=6)
    screen.write("abc猫猫x")
    assert screen.lines == ["abc猫", "猫x"]


def test_screen_drops_colour_codes_and_handles_new_lines():
    screen = Screen()
    screen.write("\x1b[2mdim\x1b[0m\nnext\r\tend")
    assert screen.lines == ["dim", "    end"]


def test_screen_shows_the_last_rows():
    screen = Screen()
    screen.write("\n".join(str(number) for number in range(50)))
    assert screen.visible(rows=3) == ["47", "48", "49"]


def test_timeline_shortens_long_waits_and_waits_for_typing():
    chunks = [Chunk(0.0, "hello\n", typed=True), Chunk(0.1, "a"), Chunk(30.0, "b"), Chunk(30.5, "c")]
    moments = timeline(chunks)
    typing = len("hello\n") / build_video.TYPING_CHARS_PER_S
    assert moments[1].video_s == pytest.approx(typing)
    assert moments[2].video_s == pytest.approx(typing + build_video.MAX_GAP_S)
    assert moments[3].video_s == pytest.approx(typing + build_video.MAX_GAP_S + 0.5)


def test_typed_text_appears_letter_by_letter():
    moments = timeline([Chunk(0.0, "hello\n", typed=True)])
    assert screen_at(moments, 2 / build_video.TYPING_CHARS_PER_S).lines == ["he"]
    assert screen_at(moments, 10.0).lines == ["hello", ""]


def test_the_clock_shows_real_time_during_a_shortened_wait():
    moments = timeline([Chunk(0.0, "a"), Chunk(30.0, "b")])
    halfway = moments[1].video_s / 2
    assert real_clock(moments, halfway) == pytest.approx(15.0)
    assert real_clock(moments, moments[1].video_s + 1) == pytest.approx(31.0)
    assert real_clock([], 1.0) == 0.0


def test_clock_label():
    assert clock_label(75.25) == "real time 01:15.2"


def test_record_shell_shows_the_command_then_its_output():
    chunks = record_shell(Step("shell", command="echo hi", display="say hi"), dict(os.environ))
    assert chunks[0] == Chunk(0.0, "$ say hi\n", typed=True)
    assert "".join(chunk.text for chunk in chunks[1:]) == "hi\n"


def test_recorder_types_each_line_once_the_prompt_is_back():
    recorder = Recorder([sys.executable, "-c", FAKE_CHAT], dict(os.environ))
    for line in ("one", "two"):
        recorder.read_until_prompt()
        recorder.type_line(line)
    recorder.read_until_prompt()
    chunks = recorder.finish()
    typed = [chunk for chunk in chunks if chunk.typed]
    shown = "".join(chunk.text for chunk in chunks if not chunk.typed)
    assert [chunk.text for chunk in typed] == ["one\n", "two\n"]
    assert shown.index("echo one") < shown.index("echo two")


@pytest.mark.skipif(not FONTS_INSTALLED, reason="DejaVu Sans Mono and Noto Sans CJK are not installed")
def test_title_and_terminal_frames_have_the_video_size():
    feature = load_features()[0]
    assert build_video.draw_title(feature).size == (build_video.WIDTH, build_video.HEIGHT)
    frame = build_video.TerminalPainter().draw(["猫 (neko) - cat."], "B1", clock_label(1.0))
    assert frame.size == (build_video.WIDTH, build_video.HEIGHT)
