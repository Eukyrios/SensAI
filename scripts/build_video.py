# Feature video: for each feature of scripts/video_features.toml, a title card in the features catalogue's
# terms, then a real run of Sensai showing it, recorded and drawn as a terminal; ffmpeg joins it all into
# one MP4. Run `python -m scripts.build_video` with Ollama running.

import re
import textwrap
import tomllib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from sensai.core.config import resolve

FEATURES_PATH = resolve("scripts/video_features.toml")
WIDTH, HEIGHT = 1280, 720
LATIN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BACKGROUND = (24, 26, 33)
TEXT = (220, 223, 228)
DIM = (130, 136, 148)
ACCENT = (235, 87, 87)
STAR = (242, 201, 76)
TERMINAL_FONT_SIZE = 18
CELL_WIDTH, LINE_HEIGHT = 11, 24
MARGIN, HEADER_HEIGHT = 24, 56
COLUMNS = (WIDTH - 2 * MARGIN) // CELL_WIDTH
ROWS = (HEIGHT - HEADER_HEIGHT - MARGIN) // LINE_HEIGHT
ANSI_CODE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


@dataclass(frozen=True)
class Step:
    kind: str
    args: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    command: str = ""


@dataclass(frozen=True)
class Feature:
    category: str
    code: str
    name: str
    rating: int
    built: str
    steps: tuple[Step, ...] = field(default_factory=tuple)


def load_features(path: Path = FEATURES_PATH) -> list[Feature]:
    with open(path, "rb") as file:
        data = tomllib.load(file)
    return [feature_from(entry) for entry in data["features"]]


def feature_from(entry: dict) -> Feature:
    steps = tuple(Step(step["kind"], tuple(step.get("args", ())), tuple(step.get("inputs", ())),
                       step.get("command", "")) for step in entry["steps"])
    return Feature(entry["category"], entry["code"], entry["name"], entry["rating"], entry["built"], steps)


def rating_label(rating: int) -> str:
    return "base feature" if rating == 0 else "★" * rating


def draw_title(feature: Feature) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((80, 150), feature.category.upper(), font=ImageFont.truetype(LATIN_FONT, 26), fill=DIM)
    draw.text((80, 200), feature.code, font=ImageFont.truetype(LATIN_FONT, 96), fill=ACCENT)
    draw.text((80, 330), feature.name, font=ImageFont.truetype(LATIN_FONT, 48), fill=TEXT)
    draw.text((80, 410), rating_label(feature.rating), font=ImageFont.truetype(CJK_FONT, 40), fill=STAR)
    built_font = ImageFont.truetype(LATIN_FONT, 22)
    for number, line in enumerate(textwrap.wrap(feature.built, 80)):
        draw.text((80, 500 + number * 32), line, font=built_font, fill=TEXT)
    return image


def cell_width(char: str) -> int:
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


class Screen:

    def __init__(self, columns: int = COLUMNS):
        self.columns = columns
        self.lines: list[str] = [""]

    def write(self, text: str) -> None:
        for char in ANSI_CODE.sub("", text):
            self._put(char)

    def visible(self, rows: int = ROWS) -> list[str]:
        return self.lines[-rows:]

    def _put(self, char: str) -> None:
        if char == "\n":
            self.lines.append("")
        elif char == "\r":
            self.lines[-1] = ""
        elif char.isprintable() or char == "\t":
            self._put_printable(" " * 4 if char == "\t" else char)

    def _put_printable(self, text: str) -> None:
        for char in text:
            if line_width(self.lines[-1]) + cell_width(char) > self.columns:
                self.lines.append("")
            self.lines[-1] += char


def line_width(line: str) -> int:
    return sum(cell_width(char) for char in line)


class TerminalPainter:

    def __init__(self):
        self.latin = ImageFont.truetype(LATIN_FONT, TERMINAL_FONT_SIZE)
        self.cjk = ImageFont.truetype(CJK_FONT, TERMINAL_FONT_SIZE)
        self.header_font = ImageFont.truetype(LATIN_FONT, 20)

    def draw(self, lines: list[str], header: str, clock: str) -> Image.Image:
        image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.text((MARGIN, 16), header, font=self.header_font, fill=ACCENT)
        draw.text((WIDTH - MARGIN - 260, 16), clock, font=self.header_font, fill=DIM)
        for row, line in enumerate(lines):
            self._draw_line(draw, line, HEADER_HEIGHT + row * LINE_HEIGHT)
        return image

    def _draw_line(self, draw: ImageDraw.ImageDraw, line: str, top: int) -> None:
        column = 0
        for char in line:
            font = self.cjk if cell_width(char) == 2 or not self.latin.getmask(char).getbbox() else self.latin
            draw.text((MARGIN + column * CELL_WIDTH, top), char, font=font, fill=TEXT)
            column += cell_width(char)
