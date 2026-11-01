# Feature video: for each feature of scripts/video_features.toml, a title card in the features catalogue's
# terms, then a real run of Sensai showing it, recorded and drawn as a terminal; ffmpeg joins it all into
# one MP4. Run `python -m scripts.build_video` with Ollama running.

import argparse
import codecs
import os
import re
import select
import subprocess
import sys
import tempfile
import textwrap
import time
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from sensai.core.config import load_toml, resolve

from . import seed_demo

FEATURES_PATH = resolve("scripts/video_features.toml")
VIDEO_PATH = resolve("videos/sensai_features.mp4")
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
PROMPT = "\n> "
READ_SLICE_S = 0.1
STEP_TIMEOUT_S = 300
FPS = 15
TYPING_CHARS_PER_S = 14
MAX_GAP_S = 1.5
READ_PAUSE_S = 4
TITLE_S = 7
END_HOLD_S = 6


@dataclass(frozen=True)
class Step:
    kind: str
    args: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    command: str = ""
    display: str = ""


@dataclass(frozen=True)
class Feature:
    category: str
    code: str
    name: str
    rating: int
    built: str
    steps: tuple[Step, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Chunk:
    at_s: float
    text: str
    typed: bool = False


@dataclass(frozen=True)
class Moment:
    video_s: float
    real_s: float
    chunk: Chunk


def load_features(path: Path = FEATURES_PATH) -> list[Feature]:
    return [feature_from(entry) for entry in load_toml(path)["features"]]


def feature_from(entry: dict) -> Feature:
    steps = tuple(Step(step["kind"], tuple(step.get("args", ())), tuple(step.get("inputs", ())),
                       step.get("command", ""), step.get("display", "")) for step in entry["steps"])
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


class Recorder:

    def __init__(self, command: list[str], env: dict[str, str]):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, env=env, cwd=resolve("."))
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.started = time.monotonic()
        self.chunks: list[Chunk] = []
        self.output = ""
        self.answered_from = 0

    def now(self) -> float:
        return time.monotonic() - self.started

    def read_until_prompt(self) -> None:
        deadline = time.monotonic() + STEP_TIMEOUT_S
        while not self.prompt_is_back() and time.monotonic() < deadline:
            if not self._read_some():
                return

    def prompt_is_back(self) -> bool:
        return len(self.output) > self.answered_from and self.output.endswith(PROMPT)

    def type_line(self, line: str) -> None:
        self.chunks.append(Chunk(self.now(), line + "\n", typed=True))
        self.process.stdin.write((line + "\n").encode())
        self.process.stdin.flush()
        self.answered_from = len(self.output)

    def finish(self) -> list[Chunk]:
        if self.process.stdin:
            self.process.stdin.close()
        while self._read_some():
            pass
        self.process.wait()
        return self.chunks

    def _read_some(self) -> bool:
        ready, _, _ = select.select([self.process.stdout], [], [], READ_SLICE_S)
        if not ready:
            return self.process.poll() is None
        data = os.read(self.process.stdout.fileno(), 4096)
        if not data:
            return False
        text = self.decoder.decode(data)
        self.output += text
        self.chunks.append(Chunk(self.now(), text))
        return True


def record_chat(step: Step, env: dict[str, str]) -> list[Chunk]:
    recorder = Recorder([sys.executable, "-m", "sensai", *step.args], env)
    for line in step.inputs:
        recorder.read_until_prompt()
        recorder.type_line(line)
    recorder.read_until_prompt()
    return recorder.finish()


def record_shell(step: Step, env: dict[str, str]) -> list[Chunk]:
    recorder = Recorder(["bash", "-c", step.command], env)
    recorder.chunks.append(Chunk(0.0, f"$ {step.display or step.command}\n", typed=True))
    return recorder.finish()


def timeline(chunks: list[Chunk]) -> list[Moment]:
    moments: list[Moment] = []
    video_s, real_s, free_at = 0.0, 0.0, 0.0
    for chunk in chunks:
        pause = READ_PAUSE_S if chunk.typed and moments else 0.0
        video_s = max(video_s + min(chunk.at_s - real_s, MAX_GAP_S), free_at) + pause
        real_s = chunk.at_s
        moments.append(Moment(video_s, real_s, chunk))
        if chunk.typed:
            free_at = video_s + typing_seconds(chunk.text)
    return moments


def typing_seconds(text: str) -> float:
    return len(text) / TYPING_CHARS_PER_S


def screen_at(moments: list[Moment], video_s: float) -> Screen:
    screen = Screen()
    for moment in moments:
        if moment.video_s > video_s:
            break
        screen.write(shown_text(moment, video_s))
    return screen


def shown_text(moment: Moment, video_s: float) -> str:
    if not moment.chunk.typed:
        return moment.chunk.text
    typed_chars = int((video_s - moment.video_s) * TYPING_CHARS_PER_S)
    return moment.chunk.text[:typed_chars]


def real_clock(moments: list[Moment], video_s: float) -> float:
    shown = [moment for moment in moments if moment.video_s <= video_s]
    if not shown:
        return 0.0
    last = shown[-1]
    following = moments[len(shown)] if len(shown) < len(moments) else None
    if following is None or following.video_s == last.video_s:
        return last.real_s + (video_s - last.video_s)
    ratio = (following.real_s - last.real_s) / (following.video_s - last.video_s)
    return last.real_s + (video_s - last.video_s) * ratio


def clock_label(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"real time {int(minutes):02d}:{rest:04.1f}"


def step_frames(moments: list[Moment], header: str, painter: TerminalPainter) -> Iterator[Image.Image]:
    end_s = (moments[-1].video_s if moments else 0.0) + END_HOLD_S
    for frame in range(int(end_s * FPS)):
        video_s = frame / FPS
        lines = screen_at(moments, video_s).visible()
        yield painter.draw(lines, header, clock_label(real_clock(moments, video_s)))


def title_frames(feature: Feature) -> Iterator[Image.Image]:
    card = draw_title(feature)
    for _ in range(TITLE_S * FPS):
        yield card


class Encoder:

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.partial = path.with_name(f"{path.stem}.part{path.suffix}")
        self.process = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(self.partial)],
            stdin=subprocess.PIPE)

    def add(self, frames: Iterator[Image.Image]) -> int:
        count = 0
        for image in frames:
            self.process.stdin.write(image.tobytes())
            count += 1
        return count

    def close(self) -> None:
        self.process.stdin.close()
        if self.process.wait() == 0:
            self.partial.replace(self.path)


def record_step(step: Step, env: dict[str, str]) -> list[Chunk]:
    return record_shell(step, env) if step.kind == "shell" else record_chat(step, env)


def demo_env(workdir: Path) -> dict[str, str]:
    db_path = workdir / "demo.db"
    seed_demo.seed(db_path).conn.close()
    return {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHON": sys.executable, "SENSAI_STORAGE__DB_PATH": str(db_path),
            "SENSAI_LOGGING__PATH": str(workdir / "sensai.jsonl"), "DEMO_DB": str(db_path),
            "DEMO_LOG": str(workdir / "sensai.jsonl")}


def add_feature(encoder: Encoder, painter: TerminalPainter, feature: Feature, env: dict[str, str]) -> None:
    encoder.add(title_frames(feature))
    header = f"{feature.code}  {feature.name}"
    for step in feature.steps:
        print(f"  {feature.code}: {step.command or ' / '.join(step.inputs)}", flush=True)
        encoder.add(step_frames(timeline(record_step(step, env)), header, painter))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the feature video from real Sensai runs")
    parser.add_argument("--only", nargs="*", default=[], help="feature codes to include (default: all)")
    parser.add_argument("--out", type=Path, default=VIDEO_PATH, help=f"video file (default: {VIDEO_PATH})")
    args = parser.parse_args(argv)
    features = [feature for feature in load_features() if not args.only or feature.code in args.only]
    encoder, painter = Encoder(args.out), TerminalPainter()
    with tempfile.TemporaryDirectory(prefix="sensai-video-") as workdir:
        env = demo_env(Path(workdir))
        for feature in features:
            add_feature(encoder, painter, feature, env)
    encoder.close()
    print(f"Video written to {args.out}")


if __name__ == "__main__":
    main()
