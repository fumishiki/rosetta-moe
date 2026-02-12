#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Run per-language convergence checks and render SVG plots + animated GIF."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON_DIR = ROOT / "benchmarks" / "convergence"
OUT_SVG_DIR = ROOT / "docs" / "assets" / "convergence"
OUT_GIF_PATH = OUT_SVG_DIR / "convergence-demo.gif"
GO_CACHE_DIR = Path("/tmp/rosetta-moe-go-build-cache")
GO_MOD_CACHE_DIR = Path("/tmp/rosetta-moe-go-mod-cache")

LANGUAGE_RUNS: list[tuple[str, Path, list[str]]] = [
    ("rust", ROOT / "rust", ["cargo", "run", "--release", "--bin", "convergence"]),
    ("go", ROOT / "go", ["go", "test", "-run", "TestConvergence", "-v", "-count=1"]),
    ("python", ROOT, ["python3", "scripts/convergence_python.py"]),
    ("julia", ROOT, ["julia", "scripts/convergence_julia.jl"]),
]

LANGUAGE_COLORS = {
    "rust": "#D7671D",
    "go": "#00ADD8",
    "python": "#3776AB",
    "julia": "#9558B2",
}


def _try_font(size: int) -> ImageFont.ImageFont:
    for name in ("Menlo.ttc", "SFNS.ttf", "Helvetica.ttc", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def run_and_parse(language: str, cwd: Path, cmd: Sequence[str]) -> dict[str, object]:
    """Run one language convergence command and parse JSON payload."""
    print(f"[{language}] running: {' '.join(cmd)}", file=sys.stderr)
    env = os.environ.copy()
    if language == "go":
        env["GOCACHE"] = str(GO_CACHE_DIR)
        env["GOMODCACHE"] = str(GO_MOD_CACHE_DIR)

    proc = subprocess.run(
        list(cmd),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        tail = "\n".join(combined.splitlines()[-20:])
        raise RuntimeError(f"{language} failed with exit={proc.returncode}\n{tail}")

    for line in reversed(combined.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{") or not candidate.endswith("}"):
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "language" in parsed and "losses" in parsed:
            return parsed

    raise RuntimeError(f"{language} output did not contain convergence JSON payload.")


def moving_average(values: list[float], window: int = 25) -> list[float]:
    """Compute simple moving average with fixed trailing window."""
    if window <= 1:
        return values[:]
    out: list[float] = []
    acc = 0.0
    for i, value in enumerate(values):
        acc += value
        if i >= window:
            acc -= values[i - window]
            out.append(acc / window)
        else:
            out.append(acc / (i + 1))
    return out


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    *,
    language: str,
    losses: list[float],
    smoothed: list[float],
    upto: int,
    panel_x: int,
    panel_y: int,
    panel_w: int,
    panel_h: int,
    title_font: ImageFont.ImageFont,
    text_font: ImageFont.ImageFont,
) -> None:
    color = LANGUAGE_COLORS[language]
    draw.rounded_rectangle(
        [panel_x, panel_y, panel_x + panel_w, panel_y + panel_h],
        radius=12,
        fill="#FFFFFF",
        outline="#D8DDE7",
        width=2,
    )

    top = panel_y + 44
    left = panel_x + 48
    right = panel_x + panel_w - 18
    bottom = panel_y + panel_h - 34
    plot_w = right - left
    plot_h = bottom - top

    y_min = 0.0
    y_max = max(losses) * 1.05
    x_max = max(len(losses) - 1, 1)

    def sx(i: int) -> float:
        return left + (i / x_max) * plot_w

    def sy(v: float) -> float:
        if y_max <= y_min:
            return top + plot_h / 2
        return top + (1.0 - (v - y_min) / (y_max - y_min)) * plot_h

    for i in range(5):
        frac = i / 4.0
        yy = top + int(plot_h * frac)
        draw.line([(left, yy), (right, yy)], fill="#ECEFF5", width=1)

    draw.line([(left, bottom), (right, bottom)], fill="#AAB3C3", width=1)

    if upto > 0:
        points = [(sx(i), sy(smoothed[i])) for i in range(0, upto + 1)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3, joint="curve")
        px, py = points[-1]
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)

    initial = losses[0]
    current = losses[upto]
    final = losses[-1]

    draw.text((panel_x + 14, panel_y + 12), f"{language.title()}", font=title_font, fill="#162033")
    draw.text(
        (panel_x + panel_w - 14, panel_y + 14),
        f"step {upto + 1}/{len(losses)}",
        font=text_font,
        fill="#5B6474",
        anchor="ra",
    )
    draw.text(
        (panel_x + 14, panel_y + panel_h - 18),
        f"{initial:.4f} -> {current:.4f} (target {final:.4f})",
        font=text_font,
        fill="#5B6474",
    )


def render_demo_gif(losses_by_lang: dict[str, list[float]], out_path: Path) -> None:
    # Keep language order fixed for a stable visual layout.
    order = ["rust", "go", "python", "julia"]
    steps = min(len(losses_by_lang[lang]) for lang in order)
    smoothed = {lang: moving_average(losses_by_lang[lang], window=25) for lang in order}

    width, height = 1180, 720
    panel_w, panel_h = 560, 300
    left_margin, top_margin = 28, 84
    col_gap, row_gap = 24, 26
    n_frames = 84

    title_font = _try_font(34)
    panel_title_font = _try_font(24)
    text_font = _try_font(16)

    frame_steps: list[int] = []
    for fi in range(n_frames):
        t = fi / (n_frames - 1)
        # Ease-out: move faster in the early phase where loss changes are steep.
        idx = int(round((t**0.8) * (steps - 1)))
        if not frame_steps or idx != frame_steps[-1]:
            frame_steps.append(idx)

    frames: list[Image.Image] = []
    for upto in frame_steps:
        img = Image.new("RGB", (width, height), "#F7F9FD")
        draw = ImageDraw.Draw(img)

        draw.text((30, 24), "Loss Convergence Demo (Python / Go / Julia / Rust)", font=title_font, fill="#132033")
        draw.text(
            (30, 58),
            "Generated by scripts/convergence_plots.py (1000 training steps)",
            font=text_font,
            fill="#556071",
        )

        for i, lang in enumerate(order):
            row, col = divmod(i, 2)
            px = left_margin + col * (panel_w + col_gap)
            py = top_margin + row * (panel_h + row_gap)
            _draw_panel(
                draw,
                language=lang,
                losses=losses_by_lang[lang],
                smoothed=smoothed[lang],
                upto=upto,
                panel_x=px,
                panel_y=py,
                panel_w=panel_w,
                panel_h=panel_h,
                title_font=panel_title_font,
                text_font=text_font,
            )

        frames.append(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=70,
        loop=0,
        optimize=True,
        disposal=2,
    )


def render_svg(language: str, losses: list[float]) -> str:
    """Render one convergence line chart as standalone SVG."""
    width, height = 980, 560
    margin_left, margin_right = 84, 34
    margin_top, margin_bottom = 54, 74
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    color = LANGUAGE_COLORS[language]
    steps = len(losses)
    x_max = float(max(steps - 1, 1))
    y_min = 0.0
    y_max = max(losses) * 1.05

    def x_scale(i: int) -> float:
        return margin_left + (i / x_max) * plot_w

    def y_scale(v: float) -> float:
        if y_max <= y_min:
            return margin_top + plot_h / 2
        return margin_top + (1.0 - (v - y_min) / (y_max - y_min)) * plot_h

    raw_points = " ".join(f"{x_scale(i):.2f},{y_scale(v):.2f}" for i, v in enumerate(losses))
    smooth = moving_average(losses, window=25)
    smooth_points = " ".join(f"{x_scale(i):.2f},{y_scale(v):.2f}" for i, v in enumerate(smooth))

    grid_lines: list[str] = []
    for i in range(6):
        frac = i / 5.0
        y_value = y_min + (y_max - y_min) * frac
        y = y_scale(y_value)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" '
            f'stroke="#E7E9EF" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{margin_left - 10}" y="{y + 5:.2f}" text-anchor="end" '
            f'font-family="Menlo, Monaco, monospace" font-size="12" fill="#6B7380">{y_value:.2f}</text>'
        )

    x_tick_labels: list[str] = []
    for step in (0, 200, 400, 600, 800, 999):
        x = x_scale(step)
        x_tick_labels.append(
            f'<line x1="{x:.2f}" y1="{margin_top + plot_h}" x2="{x:.2f}" y2="{margin_top + plot_h + 6}" '
            f'stroke="#B8C0CC" stroke-width="1" />'
        )
        x_tick_labels.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_h + 24}" text-anchor="middle" '
            f'font-family="Menlo, Monaco, monospace" font-size="12" fill="#6B7380">{step}</text>'
        )

    initial = losses[0]
    final = losses[-1]
    reduction = (1.0 - (final / initial)) * 100 if initial > 0 else 0.0
    last_x, last_y = x_scale(steps - 1), y_scale(final)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{language} loss convergence">
  <rect width="100%" height="100%" fill="#FCFCFE" />
  <rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#FFFFFF" stroke="#D7DCE5" stroke-width="1" rx="8" />
  {''.join(grid_lines)}
  <line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#9FA8B8" stroke-width="1.2" />
  {''.join(x_tick_labels)}

  <polyline fill="none" stroke="{color}" stroke-opacity="0.28" stroke-width="1.4" points="{raw_points}" />
  <polyline fill="none" stroke="{color}" stroke-width="3" points="{smooth_points}" />
  <circle cx="{last_x:.2f}" cy="{last_y:.2f}" r="4.5" fill="{color}" />
  <text x="{last_x - 8:.2f}" y="{max(last_y - 12, margin_top + 16):.2f}" text-anchor="end"
        font-family="Menlo, Monaco, monospace" font-size="12" fill="{color}">final {final:.4f}</text>

  <text x="{margin_left}" y="30" font-family="Avenir Next, Helvetica, Arial, sans-serif" font-size="24" fill="#19202D">{language.title()} Loss Convergence</text>
  <text x="{margin_left}" y="48" font-family="Menlo, Monaco, monospace" font-size="13" fill="#4C5566">initial={initial:.4f}  final={final:.4f}  reduction={reduction:.2f}%  steps={steps}</text>
  <text x="{margin_left + plot_w / 2:.2f}" y="{height - 20}" text-anchor="middle" font-family="Menlo, Monaco, monospace" font-size="12" fill="#6B7380">training step</text>
  <text x="20" y="{margin_top + plot_h / 2:.2f}" transform="rotate(-90 20 {margin_top + plot_h / 2:.2f})" text-anchor="middle" font-family="Menlo, Monaco, monospace" font-size="12" fill="#6B7380">loss</text>
</svg>
"""


def main() -> int:
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SVG_DIR.mkdir(parents=True, exist_ok=True)
    GO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GO_MOD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    summary: list[tuple[str, float, float, float]] = []
    losses_by_lang: dict[str, list[float]] = {}

    for language, cwd, cmd in LANGUAGE_RUNS:
        payload = run_and_parse(language, cwd, cmd)
        losses = [float(v) for v in payload["losses"]]
        if not losses:
            raise RuntimeError(f"{language} returned an empty loss sequence.")

        normalized = {
            "language": language,
            "steps": len(losses),
            "losses": [round(v, 6) for v in losses],
        }

        json_path = OUT_JSON_DIR / f"{language}.json"
        json_path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")

        svg_path = OUT_SVG_DIR / f"{language}.svg"
        svg_path.write_text(render_svg(language, losses), encoding="utf-8")
        losses_by_lang[language] = losses

        initial, final = losses[0], losses[-1]
        reduction = (1.0 - (final / initial)) * 100 if initial > 0 else 0.0
        summary.append((language, initial, final, reduction))

    render_demo_gif(losses_by_lang, OUT_GIF_PATH)

    print("\nConvergence summary:")
    for language, initial, final, reduction in summary:
        print(f"  {language:>6}: {initial:8.4f} -> {final:8.4f}  ({reduction:6.2f}% reduction)")

    print(f"\nSaved JSON: {OUT_JSON_DIR}")
    print(f"Saved SVG : {OUT_SVG_DIR}")
    print(f"Saved GIF : {OUT_GIF_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
