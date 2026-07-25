#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Run per-language convergence checks and render SVG plots + animated GIF.

Supports N=30 trial runs with BCa bootstrap confidence intervals,
sign-flip permutation tests, and robust statistics (trimmed mean,
winsorized std). Runs both CPU and GPU modes when available.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON_DIR = ROOT / "benchmarks" / "convergence"
OUT_SVG_DIR = ROOT / "docs" / "assets" / "convergence"
OUT_GIF_PATH = OUT_SVG_DIR / "convergence-demo.gif"
GO_CACHE_DIR = Path("/tmp/rosetta-moe-go-build-cache")
GO_MOD_CACHE_DIR = Path("/tmp/rosetta-moe-go-mod-cache")

N_TRIALS = 30
N_BOOTSTRAP = 10000
ALPHA = 0.05
TRIM_FRACTION = 0.2

LANGUAGE_RUNS_CPU: list[tuple[str, Path, list[str]]] = [
    ("rust", ROOT / "rust", ["cargo", "run", "--release", "--bin", "convergence", "--"]),
    ("go", ROOT / "go", ["go", "test", "-run", "^TestConvergence$", "-v", "-count=1"]),
    ("python", ROOT, ["python3", "scripts/convergence_python.py"]),
    ("julia", ROOT, ["julia", "scripts/convergence_julia.jl"]),
]

LANGUAGE_RUNS_GPU: list[tuple[str, Path, list[str]]] = [
    ("rust", ROOT / "rust", ["cargo", "run", "--release", "--features", "metal", "--bin", "convergence", "--"]),
    ("go", ROOT / "go", ["go", "test", "-run", "^TestConvergenceGpu$", "-v", "-count=1"]),
    ("python", ROOT, ["python3", "scripts/convergence_python_gpu.py"]),
    ("julia", ROOT, ["julia", "scripts/convergence_julia_gpu.jl"]),
]

# Batch scripts for Julia (single-process, amortized JIT)
JULIA_BATCH_CMD: dict[str, tuple[Path, list[str]]] = {
    "cpu": (ROOT, ["julia", "scripts/convergence_julia_batch.jl"]),
    "gpu": (ROOT, ["julia", "scripts/convergence_julia_gpu_batch.jl"]),
}

RAW_CACHE_DIR = OUT_JSON_DIR

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


def run_and_parse(
    language: str,
    cwd: Path,
    cmd: Sequence[str],
    extra_env: dict[str, str] | None = None,
) -> dict[str, object]:
    """Run one language convergence command and parse JSON payload."""
    print(f"[{language}] running: {' '.join(cmd)}", file=sys.stderr)
    env = os.environ.copy()
    if language == "go":
        env["GOCACHE"] = str(GO_CACHE_DIR)
        env["GOMODCACHE"] = str(GO_MOD_CACHE_DIR)
    if extra_env:
        env.update(extra_env)

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


def run_trials(
    language: str,
    cwd: Path,
    cmd: list[str],
    n_trials: int = N_TRIALS,
    device: str = "cpu",
) -> list[list[float]]:
    """Run convergence N times with seeds 1..N, return list of loss trajectories."""
    all_losses: list[list[float]] = []
    for seed in range(1, n_trials + 1):
        print(f"[{language}/{device}] trial {seed}/{n_trials}...", file=sys.stderr)
        seed_cmd = list(cmd)
        seed_env: dict[str, str] | None = None
        if language == "go":
            seed_env = {"CONV_SEED": str(seed)}
        else:
            seed_cmd.extend(["--seed", str(seed)])
        payload = run_and_parse(language, cwd, seed_cmd, extra_env=seed_env)
        losses = [float(v) for v in payload["losses"]]
        if not losses:
            raise RuntimeError(f"{language} trial {seed} returned empty loss sequence.")
        all_losses.append(losses)
    return all_losses


def run_trials_batch(
    language: str,
    cwd: Path,
    cmd: list[str],
    n_trials: int = N_TRIALS,
    device: str = "cpu",
) -> list[list[float]]:
    """Run batch convergence script that outputs all trials in a single JSON."""
    full_cmd = list(cmd) + ["--trials", str(n_trials)]
    print(f"[{language}/{device}] batch: {' '.join(full_cmd)}", file=sys.stderr)

    env = os.environ.copy()
    proc = subprocess.run(
        full_cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        tail = "\n".join(combined.splitlines()[-20:])
        raise RuntimeError(f"{language} batch failed with exit={proc.returncode}\n{tail}")

    # Parse the single JSON line from stdout
    for line in reversed(proc.stdout.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{") or not candidate.endswith("}"):
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "trials" in parsed:
            trials = [[float(v) for v in t] for t in parsed["trials"]]
            if not trials:
                raise RuntimeError(f"{language} batch returned empty trials.")
            print(f"[{language}/{device}] batch complete: {len(trials)} trials", file=sys.stderr)
            return trials

    raise RuntimeError(f"{language} batch output did not contain trials JSON payload.")


def _save_raw_cache(language: str, device: str, trials: list[list[float]]) -> None:
    """Save raw trial data to cache file."""
    cache_path = RAW_CACHE_DIR / f"{language}_{device}_raw.json"
    payload = {
        "language": language,
        "device": device,
        "n_trials": len(trials),
        "trials": [[round(v, 6) for v in t] for t in trials],
    }
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[{language}/{device}] cached raw data -> {cache_path}", file=sys.stderr)


def _load_raw_cache(language: str, device: str) -> list[list[float]] | None:
    """Load raw trial data from cache file, or return None if absent."""
    cache_path = RAW_CACHE_DIR / f"{language}_{device}_raw.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        trials = [[float(v) for v in t] for t in data["trials"]]
        print(f"[{language}/{device}] loaded from cache ({len(trials)} trials)", file=sys.stderr)
        return trials
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"[{language}/{device}] cache corrupt, ignoring: {e}", file=sys.stderr)
        return None


def _parse_only_flag(argv: list[str]) -> set[tuple[str, str]] | None:
    """Parse --only flag from argv. Returns set of (lang, device) or None for all."""
    for i, arg in enumerate(argv):
        if arg == "--only" and i + 1 < len(argv):
            combos = set()
            for token in argv[i + 1].split(","):
                token = token.strip()
                parts = token.rsplit("_", 1)
                if len(parts) == 2 and parts[1] in ("cpu", "gpu"):
                    combos.add((parts[0], parts[1]))
                else:
                    print(f"WARNING: ignoring invalid --only token: {token!r}", file=sys.stderr)
            return combos if combos else None
    return None


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


def detect_spikes(
    losses: list[float],
    ema_alpha: float = 0.1,
    ema_multiplier: float = 3.0,
    abs_floor: float = 0.001,
) -> list[dict]:
    """Detect loss spikes using EMA-relative threshold with absolute floor.

    A step is a spike when BOTH conditions hold:
      1. loss[t] > ema[t] * ema_multiplier   (relative to recent trend)
      2. loss[t] - loss[t-1] > abs_floor      (absolute increase is non-trivial)

    This avoids false positives from percentage-based detection at very low
    loss values (e.g. 0.00001 -> 0.0001 is +900% but only delta=0.00009).

    Returns list of dicts: {step, loss, prev_loss, ema, delta_abs, ratio}.
    """
    if len(losses) < 2:
        return []
    spikes = []
    ema = losses[0]
    for t in range(1, len(losses)):
        delta = losses[t] - losses[t - 1]
        ratio = losses[t] / max(ema, 1e-12)
        if ratio > ema_multiplier and delta > abs_floor:
            spikes.append({
                "step": t,
                "loss": round(losses[t], 6),
                "prev_loss": round(losses[t - 1], 6),
                "ema": round(ema, 6),
                "delta_abs": round(delta, 6),
                "ratio": round(ratio, 2),
            })
        # Update EMA after spike check (use previous step's loss)
        ema = ema_alpha * losses[t] + (1 - ema_alpha) * ema
    return spikes


# ---------------------------------------------------------------------------
# Statistics helpers (no scipy dependency)
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF. Abramowitz & Stegun 26.2.23."""
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p < 0.5:
        return -_norm_ppf(1 - p)
    t = math.sqrt(-2.0 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)


def bca_bootstrap_ci(
    data: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval."""
    n = len(data)
    data_arr = np.array(data)
    theta_hat = float(np.mean(data_arr))

    # Bootstrap distribution
    rng = np.random.default_rng(42)
    boot_thetas = np.array([
        float(np.mean(rng.choice(data_arr, size=n, replace=True)))
        for _ in range(n_bootstrap)
    ])

    # Bias correction: z0
    prop_below = float(np.mean(boot_thetas < theta_hat))
    prop_below = max(1e-10, min(1 - 1e-10, prop_below))
    z0 = _norm_ppf(prop_below)

    # Acceleration: a (jackknife)
    jackknife = np.array([
        float(np.mean(np.concatenate([data_arr[:i], data_arr[i + 1:]])))
        for i in range(n)
    ])
    jack_mean = float(np.mean(jackknife))
    diff = jack_mean - jackknife
    num = float(np.sum(diff ** 3))
    denom = 6.0 * float(np.sum(diff ** 2) ** 1.5)
    a = num / denom if abs(denom) > 1e-12 else 0.0

    # Adjusted quantiles
    z_lo = _norm_ppf(alpha / 2)
    z_hi = _norm_ppf(1 - alpha / 2)

    def _adjust(z: float) -> float:
        numer = z0 + z
        return _norm_cdf(z0 + numer / (1 - a * numer))

    alpha1 = _adjust(z_lo)
    alpha2 = _adjust(z_hi)

    ci_lower = float(np.percentile(boot_thetas, max(0.0, alpha1) * 100))
    ci_upper = float(np.percentile(boot_thetas, min(1.0, alpha2) * 100))

    return ci_lower, ci_upper


def sign_flip_permutation_p(deltas: list[float], n_perms: int = 10000) -> float:
    """Two-sided sign-flip permutation test for H0: mean delta = 0."""
    deltas_arr = np.array(deltas)
    k = len(deltas_arr)
    t_obs = abs(float(np.mean(deltas_arr)))

    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perms):
        signs = rng.choice([-1.0, 1.0], size=k)
        t_perm = abs(float(np.mean(signs * deltas_arr)))
        if t_perm >= t_obs:
            count += 1

    return (1 + count) / (1 + n_perms)


def trimmed_mean(data: list[float], fraction: float = TRIM_FRACTION) -> float:
    """Compute trimmed mean (remove fraction from each tail)."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    k = int(n * fraction)
    if 2 * k >= n:
        return float(np.median(data))
    return float(np.mean(sorted_data[k : n - k]))


def winsorized_std(data: list[float], fraction: float = TRIM_FRACTION) -> float:
    """Compute Winsorized standard deviation."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    k = int(n * fraction)
    winsorized = sorted_data[:]
    for i in range(k):
        winsorized[i] = sorted_data[k]
    for i in range(n - k, n):
        winsorized[i] = sorted_data[n - k - 1]
    return float(np.std(winsorized, ddof=1))


# ---------------------------------------------------------------------------
# Rendering (unchanged)
# ---------------------------------------------------------------------------


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
            "Generated by scripts/convergence_plots.py (500 training steps)",
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


def render_svg(language: str, losses: list[float], spikes: list[dict] | None = None) -> str:
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
    for step in (0, 100, 200, 300, 400, 499):
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

    # Spike markers
    spike_markers = ""
    if spikes:
        for s in spikes:
            sx_val = x_scale(s["step"])
            sy_val = y_scale(s["loss"])
            spike_markers += (
                f'  <circle cx="{sx_val:.2f}" cy="{sy_val:.2f}" r="5" '
                f'fill="none" stroke="#E53E3E" stroke-width="2" />\n'
            )
        # Add spike count annotation
        spike_markers += (
            f'  <text x="{width - margin_right}" y="48" text-anchor="end" '
            f'font-family="Menlo, Monaco, monospace" font-size="13" fill="#E53E3E">'
            f'{len(spikes)} spike{"s" if len(spikes) != 1 else ""}</text>\n'
        )

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
{spike_markers}
  <text x="{margin_left}" y="30" font-family="Avenir Next, Helvetica, Arial, sans-serif" font-size="24" fill="#19202D">{language.title()} Loss Convergence</text>
  <text x="{margin_left}" y="48" font-family="Menlo, Monaco, monospace" font-size="13" fill="#4C5566">initial={initial:.4f}  final={final:.4f}  reduction={reduction:.2f}%  steps={steps}</text>
  <text x="{margin_left + plot_w / 2:.2f}" y="{height - 20}" text-anchor="middle" font-family="Menlo, Monaco, monospace" font-size="12" fill="#6B7380">training step</text>
  <text x="20" y="{margin_top + plot_h / 2:.2f}" transform="rotate(-90 20 {margin_top + plot_h / 2:.2f})" text-anchor="middle" font-family="Menlo, Monaco, monospace" font-size="12" fill="#6B7380">loss</text>
</svg>
"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SVG_DIR.mkdir(parents=True, exist_ok=True)
    GO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GO_MOD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    only = _parse_only_flag(sys.argv)
    if only is not None:
        print(f"--only: running {', '.join(f'{l}_{d}' for l, d in sorted(only))}", file=sys.stderr)

    results: dict[tuple[str, str], list[list[float]]] = {}

    def _should_run(language: str, device: str) -> bool:
        """True if this combination needs fresh execution (not cache)."""
        return only is None or (language, device) in only

    def _run_or_cache(language: str, device: str, cwd: Path, cmd: list[str]) -> None:
        """Run trials or load from cache, storing into results."""
        key = (language, device)
        if _should_run(language, device):
            # Always run: use batch for Julia, per-process for others
            try:
                if language == "julia" and device in JULIA_BATCH_CMD:
                    batch_cwd, batch_cmd = JULIA_BATCH_CMD[device]
                    trials = run_trials_batch(language, batch_cwd, batch_cmd, device=device)
                else:
                    trials = run_trials(language, cwd, cmd, device=device)
                results[key] = trials
                _save_raw_cache(language, device, trials)
            except Exception as e:
                label = "FAILED" if device == "cpu" else "SKIPPED"
                print(f"[{language}/{device}] {label}: {e}", file=sys.stderr)
                # Fall back to cache on failure
                cached = _load_raw_cache(language, device)
                if cached is not None:
                    results[key] = cached
                    print(f"[{language}/{device}] using cached data as fallback", file=sys.stderr)
        else:
            # Not in --only: try loading from cache
            cached = _load_raw_cache(language, device)
            if cached is not None:
                results[key] = cached

    # CPU trials
    for language, cwd, cmd in LANGUAGE_RUNS_CPU:
        _run_or_cache(language, "cpu", cwd, cmd)

    # GPU trials (graceful skip if unavailable)
    for language, cwd, cmd in LANGUAGE_RUNS_GPU:
        _run_or_cache(language, "gpu", cwd, cmd)

    if not results:
        print("No convergence results collected.", file=sys.stderr)
        return 1

    # Compute per-(language, device) statistics
    stats: dict[tuple[str, str], dict] = {}
    for key, trials in results.items():
        language, device = key
        final_losses = [t[-1] for t in trials]
        spike_counts = [len(detect_spikes(t)) for t in trials]

        bca_ci = bca_bootstrap_ci(final_losses)

        stats[key] = {
            "language": language,
            "device": device,
            "n_trials": len(trials),
            "final_loss": {
                "mean": float(np.mean(final_losses)),
                "trimmed_mean": trimmed_mean(final_losses),
                "std": float(np.std(final_losses, ddof=1)),
                "winsorized_std": winsorized_std(final_losses),
                "bca_ci_lower": bca_ci[0],
                "bca_ci_upper": bca_ci[1],
                "median": float(np.median(final_losses)),
                "min": float(np.min(final_losses)),
                "max": float(np.max(final_losses)),
            },
            "spikes": {
                "median": float(np.median(spike_counts)),
                "iqr_25": float(np.percentile(spike_counts, 25)),
                "iqr_75": float(np.percentile(spike_counts, 75)),
                "counts": spike_counts,
            },
        }

    # Paired comparison: CPU vs GPU per language
    comparisons: dict[str, dict] = {}
    for language in ["rust", "go", "python", "julia"]:
        cpu_key = (language, "cpu")
        gpu_key = (language, "gpu")
        if cpu_key in results and gpu_key in results:
            cpu_finals = [t[-1] for t in results[cpu_key]]
            gpu_finals = [t[-1] for t in results[gpu_key]]
            n_paired = min(len(cpu_finals), len(gpu_finals))
            deltas = [cpu_finals[i] - gpu_finals[i] for i in range(n_paired)]
            perm_p = sign_flip_permutation_p(deltas)
            delta_ci = bca_bootstrap_ci(deltas)
            comparisons[language] = {
                "mean_delta": float(np.mean(deltas)),
                "bca_ci": [delta_ci[0], delta_ci[1]],
                "permutation_p": perm_p,
                "significant": delta_ci[0] > 0 or delta_ci[1] < 0,
            }

    # Save convergence_stats.json
    summary_json = {
        "n_trials": N_TRIALS,
        "n_bootstrap": N_BOOTSTRAP,
        "alpha": ALPHA,
        "stats": {f"{k[0]}_{k[1]}": v for k, v in stats.items()},
        "comparisons": comparisons,
    }
    (OUT_JSON_DIR / "convergence_stats.json").write_text(
        json.dumps(summary_json, indent=2) + "\n", encoding="utf-8"
    )

    # Render SVG per (language, device) using median trial
    for key, trials in results.items():
        language, device = key
        final_losses = [t[-1] for t in trials]
        median_idx = int(np.argsort(final_losses)[len(final_losses) // 2])
        representative = trials[median_idx]
        spikes = detect_spikes(representative)

        suffix = f"_{device}" if device == "gpu" else ""
        svg_path = OUT_SVG_DIR / f"{language}{suffix}.svg"
        svg_path.write_text(render_svg(language, representative, spikes=spikes), encoding="utf-8")

    # Also save per-trial JSON (one per language+device, preserving original format)
    for key, trials in results.items():
        language, device = key
        final_losses = [t[-1] for t in trials]
        median_idx = int(np.argsort(final_losses)[len(final_losses) // 2])
        representative = trials[median_idx]
        spikes = detect_spikes(representative)

        normalized = {
            "language": language,
            "device": device,
            "steps": len(representative),
            "losses": [round(v, 6) for v in representative],
            "spikes": spikes,
        }
        suffix = f"_{device}" if device == "gpu" else ""
        json_path = OUT_JSON_DIR / f"{language}{suffix}.json"
        json_path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")

    # Render GIF (CPU only, median trials)
    cpu_losses: dict[str, list[float]] = {}
    for language in ["rust", "go", "python", "julia"]:
        if (language, "cpu") in results:
            trials = results[(language, "cpu")]
            final_losses = [t[-1] for t in trials]
            median_idx = int(np.argsort(final_losses)[len(final_losses) // 2])
            cpu_losses[language] = trials[median_idx]
    if len(cpu_losses) == 4:
        render_demo_gif(cpu_losses, OUT_GIF_PATH)

    # Print summary table
    print(f"\nConvergence Summary (N={N_TRIALS} trials, BCa {int((1 - ALPHA) * 100)}% CI)")
    print(f"{'Lang':>8} {'Device':>6} {'TrimMean':>10} {'BCa CI':>24} {'Spikes':>14}")
    print("-" * 65)
    for key in sorted(stats.keys()):
        s = stats[key]
        fl = s["final_loss"]
        sp = s["spikes"]
        ci_str = f"[{fl['bca_ci_lower']:.6f}, {fl['bca_ci_upper']:.6f}]"
        spike_str = f"{sp['median']:.0f} ({sp['iqr_25']:.0f}-{sp['iqr_75']:.0f})"
        print(f"{s['language']:>8} {s['device']:>6} {fl['trimmed_mean']:>10.6f}  {ci_str:>22} {spike_str:>14}")

    if comparisons:
        print(f"\nCPU vs GPU Paired Comparison (sign-flip permutation)")
        print(f"{'Lang':>8} {'MeanDelta':>10} {'BCa CI':>24} {'p-value':>10} {'Sig':>5}")
        print("-" * 60)
        for lang, c in comparisons.items():
            ci_str = f"[{c['bca_ci'][0]:.6f}, {c['bca_ci'][1]:.6f}]"
            sig = "YES" if c["significant"] else "no"
            print(f"{lang:>8} {c['mean_delta']:>10.6f}  {ci_str:>22} {c['permutation_p']:>10.4f} {sig:>5}")

    print(f"\nSaved JSON : {OUT_JSON_DIR}")
    print(f"Saved SVG  : {OUT_SVG_DIR}")
    print(f"Saved GIF  : {OUT_GIF_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
