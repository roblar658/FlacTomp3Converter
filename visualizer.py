"""
Step-by-step visualization generator for the FLAC -> MP3 compiler.
Produces 7 high-resolution matplotlib charts (PNG) covering every stage of the
pipeline plus an interactive HTML report (index.html).
"""

from __future__ import annotations

import base64
import html
import os
from typing import Dict
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from flac_decoder import FlacMetadata
from mp3_encoder import EncodeStats
from mp3_tables import SFB_LONG


_DARK_BG = "#0f172a"
_PANEL_BG = "#1e293b"
_GRID_CLR = "#334155"
_TEXT_CLR = "#e2e8f0"
_MUTED_CLR = "#94a3b8"
_ACCENT_1 = "#38bdf8"  # Cyan
_ACCENT_2 = "#f472b6"  # Pink
_ACCENT_3 = "#34d399"  # Emerald
_ACCENT_4 = "#fbbf24"  # Amber
_ACCENT_5 = "#a78bfa"  # Violet


def _style_ax(ax) -> None:
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors=_TEXT_CLR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_GRID_CLR)
    ax.grid(True, color=_GRID_CLR, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.xaxis.label.set_color(_TEXT_CLR)
    ax.yaxis.label.set_color(_TEXT_CLR)
    ax.title.set_color("#ffffff")


def _new_fig(figsize=(11.5, 6.2)):
    return plt.figure(figsize=figsize, dpi=140, facecolor=_DARK_BG)


def _plot_step1_flac_pcm(
    out_dir: str,
    pcm: np.ndarray,
    flac_meta: FlacMetadata,
    stats: EncodeStats,
    flac_size_bytes: int,
) -> str:
    fig = _new_fig((11.5, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.1], hspace=0.38)

    ax1 = fig.add_subplot(gs[0])
    _style_ax(ax1)
    n = len(pcm)
    step = max(1, n // 4000)
    t_all = np.arange(0, n, step) / float(stats.sample_rate)
    ax1.plot(t_all, pcm[::step, 0], color=_ACCENT_1, lw=0.9, label="Channel 0 (Left)")
    if pcm.shape[1] > 1:
        ax1.plot(t_all, pcm[::step, 1], color=_ACCENT_2, lw=0.7, alpha=0.65, label="Channel 1 (Right)")
    vis_t0 = (stats.vis_frame_idx * 1152) / float(stats.sample_rate)
    vis_t1 = ((stats.vis_frame_idx + 1) * 1152) / float(stats.sample_rate)
    ax1.axvspan(vis_t0, vis_t1, color=_ACCENT_4, alpha=0.28, label=f"Analyzed frame #{stats.vis_frame_idx}")
    ax1.set_title("Step 1: Native FLAC Decoding -> Normalized PCM Waveform", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(-1.05, 1.05)
    ax1.legend(loc="upper right", facecolor=_DARK_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=8.5)

    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2)
    frame = stats.vis_pcm_frame
    if frame is None:
        frame = np.zeros(1152)
    idx = np.arange(len(frame))
    ax2.plot(idx[:576], frame[:576], color=_ACCENT_1, lw=1.2, label="Granule 0 (576 samples)")
    ax2.plot(idx[576:], frame[576:], color=_ACCENT_3, lw=1.2, label="Granule 1 (576 samples)")
    ax2.axvline(576, color=_ACCENT_4, linestyle="--", lw=1.2)
    ax2.set_title(
        f"Zoom on MP3 Frame #{stats.vis_frame_idx} (1152 samples = 2 granules) | "
        f"FLAC: {flac_meta.sample_rate} Hz, {flac_meta.bits_per_sample}-bit, {flac_meta.channels} ch ({flac_size_bytes/1024:.1f} KB)",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax2.set_xlabel("Sample index within frame (0..1151)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper right", facecolor=_DARK_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=8.5)

    kinds: Dict[str, int] = {}
    for fs in flac_meta.frame_stats:
        for k in fs["subframes"]:
            kinds[k] = kinds.get(k, 0) + 1
    top_kinds = ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items(), key=lambda x: -x[1])[:4])
    fig.text(
        0.5,
        0.015,
        f"Observed FLAC predictors: {top_kinds or 'N/A'} | Target MPEG-1 Layer III rate: {stats.sample_rate} Hz",
        ha="center",
        color=_MUTED_CLR,
        fontsize=9,
    )

    path = os.path.join(out_dir, "step_1_flac_to_pcm.png")
    fig.savefig(path, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    return path


def _plot_step2_polyphase(out_dir: str, stats: EncodeStats) -> str:
    fig = _new_fig((11.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.22)

    sb = stats.vis_subbands
    if sb is None:
        sb = np.zeros((18, 32))
    sb_db = 20.0 * np.log10(np.maximum(np.abs(sb), 1e-6))

    ax1 = fig.add_subplot(gs[0])
    _style_ax(ax1)
    nyq_khz = (stats.sample_rate / 2.0) / 1000.0
    im = ax1.imshow(
        sb_db.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, 18, 0, 32],
        vmin=-90,
        vmax=max(-10, float(np.max(sb_db))),
    )
    ax1.set_title("Step 2: 32-Band Polyphase Analysis Filterbank (32 subbands x 18 slots)", fontsize=11.5, fontweight="bold", pad=8)
    ax1.set_xlabel("Time slot in granule 0 (0..17, 32 samples per slot)")
    ax1.set_ylabel(f"Subband (0..31, covering 0 - {nyq_khz:.1f} kHz)")
    cbar = fig.colorbar(im, ax=ax1, pad=0.02)
    cbar.set_label("Amplitude (dB)", color=_TEXT_CLR)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_CLR, labelcolor=_TEXT_CLR)

    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2)
    mean_db = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(sb ** 2, axis=0)), 1e-6))
    bands = np.arange(32)
    ax2.barh(bands, mean_db + 90.0, left=-90.0, color=_ACCENT_1, edgecolor=_DARK_BG, height=0.78)
    ax2.set_title("Mean RMS Energy per Subband", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("RMS Level (dB)")
    ax2.set_ylabel("Subband Index")
    ax2.set_xlim(-90, max(-5, float(np.max(mean_db)) + 5))
    ax2.set_ylim(-0.5, 31.5)

    path = os.path.join(out_dir, "step_2_polyphase_filterbank.png")
    fig.savefig(path, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    return path


def _plot_step3_mdct(out_dir: str, stats: EncodeStats) -> str:
    fig = _new_fig((11.5, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.75, 1.3], hspace=0.40)

    ax1 = fig.add_subplot(gs[0])
    _style_ax(ax1)
    n = np.arange(36)
    w = np.sin(np.pi / 36.0 * (n + 0.5))
    ax1.plot(n, w, color=_ACCENT_3, lw=2.2, label="36-point sine window w[n]")
    ax1.fill_between(n[:18], 0, w[:18], color=_ACCENT_1, alpha=0.22, label="Previous granule (18 subband samples)")
    ax1.fill_between(n[18:], 0, w[18:], color=_ACCENT_2, alpha=0.22, label="Current granule (18 subband samples)")
    ax1.axvline(18, color=_ACCENT_4, linestyle="--", lw=1.1)
    ax1.set_title("Step 3a: Overlapping 36-Point MDCT Window (TDAC) per Subband -> 18 Frequency Lines", fontsize=11, fontweight="bold", pad=6)
    ax1.set_xlabel("Time slot n (0..35)")
    ax1.set_ylabel("Window Weight")
    ax1.legend(loc="lower center", ncol=3, facecolor=_DARK_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=8.5)

    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2)
    raw = stats.vis_mdct_raw if stats.vis_mdct_raw is not None else np.zeros(576)
    ali = stats.vis_mdct_aliased if stats.vis_mdct_aliased is not None else np.zeros(576)
    raw_db = 20.0 * np.log10(np.maximum(np.abs(raw), 1e-7))
    ali_db = 20.0 * np.log10(np.maximum(np.abs(ali), 1e-7))
    lines = np.arange(576)
    ax2.plot(lines, raw_db, color=_ACCENT_1, lw=1.0, alpha=0.8, label="MDCT before aliasing reduction")
    ax2.plot(lines, ali_db, color=_ACCENT_2, lw=1.0, alpha=0.85, linestyle="--", label="MDCT after forward aliasing reduction (butterfly)")
    for b in range(18, 576, 18):
        ax2.axvline(b, color=_GRID_CLR, lw=0.5, alpha=0.45)
    ax2.set_title("Step 3b: 576 MDCT Spectral Lines (32 Subbands x 18 Lines) & Aliasing Reduction", fontsize=11, fontweight="bold", pad=6)
    ax2.set_xlabel("MDCT Frequency Line (0..575)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_xlim(0, 575)
    ax2.set_ylim(-100, max(0, float(np.max(ali_db)) + 5))
    ax2.legend(loc="upper right", facecolor=_DARK_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=8.5)

    path = os.path.join(out_dir, "step_3_mdct_and_aliasing.png")
    fig.savefig(path, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    return path


def _plot_step4_psychoacoustic(out_dir: str, stats: EncodeStats) -> str:
    fig = _new_fig((11.5, 6.2))
    ax = fig.add_subplot(111)
    _style_ax(ax)

    freqs = stats.vis_fft_freqs if stats.vis_fft_freqs is not None else np.linspace(20, 22050, 513)
    fft_db = stats.vis_fft_db if stats.vis_fft_db is not None else np.full(513, -100.0)
    sfb_hz = stats.vis_sfb_centers_hz if stats.vis_sfb_centers_hz is not None else np.linspace(100, 18000, 21)
    sfb_e = stats.vis_sfb_energy_db if stats.vis_sfb_energy_db is not None else np.full(21, -80.0)
    sfb_ath = stats.vis_sfb_ath_db if stats.vis_sfb_ath_db is not None else np.full(21, -80.0)
    sfb_mask = stats.vis_sfb_mask_db if stats.vis_sfb_mask_db is not None else np.full(21, -70.0)

    ax.plot(freqs[1:], fft_db[1:], color=_ACCENT_1, lw=0.9, alpha=0.65, label="1024-pt FFT Power Spectrum")
    ax.plot(sfb_hz, sfb_e, "o-", color=_ACCENT_4, lw=1.8, markersize=4.5, label="Energy per Scalefactor Band (21 bands)")
    ax.plot(sfb_hz, sfb_ath, "--", color=_MUTED_CLR, lw=1.6, label="Absolute Threshold of Hearing (ATH - Terhardt)")
    ax.plot(sfb_hz, sfb_mask, "s-", color=_ACCENT_2, lw=2.0, markersize=4.5, label="Masking Threshold (Schroeder spreading)")
    ax.fill_between(sfb_hz, -120, sfb_mask, color=_ACCENT_2, alpha=0.16, label="Inaudible Zone (masked quantization noise)")

    ax.set_xscale("log")
    ax.set_xlim(30, stats.sample_rate / 2.0)
    ax.set_ylim(-110, max(10, float(np.max(sfb_e)) + 10))
    ax.set_title("Step 4: Psychoacoustic Model — Audible Signal vs. Masking Threshold", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Frequency (Hz, log scale)")
    ax.set_ylabel("Level (dB)")
    ax.legend(loc="lower left", facecolor=_DARK_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=8.5)

    path = os.path.join(out_dir, "step_4_psychoacoustic_model.png")
    fig.savefig(path, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    return path


def _plot_step5_quantization_rd(out_dir: str, stats: EncodeStats) -> str:
    fig = _new_fig((11.5, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.95], hspace=0.40, wspace=0.22)

    xr = stats.vis_mdct_aliased if stats.vis_mdct_aliased is not None else np.zeros(576)
    xr_deq = stats.vis_xr_dequant if stats.vis_xr_dequant is not None else np.zeros(576)
    ix = stats.vis_ix if stats.vis_ix is not None else np.zeros(576, dtype=np.int32)
    sfs = stats.vis_scalefacs if stats.vis_scalefacs is not None else [0] * 21

    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1)
    lines = np.arange(576)
    ax1.plot(lines, 20 * np.log10(np.maximum(np.abs(xr), 1e-7)), color=_ACCENT_1, lw=1.0, label="Original |xr|")
    ax1.plot(lines, 20 * np.log10(np.maximum(np.abs(xr_deq), 1e-7)), color=_ACCENT_3, lw=0.9, alpha=0.85, linestyle="--", label="Dequantized |xr_deq|")
    err_db = 20 * np.log10(np.maximum(np.abs(xr - xr_deq), 1e-7))
    ax1.plot(lines, err_db, color=_ACCENT_2, lw=0.7, alpha=0.65, label="Quantization noise |xr - xr_deq|")
    ax1.set_title("Step 5a: Original vs. Quantized Spectrum", fontsize=10.5, fontweight="bold", pad=6)
    ax1.set_xlabel("Frequency Line (0..575)")
    ax1.set_ylabel("Level (dB)")
    ax1.set_ylim(-100, max(5, float(np.max(20 * np.log10(np.maximum(np.abs(xr), 1e-7)))) + 5))
    ax1.legend(loc="upper right", facecolor=_DARK_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=7.8)

    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2)
    ax2.stem(lines, ix, linefmt=_ACCENT_4, markerfmt=" ", basefmt=_GRID_CLR)
    ax2.set_title(f"Step 5b: Non-Linear Quantized Integers ix (|xr|^(3/4)), global_gain={stats.vis_global_gain}", fontsize=10.5, fontweight="bold", pad=6)
    ax2.set_xlabel("Frequency Line (0..575)")
    ax2.set_ylabel("Integer Value ix")

    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3)
    hist = stats.vis_gain_history or [(stats.vis_global_gain, 0)]
    steps = np.arange(1, len(hist) + 1)
    gains = [h[0] for h in hist]
    bits = [h[1] for h in hist]
    ax3.plot(steps, bits, "o-", color=_ACCENT_1, lw=1.8, markersize=5, label="Huffman + SF bits")
    for s_i, g_i, b_i in zip(steps, gains, bits):
        ax3.annotate(f"g={g_i}", (s_i, b_i), textcoords="offset points", xytext=(0, 6), ha="center", color=_ACCENT_4, fontsize=7.5)
    ax3.set_title("Step 5c: Inner Rate Loop (Binary Search on global_gain)", fontsize=10.5, fontweight="bold", pad=6)
    ax3.set_xlabel("Search Iteration")
    ax3.set_ylabel("Bits Used")
    ax3.set_ylim(0, max(bits) * 1.18 if bits else 100)
    ax3.legend(loc="upper left", facecolor=_DARK_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4)
    ax4.bar(np.arange(21), sfs, color=_ACCENT_5, edgecolor=_DARK_BG)
    ax4.set_title("Step 5d: Band Scalefactors (scalefac_l) Across Bands 0..20", fontsize=10.5, fontweight="bold", pad=6)
    ax4.set_xlabel("Scalefactor Band (sfb)")
    ax4.set_ylabel("Scalefactor Index")
    ax4.set_xticks(range(0, 21, 2))
    if max(sfs) == 0:
        ax4.set_ylim(0, 4)
        ax4.text(
            10,
            2.0,
            "All scalefactors = 0\n(quantization noise is already below masking threshold)",
            ha="center",
            va="center",
            color=_MUTED_CLR,
            fontsize=8.5,
        )

    path = os.path.join(out_dir, "step_5_quantization_and_rd.png")
    fig.savefig(path, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    return path


def _plot_step6_huffman_reservoir(out_dir: str, stats: EncodeStats) -> str:
    fig = _new_fig((11.5, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.42)

    ax1 = fig.add_subplot(gs[0])
    _style_ax(ax1)
    ix = stats.vis_ix if stats.vis_ix is not None else np.zeros(576, dtype=np.int32)
    plan = stats.vis_region_info or {
        "big_values": 0,
        "region0_count": 0,
        "region1_count": 0,
        "table_select": [0, 0, 0],
        "count1table_select": 0,
        "rzero_start": 0,
        "part2_bits": 0,
        "part2_3_length": 0,
    }
    bounds = SFB_LONG[stats.sample_rate]
    bv_end = plan["big_values"] * 2
    r1 = min(bounds[plan["region0_count"] + 1], bv_end)
    r2 = min(bounds[min(22, plan["region0_count"] + plan["region1_count"] + 2)], bv_end)
    rz = plan["rzero_start"]
    t_sel = plan["table_select"]

    ax1.axvspan(0, r1, color=_ACCENT_1, alpha=0.22)
    ax1.axvspan(r1, r2, color=_ACCENT_4, alpha=0.22)
    ax1.axvspan(r2, bv_end, color=_ACCENT_5, alpha=0.22)
    ax1.axvspan(bv_end, rz, color=_ACCENT_3, alpha=0.22)
    ax1.axvspan(rz, 576, color=_MUTED_CLR, alpha=0.14)
    ax1.plot(np.arange(576), np.abs(ix), color="#ffffff", lw=0.9)

    c1_name = "Table A (32)" if plan["count1table_select"] == 0 else "Table B (33)"
    legend_patches = [
        Patch(facecolor=_ACCENT_1, alpha=0.45, label=f"Region 0 [0:{r1}] -> Huff #{t_sel[0]}"),
        Patch(facecolor=_ACCENT_4, alpha=0.45, label=f"Region 1 [{r1}:{r2}] -> Huff #{t_sel[1]}"),
        Patch(facecolor=_ACCENT_5, alpha=0.45, label=f"Region 2 [{r2}:{bv_end}] -> Huff #{t_sel[2]}"),
        Patch(facecolor=_ACCENT_3, alpha=0.45, label=f"count1 [{bv_end}:{rz}] -> {c1_name}"),
        Patch(facecolor=_MUTED_CLR, alpha=0.35, label=f"rzero [{rz}:576] -> 0 bits"),
    ]
    ax1.set_title(
        f"Step 6a: Huffman Partitioning of 576 Lines (SF={plan['part2_bits']}b + Huff={plan['part2_3_length']-plan['part2_bits']}b = {plan['part2_3_length']} bits)",
        fontsize=11,
        fontweight="bold",
        pad=6,
    )
    ax1.set_xlabel("Frequency Line (0..575)")
    ax1.set_ylabel("|ix|")
    ax1.set_xlim(0, 575)
    ax1.legend(handles=legend_patches, loc="upper right", facecolor=_DARK_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=8)

    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2)
    frames = np.arange(len(stats.frame_main_bits))
    ax2.plot(frames, stats.frame_main_bits, color=_ACCENT_1, lw=1.3, label="Used main_data bits per frame")
    ax2.plot(frames, stats.frame_slot_bits, color=_ACCENT_4, lw=1.1, linestyle="--", label="Fixed frame slot capacity (slot_bits)")
    ax2_r = ax2.twinx()
    ax2_r.plot(frames, stats.frame_reservoir_bytes, color=_ACCENT_2, lw=1.2, alpha=0.85, label="Bit reservoir (main_data_begin bytes)")
    ax2_r.set_ylabel("Bit Reservoir (bytes, max 511)", color=_ACCENT_2)
    ax2_r.tick_params(colors=_ACCENT_2, labelsize=8.5)
    ax2.set_title("Step 6b: Bit Allocation and Bit Reservoir Across the Full MP3 Stream", fontsize=11, fontweight="bold", pad=6)
    ax2.set_xlabel("MP3 Frame Index")
    ax2.set_ylabel("Bits per Frame")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", facecolor=_DARK_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=8)

    path = os.path.join(out_dir, "step_6_huffman_and_reservoir.png")
    fig.savefig(path, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    return path


def _plot_step7_summary(
    out_dir: str,
    stats: EncodeStats,
    flac_size_bytes: int,
) -> str:
    fig = _new_fig((11.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.25)

    ax1 = fig.add_subplot(gs[0])
    _style_ax(ax1)
    labels = ["Uncompressed PCM\n(16-bit)", "Input FLAC\n(Lossless)", f"Custom MP3\n({stats.bitrate_kbps} kbps)"]
    sizes_kb = [stats.raw_pcm_bytes / 1024.0, flac_size_bytes / 1024.0, stats.mp3_bytes / 1024.0]
    colors = [_MUTED_CLR, _ACCENT_1, _ACCENT_3]
    bars = ax1.bar(labels, sizes_kb, color=colors, width=0.55, edgecolor=_DARK_BG)
    for bar, sz in zip(bars, sizes_kb):
        ratio = stats.raw_pcm_bytes / max(sz * 1024.0, 1.0)
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(sizes_kb) * 0.02,
            f"{sz:.1f} KB\n({ratio:.1f}x)",
            ha="center",
            va="bottom",
            color=_TEXT_CLR,
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_title("Step 7a: File Size & Compression Ratio", fontsize=11, fontweight="bold", pad=8)
    ax1.set_ylabel("Size (KB)")
    ax1.set_ylim(0, max(sizes_kb) * 1.22)

    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2)
    frames = np.arange(len(stats.frame_snr_db))
    ax2.plot(frames, stats.frame_snr_db, color=_ACCENT_3, lw=1.4, label="Spectral SNR per frame (dB)")
    ax2_r = ax2.twinx()
    ax2_r.plot(frames, stats.frame_global_gains, color=_ACCENT_4, lw=1.1, alpha=0.85, label="Mean global_gain")
    ax2_r.set_ylabel("global_gain (0..255)", color=_ACCENT_4)
    ax2_r.tick_params(colors=_ACCENT_4, labelsize=8.5)
    avg_snr = float(np.mean(stats.frame_snr_db)) if stats.frame_snr_db else 0.0
    ax2.set_title(f"Step 7b: Quantization SNR Across Track (avg {avg_snr:.1f} dB)", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("MP3 Frame Index")
    ax2.set_ylabel("SNR (dB)")
    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2_r.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, loc="lower right", facecolor=_DARK_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=8.5)

    path = os.path.join(out_dir, "step_7_result_and_comparison.png")
    fig.savefig(path, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    return path


def generate_visualizations(
    out_dir: str,
    input_flac: str,
    output_mp3: str,
    pcm: np.ndarray,
    flac_meta: FlacMetadata,
    stats: EncodeStats,
    flac_size_bytes: int,
) -> Dict[str, object]:
    """
    Generates 7 step-by-step PNG charts and an interactive HTML report (index.html).
    Returns a dict with file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    pngs = [
        _plot_step1_flac_pcm(out_dir, pcm, flac_meta, stats, flac_size_bytes),
        _plot_step2_polyphase(out_dir, stats),
        _plot_step3_mdct(out_dir, stats),
        _plot_step4_psychoacoustic(out_dir, stats),
        _plot_step5_quantization_rd(out_dir, stats),
        _plot_step6_huffman_reservoir(out_dir, stats),
        _plot_step7_summary(out_dir, stats, flac_size_bytes),
    ]

    steps_info = [
        (
            "Step 1: Native FLAC Decoding to PCM",
            os.path.basename(pngs[0]),
            f"Pure Python decoder read <code>{html.escape(os.path.basename(input_flac))}</code> ({flac_size_bytes/1024:.1f} KB), "
            f"unpacked Rice-coded residuals and LPC/FIXED linear predictors, and reconstructed {flac_meta.total_samples:,} samples "
            f"({flac_meta.channels} ch, {flac_meta.bits_per_sample}-bit @ {flac_meta.sample_rate} Hz).",
        ),
        (
            "Step 2: 32-Band Polyphase Analysis Filterbank",
            os.path.basename(pngs[1]),
            "Each granule (576 samples) is passed through a 512-tap prototype FIR window (ISO 11172-3 Table 3-C.1) "
            "and 64->32 cosine modulation, splitting the signal into 32 uniform frequency subbands over 18 time slots.",
        ),
        (
            "Step 3: MDCT & Forward Aliasing Reduction",
            os.path.basename(pngs[2]),
            "Odd-indexed subbands undergo time-slot sign inversion before 18 previous + 18 current subband slots are multiplied "
            "by a 36-point sine window and transformed via MDCT into 576 spectral lines (32 x 18). Adjacent-band aliasing "
            "is then cancelled via 8 orthogonal butterfly rotations.",
        ),
        (
            "Step 4: Psychoacoustic Model (Masking & ATH)",
            os.path.basename(pngs[3]),
            "A 1024-point Hann-windowed FFT and Schroeder Bark-scale spreading compute which frequencies are masked by "
            "nearby louder tones or fall below the Absolute Threshold of Hearing (ATH). Quantization noise below the pink curve is inaudible.",
        ),
        (
            "Step 5: Non-Linear Quantization & Rate-Distortion Loop",
            os.path.basename(pngs[4]),
            f"Spectral lines are compressed via power-law companding |xr|^(3/4). The inner loop binary-searches "
            f"<code>global_gain</code> (converging at {stats.vis_global_gain}), while the outer loop boosts band scalefactors "
            "whenever quantization distortion exceeds the masking threshold.",
        ),
        (
            "Step 6: Huffman Entropy Coding & Bit Reservoir",
            os.path.basename(pngs[5]),
            "The 576 quantized integers are partitioned into <code>big_values</code> (3 regions using optimal ISO Huffman tables), "
            "<code>count1</code> (-1/0/+1 quadruples), and <code>rzero</code> (free zero tail). "
            "The bit reservoir (<code>main_data_begin</code> up to 511 bytes) smooths bit demand across frames.",
        ),
        (
            "Step 7: Compression & Audio Quality Summary",
            os.path.basename(pngs[6]),
            f"Compiled MP3: <code>{html.escape(os.path.basename(output_mp3))}</code> ({stats.mp3_bytes/1024:.1f} KB, "
            f"{stats.num_frames} frames @ {stats.bitrate_kbps} kbps). Compression ratio vs. raw PCM: "
            f"{stats.raw_pcm_bytes / max(stats.mp3_bytes, 1):.1f}x, average spectral SNR: "
            f"{(sum(stats.frame_snr_db)/len(stats.frame_snr_db)) if stats.frame_snr_db else 0:.1f} dB.",
        ),
    ]

    cards_html = []
    nav_html = []
    for i, (title, img_name, desc) in enumerate(steps_info, start=1):
        img_path = os.path.join(out_dir, img_name)
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        active_cls = " active" if i == 1 else ""
        nav_html.append(
            f'<button class="tab-btn{active_cls}" onclick="showStep({i})" id="btn-{i}">'
            f'<span class="step-num">Step {i}</span><span class="step-lbl">{html.escape(title.split(": ", 1)[1])}</span></button>'
        )
        cards_html.append(
            f'<section class="step-card{active_cls}" id="step-{i}">'
            f'<h2>{html.escape(title)}</h2>'
            f'<p class="desc">{desc}</p>'
            f'<div class="img-wrap"><img src="data:image/png;base64,{b64}" alt="{html.escape(title)}"/></div>'
            f'</section>'
        )

    avg_snr = (sum(stats.frame_snr_db) / len(stats.frame_snr_db)) if stats.frame_snr_db else 0.0
    comp_ratio = stats.raw_pcm_bytes / max(stats.mp3_bytes, 1)
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Custom MP3 Compiler — Step-by-Step Visualization</title>
<style>
  :root {{ --bg:#0f172a; --card:#1e293b; --border:#334155; --text:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }}
  header {{ padding:20px 28px; border-bottom:1px solid var(--border); background:#0b1222; display:flex; flex-wrap:wrap; gap:16px; align-items:center; justify-content:space-between; }}
  header h1 {{ margin:0; font-size:20px; color:#fff; }}
  .badges {{ display:flex; gap:10px; flex-wrap:wrap; }}
  .badge {{ background:var(--card); border:1px solid var(--border); padding:6px 12px; border-radius:999px; font-size:12.5px; color:var(--accent); }}
  main {{ max-width:1200px; margin:0 auto; padding:22px; }}
  .tabs {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:8px; margin-bottom:18px; }}
  .tab-btn {{ background:var(--card); border:1px solid var(--border); color:var(--text); padding:10px 12px; border-radius:10px; cursor:pointer; text-align:left; transition:0.15s; display:flex; flex-direction:column; gap:3px; }}
  .tab-btn:hover {{ border-color:var(--accent); }}
  .tab-btn.active {{ background:#0c2d48; border-color:var(--accent); box-shadow:0 0 0 1px var(--accent); }}
  .step-num {{ font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:var(--accent); font-weight:700; }}
  .step-lbl {{ font-size:12.5px; color:#fff; font-weight:500; }}
  .controls {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }}
  .nav-actions button {{ background:var(--card); border:1px solid var(--border); color:#fff; padding:8px 14px; border-radius:8px; cursor:pointer; font-size:13px; }}
  .nav-actions button:hover {{ border-color:var(--accent); }}
  .step-card {{ display:none; background:var(--card); border:1px solid var(--border); border-radius:14px; padding:20px; }}
  .step-card.active {{ display:block; }}
  .step-card h2 {{ margin:0 0 8px 0; font-size:18px; color:#fff; }}
  .desc {{ margin:0 0 16px 0; color:var(--muted); line-height:1.55; font-size:14px; }}
  .desc code {{ background:#0f172a; padding:2px 6px; border-radius:4px; color:var(--accent); }}
  .img-wrap {{ background:#0f172a; border:1px solid var(--border); border-radius:10px; padding:8px; text-align:center; }}
  .img-wrap img {{ max-width:100%; height:auto; border-radius:6px; }}
  .all-mode .step-card {{ display:block; margin-bottom:20px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>Custom Python 3 MP3 Compiler — Pipeline Visualization</h1>
    <div style="color:var(--muted); font-size:13px; margin-top:4px;">
      {html.escape(os.path.basename(input_flac))} &rarr; {html.escape(os.path.basename(output_mp3))}
    </div>
  </div>
  <div class="badges">
    <span class="badge">{stats.sample_rate} Hz &bull; {stats.channels} ch</span>
    <span class="badge">{stats.bitrate_kbps} kbps CBR</span>
    <span class="badge">{stats.num_frames} MP3 frames</span>
    <span class="badge">{comp_ratio:.1f}x compression</span>
    <span class="badge">Avg SNR: {avg_snr:.1f} dB</span>
  </div>
</header>
<main id="container">
  <div class="controls">
    <div class="nav-actions">
      <button onclick="prevStep()">&larr; Previous Step</button>
      <button onclick="nextStep()">Next Step &rarr;</button>
    </div>
    <div class="nav-actions">
      <button onclick="toggleAll()" id="all-btn">Show All Steps</button>
    </div>
  </div>
  <div class="tabs">{''.join(nav_html)}</div>
  {''.join(cards_html)}
</main>
<script>
  let current = 1;
  let showAll = false;
  function showStep(n) {{
    showAll = false;
    document.getElementById('container').classList.remove('all-mode');
    document.getElementById('all-btn').textContent = 'Show All Steps';
    current = n;
    for (let i = 1; i <= 7; i++) {{
      document.getElementById('step-' + i).classList.toggle('active', i === n);
      document.getElementById('btn-' + i).classList.toggle('active', i === n);
    }}
  }}
  function prevStep() {{ showStep(current > 1 ? current - 1 : 7); }}
  function nextStep() {{ showStep(current < 7 ? current + 1 : 1); }}
  function toggleAll() {{
    showAll = !showAll;
    document.getElementById('container').classList.toggle('all-mode', showAll);
    document.getElementById('all-btn').textContent = showAll ? 'Show One Step at a Time' : 'Show All Steps';
  }}
  document.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight') nextStep();
    if (e.key === 'ArrowLeft') prevStep();
  }});
</script>
</body>
</html>
"""
    html_path = os.path.join(out_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    return {"html": html_path, "pngs": pngs}
