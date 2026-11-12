"""
FLAC-to-MP3 compiler in Python 3 (custom-built MP3 encoder + native FLAC decoder).
Reads a .flac file, decodes it to PCM, compiles a standard MPEG-1 Layer III (.mp3),
and generates step-by-step visualizations of the entire encoding pipeline.

Usage:
    python flac2mp3.py input.flac [output.mp3] [--bitrate 192] [--open]
"""

from __future__ import annotations

import argparse
import os
import time
import webbrowser
import numpy as np

from flac_decoder import decode_flac
from mp3_encoder import CustomMP3Encoder, EncodeStats, resample_audio
from mp3_tables import BITRATE_INDEX_MPEG1, SAMPLERATE_INDEX_MPEG1
from visualizer import generate_visualizations


def _pick_target_samplerate(src_sr: int) -> int:
    if src_sr in SAMPLERATE_INDEX_MPEG1:
        return src_sr
    if src_sr % 48000 == 0 or src_sr == 24000 or src_sr == 96000:
        return 48000
    return 44100


def _prepare_channels(pcm: np.ndarray) -> np.ndarray:
    ch = pcm.shape[1]
    if ch in (1, 2):
        return pcm
    # Downmix multi-channel audio (e.g. 5.1) to stereo
    left = pcm[:, 0]
    right = pcm[:, 1]
    if ch >= 3:
        center = pcm[:, 2] * 0.7071
        left = left + center
        right = right + center
    if ch >= 5:
        left = left + pcm[:, 4] * 0.5
        right = right + (pcm[:, 5] * 0.5 if ch >= 6 else pcm[:, 4] * 0.5)
    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo)))
    if peak > 1.0:
        stereo = stereo / peak
    return stereo


def compile_flac_to_mp3(
    input_flac: str,
    output_mp3: str | None = None,
    bitrate_kbps: int = 192,
    visualize: bool = True,
    vis_dir: str | None = None,
    open_browser: bool = False,
) -> tuple[str, EncodeStats, dict | None]:
    t0 = time.perf_counter()
    if not os.path.isfile(input_flac):
        raise FileNotFoundError(f"FLAC file not found: {input_flac}")

    if output_mp3 is None:
        stem, _ = os.path.splitext(input_flac)
        output_mp3 = stem + ".mp3"

    flac_size = os.path.getsize(input_flac)
    print("\n=== Custom Python 3 FLAC -> MP3 Compiler ===")
    print(f"[Step 1/7] Decoding FLAC file: {input_flac} ({flac_size / 1024:.1f} KB)")
    pcm, flac_meta = decode_flac(input_flac)
    print(
        f"           -> {flac_meta.sample_rate} Hz, {flac_meta.channels} channel(s), "
        f"{flac_meta.bits_per_sample}-bit, {len(pcm):,} samples "
        f"({len(pcm) / flac_meta.sample_rate:.2f} s)"
    )

    pcm = _prepare_channels(pcm)
    target_sr = _pick_target_samplerate(flac_meta.sample_rate)
    if target_sr != flac_meta.sample_rate:
        print(f"           -> Resampling from {flac_meta.sample_rate} Hz to {target_sr} Hz (MPEG-1)...")
        pcm = resample_audio(pcm, flac_meta.sample_rate, target_sr)

    print("[Step 2/7] Initializing 32-band polyphase analysis filterbank (512-tap ISO 11172-3)...")
    print("[Step 3/7] Initializing 36->18 MDCT + forward aliasing reduction (576 spectral lines)...")
    print("[Step 4/7] Initializing psychoacoustic model (1024-pt FFT, Bark spreading, ATH)...")
    print(f"[Step 5/7] Running non-uniform quantization & Rate-Distortion loop ({bitrate_kbps} kbps)...")
    print("[Step 6/7] Packing Huffman codes (34 ISO tables) and bit reservoir...")

    encoder = CustomMP3Encoder(
        sample_rate=target_sr,
        channels=pcm.shape[1],
        bitrate_kbps=bitrate_kbps,
    )

    def on_progress(done: int, total: int) -> None:
        pct = 100.0 * done / max(total, 1)
        print(f"           Compiling MP3 frames: {done}/{total} ({pct:5.1f}%)", end="\r", flush=True)

    mp3_bytes, stats = encoder.encode(pcm, original_sr=flac_meta.sample_rate, progress_cb=on_progress)
    print()

    out_parent = os.path.dirname(os.path.abspath(output_mp3))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    with open(output_mp3, "wb") as f:
        f.write(mp3_bytes)

    avg_snr = (sum(stats.frame_snr_db) / len(stats.frame_snr_db)) if stats.frame_snr_db else 0.0
    elapsed = time.perf_counter() - t0
    print(
        f"[Step 7/7] Compilation complete! -> {output_mp3} ({stats.mp3_bytes / 1024:.1f} KB, "
        f"{stats.num_frames} frames, avg SNR={avg_snr:.1f} dB, time={elapsed:.2f} s)"
    )

    vis_info = None
    if visualize:
        if vis_dir is None:
            stem, _ = os.path.splitext(output_mp3)
            vis_dir = stem + "_visualization"
        print(f"\nGenerating step-by-step visualizations in: {vis_dir}")
        vis_info = generate_visualizations(
            out_dir=vis_dir,
            input_flac=input_flac,
            output_mp3=output_mp3,
            pcm=pcm,
            flac_meta=flac_meta,
            stats=stats,
            flac_size_bytes=flac_size,
        )
        print(f"  -> HTML report: {vis_info['html']}")
        for p in vis_info["pngs"]:
            print(f"  -> Image: {os.path.basename(p)}")
        if open_browser:
            webbrowser.open("file:///" + os.path.abspath(vis_info["html"]).replace("\\", "/"))

    return output_mp3, stats, vis_info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Custom Python 3 FLAC-to-MP3 compiler with step-by-step visualizations."
    )
    parser.add_argument("input_flac", help="Path to input .flac file")
    parser.add_argument("output_mp3", nargs="?", default=None, help="Path to output .mp3 file (optional)")
    parser.add_argument(
        "-b", "--bitrate",
        type=int,
        default=192,
        choices=sorted(BITRATE_INDEX_MPEG1.keys()),
        help="MP3 bitrate in kbps (default: 192)",
    )
    parser.add_argument("--vis-dir", default=None, help="Output directory for visualizations (PNG + HTML)")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization generation")
    parser.add_argument("--open", action="store_true", help="Open HTML visualization in browser after compilation")

    args = parser.parse_args(argv)
    compile_flac_to_mp3(
        input_flac=args.input_flac,
        output_mp3=args.output_mp3,
        bitrate_kbps=args.bitrate,
        visualize=not args.no_vis,
        vis_dir=args.vis_dir,
        open_browser=args.open,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
