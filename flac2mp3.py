"""
FLAC til MP3-kompilator i Python 3 (helt egenlaget MP3-enkoder + FLAC-dekoder).
Tar inn en .flac-fil, dekoder den til PCM, kompilerer en standard MPEG-1 Layer III (.mp3),
og lager steg-for-steg visualiseringer av hele kompileringen.

Bruk:
    python flac2mp3.py inn.flac [ut.mp3] [--bitrate 192] [--open]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import webbrowser
import numpy as np

from flac_decoder import decode_flac, FlacMetadata
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
    # Enkel nedmiksing fra flerkanals (f.eks. 5.1) til stereo
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
        raise FileNotFoundError(f"Fant ikke FLAC-filen: {input_flac}")

    if output_mp3 is None:
        stem, _ = os.path.splitext(input_flac)
        output_mp3 = stem + ".mp3"

    flac_size = os.path.getsize(input_flac)
    print(f"\n=== Egenlaget Python3 FLAC -> MP3-kompilator ===")
    print(f"[Steg 1/7] Dekoder FLAC-fil: {input_flac} ({flac_size / 1024:.1f} KB)")
    pcm, flac_meta = decode_flac(input_flac)
    print(
        f"           -> {flac_meta.sample_rate} Hz, {flac_meta.channels} kanal(er), "
        f"{flac_meta.bits_per_sample}-bit, {len(pcm):,} samples "
        f"({len(pcm) / flac_meta.sample_rate:.2f} s)"
    )

    pcm = _prepare_channels(pcm)
    target_sr = _pick_target_samplerate(flac_meta.sample_rate)
    if target_sr != flac_meta.sample_rate:
        print(f"           -> Resampler fra {flac_meta.sample_rate} Hz til {target_sr} Hz (MPEG-1)...")
        pcm = resample_audio(pcm, flac_meta.sample_rate, target_sr)

    print(f"[Steg 2/7] Initialiserer 32-bånds polyfase analysefilterbank (512-tap ISO 11172-3)...")
    print(f"[Steg 3/7] Initialiserer 36->18 MDCT + forover alias-reduksjon (576 spektrallinjer)...")
    print(f"[Steg 4/7] Initialiserer psykoakustisk modell (1024-pt FFT, Bark-spredning, ATH)...")
    print(f"[Steg 5/7] Kjorer ikke-lineær kvantisering & Rate-Distortion-lokke ({bitrate_kbps} kbps)...")
    print(f"[Steg 6/7] Pakker Huffman-koder (34 ISO-tabeller) og bitreservoar...")

    encoder = CustomMP3Encoder(
        sample_rate=target_sr,
        channels=pcm.shape[1],
        bitrate_kbps=bitrate_kbps,
    )

    def on_progress(done: int, total: int) -> None:
        pct = 100.0 * done / max(total, 1)
        print(f"           Kompilerer MP3-rammer: {done}/{total} ({pct:5.1f}%)", end="\r", flush=True)

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
        f"[Steg 7/7] Ferdig kompilert! -> {output_mp3} ({stats.mp3_bytes / 1024:.1f} KB, "
        f"{stats.num_frames} rammer, snitt SNR={avg_snr:.1f} dB, tid={elapsed:.2f} s)"
    )

    vis_info = None
    if visualize:
        if vis_dir is None:
            stem, _ = os.path.splitext(output_mp3)
            vis_dir = stem + "_visualisering"
        print(f"\nGenererer steg-for-steg visualiseringer i: {vis_dir}")
        vis_info = generate_visualizations(
            out_dir=vis_dir,
            input_flac=input_flac,
            output_mp3=output_mp3,
            pcm=pcm,
            flac_meta=flac_meta,
            stats=stats,
            flac_size_bytes=flac_size,
        )
        print(f"  -> HTML-rapport: {vis_info['html']}")
        for p in vis_info["pngs"]:
            print(f"  -> Bilde: {os.path.basename(p)}")
        if open_browser:
            webbrowser.open("file:///" + os.path.abspath(vis_info["html"]).replace("\\", "/"))

    return output_mp3, stats, vis_info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Egenlaget Python 3 FLAC-til-MP3 kompilator med steg-for-steg visualiseringer."
    )
    parser.add_argument("input_flac", help="Sti til input .flac-fil")
    parser.add_argument("output_mp3", nargs="?", default=None, help="Sti til output .mp3-fil (valgfri)")
    parser.add_argument(
        "-b", "--bitrate",
        type=int,
        default=192,
        choices=sorted(BITRATE_INDEX_MPEG1.keys()),
        help="MP3 bitrate i kbps (standard: 192)",
    )
    parser.add_argument("--vis-dir", default=None, help="Mappe for visualiseringer (PNG + HTML)")
    parser.add_argument("--no-vis", action="store_true", help="Deaktiver generering av visualiseringer")
    parser.add_argument("--open", action="store_true", help="Åpne HTML-visualiseringen i nettleser etter kompilering")

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

