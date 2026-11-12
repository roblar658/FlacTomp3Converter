"""
Egenlaget MPEG-1 Audio Layer III (.mp3) kompilator i Python 3.
Implementerer hele MP3-kodingskjeden fra bunnen av:
  1. Valgfri resampling til stottet MPEG-1 samplingsfrekvens (44100 / 48000 / 32000 Hz)
  2. 32-bånds polyfase analysefilterbank (512-tappers ISO 11172-3 vindu)
  3. Oddetalls-delbånd frekvensinversjon + 36->18 MDCT med sinusvindu + alias-reduksjon
  4. Psykoakustisk modell (1024-pt FFT, Bark-skala spredning og ATH-terskel)
  5. Ikke-lineær kvantisering (|xr|^(3/4)) med indre/ytre Rate-Distortion-lokke
  6. Optimal Huffman-koding (34 ISO-tabeller: big_values + count1 + rzero)
  7. Bitreservoar (main_data_begin) og MP3-rammepakking

Samler ogsa detaljert telemetri for hvert steg til visualisering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from mp3_tables import (
    BITRATE_INDEX_MPEG1,
    CA,
    CS,
    ENWINDOW,
    HUFFMAN_TABLES,
    PRETAB,
    SAMPLERATE_INDEX_MPEG1,
    SFB_LONG,
    SLEN1_TAB,
    SLEN2_TAB,
)


class BitWriter:
    """Rask MSB-først bitstrømskriver."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.acc = 0
        self.bits = 0

    def write_bits(self, val: int, n: int) -> None:
        if n == 0:
            return
        val &= (1 << n) - 1
        self.acc = (self.acc << n) | val
        self.bits += n
        while self.bits >= 8:
            self.bits -= 8
            self.buf.append((self.acc >> self.bits) & 0xFF)
            self.acc &= (1 << self.bits) - 1

    def total_bits(self) -> int:
        return len(self.buf) * 8 + self.bits

    def align_byte(self) -> int:
        pad = (8 - (self.bits & 7)) & 7
        if pad:
            self.write_bits(0, pad)
        return pad

    def to_bytes(self) -> bytes:
        if self.bits > 0:
            self.align_byte()
        return bytes(self.buf)


@dataclass
class EncodeStats:
    sample_rate: int
    original_sample_rate: int
    channels: int
    bitrate_kbps: int
    num_samples: int
    num_frames: int
    duration_sec: float
    mp3_bytes: int
    raw_pcm_bytes: int
    # Detaljert telemetri fra en aktiv representativ ramme/granul til visualisering
    vis_frame_idx: int = 0
    vis_pcm_frame: Optional[np.ndarray] = None           # (1152,) mono/ch0
    vis_subbands: Optional[np.ndarray] = None            # (18, 32) granul 0 ch 0
    vis_mdct_raw: Optional[np.ndarray] = None            # (576,) før alias-reduksjon
    vis_mdct_aliased: Optional[np.ndarray] = None        # (576,) etter alias-reduksjon
    vis_fft_freqs: Optional[np.ndarray] = None           # (513,) Hz
    vis_fft_db: Optional[np.ndarray] = None              # (513,) dB
    vis_sfb_centers_hz: Optional[np.ndarray] = None      # (21,) Hz
    vis_sfb_energy_db: Optional[np.ndarray] = None       # (21,) dB
    vis_sfb_ath_db: Optional[np.ndarray] = None          # (21,) dB
    vis_sfb_mask_db: Optional[np.ndarray] = None         # (21,) dB
    vis_ix: Optional[np.ndarray] = None                  # (576,) kvantiserte linjer
    vis_xr_dequant: Optional[np.ndarray] = None          # (576,) dekvantisert spektrum
    vis_scalefacs: Optional[List[int]] = None            # (21,)
    vis_global_gain: int = 0
    vis_gain_history: Optional[List[Tuple[int, int]]] = None  # (global_gain, total_bits)
    vis_region_info: Optional[dict] = None               # big_values, count1, rzero, tables...
    # Per-ramme statistikk over hele filen
    frame_reservoir_bytes: List[int] = field(default_factory=list)
    frame_main_bits: List[int] = field(default_factory=list)
    frame_slot_bits: List[int] = field(default_factory=list)
    frame_global_gains: List[int] = field(default_factory=list)
    frame_snr_db: List[float] = field(default_factory=list)


def resample_audio(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Enkel hoykvalitets FFT/sinc-basert resampling i numpy."""
    if src_sr == dst_sr:
        return pcm
    n_src = pcm.shape[0]
    n_dst = int(round(n_src * (dst_sr / src_sr)))
    if n_src == 0 or n_dst == 0:
        return np.zeros((0, pcm.shape[1]), dtype=np.float64)

    # Vindubasert lineær/Lanczos-3 resampling (rask og uten ring-artefakter på lange filer)
    t_dst = np.linspace(0.0, n_src - 1.0, n_dst, endpoint=False)
    idx0 = np.floor(t_dst).astype(np.int64)
    frac = t_dst - idx0
    a = 3
    out = np.zeros((n_dst, pcm.shape[1]), dtype=np.float64)
    w_sum = np.zeros(n_dst, dtype=np.float64)

    for k in range(-a + 1, a + 1):
        idx = np.clip(idx0 + k, 0, n_src - 1)
        x = frac - k
        x_safe = np.where(np.abs(x) < 1e-12, 1.0, x)
        # Lanczos kjerne
        w = np.where(np.abs(x) < 1e-12, 1.0,
                     np.where(np.abs(x) < a,
                              (a * np.sin(np.pi * x_safe) * np.sin(np.pi * x_safe / a)) / (np.pi * np.pi * x_safe * x_safe),
                              0.0))
        out += pcm[idx] * w[:, None]
        w_sum += w

    out /= np.maximum(w_sum[:, None], 1e-12)
    return np.clip(out, -1.0, 1.0)


def _hz_to_bark(f: np.ndarray | float) -> np.ndarray | float:
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def _ath_db(f_hz: np.ndarray | float) -> np.ndarray | float:
    f_khz = np.maximum(f_hz, 20.0) / 1000.0
    # Terhardt (1979) absolutt horeterskel i dB SPL, forskjøvet til fullskala dB (-96 dBFS ~ 0 dB SPL)
    ath_spl = 3.64 * (f_khz ** -0.8) - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2) + 1e-3 * (f_khz ** 4)
    return ath_spl - 96.0


class CustomMP3Encoder:
    """Egenlaget MPEG-1 Layer III (MP3) enkoder."""

    def __init__(self, sample_rate: int = 44100, channels: int = 2, bitrate_kbps: int = 192) -> None:
        if sample_rate not in SAMPLERATE_INDEX_MPEG1:
            raise ValueError(f"Ikke-støttet MPEG-1 samplingsfrekvens: {sample_rate}")
        if bitrate_kbps not in BITRATE_INDEX_MPEG1:
            raise ValueError(f"Ikke-støttet MPEG-1 bitrate: {bitrate_kbps}")
        if channels not in (1, 2):
            raise ValueError(f"Kun mono (1) eller stereo (2) støttes, fikk {channels}")

        self.sample_rate = sample_rate
        self.sr_idx = SAMPLERATE_INDEX_MPEG1[sample_rate]
        self.channels = channels
        self.mode = 3 if channels == 1 else 0  # 3=Mono, 0=Stereo
        self.bitrate_kbps = bitrate_kbps
        self.br_idx = BITRATE_INDEX_MPEG1[bitrate_kbps]
        self.sfb_bounds = SFB_LONG[sample_rate]

        # Forhåndsberegn polyfase analyse-cosinusmatrise (32 x 64)
        k = np.arange(32, dtype=np.float64)[:, None]
        j = np.arange(64, dtype=np.float64)[None, :]
        self.M_ana = np.cos((2.0 * k + 1.0) * (j - 16.0) * np.pi / 64.0)

        # Forhåndsberegn MDCT-matrise (18 x 36) med 36-pt sinusvindu og 1/9-normalisering
        n = np.arange(36, dtype=np.float64)
        km = np.arange(18, dtype=np.float64)[:, None]
        self.mdct_win = np.sin(np.pi / 36.0 * (n + 0.5))
        self.mdct_mat = (1.0 / 9.0) * (np.cos(np.pi / 72.0 * (2.0 * n + 19.0) * (2.0 * km + 1.0)) * self.mdct_win[None, :])

        # Tilstandsbuffer for polyfase (512 per kanal) og forrige subband-granul (32 x 18 per kanal)
        self.fifo = np.zeros((channels, 512), dtype=np.float64)
        self.prev_sb = np.zeros((channels, 32, 18), dtype=np.float64)

        # Psykoakustiske tabeller for de 21 skalafaktorbåndene
        self._init_psychoacoustic()

        # Forhåndsbygg oppslag for rask Huffman-lengdeberegning
        self._init_huffman_helpers()

    def _init_psychoacoustic(self) -> None:
        nyq = self.sample_rate / 2.0
        bounds = np.array(self.sfb_bounds, dtype=np.float64)
        centers_hz = ((bounds[:-1] + bounds[1:]) * 0.5 / 576.0) * nyq
        self.sfb_centers_hz = centers_hz[:21]
        self.sfb_barks = _hz_to_bark(self.sfb_centers_hz)

        # Schroeder spredningsmatrise (21 x 21)
        dz = self.sfb_barks[None, :] - self.sfb_barks[:, None]  # target - masker
        # +15.81 + 7.5*(dz+0.474) - 17.5*sqrt(1+(dz+0.474)^2) dB
        t = dz + 0.474
        spread_db = 15.81 + 7.5 * t - 17.5 * np.sqrt(1.0 + t * t)
        spread = 10.0 ** (spread_db / 10.0)
        # Normaliser hver rad slik at summen bevarer skala
        self.spread_mat = spread / np.maximum(spread.sum(axis=1, keepdims=True), 1e-12)

        # ATH per sfb i energi-ekvivalent (relativt til MDCT-linjeenergi)
        ath_db = _ath_db(self.sfb_centers_hz)
        # Skalering slik at 0 dBFS sinus (~1.0) gir ~0 dB i spektrum og ATH ligger rundt -85..-45 dB
        self.sfb_ath_db = ath_db
        self.sfb_ath_lin = 10.0 ** (ath_db / 10.0)

        # 1024-punkts Hann-vindu for FFT i psykoakustisk modell
        self.psy_win = np.hanning(1024)
        self.psy_freqs = np.fft.rfftfreq(1024, d=1.0 / self.sample_rate)

    def _init_huffman_helpers(self) -> None:
        self.huff_max = {}
        self.hl_np = {}
        for idx in range(34):
            tab = HUFFMAN_TABLES[idx]
            if tab is None:
                self.huff_max[idx] = 0
            else:
                xlen, ylen, linbits, linmax, _hb, hl = tab
                self.huff_max[idx] = (15 + linmax) if linbits > 0 else (xlen - 1)
                self.hl_np[idx] = np.array(hl, dtype=np.int32).reshape(xlen, ylen)

    def _polyphase_granule(self, pcm_576: np.ndarray, ch: int) -> np.ndarray:
        """Kjorer 576 samples gjennom 32-bånds polyfase analysefilterbank -> (18, 32)."""
        fifo = self.fifo[ch]
        sb_out = np.empty((18, 32), dtype=np.float64)
        for s in range(18):
            chunk = pcm_576[s * 32:(s + 1) * 32]
            fifo[32:] = fifo[:-32]
            fifo[:32] = chunk[::-1]
            z = fifo * ENWINDOW
            y = z.reshape(8, 64).sum(axis=0)
            sb_out[s] = self.M_ana @ y
        return sb_out

    def _mdct_and_alias(self, sb_18x32: np.ndarray, ch: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Utforer oddetalls-delbånd-inversjon, 36->18 MDCT og forover alias-reduksjon.
        Returnerer (mdct_raw_576, mdct_aliased_576).
        """
        cur = sb_18x32.T.copy()  # (32, 18)
        # Inverser fortegn på oddetalls-tidsluker (1,3,5..) i oddetalls-delbånd (1,3,5..)
        cur[1::2, 1::2] = -cur[1::2, 1::2]

        # Sett sammen 36 samples per delbånd: [prev_18, cur_18]
        blk36 = np.concatenate([self.prev_sb[ch], cur], axis=1)  # (32, 36)
        self.prev_sb[ch] = cur

        # 36 -> 18 MDCT for alle 32 delbånd samtidig
        mdct_raw = blk36 @ self.mdct_mat.T  # (32, 18)
        raw_576 = mdct_raw.reshape(576).copy()

        # Forover alias-reduksjon (invers av dekoderens sommerfugl)
        aliased = mdct_raw.copy()
        for sb in range(31):
            for i in range(8):
                cs_i = CS[i]
                ca_i = CA[i]
                bu = aliased[sb, 17 - i]
                bd = aliased[sb + 1, i]
                # Dekoder gjør: out_u = in_u*cs - in_d*ca; out_d = in_d*cs + in_u*ca
                # Invers rotasjon i enkoder:
                aliased[sb, 17 - i] = bu * cs_i + bd * ca_i
                aliased[sb + 1, i] = bd * cs_i - bu * ca_i

        return raw_576, aliased.reshape(576)

    def _psychoacoustic_thresholds(self, pcm_1024: np.ndarray, xr_576: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Beregner maskeringsterskel (linær energi per bånd) og FFT-spektrum for visualisering.
        Returnerer: (mask_thresh_21, sfb_energy_db_21, sfb_mask_db_21, fft_db_513)
        """
        # 1024-pt FFT for spektral flathet / tonalitet
        spec = np.fft.rfft(pcm_1024 * self.psy_win)
        mag2 = (np.abs(spec) / (1024.0 * 0.5)) ** 2
        fft_db = 10.0 * np.log10(np.maximum(mag2, 1e-16))

        # Energi per skalafaktorbånd fra MDCT-spekteret (21 bånd)
        sfb_e = np.empty(21, dtype=np.float64)
        bounds = self.sfb_bounds
        xr2 = xr_576 * xr_576
        for b in range(21):
            sfb_e[b] = float(np.sum(xr2[bounds[b]:bounds[b + 1]]))

        # Spredt maskeringsenergi via Schroeder-matrisen
        spread_e = self.spread_mat @ sfb_e

        # Spektral flathet (geom_mean / arith_mean) for å estimere tonalitet
        pos_mag = np.maximum(mag2[1:], 1e-16)
        sfm = float(np.exp(np.mean(np.log(pos_mag))) / np.mean(pos_mag))
        tonality = max(0.0, min(1.0, -np.log10(max(sfm, 1e-6)) / 3.0))
        # Tonal lyd maskerer ~18-22 dB under masker, støy ~8-10 dB under
        offset_db = 9.0 + 13.0 * tonality

        mask_lin = np.maximum(self.sfb_ath_lin * 0.25, spread_e * (10.0 ** (-offset_db / 10.0)))
        sfb_e_db = 10.0 * np.log10(np.maximum(sfb_e, 1e-16))
        sfb_mask_db = 10.0 * np.log10(np.maximum(mask_lin, 1e-16))
        return mask_lin, sfb_e_db, sfb_mask_db, fft_db

    def _quantize_xr(
        self,
        xr_abs: np.ndarray,
        xr_sign: np.ndarray,
        global_gain: int,
        scalefacs: np.ndarray,
        scalefac_scale: int,
        preflag: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Kvantiserer 576 MDCT-linjer med ISO 11172-3 ikke-lineær kvantiserer."""
        bounds = self.sfb_bounds
        sf_shift = 2 if scalefac_scale == 0 else 4
        ix_abs = np.empty(576, dtype=np.int32)
        xr_deq = np.empty(576, dtype=np.float64)

        for b in range(22):
            s, e = bounds[b], bounds[b + 1]
            sf = int(scalefacs[b]) if b < 21 else 0
            if preflag and b < 21:
                sf += PRETAB[b]
            step_pow = global_gain - 210 - sf_shift * sf
            step = 2.0 ** (step_pow / 4.0)
            istep = 1.0 / step
            q = np.floor(((xr_abs[s:e] * istep) ** 0.75) + 0.4054).astype(np.int32)
            ix_abs[s:e] = q
            xr_deq[s:e] = (q.astype(np.float64) ** (4.0 / 3.0)) * step

        max_q = int(np.max(ix_abs))
        ix = np.where(xr_sign < 0, -ix_abs, ix_abs)
        xr_deq = np.where(xr_sign < 0, -xr_deq, xr_deq)
        return ix, xr_deq, max_q

    def _count_huff_bits_region(self, abs_ix: np.ndarray, start: int, end: int, table_idx: int) -> int:
        if start >= end or table_idx == 0:
            return 0
        tab = HUFFMAN_TABLES[table_idx]
        if tab is None:
            return 10**9
        linbits = tab[2]
        ax = abs_ix[start:end:2]
        ay = abs_ix[start + 1:end:2]
        if linbits > 0:
            hx = np.minimum(ax, 15)
            hy = np.minimum(ay, 15)
            bits = int(self.hl_np[table_idx][hx, hy].sum())
            bits += int((np.count_nonzero(ax >= 15) + np.count_nonzero(ay >= 15)) * linbits)
        else:
            bits = int(self.hl_np[table_idx][ax, ay].sum())
        bits += int(np.count_nonzero(ax) + np.count_nonzero(ay))
        return bits

    def _best_table_for_region(self, abs_ix: np.ndarray, start: int, end: int) -> Tuple[int, int]:
        if start >= end:
            return 0, 0
        sub = abs_ix[start:end]
        max_abs = int(np.max(sub))
        if max_abs == 0:
            return 0, 0

        if max_abs == 1:
            candidates = (1, 2, 3, 5, 6)
        elif max_abs == 2:
            candidates = (2, 3, 5, 6, 7, 8, 9)
        elif max_abs == 3:
            candidates = (5, 6, 7, 8, 9, 10, 11, 12)
        elif max_abs <= 5:
            candidates = (7, 8, 9, 10, 11, 12, 13, 15)
        elif max_abs <= 7:
            candidates = (10, 11, 12, 13, 15)
        elif max_abs <= 15:
            candidates = (13, 15, 16, 24)
        else:
            candidates = tuple(t for t in range(16, 32) if self.huff_max[t] >= max_abs)

        ax = sub[0::2]
        ay = sub[1::2]
        sign_bits = int(np.count_nonzero(sub))
        best_t = -1
        best_b = 10**9
        for t in candidates:
            if self.huff_max[t] < max_abs:
                continue
            linbits = HUFFMAN_TABLES[t][2]
            if linbits > 0:
                hx = np.minimum(ax, 15)
                hy = np.minimum(ay, 15)
                b = int(self.hl_np[t][hx, hy].sum()) + int((np.count_nonzero(ax >= 15) + np.count_nonzero(ay >= 15)) * linbits) + sign_bits
            else:
                b = int(self.hl_np[t][ax, ay].sum()) + sign_bits
            if b < best_b:
                best_b = b
                best_t = t
        return best_t, best_b

    def _encode_huffman_plan(self, ix: np.ndarray) -> dict:
        """Finner optimal oppdeling i big_values (3 regioner), count1 (A/B) og rzero."""
        nz = np.nonzero(ix)[0]
        if len(nz) == 0:
            return {
                "big_values": 0,
                "region0_count": 0,
                "region1_count": 0,
                "table_select": [0, 0, 0],
                "count1table_select": 0,
                "count1_end": 0,
                "rzero_start": 0,
                "huff_bits": 0,
            }

        last_nz = int(nz[-1]) + 1
        rzero_start = (last_nz + 1) & ~1  # rund opp til partall

        # Finn count1-kvadrupler baklengs fra rzero_start der |val| <= 1
        c1_start = rzero_start
        while c1_start >= 4:
            q = ix[c1_start - 4:c1_start]
            if int(np.max(np.abs(q))) <= 1:
                c1_start -= 4
            else:
                break

        bv_end = c1_start
        big_values = bv_end // 2

        # Tell count1-bits for tabell A (32) og tabell B (33)
        c1_bits_a = 0
        c1_bits_b = 0
        hl32 = HUFFMAN_TABLES[32][5]
        for pos in range(c1_start, rzero_start, 4):
            v, w, x, y = int(ix[pos]), int(ix[pos + 1]), int(ix[pos + 2]), int(ix[pos + 3])
            av, aw, ax, ay = abs(v), abs(w), abs(x), abs(y)
            p = (av << 3) | (aw << 2) | (ax << 1) | ay
            signs = av + aw + ax + ay
            c1_bits_a += hl32[p] + signs
            c1_bits_b += 4 + signs

        if c1_bits_a <= c1_bits_b:
            count1table_select = 0
            c1_bits = c1_bits_a
        else:
            count1table_select = 1
            c1_bits = c1_bits_b

        if bv_end == 0:
            return {
                "big_values": 0,
                "region0_count": 0,
                "region1_count": 0,
                "table_select": [0, 0, 0],
                "count1table_select": count1table_select,
                "count1_end": rzero_start,
                "rzero_start": rzero_start,
                "huff_bits": c1_bits,
            }

        # Finn beste region0_count (0..15) og region1_count (0..7) langs sfb_bounds
        bounds = self.sfb_bounds
        # Finn maks bånd-indeks innenfor bv_end
        max_b = 0
        for b in range(1, 23):
            if bounds[b] >= bv_end:
                max_b = b
                break
        if max_b == 0:
            max_b = 22

        # Kandidater rundt energideling av bv_end for rask og nær-optimal regiondeling
        r0_cands = set()
        for frac in (0.15, 0.25, 0.35, 0.45, 0.55, 0.7):
            target = int(bv_end * frac)
            best_r0 = min(15, max(0, max_b - 2))
            for r0 in range(min(16, max_b)):
                if bounds[r0 + 1] >= target:
                    best_r0 = r0
                    break
            r0_cands.add(best_r0)
        if not r0_cands:
            r0_cands.add(0)

        abs_ix = np.abs(ix)
        best_bv_bits = 10**9
        best_cfg = (0, 0, [0, 0, 0])
        region_cache: Dict[Tuple[int, int], Tuple[int, int]] = {}

        def get_best(s: int, e: int) -> Tuple[int, int]:
            key = (s, e)
            res = region_cache.get(key)
            if res is None:
                res = self._best_table_for_region(abs_ix, s, e)
                region_cache[key] = res
            return res

        for r0 in sorted(r0_cands):
            r1_start = min(bounds[r0 + 1], bv_end)
            t0, b0 = get_best(0, r1_start)
            for r1 in range(8):
                b2_idx = min(22, r0 + r1 + 2)
                r2_start = min(bounds[b2_idx], bv_end)
                if r2_start == r1_start and r1 > 0:
                    continue
                t1, b1 = get_best(r1_start, r2_start)
                t2, b2 = get_best(r2_start, bv_end)
                tot = b0 + b1 + b2
                if tot < best_bv_bits:
                    best_bv_bits = tot
                    best_cfg = (r0, r1, [t0, t1, t2])
                if r2_start >= bv_end:
                    break

        return {
            "big_values": big_values,
            "region0_count": best_cfg[0],
            "region1_count": best_cfg[1],
            "table_select": best_cfg[2],
            "count1table_select": count1table_select,
            "count1_end": rzero_start,
            "rzero_start": rzero_start,
            "huff_bits": best_bv_bits + c1_bits,
        }

    def _choose_scalefac_compress(self, scalefacs: np.ndarray) -> Tuple[int, int]:
        """Velger minste (scalefac_compress, part2_bits) som rommer alle 21 skalafaktorer."""
        max_sf1 = int(np.max(scalefacs[0:11]))
        max_sf2 = int(np.max(scalefacs[11:21]))
        best_c = 15
        best_bits = 10**9
        for c in range(16):
            s1 = SLEN1_TAB[c]
            s2 = SLEN2_TAB[c]
            if max_sf1 < (1 << s1) and max_sf2 < (1 << s2):
                bits = 11 * s1 + 10 * s2
                if bits < best_bits:
                    best_bits = bits
                    best_c = c
        return best_c, best_bits

    def _encode_granule_rd(
        self,
        xr: np.ndarray,
        mask_lin: np.ndarray,
        target_bits: int,
        max_bits: int,
        record_history: bool = False,
    ) -> dict:
        """
        Indre/ytre Rate-Distortion-lokke for én granul og én kanal.
        Justerer global_gain (indreløkke) og scalefac_l (ytreløkke).
        """
        xr_abs = np.abs(xr)
        xr_sign = np.sign(xr)
        max_xr = float(np.max(xr_abs))

        if max_xr < 1e-12:
            return {
                "ix": np.zeros(576, dtype=np.int32),
                "xr_deq": np.zeros(576, dtype=np.float64),
                "scalefacs": np.zeros(21, dtype=np.int32),
                "global_gain": 210,
                "scalefac_compress": 0,
                "scalefac_scale": 0,
                "preflag": 0,
                "part2_bits": 0,
                "part2_3_length": 0,
                "plan": self._encode_huffman_plan(np.zeros(576, dtype=np.int32)),
                "gain_history": [(210, 0)] if record_history else None,
            }

        # Beregn nedre grense for global_gain slik at maks |ix| <= 8191
        # (max_xr / 2^((g-210)/4))^(3/4) <= 8191 => 2^((g-210)/4) >= max_xr / 8191^(4/3)
        min_step = max_xr / (8191.0 ** (4.0 / 3.0))
        min_gain = max(0, int(math.ceil(210.0 + 4.0 * math.log2(max(min_step, 1e-30)))))

        scalefacs = np.zeros(21, dtype=np.int32)
        scalefac_scale = 1  # 1 gir dobbel dynamikk per steg (maks 15 -> -60 dB)
        preflag = 0
        gain_history: List[Tuple[int, int]] = []

        def run_inner_loop(sf: np.ndarray, budget: int) -> dict:
            sf_comp, p2_bits = self._choose_scalefac_compress(sf)
            # Sikre at max_ix ikke sprenger 8191 selv med største scalefactor
            max_sf_eff = int(np.max(sf)) * (2 if scalefac_scale == 0 else 4)
            lo = max(min_gain, min(255, min_gain + max_sf_eff))
            # Startbinærsøk etter minste global_gain som får plass i budget
            lo = max(0, min_gain)
            hi = 255
            best_res = None

            # Kjør hurtig binærsøk (8 steg)
            for _ in range(9):
                if lo > hi:
                    break
                mid = (lo + hi) >> 1
                ix, xr_deq, max_q = self._quantize_xr(xr_abs, xr_sign, mid, sf, scalefac_scale, preflag)
                if max_q > 8206:
                    lo = mid + 1
                    continue
                plan = self._encode_huffman_plan(ix)
                tot_bits = p2_bits + plan["huff_bits"]
                if record_history:
                    gain_history.append((mid, tot_bits))
                if tot_bits <= budget:
                    best_res = (mid, ix, xr_deq, plan, p2_bits, sf_comp, tot_bits)
                    hi = mid - 1
                else:
                    lo = mid + 1

            if best_res is None:
                # Fallback på maks gain = 255
                ix, xr_deq, _ = self._quantize_xr(xr_abs, xr_sign, 255, sf, scalefac_scale, preflag)
                plan = self._encode_huffman_plan(ix)
                tot_bits = p2_bits + plan["huff_bits"]
                best_res = (255, ix, xr_deq, plan, p2_bits, sf_comp, tot_bits)
            return {
                "global_gain": best_res[0],
                "ix": best_res[1],
                "xr_deq": best_res[2],
                "plan": best_res[3],
                "part2_bits": best_res[4],
                "scalefac_compress": best_res[5],
                "part2_3_length": best_res[6],
                "scalefacs": sf.copy(),
                "scalefac_scale": scalefac_scale,
                "preflag": preflag,
            }

        best_overall = run_inner_loop(scalefacs, target_bits)
        bounds = self.sfb_bounds

        # Ytreløkke: opptil 4 runder der bånd med støy > maskering får økt scalefactor
        for _outer in range(4):
            err = (xr - best_overall["xr_deq"]) ** 2
            over_bands = []
            for b in range(21):
                band_noise = float(np.sum(err[bounds[b]:bounds[b + 1]]))
                if band_noise > mask_lin[b] and scalefacs[b] < 7:
                    over_bands.append((band_noise / max(mask_lin[b], 1e-16), b))
            if not over_bands:
                break
            over_bands.sort(reverse=True)
            cand_sf = scalefacs.copy()
            for _ratio, b in over_bands[:6]:
                cand_sf[b] += 1
            cand_res = run_inner_loop(cand_sf, max_bits)
            # Godta dersom total støy-til-maske bedres og budsjettet holder
            if cand_res["part2_3_length"] <= max_bits:
                old_max_nmr = max(
                    float(np.sum(((xr - best_overall["xr_deq"]) ** 2)[bounds[b]:bounds[b + 1]])) / max(mask_lin[b], 1e-16)
                    for b in range(21)
                )
                new_max_nmr = max(
                    float(np.sum(((xr - cand_res["xr_deq"]) ** 2)[bounds[b]:bounds[b + 1]])) / max(mask_lin[b], 1e-16)
                    for b in range(21)
                )
                if new_max_nmr <= old_max_nmr:
                    scalefacs = cand_sf
                    best_overall = cand_res
                else:
                    break
            else:
                break

        best_overall["gain_history"] = gain_history if record_history else None
        return best_overall

    def _write_main_data_granule(self, bw: BitWriter, gr_res: dict) -> None:
        sf = gr_res["scalefacs"]
        sf_comp = gr_res["scalefac_compress"]
        slen1 = SLEN1_TAB[sf_comp]
        slen2 = SLEN2_TAB[sf_comp]

        for b in range(11):
            bw.write_bits(int(sf[b]), slen1)
        for b in range(11, 21):
            bw.write_bits(int(sf[b]), slen2)

        ix = gr_res["ix"]
        plan = gr_res["plan"]
        bv_end = plan["big_values"] * 2
        if bv_end > 0:
            bounds = self.sfb_bounds
            r1_start = min(bounds[plan["region0_count"] + 1], bv_end)
            r2_start = min(bounds[min(22, plan["region0_count"] + plan["region1_count"] + 2)], bv_end)
            regions = (
                (0, r1_start, plan["table_select"][0]),
                (r1_start, r2_start, plan["table_select"][1]),
                (r2_start, bv_end, plan["table_select"][2]),
            )
            for s, e, t_idx in regions:
                if s >= e or t_idx == 0:
                    continue
                _xlen, ylen, linbits, _linmax, hb, hl = HUFFMAN_TABLES[t_idx]
                for pos in range(s, e, 2):
                    vx = int(ix[pos])
                    vy = int(ix[pos + 1])
                    ax, ay = abs(vx), abs(vy)
                    hx = 15 if (ax >= 15 and linbits > 0) else ax
                    hy = 15 if (ay >= 15 and linbits > 0) else ay
                    tpos = hx * ylen + hy
                    bw.write_bits(hb[tpos], hl[tpos])
                    if linbits > 0 and ax >= 15:
                        bw.write_bits(ax - 15, linbits)
                    if vx != 0:
                        bw.write_bits(1 if vx < 0 else 0, 1)
                    if linbits > 0 and ay >= 15:
                        bw.write_bits(ay - 15, linbits)
                    if vy != 0:
                        bw.write_bits(1 if vy < 0 else 0, 1)

        # count1-kvadrupler
        c1_tab = 32 if plan["count1table_select"] == 0 else 33
        hb_c1 = HUFFMAN_TABLES[c1_tab][4]
        hl_c1 = HUFFMAN_TABLES[c1_tab][5]
        for pos in range(bv_end, plan["rzero_start"], 4):
            v, w, x, y = int(ix[pos]), int(ix[pos + 1]), int(ix[pos + 2]), int(ix[pos + 3])
            av, aw, ax, ay = abs(v), abs(w), abs(x), abs(y)
            p = (av << 3) | (aw << 2) | (ax << 1) | ay
            bw.write_bits(hb_c1[p], hl_c1[p])
            if v != 0:
                bw.write_bits(1 if v < 0 else 0, 1)
            if w != 0:
                bw.write_bits(1 if w < 0 else 0, 1)
            if x != 0:
                bw.write_bits(1 if x < 0 else 0, 1)
            if y != 0:
                bw.write_bits(1 if y < 0 else 0, 1)

    def encode(self, pcm: np.ndarray, original_sr: Optional[int] = None, progress_cb=None) -> Tuple[bytes, EncodeStats]:
        """
        Kompilerer PCM (shape: (N, ch) float64 i [-1, 1]) til en komplett MP3-bitstrøm.
        Returnerer (mp3_bytes, EncodeStats).
        """
        if pcm.ndim == 1:
            pcm = pcm[:, None]
        if pcm.shape[1] != self.channels:
            raise ValueError(f"Kanalantall stemmer ikke: forventet {self.channels}, fikk {pcm.shape[1]}")

        n_orig = len(pcm)
        # Pad med ekstra nullramme for filterbank-forsinkelse + 1152-multippel
        pad_tail = (1152 - (n_orig % 1152)) % 1152 + 1152
        padded = np.vstack([pcm, np.zeros((pad_tail, self.channels), dtype=np.float64)])
        num_frames = len(padded) // 1152

        # Finn en aktiv representativ ramme for visualisering (høyest RMS etter ramme 1)
        frame_rms = [float(np.sqrt(np.mean(padded[f * 1152:(f + 1) * 1152] ** 2))) for f in range(num_frames)]
        if num_frames > 2:
            vis_frame = 1 + int(np.argmax(frame_rms[1:-1]))
        else:
            vis_frame = 0

        stats = EncodeStats(
            sample_rate=self.sample_rate,
            original_sample_rate=original_sr or self.sample_rate,
            channels=self.channels,
            bitrate_kbps=self.bitrate_kbps,
            num_samples=n_orig,
            num_frames=num_frames,
            duration_sec=n_orig / float(self.sample_rate),
            mp3_bytes=0,
            raw_pcm_bytes=n_orig * self.channels * 2,
            vis_frame_idx=vis_frame,
        )

        side_info_bytes = 17 if self.channels == 1 else 32
        header_bytes = 4
        bitrate_bps = self.bitrate_kbps * 1000
        # MPEG-1 Layer III padding-akkumulator
        pad_acc = 0
        base_frame_bytes = (144 * bitrate_bps) // self.sample_rate
        pad_rem = (144 * bitrate_bps) % self.sample_rate

        main_data_stream = bytearray()
        frame_meta_list = []
        reservoir_bytes = 0  # ubrukte main_data-bytes fra tidligere rammer (0..511)

        for f in range(num_frames):
            if progress_cb and (f % 10 == 0 or f == num_frames - 1):
                progress_cb(f + 1, num_frames)

            pad_acc += pad_rem
            if pad_acc >= self.sample_rate:
                pad_acc -= self.sample_rate
                padding = 1
            else:
                padding = 0

            frame_bytes = base_frame_bytes + padding
            slot_bytes = frame_bytes - header_bytes - side_info_bytes
            main_data_begin = min(reservoir_bytes, 511)

            # Maks antall main_data bytes denne rammen kan skrive
            max_main_bytes = slot_bytes + main_data_begin
            # Normalt sikter vi mot slot_bytes + litt fra reservoar, og passer på at ny rest <= 511
            target_total_bits = (slot_bytes + min(main_data_begin, 120)) * 8
            hard_max_bits = max_main_bytes * 8

            frame_pcm = padded[f * 1152:(f + 1) * 1152]
            granules_res = [[None] * self.channels for _ in range(2)]

            # Fordel bits mellom de 2x(1 eller 2) granul-kanal-parene proporsjonalt med energi/kompleksitet
            n_gc = 2 * self.channels
            xr_cache = [[None] * self.channels for _ in range(2)]
            psy_cache = [[None] * self.channels for _ in range(2)]
            perceptual_weights = []

            for gr in range(2):
                gr_start = gr * 576
                gr_pcm = frame_pcm[gr_start:gr_start + 576]
                for ch in range(self.channels):
                    sb_18x32 = self._polyphase_granule(gr_pcm[:, ch], ch)
                    raw_576, xr_576 = self._mdct_and_alias(sb_18x32, ch)
                    # Hent 1024-punkts vindu sentrert rundt denne granulen
                    center = f * 1152 + gr_start + 288
                    w_start = center - 512
                    if w_start < 0:
                        win1024 = np.concatenate([np.zeros(-w_start), padded[:w_start + 1024, ch]])
                    elif w_start + 1024 > len(padded):
                        avail = padded[w_start:, ch]
                        win1024 = np.concatenate([avail, np.zeros(1024 - len(avail))])
                    else:
                        win1024 = padded[w_start:w_start + 1024, ch]

                    mask_lin, sfb_e_db, sfb_mask_db, fft_db = self._psychoacoustic_thresholds(win1024, xr_576)
                    xr_cache[gr][ch] = (sb_18x32, raw_576, xr_576)
                    psy_cache[gr][ch] = (mask_lin, sfb_e_db, sfb_mask_db, fft_db)
                    energy = float(np.sum(xr_576 * xr_576))
                    perceptual_weights.append(max(0.25, math.log10(1.0 + energy * 1e6)))

            w_sum = sum(perceptual_weights)
            # Reserver litt margin (opptil 7 bit byte-padding på slutten)
            usable_target = max(64, target_total_bits - 16)
            usable_hard = max(64, hard_max_bits - 16)

            idx_gc = 0
            bits_used_so_far = 0
            frame_snr_vals = []
            frame_gains = []

            for gr in range(2):
                for ch in range(self.channels):
                    frac = perceptual_weights[idx_gc] / w_sum
                    rem_slots = n_gc - idx_gc - 1
                    g_target = int(usable_target * frac)
                    g_max = min(4095, int(usable_hard * frac) + 400, usable_hard - bits_used_so_far - rem_slots * 16)
                    g_max = max(32, g_max)
                    g_target = max(32, min(g_target, g_max, 4095))

                    sb_18x32, raw_576, xr_576 = xr_cache[gr][ch]
                    mask_lin, sfb_e_db, sfb_mask_db, fft_db = psy_cache[gr][ch]
                    is_vis = (f == vis_frame and gr == 0 and ch == 0)

                    res = self._encode_granule_rd(xr_576, mask_lin, g_target, g_max, record_history=is_vis)
                    granules_res[gr][ch] = res
                    bits_used_so_far += res["part2_3_length"]
                    frame_gains.append(res["global_gain"])

                    sig_pow = float(np.sum(xr_576 * xr_576))
                    err_pow = float(np.sum((xr_576 - res["xr_deq"]) ** 2))
                    if sig_pow > 1e-12:
                        frame_snr_vals.append(10.0 * math.log10(sig_pow / max(err_pow, 1e-16)))

                    if is_vis:
                        stats.vis_pcm_frame = frame_pcm[:, 0].copy()
                        stats.vis_subbands = sb_18x32.copy()
                        stats.vis_mdct_raw = raw_576.copy()
                        stats.vis_mdct_aliased = xr_576.copy()
                        stats.vis_fft_freqs = self.psy_freqs.copy()
                        stats.vis_fft_db = fft_db.copy()
                        stats.vis_sfb_centers_hz = self.sfb_centers_hz.copy()
                        stats.vis_sfb_energy_db = sfb_e_db.copy()
                        stats.vis_sfb_ath_db = self.sfb_ath_db.copy()
                        stats.vis_sfb_mask_db = sfb_mask_db.copy()
                        stats.vis_ix = res["ix"].copy()
                        stats.vis_xr_dequant = res["xr_deq"].copy()
                        stats.vis_scalefacs = res["scalefacs"].tolist()
                        stats.vis_global_gain = res["global_gain"]
                        stats.vis_gain_history = res["gain_history"]
                        stats.vis_region_info = {
                            **res["plan"],
                            "part2_bits": res["part2_bits"],
                            "part2_3_length": res["part2_3_length"],
                            "scalefac_compress": res["scalefac_compress"],
                        }
                    idx_gc += 1

            # Skriv main_data for denne rammen
            mbw = BitWriter()
            for gr in range(2):
                for ch in range(self.channels):
                    self._write_main_data_granule(mbw, granules_res[gr][ch])
            mbw.align_byte()
            frame_main = mbw.to_bytes()

            # Beregn nytt reservoar og legg til stuffing hvis reservoaret ville overstige 511 bytes
            new_res = main_data_begin + slot_bytes - len(frame_main)
            if new_res > 511:
                stuff_bytes = new_res - 511
                frame_main = frame_main + (b"\x00" * stuff_bytes)
                new_res = 511
            elif new_res < 0:
                # Sikkerhetsnett (skal aldri inntreffe pga g_max)
                raise RuntimeError(f"Bitreservoar-underflow i ramme {f}: {new_res}")

            main_data_stream.extend(frame_main)
            reservoir_bytes = new_res

            stats.frame_reservoir_bytes.append(main_data_begin)
            stats.frame_main_bits.append(bits_used_so_far)
            stats.frame_slot_bits.append(slot_bytes * 8)
            stats.frame_global_gains.append(int(round(sum(frame_gains) / len(frame_gains))))
            stats.frame_snr_db.append(sum(frame_snr_vals) / len(frame_snr_vals) if frame_snr_vals else 60.0)

            frame_meta_list.append((padding, slot_bytes, main_data_begin, granules_res))

        # Sørg for at main_data_stream har nok bytes til siste rammes slot
        total_slots = sum(m[1] for m in frame_meta_list)
        if len(main_data_stream) < total_slots:
            main_data_stream.extend(b"\x00" * (total_slots - len(main_data_stream)))

        # Sett sammen alle MP3-rammer: Header (4B) + Side Info (17/32B) + Slot fra main_data_stream
        mp3_out = bytearray()
        m_pos = 0
        for padding, slot_bytes, main_data_begin, granules_res in frame_meta_list:
            # 32-bit MP3 frame header
            hdr = (
                (0x7FF << 21)
                | (3 << 19)          # MPEG-1
                | (1 << 17)          # Layer III
                | (1 << 16)          # Ingen CRC-16
                | (self.br_idx << 12)
                | (self.sr_idx << 10)
                | (padding << 9)
                | (0 << 8)           # private
                | (self.mode << 6)   # mode (0=stereo, 3=mono)
                | (0 << 4)           # mode_ext
                | (0 << 3)           # copyright
                | (1 << 2)           # original
                | 0                  # emphasis
            )
            mp3_out.extend(hdr.to_bytes(4, "big"))

            sbw = BitWriter()
            sbw.write_bits(main_data_begin, 9)
            sbw.write_bits(0, 5 if self.channels == 1 else 3)  # private_bits
            for _ch in range(self.channels):
                sbw.write_bits(0, 4)  # scfsi = 0000
            for gr in range(2):
                for ch in range(self.channels):
                    g = granules_res[gr][ch]
                    p = g["plan"]
                    sbw.write_bits(g["part2_3_length"], 12)
                    sbw.write_bits(p["big_values"], 9)
                    sbw.write_bits(g["global_gain"], 8)
                    sbw.write_bits(g["scalefac_compress"], 4)
                    sbw.write_bits(0, 1)  # window_switching_flag = 0 (normal long block)
                    for t_idx in p["table_select"]:
                        sbw.write_bits(t_idx, 5)
                    sbw.write_bits(p["region0_count"], 4)
                    sbw.write_bits(p["region1_count"], 3)
                    sbw.write_bits(g["preflag"], 1)
                    sbw.write_bits(g["scalefac_scale"], 1)
                    sbw.write_bits(p["count1table_select"], 1)
            side_bytes = sbw.to_bytes()
            assert len(side_bytes) == side_info_bytes
            mp3_out.extend(side_bytes)
            mp3_out.extend(main_data_stream[m_pos:m_pos + slot_bytes])
            m_pos += slot_bytes

        mp3_bytes = bytes(mp3_out)
        stats.mp3_bytes = len(mp3_bytes)
        return mp3_bytes, stats
