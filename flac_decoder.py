"""
Pure Python 3 native FLAC decoder (RFC 9639).
Uses zero external audio libraries — parses FLAC metadata blocks, decodes
CONSTANT, VERBATIM, FIXED (0-4), and LPC (1-32) subframes with Rice coding,
and performs stereo channel decorrelation (L/S, R/S, M/S) to normalized PCM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


_BLOCKSIZE_TABLE = {
    1: 192,
    2: 576,
    3: 1152,
    4: 2304,
    5: 4608,
    8: 256,
    9: 512,
    10: 1024,
    11: 2048,
    12: 4096,
    13: 8192,
    14: 16384,
    15: 32768,
}

_SAMPLERATE_TABLE = {
    1: 88200,
    2: 176400,
    3: 192000,
    4: 8000,
    5: 16000,
    6: 22050,
    7: 24000,
    8: 32000,
    9: 44100,
    10: 48000,
    11: 96000,
}

_BPS_TABLE = {
    1: 8,
    2: 12,
    4: 16,
    5: 20,
    6: 24,
    7: 32,
}


@dataclass
class FlacMetadata:
    sample_rate: int
    channels: int
    bits_per_sample: int
    total_samples: int
    min_blocksize: int
    max_blocksize: int
    min_framesize: int
    max_framesize: int
    md5: bytes
    vorbis_comments: Dict[str, str] = field(default_factory=dict)
    frame_stats: List[dict] = field(default_factory=list)


class BitReader:
    """Fast MSB-first bit reader over bytes using an integer accumulator."""

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data = data
        self.length = len(data)
        self.byte_pos = pos
        self.buf = 0
        self.bits = 0

    def _fill(self, n: int) -> None:
        while self.bits < n and self.byte_pos < self.length:
            self.buf = (self.buf << 8) | self.data[self.byte_pos]
            self.byte_pos += 1
            self.bits += 8

    def read_bits(self, n: int) -> int:
        if n == 0:
            return 0
        if self.bits < n:
            self._fill(n)
            if self.bits < n:
                raise EOFError("Unexpected end of FLAC bitstream")
        shift = self.bits - n
        val = (self.buf >> shift) & ((1 << n) - 1)
        self.bits = shift
        self.buf &= (1 << shift) - 1
        return val

    def read_signed(self, n: int) -> int:
        val = self.read_bits(n)
        sign_bit = 1 << (n - 1)
        if val & sign_bit:
            val -= 1 << n
        return val

    def read_unary_zeros(self) -> int:
        """Reads the count of leading 0 bits before the terminating 1 bit."""
        zeros = 0
        while True:
            if self.bits == 0:
                if self.byte_pos >= self.length:
                    raise EOFError("Unexpected end of stream during unary read")
                b = self.data[self.byte_pos]
                self.byte_pos += 1
                if b == 0:
                    zeros += 8
                    continue
                self.buf = b
                self.bits = 8
            elif self.buf == 0:
                zeros += self.bits
                self.bits = 0
                continue
            # Locate highest set bit in self.buf
            bl = self.buf.bit_length()
            leading_zeros = self.bits - bl
            zeros += leading_zeros
            # Consume leading_zeros + 1 bit
            rem = bl - 1
            self.bits = rem
            self.buf &= (1 << rem) - 1
            return zeros

    def align_to_byte(self) -> None:
        drop = self.bits & 7
        if drop:
            self.bits -= drop
            self.buf &= (1 << self.bits) - 1
        while self.bits >= 8:
            self.bits -= 8
            self.byte_pos -= 1
        self.buf = 0
        self.bits = 0

    def read_bytes(self, n: int) -> bytes:
        self.align_to_byte()
        end = self.byte_pos + n
        if end > self.length:
            raise EOFError("Unexpected end of stream while reading bytes")
        out = self.data[self.byte_pos:end]
        self.byte_pos = end
        return out

    def read_utf8_uint(self) -> int:
        first = self.read_bits(8)
        if (first & 0x80) == 0:
            return first
        ones = 0
        mask = 0x80
        while first & mask:
            ones += 1
            mask >>= 1
        if ones < 2 or ones > 7:
            raise ValueError(f"Invalid UTF-8 coded FLAC number: {first:#x}")
        val = first & ((1 << (7 - ones)) - 1)
        for _ in range(ones - 1):
            cont = self.read_bits(8)
            val = (val << 6) | (cont & 0x3F)
        return val

    def has_more(self) -> bool:
        return self.byte_pos < self.length or self.bits > 0


def _decode_residual(br: BitReader, blocksize: int, predictor_order: int) -> List[int]:
    method = br.read_bits(2)
    if method > 1:
        raise ValueError(f"Unsupported FLAC residual coding method: {method}")
    param_bits = 4 if method == 0 else 5
    escape_val = (1 << param_bits) - 1

    partition_order = br.read_bits(4)
    num_partitions = 1 << partition_order
    base_samples = blocksize >> partition_order

    residual: List[int] = []
    read_bits = br.read_bits
    read_signed = br.read_signed
    read_unary = br.read_unary_zeros

    for p in range(num_partitions):
        count = base_samples - predictor_order if p == 0 else base_samples
        rice_param = read_bits(param_bits)
        if rice_param == escape_val:
            bps = read_bits(5)
            if bps == 0:
                residual.extend([0] * count)
            else:
                for _ in range(count):
                    residual.append(read_signed(bps))
        else:
            if rice_param == 0:
                for _ in range(count):
                    q = read_unary()
                    residual.append((q >> 1) ^ -(q & 1))
            else:
                for _ in range(count):
                    q = read_unary()
                    r = read_bits(rice_param)
                    folded = (q << rice_param) | r
                    residual.append((folded >> 1) ^ -(folded & 1))
    return residual


def _decode_subframe(
    br: BitReader, blocksize: int, bps: int
) -> Tuple[List[int], str]:
    zero_pad = br.read_bits(1)
    if zero_pad != 0:
        raise ValueError("Invalid FLAC subframe padding bit (expected 0)")

    sub_type = br.read_bits(6)
    wasted_flag = br.read_bits(1)
    wasted_bits = 0
    if wasted_flag:
        wasted_bits = br.read_unary_zeros() + 1
    eff_bps = bps - wasted_bits

    if sub_type == 0:
        val = br.read_signed(eff_bps)
        samples = [val] * blocksize
        kind = "CONSTANT"
    elif sub_type == 1:
        samples = [br.read_signed(eff_bps) for _ in range(blocksize)]
        kind = "VERBATIM"
    elif 8 <= sub_type <= 12:
        order = sub_type - 8
        samples = [br.read_signed(eff_bps) for _ in range(order)]
        res = _decode_residual(br, blocksize, order)
        if order == 0:
            samples.extend(res)
        elif order == 1:
            prev = samples[0]
            for r in res:
                prev = prev + r
                samples.append(prev)
        elif order == 2:
            s1, s2 = samples[1], samples[0]
            for r in res:
                cur = 2 * s1 - s2 + r
                samples.append(cur)
                s2, s1 = s1, cur
        elif order == 3:
            s1, s2, s3 = samples[2], samples[1], samples[0]
            for r in res:
                cur = 3 * s1 - 3 * s2 + s3 + r
                samples.append(cur)
                s3, s2, s1 = s2, s1, cur
        else:  # order == 4
            s1, s2, s3, s4 = samples[3], samples[2], samples[1], samples[0]
            for r in res:
                cur = 4 * s1 - 6 * s2 + 4 * s3 - s4 + r
                samples.append(cur)
                s4, s3, s2, s1 = s3, s2, s1, cur
        kind = f"FIXED({order})"
    elif 32 <= sub_type <= 63:
        order = sub_type - 31
        samples = [br.read_signed(eff_bps) for _ in range(order)]
        prec = br.read_bits(4) + 1
        if prec == 16:
            raise ValueError("Reserved QLP coefficient precision (16)")
        qlp_shift = br.read_signed(5)
        coeffs = [br.read_signed(prec) for _ in range(order)]
        res = _decode_residual(br, blocksize, order)

        for r in res:
            idx = len(samples)
            acc = 0
            for j in range(order):
                acc += coeffs[j] * samples[idx - 1 - j]
            samples.append((acc >> qlp_shift) + r)
        kind = f"LPC({order})"
    else:
        raise ValueError(f"Invalid FLAC subframe type: {sub_type}")

    if wasted_bits > 0:
        samples = [s << wasted_bits for s in samples]

    return samples, kind


def _parse_vorbis_comments(raw: bytes) -> Dict[str, str]:
    comments: Dict[str, str] = {}
    if len(raw) < 8:
        return comments
    pos = 0
    vendor_len = int.from_bytes(raw[pos:pos + 4], "little")
    pos += 4 + vendor_len
    if pos + 4 > len(raw):
        return comments
    count = int.from_bytes(raw[pos:pos + 4], "little")
    pos += 4
    for _ in range(count):
        if pos + 4 > len(raw):
            break
        clen = int.from_bytes(raw[pos:pos + 4], "little")
        pos += 4
        entry = raw[pos:pos + clen].decode("utf-8", errors="replace")
        pos += clen
        if "=" in entry:
            k, v = entry.split("=", 1)
            comments[k.upper()] = v
    return comments


def decode_flac(path: str) -> Tuple[np.ndarray, FlacMetadata]:
    """
    Decodes a native .flac file completely in Python 3.
    Returns:
        pcm: numpy float64 array with shape (num_samples, channels) in [-1.0, 1.0]
        metadata: FlacMetadata with stream info and frame statistics for visualization
    """
    with open(path, "rb") as f:
        data = f.read()

    pos = 0
    # Skip optional ID3v2 header before the fLaC marker
    if data[:3] == b"ID3" and len(data) >= 10:
        s = data[6:10]
        id3_size = ((s[0] & 0x7F) << 21) | ((s[1] & 0x7F) << 14) | ((s[2] & 0x7F) << 7) | (s[3] & 0x7F)
        pos = 10 + id3_size

    if data[pos:pos + 4] != b"fLaC":
        raise ValueError("File is not a valid native FLAC file (missing 'fLaC' marker)")
    pos += 4

    streaminfo: Optional[FlacMetadata] = None
    vorbis_comments: Dict[str, str] = {}

    while True:
        if pos + 4 > len(data):
            raise EOFError("Truncated FLAC metadata block")
        header = data[pos]
        is_last = bool(header & 0x80)
        block_type = header & 0x7F
        block_len = int.from_bytes(data[pos + 1:pos + 4], "big")
        pos += 4
        block_data = data[pos:pos + block_len]
        pos += block_len

        if block_type == 0:  # STREAMINFO
            min_bs = int.from_bytes(block_data[0:2], "big")
            max_bs = int.from_bytes(block_data[2:4], "big")
            min_fs = int.from_bytes(block_data[4:7], "big")
            max_fs = int.from_bytes(block_data[7:10], "big")
            packed = int.from_bytes(block_data[10:18], "big")
            sr = (packed >> 44) & 0xFFFFF
            ch = ((packed >> 41) & 0x7) + 1
            bps = ((packed >> 36) & 0x1F) + 1
            total_samples = packed & ((1 << 36) - 1)
            md5 = block_data[18:34]
            streaminfo = FlacMetadata(
                sample_rate=sr,
                channels=ch,
                bits_per_sample=bps,
                total_samples=total_samples,
                min_blocksize=min_bs,
                max_blocksize=max_bs,
                min_framesize=min_fs,
                max_framesize=max_fs,
                md5=md5,
            )
        elif block_type == 4:  # VORBIS_COMMENT
            vorbis_comments = _parse_vorbis_comments(block_data)

        if is_last:
            break

    if streaminfo is None:
        raise ValueError("FLAC file is missing STREAMINFO metadata block")
    streaminfo.vorbis_comments = vorbis_comments

    br = BitReader(data, pos)
    channel_chunks: List[List[List[int]]] = [[] for _ in range(streaminfo.channels)]
    frame_idx = 0

    while br.has_more():
        sync = br.read_bits(14)
        if sync != 0x3FFE:
            raise ValueError(f"Lost FLAC frame sync at frame {frame_idx}: {sync:#x}")
        _reserved = br.read_bits(1)
        _blocking_strategy = br.read_bits(1)

        bs_code = br.read_bits(4)
        sr_code = br.read_bits(4)
        ch_assign = br.read_bits(4)
        bps_code = br.read_bits(3)
        _reserved2 = br.read_bits(1)

        _frame_or_sample_num = br.read_utf8_uint()

        if bs_code == 6:
            blocksize = br.read_bits(8) + 1
        elif bs_code == 7:
            blocksize = br.read_bits(16) + 1
        elif bs_code in _BLOCKSIZE_TABLE:
            blocksize = _BLOCKSIZE_TABLE[bs_code]
        else:
            raise ValueError(f"Invalid blocksize code: {bs_code}")

        if sr_code == 0:
            sample_rate = streaminfo.sample_rate
        elif sr_code == 12:
            sample_rate = br.read_bits(8) * 1000
        elif sr_code == 13:
            sample_rate = br.read_bits(16)
        elif sr_code == 14:
            sample_rate = br.read_bits(16) * 10
        elif sr_code in _SAMPLERATE_TABLE:
            sample_rate = _SAMPLERATE_TABLE[sr_code]
        else:
            raise ValueError(f"Invalid sample rate code: {sr_code}")

        bps = streaminfo.bits_per_sample if bps_code == 0 else _BPS_TABLE[bps_code]
        _crc8 = br.read_bits(8)

        if ch_assign < 8:
            num_ch = ch_assign + 1
            sub_bps_list = [bps] * num_ch
        elif ch_assign in (8, 9, 10):
            num_ch = 2
            if ch_assign == 8:  # Left/Side
                sub_bps_list = [bps, bps + 1]
            else:  # Right/Side or Mid/Side
                sub_bps_list = [bps + 1 if ch_assign == 9 else bps, bps + 1]
        else:
            raise ValueError(f"Reserved channel assignment: {ch_assign}")

        subframes = []
        sub_kinds = []
        for c in range(num_ch):
            sf_samples, sf_kind = _decode_subframe(br, blocksize, sub_bps_list[c])
            subframes.append(sf_samples)
            sub_kinds.append(sf_kind)

        # Stereo channel decorrelation
        if ch_assign == 8:  # Left / Side
            left = subframes[0]
            side = subframes[1]
            subframes[1] = [l - s for l, s in zip(left, side)]
        elif ch_assign == 9:  # Side / Right
            side = subframes[0]
            right = subframes[1]
            subframes[0] = [r + s for s, r in zip(side, right)]
        elif ch_assign == 10:  # Mid / Side
            mid = subframes[0]
            side = subframes[1]
            left = []
            right = []
            for m, s in zip(mid, side):
                m2 = (m << 1) + (s & 1)
                left.append((m2 + s) >> 1)
                right.append((m2 - s) >> 1)
            subframes[0] = left
            subframes[1] = right

        for c in range(num_ch):
            channel_chunks[c].append(subframes[c])

        # Frame footer CRC-16
        br.read_bytes(2)

        if len(streaminfo.frame_stats) < 200:
            streaminfo.frame_stats.append({
                "frame": frame_idx,
                "blocksize": blocksize,
                "sample_rate": sample_rate,
                "ch_assign": ch_assign,
                "subframes": sub_kinds,
            })
        frame_idx += 1

    scale = float(1 << (streaminfo.bits_per_sample - 1))
    arrays = [np.concatenate([np.asarray(chunk, dtype=np.float64) for chunk in ch]) / scale for ch in channel_chunks]
    pcm = np.stack(arrays, axis=1)
    if streaminfo.total_samples > 0 and len(pcm) > streaminfo.total_samples:
        pcm = pcm[:streaminfo.total_samples]

    return pcm, streaminfo
