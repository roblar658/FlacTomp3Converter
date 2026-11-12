"""
Automatisk test-suite for egenlaget FLAC -> MP3 kompilator og visualiseringer.
Genererer ekte .flac testfiler, kompilerer dem med vår Python3-kompilator,
verifiserer at alle 7 steg-visualiseringer + HTML blir laget, og dekoder
MP3-resultatet med soundfile (libmpg123) for å kontrollere lydkvalitet.
"""

import os
import tempfile
import unittest
import numpy as np
import soundfile as sf

from flac_decoder import decode_flac
from flac2mp3 import compile_flac_to_mp3


class TestCustomMp3Compiler(unittest.TestCase):
    def test_stereo_44100_chord_with_visualizations(self):
        sr = 44100
        dur = 1.5
        t = np.arange(int(sr * dur)) / sr
        # Rik stereo-akkord + høyfrekvent overtone for å aktivere mange bånd
        left = 0.40 * np.sin(2 * np.pi * 440.0 * t) + 0.25 * np.sin(2 * np.pi * 1318.5 * t) + 0.12 * np.sin(2 * np.pi * 5200.0 * t)
        right = 0.38 * np.sin(2 * np.pi * 554.37 * t) + 0.22 * np.cos(2 * np.pi * 2637.0 * t) + 0.10 * np.sin(2 * np.pi * 8800.0 * t)
        # Fade in/out for pen overgang
        env = np.minimum(1.0, np.minimum(t / 0.05, (dur - t) / 0.05))
        pcm_in = np.stack([left * env, right * env], axis=1)

        with tempfile.TemporaryDirectory() as tmp:
            flac_path = os.path.join(tmp, "chord_stereo.flac")
            mp3_path = os.path.join(tmp, "chord_stereo.mp3")
            vis_dir = os.path.join(tmp, "vis")

            sf.write(flac_path, pcm_in, sr, subtype="PCM_16")
            out_mp3, stats, vis_info = compile_flac_to_mp3(
                flac_path,
                mp3_path,
                bitrate_kbps=192,
                visualize=True,
                vis_dir=vis_dir,
                open_browser=False,
            )

            self.assertTrue(os.path.isfile(out_mp3))
            self.assertGreater(os.path.getsize(out_mp3), 10000)
            self.assertIsNotNone(vis_info)
            self.assertTrue(os.path.isfile(vis_info["html"]))
            self.assertEqual(len(vis_info["pngs"]), 7)
            for p in vis_info["pngs"]:
                self.assertTrue(os.path.isfile(p))
                self.assertGreater(os.path.getsize(p), 5000)

            # Verifiser at standard MP3-dekoder (libsndfile/libmpg123) dekoder filen korrekt
            dec, dsr = sf.read(out_mp3)
            self.assertEqual(dsr, 44100)
            self.assertEqual(dec.shape[1], 2)

            # Juster for MP3 filterbank+MDCT-forsinkelse (~1057 samples)
            ref0 = pcm_in[8000:50000, 0]
            corrs0 = [np.corrcoef(ref0, dec[8000 + d:50000 + d, 0])[0, 1] for d in range(1000, 1120)]
            best_corr0 = max(corrs0)
            best_d = 1000 + int(np.argmax(corrs0))
            corr1 = np.corrcoef(pcm_in[8000:50000, 1], dec[8000 + best_d:50000 + best_d, 1])[0, 1]

            self.assertGreater(best_corr0, 0.999)
            self.assertGreater(corr1, 0.999)

    def test_mono_48000_and_resample_22050(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 1) Mono 48000 Hz
            sr1 = 48000
            t1 = np.arange(int(sr1 * 0.8)) / sr1
            x1 = 0.5 * np.sin(2 * np.pi * 1000.0 * t1)
            f1 = os.path.join(tmp, "mono48.flac")
            m1 = os.path.join(tmp, "mono48.mp3")
            sf.write(f1, x1, sr1, subtype="PCM_16")
            compile_flac_to_mp3(f1, m1, bitrate_kbps=128, visualize=False)
            dec1, dsr1 = sf.read(m1)
            self.assertEqual(dsr1, 48000)
            self.assertEqual(dec1.ndim, 1)
            freqs1 = np.fft.rfftfreq(len(dec1), 1.0 / dsr1)
            peak1 = freqs1[np.argmax(np.abs(np.fft.rfft(dec1)))]
            self.assertAlmostEqual(peak1, 1000.0, delta=5.0)

            # 2) Resampling fra 22050 Hz -> 44100 Hz MPEG-1 MP3
            sr2 = 22050
            t2 = np.arange(int(sr2 * 0.8)) / sr2
            x2 = 0.45 * np.sin(2 * np.pi * 660.0 * t2)
            f2 = os.path.join(tmp, "mono22.flac")
            m2 = os.path.join(tmp, "mono22.mp3")
            sf.write(f2, x2, sr2, subtype="PCM_16")
            compile_flac_to_mp3(f2, m2, bitrate_kbps=160, visualize=False)
            dec2, dsr2 = sf.read(m2)
            self.assertEqual(dsr2, 44100)
            freqs2 = np.fft.rfftfreq(len(dec2), 1.0 / dsr2)
            peak2 = freqs2[np.argmax(np.abs(np.fft.rfft(dec2)))]
            self.assertAlmostEqual(peak2, 660.0, delta=5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

