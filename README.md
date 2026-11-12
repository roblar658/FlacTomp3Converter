# Egenlaget Python 3 FLAC-til-MP3 Kompilator (`c:\dev\mp3`)

En komplett, egenlaget MPEG-1 Audio Layer III (MP3)-kompilator og native FLAC-dekoder skrevet i Python 3 (uten bruk av eksterne enkodere som `ffmpeg`, `lame` eller `pydub`), med steg-for-steg visualiseringer av hele kodingsprosessen.

## Filer i mappen

- `flac2mp3.py` — Hovedprogram / CLI som tar inn en `.flac`-fil, kompilerer `.mp3` og genererer visualiseringer.
- `flac_decoder.py` — Ren Python 3 native FLAC-dekoder (`STREAMINFO`, `CONSTANT`, `VERBATIM`, `FIXED 0..4`, `LPC 1..32`, Rice-koding, L/S-, R/S- og M/S-stereodekorrelasjon).
- `mp3_encoder.py` — Egenlaget MPEG-1 Layer III MP3-kompilator (resampling, 32-bånds polyfase filterbank, 36->18 MDCT + alias-reduksjon, psykoakustisk modell, ikke-lineær kvantisering med indre/ytre Rate-Distortion-løkke, 34 ISO Huffman-tabeller, bitreservoar).
- `mp3_tables.py` — Standard ISO/IEC 11172-3-tabeller (`ENWINDOW`, `SFB_LONG`, `SLEN1/2`, `PRETAB`, `CS/CA`, Huffman-tabeller 0..33).
- `visualizer.py` — Lager 7 steg-for-steg PNG-grafer og en interaktiv `index.html`-rapport.
- `test_mp3_compiler.py` — Automatisk test-suite som genererer FLAC-testfiler, kompilerer MP3, og verifiserer dem mot en uavhengig MP3-dekoder.

## Bruk

```powershell
cd c:\dev\mp3
python flac2mp3.py min_lyd.flac ut.mp3 --bitrate 192 --open
```

Valg:
- `--bitrate` (`32`..`320`, standard `192` kbps)
- `--vis-dir` (egen mappe for visualiseringer; standard `<ut>_visualisering`)
- `--no-vis` (hopp over visualiseringer)
- `--open` (åpne `index.html` automatisk i nettleser etter kompilering)

## De 7 visualiserte stegene

1. **Steg 1 (`steg_1_flac_til_pcm.png`)**: Native FLAC-dekoding -> normalisert PCM-bølgeform og zoom på én ramme (1152 samples).
2. **Steg 2 (`steg_2_polyfase_filterbank.png`)**: 32-bånds polyfase analysefilterbank over 18 tidsluker.
3. **Steg 3 (`steg_3_mdct_og_alias.png`)**: 36-punkts TDAC sinusvindu og 576 MDCT-spektrallinjer før/etter alias-reduksjon.
4. **Steg 4 (`steg_4_psykoakustisk_modell.png`)**: 1024-pt FFT, 21 skalafaktorbånd, Absolutt høreterskel (ATH) og maskeringsterskel.
5. **Steg 5 (`steg_5_kvantisering_og_rd.png`)**: Ikke-lineær kvantisering (`ix`), kvantiseringsstøy, binærsøk på `global_gain` og skalafaktorer.
6. **Steg 6 (`steg_6_huffman_og_bitreservoar.png`)**: Oppdeling i `big_values`, `count1` og `rzero` + bitreservoar (`main_data_begin`) per ramme.
7. **Steg 7 (`steg_7_resultat_og_sammenligning.png`)**: Filstørrelse/kompresjon (PCM vs. FLAC vs. MP3) og spektral SNR gjennom hele sporet.

