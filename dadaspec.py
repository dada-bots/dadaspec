#!/usr/bin/env python3
"""dadaspec — render a mel spectrogram thumbnail for an audio file.

    python dadaspec.py song.mp3        # writes song.jpg beside it
    python dadaspec.py tracks/ -f      # a whole directory, re-rendering

Needs numpy, Pillow and soundfile.

The image is 300x60. Top half is the left channel with bass at the bottom,
bottom half is the right with bass at the top, so the halves mirror and a
stereo difference reads as asymmetry. Brightness is per-bin dB; hue is which
band holds the energy (bass red, mid green, high blue), each normalised on its
own so a quiet passage still shows colour.
"""

import argparse
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

TARGET_SR = 32000
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".aiff", ".aif", ".m4a"}


SPEC_BANDS = [
    (0, 200, (1.0, 0.0, 0.0)),      # Bass -> Red
    (200, 1500, (0.0, 1.0, 0.0)),   # Mid  -> Green
    (1500, 16000, (0.0, 0.0, 1.0)), # High -> Blue
]
SPEC_W, SPEC_H = 300, 60   # default output size; override per call or with --width/--height

_BAND_COLORS = np.array([c for _, _, c in SPEC_BANDS], dtype=np.float32)

_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0

def _hz_to_mel(hz):
    hz = np.asarray(hz, dtype=np.float64)
    # np.where evaluates both branches — mask the log input so the linear-region
    # values don't trigger log(0) warnings (they're discarded anyway).
    log_term = np.log(np.maximum(hz, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOGSTEP
    return np.where(hz >= _MIN_LOG_HZ, _MIN_LOG_MEL + log_term, hz / _F_SP)


def _mel_to_hz(mels):
    mels = np.asarray(mels, dtype=np.float64)
    return np.where(mels >= _MIN_LOG_MEL,
                    _MIN_LOG_HZ * np.exp(_LOGSTEP * (mels - _MIN_LOG_MEL)),
                    _F_SP * mels)


def _mel_frequencies(n_mels, fmax, fmin=0.0):
    return _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels))

@lru_cache(maxsize=8)
def _mel_filterbank(n_mels, n_fft, sr, fmax):
    pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), n_mels + 2))
    fft_f = np.linspace(0, sr / 2, n_fft // 2 + 1)
    filt = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        left = (fft_f - lo) / max(ce - lo, 1e-10)
        right = (hi - fft_f) / max(hi - ce, 1e-10)
        filt[i] = np.maximum(0, np.minimum(left, right))
    # Slaney area-normalization (matches librosa norm='slaney' default)
    enorm = (2.0 / (pts[2:n_mels + 2] - pts[0:n_mels])).astype(np.float32)
    filt *= enorm[:, None]
    return filt

def _stft_power_np(y_ch, n_fft, hop_length):
    """Power STFT with a periodic Hann window, shaped (n_fft // 2 + 1, n_frames).

    Uncentred on purpose: every frame holds real signal and nothing invented.
    A centred STFT has to pad the ends, and whatever it pads with is a kink
    the window reads as broadband energy, so the first and last frame light up
    white. On a pure 800 Hz tone, the fraction of frame energy falling outside
    the tone's own band is 1.6e-6 mid-signal but 8.3e-2 in the first frame
    under reflect padding, and 4.1e-2 under zero or edge padding. Dropping the
    padding puts the edges back at 8.2e-7 and 3.9e-6.

    The cost is the final partial window, under 64 ms, which no thumbnail
    resolves. Signals shorter than one window are zero-padded to fill it,
    since otherwise there would be no frame at all.
    """
    y = np.asarray(y_ch, dtype=np.float32)
    if y.shape[-1] < n_fft:
        y = np.pad(y, (0, n_fft - y.shape[-1]))
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop_length] * win
    spec = np.fft.rfft(frames, axis=-1)
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float32).T

def _melspectrogram(y_ch, sr, n_mels=30, fmax=16000, hop_length=2048, n_fft=2048):
    fb = _mel_filterbank(n_mels, n_fft, sr, fmax)
    return fb @ _stft_power_np(y_ch, n_fft, hop_length)


def resample_linear(y, sr, target_sr):
    """Linear resample, equal to torch F.interpolate(mode='linear',
    align_corners=False) sample for sample. `y` is (channels, samples) or
    (samples,); returns float32 of the same rank."""
    y = np.asarray(y, dtype=np.float32)
    if sr == target_sr:
        return y
    n_in = y.shape[-1]
    n_out = int(round(n_in * target_sr / sr))
    # Source positions in float32 like torch's kernel: over millions of samples
    # float64 positions drift into different neighbour pairs.
    f32 = np.float32
    src = (np.arange(n_out, dtype=f32) + f32(0.5)) * f32(n_in / n_out) - f32(0.5)
    np.maximum(src, 0, out=src)
    i0 = src.astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    lam = src - i0.astype(f32)
    return (1 - lam) * y[..., i0] + lam * y[..., i1]

def _power_to_db(S, top_db=80.0):
    log_spec = 10.0 * np.log10(np.maximum(S, 1e-10))
    return np.maximum(log_spec - log_spec.max(), -top_db)

def _mel_channel(y_ch, sr, n_mels=30):
    """Compute dB-scaled mel + band-tinted RGB for one channel.

    Single mel spectrogram (no redundant STFT), hop=2048 for ~4x fewer frames.
    """
    S = _melspectrogram(y_ch, sr, n_mels=n_mels, fmax=16000, hop_length=2048)

    # dB-scale with gamma for visual contrast
    S_db = _power_to_db(S)
    np.clip(S_db, -60, 0, out=S_db)
    S_db += 60.0
    S_db /= 60.0
    np.power(S_db, 0.6, out=S_db)

    # Band colors from mel bin frequencies (no separate STFT needed)
    mel_f = _mel_frequencies(n_mels, fmax=16000)
    n_frames = S.shape[1]
    # Compute per-band normalized energy, then mix into RGB
    band_norms = np.empty((3, n_frames), dtype=np.float32)
    for i, (flo, fhi, _) in enumerate(SPEC_BANDS):
        mask = (mel_f >= flo) & (mel_f < fhi)
        if mask.any():
            power = np.sum(S[mask], axis=0)
            db = 10.0 * np.log10(power + 1e-10)
            np.clip(db, -20, None, out=db)
            db -= -20
            mx = db.max()
            if mx > 0:
                db /= mx
            band_norms[i] = db
        else:
            band_norms[i] = 0.0

    # (n_frames, 3) = (n_frames, 3_bands) @ (3_bands, 3_rgb)
    rgb = band_norms.T @ _BAND_COLORS
    for c in range(3):
        mx = rgb[:, c].max()
        if mx > 0:
            rgb[:, c] /= mx

    return S_db, rgb


def render_to_jpg(y, sr, jpg_path, width=SPEC_W, height=SPEC_H):
    """Render a 3-band tinted stereo mel spectrogram from an array.

    `y` is (channels, samples) or (samples,). int16 is scaled to float; float
    is used as-is. Mono is duplicated to stereo.

    `width`/`height` default to 300x60. Raising `width` resolves real detail
    up to one pixel per frame (sr/hop_length ~= 15.6 frames/sec, so ~2800 for
    a 3-minute track); past that it interpolates. `height` is interpolated
    above 60, which is all there is: 30 mel bins per channel, stacked.

    Raises on failure so the caller decides whether that is fatal.
    """
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float32) / 32768.0

    # Force stereo, accepting either (samples,) or (1, samples).
    if y.ndim == 1:
        y = np.stack([y, y])
    elif y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)

    S_L, rgb_L = _mel_channel(y[0], sr)
    S_R, rgb_R = _mel_channel(y[1], sr)

    nf = min(S_L.shape[1], S_R.shape[1])
    S_L, S_R = S_L[:, :nf], S_R[:, :nf]
    rgb_L, rgb_R = rgb_L[:nf], rgb_R[:nf]
    nm = S_L.shape[0]

    S_L = S_L[::-1]  # L: flip so bass at bottom

    # Vectorized compositing — no Python loop over frames
    img = np.empty((nm * 2, nf, 3), dtype=np.float32)
    img[:nm] = S_L[:, :, np.newaxis] * rgb_L[np.newaxis, :, :]
    img[nm:] = S_R[:, :, np.newaxis] * rgb_R[np.newaxis, :, :]

    np.clip(img, 0, 1, out=img)
    img *= 255
    Image.fromarray(img.astype(np.uint8)).resize(
        (width, height), Image.LANCZOS).save(
        str(jpg_path), quality=60, optimize=True)


# ---------------------------------------------------------------- decoding

def _decode_soundfile(path):
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y.T, sr


def _decode_ffmpeg(path):
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not on PATH")
    out = subprocess.run(
        [exe, "-v", "error", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(TARGET_SR), "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    if not out:
        raise RuntimeError("ffmpeg produced no samples")
    return np.frombuffer(out, dtype=np.float32).reshape(-1, 2).T.copy(), TARGET_SR


def load_audio(path, target_sr=TARGET_SR):
    """(channels, samples) float32 at target_sr. Tries soundfile, then ffmpeg."""
    errors = []
    for decode in (_decode_soundfile, _decode_ffmpeg):
        try:
            y, sr = decode(path)
        except Exception as e:
            errors.append(f"{decode.__name__}: {type(e).__name__}: {e}")
            continue
        if y.size == 0:
            errors.append(f"{decode.__name__}: empty")
            continue
        if target_sr and sr != target_sr:
            y = resample_linear(y, sr, target_sr)
            sr = target_sr
        return np.ascontiguousarray(y, dtype=np.float32), sr
    raise RuntimeError(f"could not decode {path}\n  " + "\n  ".join(errors))


def render_file(audio_path, image_path=None, target_sr=TARGET_SR,
                width=SPEC_W, height=SPEC_H):
    """Render one file. Defaults to the audio path with a .jpg suffix."""
    audio_path = Path(audio_path)
    image_path = Path(image_path) if image_path else audio_path.with_suffix(".jpg")
    y, sr = load_audio(audio_path, target_sr)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    render_to_jpg(y, sr, image_path, width, height)
    return image_path


# ---------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Render a mel spectrogram image beside each audio file.")
    ap.add_argument("inputs", nargs="+", metavar="FILE",
                    help="audio files, or directories of them")
    ap.add_argument("-o", "--out", metavar="PATH",
                    help="write here instead (a file for one input, else a directory)")
    ap.add_argument("-W", "--width", type=int, default=SPEC_W,
                    help=f"output width in pixels (default {SPEC_W})")
    ap.add_argument("-H", "--height", type=int, default=SPEC_H,
                    help=f"output height in pixels (default {SPEC_H})")
    ap.add_argument("--sr", type=int, default=TARGET_SR,
                    help=f"resample before analysis (default {TARGET_SR})")
    ap.add_argument("-f", "--force", action="store_true",
                    help="re-render even if the image exists")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.width < 1 or a.height < 1:
        print("--width and --height must be positive", file=sys.stderr)
        return 2

    files = []
    for p in map(Path, a.inputs):
        if p.is_dir():
            files += sorted(q for q in p.iterdir()
                            if q.is_file() and q.suffix.lower() in AUDIO_EXTS)
        else:
            files.append(p)
    if not files:
        print("no input files", file=sys.stderr)
        return 2

    out = Path(a.out) if a.out else None
    as_dir = out is not None and (len(files) > 1 or out.is_dir() or not out.suffix)

    failed = skipped = 0
    for f in files:
        dest = None if out is None else (out / (f.stem + ".jpg") if as_dir else out)
        check = dest if dest is not None else f.with_suffix(".jpg")
        if not a.force and check.exists():
            skipped += 1
            continue
        try:
            written = render_file(f, dest, a.sr, a.width, a.height)
        except Exception as e:
            print(f"{f}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
            continue
        if not a.quiet:
            print(written)

    if skipped and not a.quiet:
        print(f"({skipped} already rendered; -f to redo)", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
