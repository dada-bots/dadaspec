# dadaspec

Mel spectrogram thumbnails for audio files. Point it at audio, get a 300×60
image beside it.

Two files, no package, no install:

```bash
python dadaspec_numpy.py song.mp3     # numpy + Pillow + soundfile
python dadaspec_torch.py song.mp3     # torch (+ torchaudio if present)
```

Both write `song.jpg` next to the input.

```bash
python dadaspec_numpy.py tracks/              # every audio file in a directory
python dadaspec_numpy.py a.wav b.wav -o out/  # into a directory instead
python dadaspec_numpy.py song.mp3 -f          # re-render an existing image
```

From Python:

```python
from dadaspec_numpy import render_file, render_to_jpg
render_file("song.mp3")            # -> Path("song.jpg")
render_to_jpg(y, sr, "out.jpg")    # y is (channels, samples) or (samples,)
```

## Reading the image

The top half is the left channel with bass at the bottom; the bottom half is
the right channel with bass at the top. They mirror around the centre line, so
a stereo difference shows as asymmetry and a mono file is exactly symmetric.

Brightness is per-bin dB. Hue says which band carries the energy — **bass red,
mid green, high blue** — and each band is normalised on its own, so a quiet
passage still shows colour instead of fading to black. A uniformly loud
spectrum goes white; one with only low end goes red however loud it is.

That per-band normalisation is deliberate: these are thumbnails for scanning a
list of takes at a glance, not measurement tools. Two images are comparable in
shape, not in absolute level.

## Why two files

The DSP is 30 mel bins, a 2048-point FFT at hop 2048, a Slaney-normalised
filterbank capped at 16 kHz, over audio resampled to 32 kHz.

`dadaspec_numpy.py` does that in numpy; `dadaspec_torch.py` does it with
`torch.stft` and `F.interpolate`. The numpy version was written against the
torch one and matches it exactly — hand them the same samples and the images
are pixel-for-pixel equal.

They can still disagree on a lossy file, because the decoders do: torchaudio
returns a few hundred more samples than soundfile for the same mp3, and that
offset shifts every frame after it. Same renderer, different audio going in.

Pick numpy if you do not want a torch dependency; pick torch if you already
have one and want it on the GPU path.

## Decoding

numpy build: `soundfile`, then `ffmpeg` if it is on PATH.
torch build: `torchaudio`, then those two.

Enough for wav, flac, ogg and — on any recent libsndfile — mp3.

## Origin

Extracted from [underfit](https://github.com/dada-bots/underfit), where it
renders training-demo thumbnails. Standalone and dependency-light so it can be
used anywhere; underfit does not import it.

MIT.
