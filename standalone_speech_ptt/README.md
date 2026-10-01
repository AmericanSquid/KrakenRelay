# Standalone speech / PTT source

This program has no KrakenRelay imports. It plays a bundled excerpt of the
Gary Gilberd reading of the appendix to Thomas Paine's *Common Sense* into a
macOS virtual loopback device. KrakenRelay listens to that device and handles
its own squelch, TX, and PTT logic. The recording is public domain in the USA:
[LibriVox](https://librivox.org/common-sense-by-thomas-paine/),
[original audio](https://archive.org/download/commonsense_gg_librivox/commonsense_6_paine.mp3).

Install FFmpeg, PyAudio, and a virtual loopback driver such as
[BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole). BlackHole
creates the audio device; this application feeds it during each scheduled
signal window. Select `BlackHole 2ch` as KrakenRelay's input device.

On macOS with Homebrew:

```sh
brew install ffmpeg blackhole-2ch
python3 -m pip install -r requirements.txt
```

```sh
python app.py --list-devices
python app.py --device 'BlackHole 2ch' --monitor 'MacBook Air Speakers' \
  --count 5 --on-min 3 --on-max 8 --off-min 1 --off-max 3
```

Start KrakenRelay with BlackHole as its input before running this app:

```sh
python ../run.py --headless --input 'BlackHole 2ch' \
  --output 'MacBook Air Speakers'
```

The app logs when audio becomes active or silent. KrakenRelay's configured
tail time determines when its controller unkeys after silence. A physical
radio keys only if KrakenRelay's configured PTT interface is connected and
working; this app does not bypass that controller.

`--dry-run --count 2 --on-min .1 --on-max .1 --off-min .1 --off-max .1`
exercises the randomized signal schedule without opening audio devices.
