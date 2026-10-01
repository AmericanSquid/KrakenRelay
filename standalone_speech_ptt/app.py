"""Play a local speech clip into a virtual audio loopback on a PTT schedule."""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


ASSET = Path(__file__).resolve().parent / "assets" / "common_sense_appendix.m4a"
SAMPLE_RATE = 48000
CHANNELS = 2
FRAMES_PER_CHUNK = 960
BYTES_PER_FRAME = CHANNELS * 2


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value):
    parsed = float(value)
    if not 0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if not 0 <= parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be a nonnegative finite number")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--device", default="BlackHole 2ch", help="Virtual loopback output name or index")
    parser.add_argument("--monitor", help="Optional speaker output name or index for local listening")
    parser.add_argument("--count", type=positive_int, default=40, help="Number of keyups")
    parser.add_argument("--on-min", type=positive_float, default=3.0, help="Shortest keyup in seconds")
    parser.add_argument("--on-max", type=positive_float, default=12.0, help="Longest keyup in seconds")
    parser.add_argument("--off-min", type=nonnegative_float, default=1.0, help="Shortest gap in seconds")
    parser.add_argument("--off-max", type=nonnegative_float, default=4.0, help="Longest gap in seconds")
    parser.add_argument("--seed", type=int, help="Repeatable random schedule")
    parser.add_argument("--dry-run", action="store_true", help="Run schedule without opening audio devices")
    return parser


def make_schedule(args):
    if args.on_min > args.on_max or args.off_min > args.off_max:
        raise ValueError("minimum duration cannot exceed maximum duration")
    rng = random.Random(args.seed)
    return [
        (rng.uniform(args.on_min, args.on_max), rng.uniform(args.off_min, args.off_max))
        for _ in range(args.count)
    ]


def load_clip():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to decode the bundled speech clip")
    if not ASSET.is_file():
        raise RuntimeError(f"Bundled speech clip is missing: {ASSET}")
    command = [
        ffmpeg, "-nostdin", "-v", "error", "-i", str(ASSET),
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE), "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=30, check=True)
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot decode bundled speech clip: {exc}") from exc
    pcm = result.stdout
    if len(pcm) < FRAMES_PER_CHUNK * BYTES_PER_FRAME or not any(pcm):
        raise RuntimeError("Bundled speech clip contains no usable audio")
    return pcm[: len(pcm) - len(pcm) % BYTES_PER_FRAME]


def list_devices(pa):
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        print(
            f"{index}: {info['name']} "
            f"(input={info['maxInputChannels']}, output={info['maxOutputChannels']})"
        )


def find_device(pa, selector, *, loopback):
    devices = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]
    if str(selector).isdigit():
        matches = [device for device in devices if device["index"] == int(selector)]
    else:
        matches = [device for device in devices if selector.casefold() in device["name"].casefold()]
    matches = [device for device in matches if device["maxOutputChannels"] >= CHANNELS]
    if loopback:
        matches = [device for device in matches if device["maxInputChannels"] >= CHANNELS]
    if len(matches) != 1:
        kind = "virtual loopback" if loopback else "monitor output"
        raise RuntimeError(
            f"Expected one {kind} device matching {selector!r}; found {len(matches)}. "
            "Use --list-devices to inspect the available devices."
        )
    return matches[0]


class SignalMarkers:
    def active(self):
        logging.info("AUDIO ACTIVE; KrakenRelay should key from BlackHole input")

    def inactive(self):
        logging.info("AUDIO SILENCE; KrakenRelay tail timer starts")

class Player:
    def __init__(self, pcm, streams):
        self.pcm = pcm
        self.streams = streams
        self.offset = 0

    def play_for(self, duration):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            remaining_frames = max(1, int((deadline - time.monotonic()) * SAMPLE_RATE))
            frames = min(FRAMES_PER_CHUNK, remaining_frames)
            size = frames * BYTES_PER_FRAME
            if self.offset + size > len(self.pcm):
                self.offset = 0
            chunk = self.pcm[self.offset:self.offset + size]
            self.offset += size
            for stream in self.streams:
                stream.write(chunk, exception_on_underflow=False)


def run_schedule(schedule, player, markers, sleep=time.sleep):
    for number, (on_seconds, off_seconds) in enumerate(schedule, 1):
        logging.info("Keyup %02d/%02d: %.2fs", number, len(schedule), on_seconds)
        try:
            markers.active()
            player.play_for(on_seconds)
        finally:
            markers.inactive()
        if number < len(schedule):
            logging.info("Gap: %.2fs", off_seconds)
            sleep(off_seconds)


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        schedule = make_schedule(args)
        if args.dry_run:
            class DryPlayer:
                def play_for(self, duration):
                    time.sleep(duration)

            run_schedule(schedule, DryPlayer(), SignalMarkers())
            return 0

        try:
            import pyaudio
        except ImportError as exc:
            raise RuntimeError("Install PyAudio from requirements.txt to use audio output") from exc

        pa = pyaudio.PyAudio()
        try:
            if args.list_devices:
                list_devices(pa)
                return 0
            output = find_device(pa, args.device, loopback=True)
            monitor = find_device(pa, args.monitor, loopback=False) if args.monitor else None
            if monitor and monitor["index"] == output["index"]:
                raise ValueError("monitor and virtual output must be different devices")
            pcm = load_clip()
            streams = []
            try:
                for device in [output, monitor]:
                    if device is None:
                        continue
                    streams.append(pa.open(
                        format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        output=True,
                        output_device_index=device["index"],
                        frames_per_buffer=FRAMES_PER_CHUNK,
                    ))
                logging.info("Virtual input: %s; monitor: %s", output["name"], monitor["name"] if monitor else "off")
                run_schedule(schedule, Player(pcm, streams), SignalMarkers())
            finally:
                for stream in streams:
                    stream.stop_stream()
                    stream.close()
        finally:
            pa.terminate()
    except KeyboardInterrupt:
        logging.info("Interrupted")
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
