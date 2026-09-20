# proseprobe

Explainable scoring of how machine-like a text reads.

proseprobe reports a calibrated scale with confidence bands, names the
stylometric features that moved the score and quotes the spans they came from,
and applies a guard for non-native English, where detectors systematically fail.
It never returns a binary verdict: "written by a machine" is not a decision this
library is willing to make on your behalf.

## Status

Pre-alpha. The package installs and imports; the scoring pipeline is being built
in the open.

## Install

```bash
pip install proseprobe
```

The core runs on the standard library. NumPy is optional and only speeds up the
numeric paths:

```bash
pip install "proseprobe[numpy]"
```

Python 3.11 or newer is required.

## Usage

```python
import proseprobe

print(proseprobe.__version__)
```

## Development

```bash
pip install -e ".[numpy]"
python -m pytest
```

## License

MIT. See [LICENSE](LICENSE).

Maintained by [Shipmind Labs](https://shipmindlabs.com).
