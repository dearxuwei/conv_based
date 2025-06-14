# Conv-Based SSVEP Models

This repository contains several neural network models for steady-state visual evoked potential (SSVEP) classification. The code implements different convolutional cross-correlation architectures and training scripts for the SSVEP Benchmark dataset.

## Repository Structure

- **`convca_read.py`** – Utilities for loading the Benchmark dataset, generating template/reference signals, and applying a Chebyshev band‑pass filter.
- **`convca_read_djl.py`** – Data loader for an alternative DJL dataset with similar preprocessing routines.
- **`CONVCA.py`** – Implementation of the `ConvCA` model that extracts features via time‑domain convolutions and computes correlations with template and reference signals.
- **`FFCA.py`** – Defines `FFTCA`, a variant where the time‑domain convolutions are replaced by learnable frequency‑domain filtering.
- **`ConvFFTCA.py`** – Contains the SE fusion model `ConvFFTCA_SE` and the new `ConvFFTCA_Parallel` variant that fuses time/frequency branches during correlation.
- **`main_CONCA.py`**, **`main_FFCA.py`**, **`main_ConvFFCA.py`** – Example scripts that perform leave‑one‑subject‑out experiments on the Benchmark dataset using the corresponding models.

## Usage

1. Place the SSVEP Benchmark data under `./Benchmark/` as expected by `convca_read.py`.
2. Choose one of the training scripts, for example:
   ```bash
   python main_CONCA.py      # ConvCA model
   python main_FFCA.py       # FFTCA model
   python main_ConvFFCA.py   # ConvFFTCA‑SE model
   # ConvFFTCA_Parallel can be imported for custom experiments
   ```
   Each script will iterate through all subjects, split runs into training and testing sessions, and output accuracy logs under the specified result directory.

The network architectures are defined within the model files and can also be imported as modules for custom training routines.

## Preprocessing

`convca_read.py` filters raw EEG with a Chebyshev Type‑I band‑pass filter before building dataset windows:

```python
nyq = 0.5 * Fs
Wp = [6/nyq, 90/nyq]
Ws = [4/nyq, 100/nyq]
N, Wn = cheb1ord(Wp, Ws, 3, 40)
b, a = cheby1(N, 0.5, Wn, 'bandpass')
```

## License

This project is provided as-is for research purposes.
