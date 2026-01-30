# utils_eeg_simulation.py
# Windows-safe + notebook-safe version (no hard-coded chdir, robust leadfield loading)

from __future__ import annotations

import os
from pathlib import Path
import numpy as np

###############JO

import numpy as np

def epoch_continuous(data, srate, tmin=0.0, tmax=None, epoch_len=2.0, overlap=0.0, drop_last=True):
    """
    data: (n_channels, n_times)
    returns epochs: (n_epochs, n_channels, n_times_epoch)
    """
    if data.ndim != 2:
        raise ValueError("data must be (n_channels, n_times)")
    n_ch, n_t = data.shape

    if tmax is None:
        tmax = n_t / srate

    start_samp = int(round(tmin * srate))
    stop_samp  = int(round(tmax * srate))
    stop_samp = min(stop_samp, n_t)

    win = int(round(epoch_len * srate))
    if win <= 0:
        raise ValueError("epoch_len too small")

    step = int(round((epoch_len - overlap) * srate))
    if step <= 0:
        raise ValueError("overlap must be < epoch_len")

    starts = np.arange(start_samp, stop_samp - win + 1, step, dtype=int)

    # If you want to keep a trailing partial epoch:
    if not drop_last and (len(starts) == 0 or starts[-1] + win < stop_samp):
        last_start = stop_samp - win
        if last_start > start_samp:
            starts = np.append(starts, last_start)

    epochs = np.stack([data[:, s:s+win] for s in starts], axis=0)
    times = np.arange(win) / srate  # relative time within epoch
    return epochs, times, starts


# ============================================================
# Helpers: paths + leadfield loader (works in .py and .ipynb)
# ============================================================

def _here() -> Path:
    """Directory of this file; falls back to current working dir in notebooks."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def load_leadfield_mat(lf_path: str | os.PathLike | None = None,
                       filename: str = "If_gain.mat") -> dict:
    """
    Load a leadfield mat file (Cohen-style If_gain.mat).

    Priority:
      1) lf_path (explicit)
      2) common project-relative locations
      3) current working directory
    """
    from scipy.io import loadmat

    if lf_path is not None:
        p = Path(lf_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Leadfield file not found: {p}")
        return loadmat(str(p))

    base = _here()

    candidates = [
        base / filename,
        base / "data" / filename,
        base / ".." / "data" / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
    ]

    for p in candidates:
        p = p.resolve()
        if p.exists():
            return loadmat(str(p))

    raise FileNotFoundError(
        f"Could not find {filename}. Tried:\n" + "\n".join(str(c) for c in candidates)
    )


# Try to load a default leadfield once (non-fatal: only used if caller passes lf_mat=None)
try:
    lf_mat_default = load_leadfield_mat()
except Exception:
    lf_mat_default = None


# ============================================================
# Simple sine simulators
# ============================================================

def calc_srate(samp_factor: int, freq: int) -> int:
    """Calculate sampling rate according to Nyquist frequency."""
    return 2 * freq * samp_factor


def sinewave_single_freq(srate: int, duration: float, noise_level: float, freq: int):
    """Simulate sine wave for a single frequency."""
    time_vec = np.arange(0, duration, 1 / srate)
    sinewave = np.sin(2 * np.pi * freq * time_vec)
    noise = noise_level * np.random.randn(len(time_vec))
    signal = sinewave + noise
    return signal, time_vec


def sinewave_multi_freq(srate: int, duration: float, noise_level: float, freqs: list):
    """
    Simulate sine wave with multiple frequencies.
    Returns combined sine wave + Gaussian noise.
    """
    time_vec = np.arange(0, duration, 1 / srate)
    signal = np.zeros(len(time_vec))
    for freq in freqs:
        signal += np.sin(2 * np.pi * freq * time_vec)
    signal += noise_level * np.random.randn(len(time_vec))
    return signal, time_vec


# ============================================================
# Cohen-style EEG simulation (1/f + transient bursts)
# ============================================================

def simulate_eeg_cohen(duration: int,
                       srate: int,
                       n_channels: int,
                       noise_level: float,
                       bands=None,
                       n_bursts=3,
                       amp_jitter=0.25,
                       width_jitter=0.25,
                       white_level=0.0,
                       seed=None):
    """
    Adapted MX Cohen method with physiologically realistic oscillations
    Reference: Cohen, M. X. (2014). Analyzing Neural Time Series Data. MIT Press.
    """
    if seed is not None:
        np.random.seed(seed)

    if bands is None:
        bands = {
            'theta': (4, 7, 1.5, 0.4, 1.5),
            'alpha': (8, 12, 2.5, 0.5, 2.0),
            'beta': (13, 30, 3.5, 0.3, 1.0),
        }

    num_pnts = int(duration * srate)
    time = np.arange(num_pnts) / srate
    eeg_data = np.zeros((n_channels, num_pnts))

    freqs = np.fft.fftfreq(num_pnts, 1 / srate)
    freq_mag = np.abs(freqs) + 0.001  # avoid divide-by-zero

    for chani in range(n_channels):
        real_part = np.random.randn(num_pnts)
        imag_part = np.random.randn(num_pnts)
        fft_signal = real_part + (1j * imag_part)

        fft_signal = fft_signal / freq_mag
        fft_signal[0] = 0

        half = int(np.floor(num_pnts / 2))
        if (num_pnts % 2) == 0:
            fft_signal[half] = np.real(fft_signal[half])
        fft_signal[-half + 1:] = np.conj(fft_signal[1:half][::-1])

        background = np.real(np.fft.ifft(fft_signal))
        background = background / np.std(background)

        clean_signal = np.zeros(num_pnts)

        for bandi, (low_freq, high_freq, center, width, amplitude) in bands.items():
            if isinstance(n_bursts, dict):
                num_bursts = int(n_bursts.get(bandi, 1))
            else:
                num_bursts = int(n_bursts)

            for _ in range(num_bursts):
                osc_freq = np.random.uniform(low_freq, high_freq)
                osc_center = np.random.uniform(time[0], time[-1])

                osc_width = width * (1 + width_jitter * np.random.randn())
                if osc_width <= 0:
                    osc_width = width

                osc_amp = amplitude * (1 + amp_jitter * np.random.randn())
                osc_pure = np.sin(2 * np.pi * osc_freq * time)

                taper = np.exp(-((time - osc_center) ** 2) / osc_width)
                clean_signal += osc_pure * taper * osc_amp

        noise_white = white_level * np.random.randn(num_pnts)
        signal = (background * noise_level) + clean_signal + noise_white
        eeg_data[chani] = signal

    freqs_psd = np.arange(half + 1) * (srate / num_pnts)
    psd = np.zeros((n_channels, half + 1))
    for chani in range(n_channels):
        data_demeaned = eeg_data[chani] - np.mean(eeg_data[chani])
        data_freq_domain = np.fft.fft(data_demeaned)
        power_spect = (np.abs(data_freq_domain) ** 2) / num_pnts
        psd[chani] = power_spect[0:half + 1]

    if n_channels == 1:
        return eeg_data[0], time, freqs_psd, psd[0]
    return eeg_data, time, freqs_psd, psd


# ============================================================
# PAC dipole simulation projected through leadfield
# ============================================================

def simulate_eeg_pac_dipoles(lf_mat=None,
                             srate=500,
                             duration=30,
                             ch_names=None,
                             orient=0,
                             dipoles=None,
                             theta_freq=6.0,
                             gamma_freqs=(45.0, 55.0),
                             coupling_percent=0.5,
                             noise_level_eeg=0.0,
                             noise_level_sources=1.0,
                             white_level_sources=0.0,
                             corr_strength=0.95,
                             seed=None):
    """
    Simulate EEG from dipole/source time series projected through a leadfield.
    Cohen MX (2017) eLife.

    lf_mat can be:
      - None -> uses lf_mat_default if found
      - path string / Path to If_gain.mat
      - loaded dict from scipy.io.loadmat
    """
    from scipy import signal
    from scipy.io import loadmat as _loadmat

    # Resolve lf_mat
    if lf_mat is None:
        if lf_mat_default is None:
            raise FileNotFoundError(
                "Leadfield not provided and default If_gain.mat not found.\n"
                "Place If_gain.mat in ./data/ or pass lf_mat as a path or loaded dict."
            )
        lf_mat_local = lf_mat_default
    elif isinstance(lf_mat, (str, os.PathLike)):
        lf_mat_local = _loadmat(str(lf_mat))
    else:
        lf_mat_local = lf_mat

    lf_gain = np.asarray(lf_mat_local["lf"][0, 0]["Gain"])
    if lf_gain.ndim == 3:
        if orient < 0 or orient >= lf_gain.shape[1]:
            raise ValueError(f"orient must be 0..{lf_gain.shape[1]-1}")
        Gain = lf_gain[:, orient, :]
    elif lf_gain.ndim == 2:
        Gain = lf_gain
    else:
        raise ValueError("lf_gain must be 2D (chans,sources) or 3D (chans,orient,sources)")

    n_channels, n_sources = Gain.shape
    num_pnts = int(duration * srate)
    time = np.arange(num_pnts) / srate

    def bandpass(x_vals, cut_low, cut_high, srate):
        nyq = srate / 2.0
        if cut_low <= 0 or cut_high >= nyq or cut_low >= cut_high:
            raise ValueError("Require 0 < cut_low < cut_high < srate/2")
        b, a = signal.butter(2, [cut_low / nyq, cut_high / nyq], btype="band")
        return signal.filtfilt(b, a, x_vals)

    if seed is not None:
        np.random.seed(seed)

    if dipoles is None:
        dipoles = {'theta': 93, 'gamma1': 108, 'gamma2': 110, 'gamma3': 115}

    # Correlated source noise
    corr_mat = np.random.rand(n_sources, n_sources)
    corr_mat = corr_mat @ corr_mat.T
    corr_mat = corr_mat / np.max(corr_mat)
    corr_mat = corr_strength * corr_mat
    np.fill_diagonal(corr_mat, 1.0)

    evals, evecs = np.linalg.eigh(corr_mat)
    evals[evals < 0] = 0
    mixing_mat = evecs @ np.sqrt(np.diag(evals))

    base = np.random.randn(n_sources, num_pnts)
    sources = mixing_mat @ base
    sources = sources / np.std(sources)
    sources *= noise_level_sources

    if white_level_sources != 0:
        sources += white_level_sources * np.random.randn(n_sources, num_pnts)

    # Theta oscillator with modulation
    modulation_amp = bandpass(np.random.randn(num_pnts), 1, 30, srate)
    modulation_freq = bandpass(np.random.randn(num_pnts), 1, 30, srate)
    modulation_amp = modulation_amp / np.std(modulation_amp)
    modulation_freq = signal.detrend(modulation_freq)

    theta_amp = 8.0 + (15.0 * modulation_amp)
    freq_mod = 15.0 * modulation_freq
    theta_phase = (2 * np.pi * theta_freq * time) + ((2 * np.pi / srate) * np.cumsum(freq_mod))
    theta_wave = theta_amp * np.sin(theta_phase)

    # Coupling windows (1 sec)
    coupling_mask = np.zeros(num_pnts)
    window_samples = int(srate)
    n_windows = int(duration * coupling_percent)
    for _ in range(n_windows):
        start = np.random.randint(0, num_pnts - window_samples)
        coupling_mask[start:start + window_samples] = 1.0

    # PAC gamma
    theta_phase_inst = np.angle(signal.hilbert(theta_wave))
    theta_env = ((1.0 + np.cos(theta_phase_inst)) / 2.0)
    theta_env = (0.9 * theta_env) ** 4
    theta_env *= coupling_mask

    noise_factor = 0.3 * np.random.randn(num_pnts)
    gamma1 = theta_env * np.sin(2 * np.pi * gamma_freqs[0] * time) * (1.0 + noise_factor)
    gamma2 = np.sin(2 * np.pi * gamma_freqs[1] * time) * (1.0 + noise_factor)

    i_theta = int(dipoles['theta'])
    i_g1 = int(dipoles['gamma1'])
    i_g2 = int(dipoles['gamma2'])

    sources[i_theta, :] += theta_wave
    sources[i_g1, :] += gamma1
    sources[i_g2, :] += gamma2

    if 'gamma3' in dipoles:
        i_g3 = int(dipoles['gamma3'])
        sources[int(i_g3), :] += gamma1

    data_eeg = Gain @ sources

    if noise_level_eeg != 0:
        data_eeg += noise_level_eeg * np.random.randn(n_channels, num_pnts)

    # Identify PAC channels (heuristic)
    theta_proj = np.abs(Gain[:, i_theta])
    gamma_proj = np.zeros(n_channels)

    gamma_ids = [i_g1, i_g2]
    if 'gamma3' in dipoles:
        gamma_ids.append(int(dipoles['gamma3']))

    for gi in gamma_ids:
        gamma_proj += np.abs(Gain[:, gi])

    theta_proj /= np.max(theta_proj)
    gamma_proj /= np.max(gamma_proj)

    pac_channels = np.where((theta_proj > 0.6) & (gamma_proj > 0.6))[0]
    return data_eeg, time, sources, coupling_mask, pac_channels


# ============================================================
# Advanced phase reset simulation
# ============================================================

def sinewave_multifreq_advanced_reset(srate: int,
                                      duration: float,
                                      noise_level: float,
                                      freqs: list,
                                      reset_times: list,
                                      reset_strengths: list,
                                      freq_specific_resets: bool = False):
    """
    Advanced phase reset simulation with frequency-specific control.
    """
    time_vec = np.arange(0, duration, 1 / srate)
    n_samples = len(time_vec)

    signal = np.zeros(n_samples)
    phase_history = np.zeros((n_samples, len(freqs)))

    reset_indices = [int(t * srate) for t in reset_times]
    current_phases = np.zeros(len(freqs))

    # For speed: make dict from index->reset_idx
    reset_map = {idx: k for k, idx in enumerate(reset_indices)}

    for i, t in enumerate(time_vec):
        if i in reset_map:
            reset_idx = reset_map[i]
            if freq_specific_resets:
                for j in range(len(freqs)):
                    current_phases[j] += reset_strengths[reset_idx][j]
            else:
                current_phases += reset_strengths[reset_idx]

        for j, freq in enumerate(freqs):
            signal[i] += np.sin(2 * np.pi * freq * t + current_phases[j])
            phase_history[i, j] = current_phases[j]

    signal += noise_level * np.random.randn(n_samples)
    return signal, time_vec, phase_history


# ============================================================
# Hyperscanning EEG simulation with shared drive (no chdir)
# ============================================================

def simulate_hyperscanning_eeg_shared_drive(lf_mat,
                                            srate=500,
                                            duration=30,
                                            theta_freq=6.0,
                                            gamma_freqs=(45.0, 55.0),
                                            dipoles=None,
                                            coupling_percent=0.5,
                                            coupling_strength_A=1.0,
                                            coupling_strength_B=1.0,
                                            coupling_delay_B_sec=0.0,
                                            noise_level_sources=1.0,
                                            white_level_sources=0.0,
                                            noise_level_eeg=0.0,
                                            corr_strength=0.95,
                                            orient=0,
                                            seed=None):
    """
    Hyperscanning EEG simulation with shared task drive.
    Output dict: eeg_a, eeg_b, sources_a, sources_b, shared_drive, time
    """
    from scipy import signal

    def perturb_gain(Gain, rel_std=0.02, seed=None):
        rng = np.random.default_rng(seed)
        scaling = 1.0 + (rel_std * rng.standard_normal(Gain.shape[1]))
        return Gain * scaling[np.newaxis, :]

    def correlated_sources(n_sources, n_pnts):
        corr = np.random.rand(n_sources, n_sources)
        corr = corr @ corr.T
        corr = corr / np.max(corr)
        corr = corr_strength * corr
        np.fill_diagonal(corr, 1.0)

        evals, evecs = np.linalg.eigh(corr)
        evals[evals < 0] = 0
        mix = evecs @ np.sqrt(np.diag(evals))

        base = np.random.randn(n_sources, n_pnts)
        src = mix @ base

        stdv = np.std(src)
        if stdv > 0:
            src /= stdv

        src *= noise_level_sources
        if white_level_sources > 0:
            src += white_level_sources * np.random.randn(n_sources, n_pnts)
        return src

    if seed is not None:
        np.random.seed(seed)

    # Dipole jitter
    if dipoles is None:
        range_val = 2
        jitter = np.random.randint(-range_val, range_val + 1, size=3)
        dipoles = {
            'theta': 93 + jitter[0],
            'gamma1': 108 + jitter[1],
            'gamma2': 110 + jitter[2],
        }

    n_pnts = int(duration * srate)
    time = np.arange(n_pnts) / srate

    win = int(srate)
    n_win = int(duration * coupling_percent)

    coupling_mask = np.zeros(n_pnts)
    for _ in range(n_win):
        i0 = np.random.randint(0, n_pnts - win)
        coupling_mask[i0:i0 + win] = 1.0

    # Shared slow envelope
    shared_env = signal.filtfilt(
        *signal.butter(2, [1 / (srate / 2), 30 / (srate / 2)], btype='band'),
        np.random.randn(n_pnts)
    )
    stdv = np.std(shared_env)
    if stdv > 0:
        shared_env /= stdv
    shared_env *= coupling_mask

    def participant_env(scale, delay_sec):
        shift = int(delay_sec * srate)
        env = np.roll(shared_env, shift)
        return scale * env

    def theta_wave():
        modulation = signal.detrend(
            signal.filtfilt(
                *signal.butter(2, [1 / (srate / 2), 30 / (srate / 2)], btype='band'),
                np.random.randn(n_pnts)
            )
        )
        phase = (2 * np.pi * theta_freq * time +
                 (2 * np.pi / srate) * np.cumsum(15 * modulation))
        amplitude = 8 + 15 * (modulation / np.std(modulation))
        theta_amp = amplitude * np.sin(phase)
        theta_phase = np.angle(signal.hilbert(np.sin(phase)))
        return theta_amp, theta_phase

    def gamma(theta_phase, envelope):
        tp = (1 + np.cos(theta_phase)) / 2
        phase_env = tp ** 4
        return envelope * phase_env * np.sin(2 * np.pi * gamma_freqs[0] * time)

    env_A = participant_env(coupling_strength_A, 0.0)
    env_B = participant_env(coupling_strength_B, coupling_delay_B_sec)

    theta_A, phase_A = theta_wave()
    theta_B, phase_B = theta_wave()

    gamma_A = gamma(phase_A, env_A)
    gamma_B = gamma(phase_B, env_B)

    # Gain
    lf_gain = np.asarray(lf_mat["lf"][0, 0]["Gain"])
    if lf_gain.ndim != 3:
        raise ValueError("Expected lf_mat['lf'][0,0]['Gain'] to be 3D (chans, orient, sources).")
    Gain_A = lf_gain[:, orient, :]

    seed_gain = None if seed is None else seed + 1
    Gain_B = perturb_gain(Gain_A, rel_std=0.02, seed=seed_gain)

    n_channels, n_sources = Gain_A.shape

    src_a = correlated_sources(n_sources, n_pnts)
    src_b = correlated_sources(n_sources, n_pnts)

    i_t = int(dipoles['theta'])
    i_g = int(dipoles['gamma1'])

    src_a[i_t] += theta_A
    src_a[i_g] += gamma_A

    src_b[i_t] += theta_B
    src_b[i_g] += gamma_B

    eeg_a = Gain_A @ src_a
    eeg_b = Gain_B @ src_b

    if noise_level_eeg > 0:
        eeg_a += noise_level_eeg * np.random.randn(*eeg_a.shape)
        eeg_b += noise_level_eeg * np.random.randn(*eeg_b.shape)

    return {
        "eeg_a": eeg_a,
        "eeg_b": eeg_b,
        "sources_a": src_a,
        "sources_b": src_b,
        "shared_drive": shared_env,
        "time": time,
    }


# ============================================================
# Demo (only runs if you execute this file directly)
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load leadfield (auto-search)
    if lf_mat_default is None:
        raise FileNotFoundError(
            "If_gain.mat not found. Put it in BrainHack\\data\\If_gain.mat "
            "or pass an explicit path to load_leadfield_mat()."
        )
    lf_mat = lf_mat_default

    output = simulate_hyperscanning_eeg_shared_drive(
        lf_mat,
        srate=500,
        duration=30,
        theta_freq=6.0,
        gamma_freqs=(45.0, 55.0),
        dipoles=None,
        coupling_percent=0.5,
        coupling_strength_A=1.0,
        coupling_strength_B=1.0,
        coupling_delay_B_sec=0.2,
        noise_level_sources=1.0,
        white_level_sources=0.5,
        noise_level_eeg=0.5,
        corr_strength=0.95,
        orient=0,
        seed=None
    )

    channel = 0
    t = output["time"]

    plt.figure()
    plt.plot(t, output["eeg_a"][channel], label="Participant A")
    plt.plot(t, output["eeg_b"][channel], label="Participant B", alpha=0.7)
    plt.xlim(5, 6)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title(f"EEG channel {channel}")
    plt.show()
