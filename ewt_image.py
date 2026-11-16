"""
Image Processing using the Empirical Wavelet Transform (2D tensor EWT)
----------------------------------------------------------------------
Experiments:
  1. Adaptive sub-band decomposition of an image
  2. Perfect-reconstruction check
  3. Denoising by sub-band soft thresholding
  4. Detail enhancement by sub-band gain

Requires: numpy, scipy, matplotlib, scikit-image
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.io import imsave

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2611)          # fixed seed -> repeatable results


# ---------------------------------------------------------------- INPUT
def make_test_image(n=256):
    """Synthetic 'tiled courtyard' scene: smooth shading + three textures
    of different spatial frequency + sharp-edged shapes."""
    y, x = np.mgrid[0:n, 0:n].astype(float)
    img = 95 + 55 * np.exp(-((x - 70) ** 2 + (y - 60) ** 2) / (2 * 90 ** 2))
    img += 0.12 * (x - n / 2)                                   # illumination ramp

    # coarse checker tiles (low frequency), lower-left region
    tiles = np.sign(np.sin(2 * np.pi * x / 44) * np.sin(2 * np.pi * y / 44))
    img += 22 * tiles * ((y > 150) & (x < 140))

    # woven fabric (mid frequency), right strip
    weave = np.sin(2 * np.pi * x / 9) + np.sin(2 * np.pi * y / 9)
    img += 14 * weave * ((x > 168) & (y > 40) & (y < 215))

    # fine grille (high frequency), top band
    grille = np.sin(2 * np.pi * (x + 0.6 * y) / 4.2)
    img += 12 * grille * ((y < 34) & (x > 20) & (x < 235))

    # shapes with sharp edges
    img[(x - 92) ** 2 + (y - 96) ** 2 < 30 ** 2] = 205          # bright disc
    img[(x - 92) ** 2 + (y - 96) ** 2 < 13 ** 2] = 60           # dark core
    img[(np.abs(x - 205) + np.abs(y - 232)) < 17] = 225         # diamond
    img[(y > 118) & (y < 126) & (x > 15) & (x < 160)] = 40      # dark bar
    return np.clip(img, 0, 255)


# ------------------------------------------------- EWT BUILDING BLOCKS
def beta(x):
    """Meyer auxiliary polynomial, rises smoothly from 0 to 1 on [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    return x ** 4 * (35 - 84 * x + 70 * x ** 2 - 20 * x ** 3)


def mean_spectrum(img, axis):
    """Average magnitude spectrum of all rows (axis=1) or columns (axis=0),
    returned on the half band [0, pi]."""
    n = img.shape[axis]
    spec = np.abs(np.fft.fft(img - img.mean(), axis=axis))
    spec = spec.mean(axis=1 - axis)
    return spec[: n // 2 + 1]


def detect_boundaries(spec, n_bands, smooth=3.0, guard=4):
    """Segment the spectrum into n_bands supports.
    Keeps the (n_bands-1) strongest local maxima of the smoothed
    log-spectrum (plus the origin) and places each boundary at the
    lowest minimum between two neighbouring maxima."""
    m = len(spec)
    s = gaussian_filter1d(np.log1p(spec), smooth, mode="nearest")
    # remove the slow 1/f decay so that texture peaks stand out
    s = s - gaussian_filter1d(s, 10 * smooth, mode="nearest")
    idx = [k for k in range(guard, m - guard)
           if s[k] > s[k - 1] and s[k] >= s[k + 1]]
    idx = sorted(idx, key=lambda k: s[k], reverse=True)[: n_bands - 1]
    peaks = [0] + sorted(idx)
    bounds = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        bounds.append(a + int(np.argmin(s[a:b + 1])))
    bounds = [max(b, 2) for b in bounds]
    return np.array(bounds) * np.pi / (m - 1), s


def ewt_filter_bank(bounds, n):
    """Empirical Littlewood-Paley-Meyer filters (frequency domain).
    Returns array (len(bounds)+1, n): scaling filter + wavelets."""
    w = np.abs(2 * np.pi * np.fft.fftfreq(n))                  # |omega| in [0, pi]
    wb = np.concatenate([bounds, [np.pi]])
    r = (wb[1:] - wb[:-1]) / (wb[1:] + wb[:-1])
    gamma = 0.95 * min(r.min(), 1.0)       # transition zones must not overlap
    N = len(bounds)
    fb = np.zeros((N + 1, n))

    def rise(w0):       # 0 -> 1 across the transition around boundary w0
        return np.sin(np.pi / 2 * beta((w - (1 - gamma) * w0) / (2 * gamma * w0)))

    def fall(w0):       # 1 -> 0 across the transition around boundary w0
        return np.cos(np.pi / 2 * beta((w - (1 - gamma) * w0) / (2 * gamma * w0)))

    fb[0] = fall(bounds[0])                                    # scaling function
    for k in range(N):
        lo = rise(bounds[k])
        hi = fall(bounds[k + 1]) if k + 1 < N else 1.0         # last band runs to pi
        fb[k + 1] = lo * hi
    return fb


def ewt2d(img, n_bands=3):
    """Tensor 2D EWT. Returns sub-bands[i][j] (vertical band i, horizontal band j),
    the two filter banks and the detected boundaries."""
    H, W = img.shape
    spec_r = mean_spectrum(img, axis=1)        # along x (rows)
    spec_c = mean_spectrum(img, axis=0)        # along y (columns)
    b_r, s_r = detect_boundaries(spec_r, n_bands)
    b_c, s_c = detect_boundaries(spec_c, n_bands)
    fb_r = ewt_filter_bank(b_r, W)
    fb_c = ewt_filter_bank(b_c, H)
    F = np.fft.fft2(img)
    bands = [[np.real(np.fft.ifft2(F * fb_c[i][:, None] * fb_r[j][None, :]))
              for j in range(n_bands)] for i in range(n_bands)]
    info = dict(b_r=b_r, b_c=b_c, spec_r=spec_r, spec_c=spec_c, s_r=s_r, s_c=s_c)
    return bands, fb_r, fb_c, info


def iewt2d(bands, fb_r, fb_c):
    """Inverse tensor EWT (the filter bank is a tight frame, so the
    inverse is the sum of every sub-band filtered again by its own filter)."""
    acc = 0
    for i, row in enumerate(bands):
        for j, b in enumerate(row):
            acc = acc + np.fft.fft2(b) * fb_c[i][:, None] * fb_r[j][None, :]
    return np.real(np.fft.ifft2(acc))


# -------------------------------------------------------- APPLICATIONS
def soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def estimate_sigma(bands, fb_r, fb_c):
    """Noise level from the finest diagonal sub-band (median rule)."""
    g = np.sqrt(np.mean(fb_c[-1] ** 2) * np.mean(fb_r[-1] ** 2))
    return np.median(np.abs(bands[-1][-1])) / 0.6745 / g


def ewt_denoise(noisy, n_bands=3, k=1.5):
    bands, fb_r, fb_c, info = ewt2d(noisy, n_bands)
    sigma = estimate_sigma(bands, fb_r, fb_c)
    for i in range(n_bands):
        for j in range(n_bands):
            if i == 0 and j == 0:
                continue                                   # keep approximation
            gain = np.sqrt(np.mean(fb_c[i] ** 2) * np.mean(fb_r[j] ** 2))
            bands[i][j] = soft(bands[i][j], k * sigma * gain)
    return np.clip(iewt2d(bands, fb_r, fb_c), 0, 255), sigma, info


def ewt_enhance(img, n_bands=3, gains=(1.0, 1.6, 2.2)):
    bands, fb_r, fb_c, _ = ewt2d(img, n_bands)
    for i in range(n_bands):
        for j in range(n_bands):
            bands[i][j] = bands[i][j] * gains[max(i, j)]
    return np.clip(iewt2d(bands, fb_r, fb_c), 0, 255)


def edge_strength(img):
    gy, gx = np.gradient(img)
    return float(np.mean(np.hypot(gx, gy)))


if __name__ == "__main__":
    NB = 3          # bands per direction for decomposition / enhancement
    NB_DN = 4       # bands per direction for denoising
    img = make_test_image(256)
    imsave(f"{OUT}/01_input.png", img.astype(np.uint8))
    print(f"Input image: {img.shape}, min={img.min():.0f}, max={img.max():.0f}, "
          f"mean={img.mean():.2f}, std={img.std():.2f}")

    # ---- Experiment 1: decomposition
    bands, fb_r, fb_c, info = ewt2d(img, NB)
    print("\nEXPERIMENT 1 - Adaptive decomposition")
    print("Row boundaries    (rad):", np.round(info["b_r"], 4))
    print("Column boundaries (rad):", np.round(info["b_c"], 4))
    energy = np.array([[np.sum((b - b.mean()) ** 2) for b in row] for row in bands])
    print("Energy share of each sub-band, mean removed (%):")
    for i in range(NB):
        print("   " + "  ".join(f"{100 * energy[i, j] / energy.sum():7.3f}"
                                for j in range(NB)))

    # ---- Experiment 2: perfect reconstruction
    rec = iewt2d(bands, fb_r, fb_c)
    err = np.max(np.abs(rec - img))
    print("\nEXPERIMENT 2 - Reconstruction")
    print(f"Max abs error = {err:.3e},  MSE = {np.mean((rec - img) ** 2):.3e}")
    frame = np.max(np.abs((fb_r ** 2).sum(0) - 1))
    print(f"Tight-frame check  max|sum(filter^2) - 1| = {frame:.3e}")

    # ---- Experiment 3: denoising
    print("\nEXPERIMENT 3 - Denoising (PSNR dB / SSIM)")
    rows = []
    for sd in (10, 20, 30):
        noisy = img + rng.normal(0, sd, img.shape)
        den, s_hat, _ = ewt_denoise(noisy, NB_DN)
        gau = gaussian_filter(noisy, 1.0)
        med = median_filter(noisy, 3)
        m = lambda a: (psnr(img, np.clip(a, 0, 255), data_range=255),
                       ssim(img, np.clip(a, 0, 255), data_range=255))
        rows.append((sd, s_hat, m(noisy), m(gau), m(med), m(den)))
        print(f"sigma={sd:2d} (estimated {s_hat:5.2f}) | noisy {m(noisy)[0]:.2f}/{m(noisy)[1]:.3f}"
              f" | gaussian {m(gau)[0]:.2f}/{m(gau)[1]:.3f}"
              f" | median {m(med)[0]:.2f}/{m(med)[1]:.3f}"
              f" | EWT {m(den)[0]:.2f}/{m(den)[1]:.3f}")
        if sd == 20:
            keep = (noisy, gau, med, den)

    # ---- Experiment 4: enhancement
    enh = ewt_enhance(img, NB)
    print("\nEXPERIMENT 4 - Detail enhancement")
    print(f"Mean gradient magnitude: input {edge_strength(img):.3f} -> "
          f"enhanced {edge_strength(enh):.3f}")
    print(f"Std deviation (contrast): input {img.std():.2f} -> enhanced {enh.std():.2f}")

    # ------------------------------------------------------- FIGURES
    g = dict(cmap="gray", vmin=0, vmax=255)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, key, bkey, name in ((ax[0], "spec_r", "b_r", "rows (horizontal frequency)"),
                               (ax[1], "spec_c", "b_c", "columns (vertical frequency)")):
        w = np.linspace(0, np.pi, len(info[key]))
        a.semilogy(w, info[key] + 1, color="#1f4e79", lw=1.2)
        for b in info[bkey]:
            a.axvline(b, color="#c0392b", ls="--", lw=1.4)
        a.set(title=f"Mean spectrum of {name}", xlabel="frequency (rad/sample)",
              ylabel="magnitude (log)", xlim=(0, np.pi))
        a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/02_spectrum_boundaries.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    w = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(fb_r.shape[1]))
    for k, lab in enumerate(["scaling (low)", "wavelet 1 (mid)", "wavelet 2 (high)"]):
        ax.plot(w, np.fft.fftshift(fb_r[k]), lw=1.8, label=lab)
    ax.set(title="Empirical wavelet filter bank (row direction)",
           xlabel="frequency (rad/sample)", ylabel="gain", xlim=(-np.pi, np.pi))
    ax.legend(loc="center right", fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/03_filter_bank.png", dpi=130); plt.close()

    fig, ax = plt.subplots(NB, NB, figsize=(9, 9))
    names = ["low", "mid", "high"]
    for i in range(NB):
        for j in range(NB):
            b = bands[i][j]
            if i == 0 and j == 0:
                ax[i, j].imshow(b, cmap="gray")
            else:
                v = np.percentile(np.abs(b), 99.5) + 1e-9
                ax[i, j].imshow(b, cmap="gray", vmin=-v, vmax=v)
            ax[i, j].set_title(f"vertical {names[i]} / horizontal {names[j]}", fontsize=9)
            ax[i, j].axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/04_subbands.png", dpi=120); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.9))
    ax[0].imshow(img, **g); ax[0].set_title("Input")
    ax[1].imshow(rec, **g); ax[1].set_title("Reconstructed (inverse EWT)")
    ax[2].imshow(np.abs(rec - img), cmap="magma"); ax[2].set_title(f"|Error|, max = {err:.1e}")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/05_reconstruction.png", dpi=120); plt.close()

    noisy, gau, med, den = keep
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.7))
    for a, im, t in zip(ax, (img, noisy, gau, med, den),
                        ("Input", "Noisy (sigma = 20)", "Gaussian filter",
                         "Median 3x3", "EWT soft threshold")):
        a.imshow(np.clip(im, 0, 255), **g)
        if im is not img:
            t += f"\nPSNR {psnr(img, np.clip(im, 0, 255), data_range=255):.2f} dB"
        a.set_title(t, fontsize=10); a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/06_denoising.png", dpi=120); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.9))
    ax[0].imshow(img, **g); ax[0].set_title("Input")
    ax[1].imshow(enh, **g); ax[1].set_title("EWT detail-enhanced")
    ax[2].imshow(enh - img, cmap="gray"); ax[2].set_title("Added detail (difference)")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/07_enhancement.png", dpi=120); plt.close()
    print("\nFigures saved in", OUT)
