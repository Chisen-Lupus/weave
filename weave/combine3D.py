# nufft_utils_wt_3d_mirror.py
# 3D NUFFT weighted coadd with mirror extension in the wavelength (lam) direction
# Reference: nufft_utils_wt.py for the 2D weighted CGNR pattern [1]

import numpy as np
import finufft


# ==============================================================
# Low-level NUFFT wrappers
# ==============================================================

def _nufft3d_forward(x, y, z, a_3d, eps=1e-8):
    """
    Type 2 NUFFT 3D: uniform Fourier coefficients -> nonuniform samples
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    z = np.ascontiguousarray(z, dtype=np.float64)
    a_3d = np.ascontiguousarray(a_3d, dtype=np.complex128)
    return finufft.nufft3d2(x, y, z, a_3d, eps=eps)


def _nufft3d_adjoint(x, y, z, data, n_modes, eps=1e-8):
    """
    Type 1 NUFFT 3D: nonuniform samples -> uniform Fourier coefficients
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    z = np.ascontiguousarray(z, dtype=np.float64)
    c = np.ascontiguousarray(data.astype(np.complex128))
    return finufft.nufft3d1(x, y, z, c, n_modes, eps=eps)


def _evaluate_on_grid_3d_mirror(a_3d, n_fine_xy, n_fine_lam):
    """
    Reconstruct on a uniform grid using zero-padded 3D iFFT.
    lam direction: only the [0, pi) physical half is returned.
    """
    Nx, Ny, Nz = a_3d.shape

    # lam direction uses 2*n_fine_lam points to cover [-pi, pi),
    # then only the [0, pi) half is kept.
    n_lam_full = 2 * n_fine_lam

    padded = np.zeros((n_fine_xy, n_fine_xy, n_lam_full),
                      dtype=np.complex128)

    cx = n_fine_xy  // 2
    cy = n_fine_xy  // 2
    cz = n_lam_full // 2

    x_s = cx - Nx // 2
    y_s = cy - Ny // 2
    z_s = cz - Nz // 2

    padded[x_s:x_s+Nx, y_s:y_s+Ny, z_s:z_s+Nz] = a_3d

    padded = np.fft.ifftshift(padded)
    full_recon = np.fft.ifftn(padded) * padded.size
    full_recon = np.fft.fftshift(full_recon)

    # Only keep lam in [0, pi)
    z_recon = full_recon[:, :, n_fine_lam:2*n_fine_lam].real

    return z_recon[::-1, ::-1, :]


# ==============================================================
# Mirror extension utility
# ==============================================================

def mirror_extend_lam(x, y, lam_01, data, weight=None):
    """
    Mirror-extend lam from [0,1] to [-1,1], then map to [-pi, pi).

    Even symmetry: f(lam) -> f(-lam) = f(lam)
    This enforces Neumann boundary conditions in the lam direction,
    suppressing Gibbs ringing at the lam boundaries.

    Parameters
    ----------
    x, y     : 1D arrays, spatial coordinates (already in [-pi, pi))
    lam_01   : 1D array, wavelength coordinate normalized to [0, 1]
    data     : 1D array, sample values
    weight   : 1D array or None, per-sample weights

    Returns
    -------
    x_ext, y_ext, lam_ext, data_ext, weight_ext
    """
    # Original data: lam in [0,1] -> [0, pi)
    lam_pos = lam_01 * np.pi

    # Mirror copy: lam -> -lam, maps to (-pi, 0]
    lam_neg = -lam_pos

    # Concatenate original + mirror
    x_ext    = np.concatenate([x, x])
    y_ext    = np.concatenate([y, y])
    lam_ext  = np.concatenate([lam_pos, lam_neg])
    data_ext = np.concatenate([data, data])  # even symmetry: same value

    if weight is not None:
        weight_ext = np.concatenate([weight, weight])
    else:
        weight_ext = np.ones_like(data_ext, dtype=np.float64)

    return x_ext, y_ext, lam_ext, data_ext, weight_ext


# ==============================================================
# Weighted CGNR solver (3D, with mirror extension)
# ==============================================================

def _fit_fourier_modes_3d_nufft_weighted_mirror(
    x, y, lam_01, data, weight,
    Kmax_x, Kmax_y, Kmax_z,
    max_iter=80,
    tol=1e-8,
    eps=1e-8,
    verbose=True,
):
    """
    Solve  min sum_i weight_i * |(A a)_i - data_i|^2  using CGNR with 3D NUFFT,
    after mirror-extending the lam direction.

    This follows the weighted CGNR pattern from nufft_utils_wt.py [1]:
      - sqrt(w) is absorbed into both forward and adjoint operators
      - forward_w(a) = sqrt(w) * A a
      - adjoint_w(v) = A^H (sqrt(w) * v)

    Parameters
    ----------
    x, y       : 1D arrays, spatial coordinates in [-pi, pi)
    lam_01     : 1D array, wavelength coordinate in [0, 1]
    data       : 1D array, sample values
    weight     : 1D array, non-negative per-sample weight
    Kmax_x, Kmax_y : int, spatial max frequency
    Kmax_z     : int, lam direction max frequency (after mirror extension)
    max_iter   : int
    tol        : float, relative residual tolerance
    eps        : float, NUFFT precision
    verbose    : bool

    Returns
    -------
    a_3d : (2*Kmax_x+1, 2*Kmax_y+1, 2*Kmax_z+1) complex array
    """
    # --- Mirror extension (doubles the data) ---
    x_ext, y_ext, lam_ext, data_ext, weight_ext = mirror_extend_lam(
        x, y, lam_01, data, weight
    )

    n_modes = (2*Kmax_x+1, 2*Kmax_y+1, 2*Kmax_z+1)

    x_   = np.ascontiguousarray(x_ext,    dtype=np.float64)
    y_   = np.ascontiguousarray(y_ext,    dtype=np.float64)
    lam_ = np.ascontiguousarray(lam_ext,  dtype=np.float64)
    d    = np.ascontiguousarray(data_ext, dtype=np.complex128)
    w    = np.ascontiguousarray(weight_ext, dtype=np.float64)

    if w.shape != d.shape:
        raise ValueError('weight must have the same shape as data')
    if np.any(w < 0):
        raise ValueError('weight must be non-negative')

    M_ext = len(x_)
    if verbose:
        print(f"  Mirror extension: {len(data)} -> {M_ext} samples")
        print(f"  Modes: {n_modes}")

    # --- Weighted operators (same pattern as nufft_utils_wt.py) [1] ---
    sqrt_w = np.sqrt(w)

    def forward_w(a_3d):
        """A_w a = sqrt(W) * A a"""
        return sqrt_w * _nufft3d_forward(x_, y_, lam_, a_3d, eps=eps)

    def adjoint_w(v):
        """A_w^H v = A^H (sqrt(W) * v)"""
        return _nufft3d_adjoint(x_, y_, lam_, sqrt_w * v, n_modes, eps=eps)

    # --- Weighted CGNR iteration [1] ---
    a = np.zeros(n_modes, dtype=np.complex128, order='C')

    # Weighted initial residual
    r = sqrt_w * d.copy()
    s = adjoint_w(r)
    p = s.copy()

    gamma = np.real(np.vdot(s, s))
    gamma0 = gamma if gamma > 0 else 1.0

    rel_res = np.sqrt(gamma / gamma0)

    for iteration in range(max_iter):
        q = forward_w(p)
        delta = np.real(np.vdot(q, q))

        if delta < 1e-30:
            if verbose:
                print(f'  CGNR stopped at iter {iteration+1}: delta too small')
            break

        alpha = gamma / delta
        a += alpha * p
        r -= alpha * q

        s = adjoint_w(r)
        gamma_new = np.real(np.vdot(s, s))
        rel_res = np.sqrt(gamma_new / (gamma0 + 1e-30))

        if verbose:
            print(f'  iter {iteration+1:3d}: rel_res = {rel_res:.2e}')

        if rel_res < tol:
            if verbose:
                print(f'  CGNR converged at iter {iteration+1}, '
                      f'relative residual = {rel_res:.2e}')
            gamma = gamma_new
            break

        beta = gamma_new / (gamma + 1e-30)
        p = s + beta * p
        gamma = gamma_new
    else:
        if verbose:
            print(f'  CGNR reached max_iter={max_iter}, '
                  f'relative residual = {rel_res:.2e}')

    return a


# ==============================================================
# Physical spectrum extraction
# ==============================================================

def extract_physical_spectrum(a_3d, Kmax_z):
    """
    Extract even (cosine) modes from mirror-extended Fourier coefficients.

    Mirror symmetry means only cos components in the lam direction
    are physical; sin components should be ~zero.

    Returns
    -------
    a_cos : (2*Kmax_x+1, 2*Kmax_y+1, Kmax_z+1) complex array
    """
    Nx, Ny, Nz = a_3d.shape
    cz = Nz // 2  # zero-frequency index

    a_cos = np.zeros((Nx, Ny, Kmax_z+1), dtype=np.complex128)

    # k=0 mode
    a_cos[:, :, 0] = a_3d[:, :, cz]

    # k>0: cos mode = a(+kz) + a(-kz)
    for k in range(1, Kmax_z+1):
        a_cos[:, :, k] = a_3d[:, :, cz+k] + a_3d[:, :, cz-k]

    return a_cos


# ==============================================================
# High-level reconstruction interface
# ==============================================================

def combine_cube_nufft_mirror_weighted(
    x_all, y_all, lam_all, z_all,
    base_size,
    weight_all=None,
    oversample=2,
    enlarge=1.0,
    Kmax_xy=None,
    Kmax_z=None,
    max_iter=80,
    tol=1e-8,
    eps=1e-8,
    verbose=True,
):
    """
    3D NUFFT weighted reconstruction with mirror extension in lam direction.

    Follows the interface pattern of combine_image_nufft_from_xyz
    in nufft_utils_wt.py [1], extended to 3D with mirror boundary
    conditions in the wavelength axis.

    Parameters
    ----------
    x_all, y_all : 1D arrays, spatial sample coordinates
    lam_all      : 1D array, wavelength values (physical units)
    z_all        : 1D array, sample values
    base_size    : float, spatial coordinate normalization scale
    weight_all   : 1D array or None
        Per-sample weights for weighted least squares.
        If None, uniform weighting is used.
    oversample   : float, oversampling factor for spatial axes
    enlarge      : float, field-of-view enlargement factor
    Kmax_xy      : int or None, max spatial frequency (auto if None)
    Kmax_z       : int or None, max lam frequency (auto if None)
    max_iter     : int
    tol          : float
    eps          : float, NUFFT precision
    verbose      : bool

    Returns
    -------
    z_recon_3d : 3D real array, reconstructed data cube
    a_3d       : 3D complex array, Fourier coefficients (mirror-extended)
    """
    x_all   = np.asarray(x_all).ravel()
    y_all   = np.asarray(y_all).ravel()
    lam_all = np.asarray(lam_all).ravel()
    z_all   = np.asarray(z_all).ravel()

    if weight_all is None:
        weight_all = np.ones_like(z_all, dtype=np.float64)
    else:
        weight_all = np.asarray(weight_all).ravel().astype(np.float64)

    if not (x_all.shape == y_all.shape == z_all.shape
            == lam_all.shape == weight_all.shape):
        raise ValueError(
            'x_all, y_all, lam_all, z_all, and weight_all '
            'must have the same shape'
        )
    if np.any(weight_all < 0):
        raise ValueError('weight_all must be non-negative')

    # --------------------------------------------------
    # Normalize spatial coords to [-pi, pi)
    # --------------------------------------------------
    x_all = x_all / base_size * (2 * np.pi) * enlarge
    y_all = y_all / base_size * (2 * np.pi) * enlarge

    # Normalize lam to [0, 1]
    lam_min, lam_max = lam_all.min(), lam_all.max()
    if lam_max > lam_min:
        lam_01 = (lam_all - lam_min) / (lam_max - lam_min)
    else:
        lam_01 = np.zeros_like(lam_all)

    # Spatial canvas mask
    mask = (
        (x_all >= -np.pi) & (x_all < np.pi) &
        (y_all >= -np.pi) & (y_all < np.pi)
    )

    x_all      = x_all[mask]
    y_all      = y_all[mask]
    z_all      = z_all[mask]
    lam_01     = lam_01[mask]
    weight_all = weight_all[mask]

    M = len(x_all)

    # --------------------------------------------------
    # Fourier mode counts
    # --------------------------------------------------
    R_OUT = int((base_size / 2) / enlarge * oversample)

    if Kmax_xy is None:
        Kmax_x = R_OUT
        Kmax_y = R_OUT
    else:
        Kmax_x = int(Kmax_xy)
        Kmax_y = int(Kmax_xy)

    if Kmax_z is None:
        # Default: use R_OUT for lam as well (user should override)
        _Kmax_z = R_OUT
    else:
        _Kmax_z = int(Kmax_z)

    n_modes_x = 2 * Kmax_x + 1
    n_modes_y = 2 * Kmax_y + 1
    n_modes_z = 2 * _Kmax_z + 1
    N_modes   = n_modes_x * n_modes_y * n_modes_z

    if verbose:
        print(f'samples:        {M}')
        print(f'Fourier modes:  ({n_modes_x})x({n_modes_y})x({n_modes_z})'
              f' = {N_modes}')
        print(f'Fitting {N_modes} modes from {M} samples '
              f'(x2 after mirror) ...')
        print(f'weighted fit:   {weight_all is not None}')

    # --------------------------------------------------
    # Weighted 3D NUFFT solve with mirror extension
    # --------------------------------------------------
    a_3d = _fit_fourier_modes_3d_nufft_weighted_mirror(
        x_all, y_all, lam_01, z_all, weight_all,
        Kmax_x, Kmax_y, _Kmax_z,
        max_iter=max_iter,
        tol=tol,
        eps=eps,
        verbose=verbose,
    )

    # --------------------------------------------------
    # Reconstruct on grid
    # --------------------------------------------------
    n_fine_xy  = 2 * R_OUT + 1
    n_fine_lam = 2 * _Kmax_z + 1
    z_recon_3d = _evaluate_on_grid_3d_mirror(a_3d, n_fine_xy, n_fine_lam)

    if verbose:
        print(f'cube size:      {z_recon_3d.shape}')
        print(f'oversample:     {oversample}')

    return z_recon_3d, a_3d


# ==============================================================
# Diagnostic utilities
# ==============================================================

def check_mirror_symmetry(a_3d, Kmax_z, verbose=True):
    """
    Verify that sin components are small (mirror symmetry check).

    Returns
    -------
    ratio : float, sin/cos power ratio (should be << 1)
    """
    cz = a_3d.shape[2] // 2
    sin_power = 0.0
    cos_power = 0.0

    for k in range(1, Kmax_z + 1):
        sin_coeff = a_3d[:, :, cz+k] - a_3d[:, :, cz-k]
        cos_coeff = a_3d[:, :, cz+k] + a_3d[:, :, cz-k]
        sin_power += np.sum(np.abs(sin_coeff)**2)
        cos_power += np.sum(np.abs(cos_coeff)**2)

    ratio = sin_power / (cos_power + 1e-30)

    if verbose:
        print(f"Sin/Cos power ratio: {ratio:.2e}")
        print(f"  (should be << 1 if mirror BC is appropriate)")

    return ratio


def compute_residual(x_all, y_all, lam_01, z_all, weight_all, a_3d,
                     verbose=True):
    """
    Compute weighted and unweighted prediction errors on original samples.
    """
    x_ext, y_ext, lam_ext, data_ext, weight_ext = mirror_extend_lam(
        x_all, y_all, lam_01, z_all, weight_all
    )

    M = len(z_all)

    z_pred_ext = _nufft3d_forward(
        np.ascontiguousarray(x_ext, dtype=np.float64),
        np.ascontiguousarray(y_ext, dtype=np.float64),
        np.ascontiguousarray(lam_ext, dtype=np.float64),
        a_3d
    )

    # Only the first half is the original (non-mirrored) data
    z_pred = z_pred_ext[:M]
    err = np.abs(z_pred - z_all)

    if verbose:
        print(f"Max  error: {err.max():.2e}")
        print(f"Mean error: {err.mean():.2e}")
        if weight_all is not None:
            w_err = np.sqrt(weight_all) * err
            print(f"Weighted RMS error: "
                  f"{np.sqrt(np.mean(w_err**2)):.2e}")

    return z_pred, err

