# nufft_utils_wt.py

# from nufft_utils import _nufft_forward, _nufft_adjoint, _evaluate_on_grid_nufft

import numpy as np
import finufft


def _nufft_forward(x, y, a_2d, eps=1e-8):
    """
    Type 2 NUFFT: uniform Fourier coefficients -> nonuniform samples
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    a_2d = np.asfortranarray(a_2d.astype(np.complex128))
    return finufft.nufft2d2(x, y, a_2d, eps=eps)

def _nufft_adjoint(x, y, z, n_modes, eps=1e-8):
    """
    Type 1 NUFFT: nonuniform samples -> uniform Fourier coefficients
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    z = np.ascontiguousarray(z.astype(np.complex128))
    return finufft.nufft2d1(x, y, z, n_modes, eps=eps)

def _evaluate_on_grid_nufft(a_2d, n_fine):
    """
    Reconstruct on a uniform grid using zero-padded iFFT
    """
    Nx, Ny = a_2d.shape

    padded = np.zeros((n_fine, n_fine), dtype=np.complex128)

    cx = n_fine//2
    cy = n_fine//2
    hx = Nx//2
    hy = Ny//2

    x_start = cx - hx
    y_start = cy - hy

    padded[x_start:x_start + Nx, y_start:y_start + Ny] = a_2d
    padded = np.fft.ifftshift(padded)

    z_recon = np.fft.ifft2(padded)*(n_fine*n_fine)
    return z_recon.real


def _fit_fourier_modes_2d_nufft_weighted(
    x, y, z, weight,
    Kmax_x, Kmax_y,
    max_iter=80,
    tol=1e-8,
    eps=1e-8,
    verbose=True,
    odd=1,
):
    """
    Solve min sum_i weight_i * |(A a)_i - z_i|^2 using CGNR with NUFFT.

    Parameters
    ----------
    x, y, z : 1D arrays
        Sample coordinates and values.
    weight : 1D array
        Non-negative weight for each sample.
    """
    n_modes = (2*Kmax_x + odd, 2*Kmax_y + odd)

    x = np.ascontiguousarray(np.asarray(x), dtype=np.float64)
    y = np.ascontiguousarray(np.asarray(y), dtype=np.float64)
    z = np.ascontiguousarray(np.asarray(z).astype(np.complex128))
    weight = np.ascontiguousarray(np.asarray(weight), dtype=np.float64)

    if weight.shape!=z.shape:
        raise ValueError('weight must have the same shape as z')
    if np.any(weight < 0):
        raise ValueError('weight must be non-negative')

    sqrt_w = np.sqrt(weight)

    def forward_w(a_2d):
        # A_w a = sqrt(W) A a
        return sqrt_w*_nufft_forward(x, y, a_2d, eps=eps)

    def adjoint_w(v):
        # A_w^* v = A^* sqrt(W) v
        return _nufft_adjoint(x, y, sqrt_w*v, n_modes, eps=eps)

    a = np.zeros(n_modes, dtype=np.complex128, order='F')

    # weighted residual system
    r = sqrt_w*z.copy()
    s = adjoint_w(r)
    p = s.copy()

    gamma = np.real(np.vdot(s, s))
    gamma0 = gamma if gamma > 0 else 1.0

    rel_res = np.sqrt(gamma/gamma0)

    for iteration in range(max_iter):
        q = forward_w(p)
        delta = np.real(np.vdot(q, q))

        if delta < 1e-30:
            if verbose:
                print(f'  CGNR stopped at iter {iteration+1}: delta too small')
            break

        alpha = gamma/delta
        a += alpha*p
        r -= alpha*q

        s = adjoint_w(r)
        gamma_new = np.real(np.vdot(s, s))
        rel_res = np.sqrt(gamma_new/(gamma0 + 1e-30))

        if verbose:
            print(f'  iter {iteration+1:2d}: rel_res = {rel_res:.2e}')

        if rel_res < tol:
            if verbose:
                print(f'  CGNR converged at iter {iteration+1}, relative residual = {rel_res:.2e}')
            gamma = gamma_new
            break

        beta = gamma_new/(gamma + 1e-30)
        p = s + beta*p
        gamma = gamma_new
    else:
        if verbose:
            print(f'  CGNR reached max_iter={max_iter}, relative residual = {rel_res:.2e}')

    return a

def combine_image_nufft_from_xyz(
    x_all,
    y_all,
    z_all,
    base_size,
    weight_all=None,
    oversample=2,
    enlarge=1.0,
    Kmax=None,
    max_iter=80,
    tol=1e-8,
    eps=1e-8,
    verbose=True,
    odd=1,
):
    """
    NUFFT reconstruction from (x, y, z)

    Parameters
    ----------
    x_all, y_all, z_all : 1D arrays
        Sample coordinates and values.
    base_size : float
        Coordinate normalization scale.
    weight_all : 1D array or None
        Per-sample weights for weighted least squares.
        If None, use uniform weighting.
    """

    x_all = np.asarray(x_all).ravel()
    y_all = np.asarray(y_all).ravel()
    z_all = np.asarray(z_all).ravel()

    if weight_all is None:
        weight_all = np.ones_like(z_all, dtype=np.float64)
    else:
        weight_all = np.asarray(weight_all).ravel().astype(np.float64)

    if not (x_all.shape==y_all.shape==z_all.shape==weight_all.shape):
        raise ValueError('x_all, y_all, z_all, and weight_all must have the same shape')
    if np.any(weight_all < 0):
        raise ValueError('weight_all must be non-negative')

    # --------------------------------------------------
    # normalize -> [-pi, pi)
    # --------------------------------------------------
    x_all = x_all/base_size*(2*np.pi)*enlarge
    y_all = y_all/base_size*(2*np.pi)*enlarge

    mask = (
        (x_all >= -np.pi) & (x_all < np.pi) &
        (y_all >= -np.pi) & (y_all < np.pi)
    )

    x_all = x_all[mask]
    y_all = y_all[mask]
    z_all = z_all[mask]
    weight_all = weight_all[mask]

    M = len(x_all)

    # --------------------------------------------------
    # Fourier modes
    # --------------------------------------------------
    R_OUT = int((base_size/2)/enlarge*oversample)

    if Kmax is None:
        Kmax_x = R_OUT
        Kmax_y = R_OUT
    else:
        Kmax_x = int(Kmax)
        Kmax_y = int(Kmax)

    n_modes_x = 2*Kmax_x + odd
    n_modes_y = 2*Kmax_y + odd
    N_modes = n_modes_x*n_modes_y

    if verbose:
        print(f'samples:        {M}')
        print(f'Fourier modes:  ({n_modes_x})x({n_modes_y}) = {N_modes}')
        print(f'Fitting {N_modes} modes from {M} samples ...')
        print(f'weighted fit:   {weight_all is not None}')

    # --------------------------------------------------
    # NUFFT solve
    # --------------------------------------------------
    a_2d = _fit_fourier_modes_2d_nufft_weighted(
        x_all, y_all, z_all, weight_all,
        Kmax_x, Kmax_y,
        max_iter=max_iter,
        tol=tol,
        eps=eps,
        verbose=verbose,
        odd=odd,
    )

    # --------------------------------------------------
    # reconstruct
    # --------------------------------------------------
    n_fine = R_OUT*2 + odd
    z_recon = _evaluate_on_grid_nufft(a_2d, n_fine)

    z_recon = np.fft.fftshift(z_recon)[::-1, ::-1]

    if verbose:
        print(f'image size:     {z_recon.shape}')
        print(f'oversample:     {oversample}')

    return z_recon.T

def build_samples_from_cutouts(
    normalized_psf_tensor,
    centroids_d_tensor,
    wts_d_tensor=None,
):
    """
    Convert cutouts + centroids -> (x_all, y_all, z_all, weight_all)

    Parameters
    ----------
    wts_d_tensor : None or torch.Tensor
        If shape is (n_frames,), each frame gets one scalar weight.
        If shape is (n_frames, ny, nx), each pixel gets its own weight.
    """

    normalized_psf = normalized_psf_tensor.detach().cpu().numpy()
    centroids_d = centroids_d_tensor.detach().cpu().numpy()

    if wts_d_tensor is None:
        sample_weight = np.ones(normalized_psf.shape)
    else:
        sample_weight = wts_d_tensor.detach().cpu().numpy()

    if np.iscomplexobj(normalized_psf):
        normalized_psf = normalized_psf.real
    if np.iscomplexobj(centroids_d):
        centroids_d = centroids_d.real
    if sample_weight is not None and np.iscomplexobj(sample_weight):
        sample_weight = sample_weight.real

    n_frames, ny, nx = normalized_psf.shape

    cx = (nx - 1)/2
    cy = (ny - 1)/2

    yy0, xx0 = np.indices((ny, nx), dtype=np.float64)
    xx0 = xx0 - cx
    yy0 = yy0 - cy

    x_all = []
    y_all = []
    z_all = []
    weight_all = []

    for i in range(n_frames):
        dx, dy = centroids_d[i]

        xx = xx0 - dx
        yy = yy0 - dy
        zz = normalized_psf[i]

        x_all.append(xx.ravel())
        y_all.append(yy.ravel())
        z_all.append(zz.ravel())

        if sample_weight.ndim==1:
            ww = np.full_like(zz, sample_weight[i], dtype=np.float64)
        elif sample_weight.ndim==3:
            ww = np.asarray(sample_weight[i], dtype=np.float64)
        else:
            raise ValueError('sample_weight_tensor must have shape (n_frames,) or (n_frames, ny, nx)')

        weight_all.append(ww.ravel())

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)
    z_all = np.concatenate(z_all)
    weight_all = np.concatenate(weight_all)

    base_size = max(nx, ny)

    return x_all, y_all, z_all, weight_all, base_size