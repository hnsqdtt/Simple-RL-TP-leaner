# -----------------------------
# FILE: env/esdf.py
# -----------------------------

import numpy as np

# Felzenszwalb & Huttenlocher 1D squared distance transform
# Ported to numpy; returns squared distances for 1D array of 0 (feature) and inf (others)

def edt_1d(f: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    v = np.zeros(n, dtype=int)
    z = np.zeros(n + 1, dtype=float)
    d = np.zeros(n, dtype=float)

    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf

    def sep(u: int, w: int) -> float:
        # Intersection of parabolas
        return ((f[w] + w*w) - (f[u] + u*u)) / (2*(w - u))

    for q in range(1, n):
        s = sep(v[k], q)
        while s <= z[k]:
            k -= 1
            s = sep(v[k], q)
        k += 1
        v[k] = q
        z[k] = s
        z[k+1] = np.inf

    k = 0
    for q in range(n):
        while z[k+1] < q:
            k += 1
        dq = q - v[k]
        d[q] = dq*dq + f[v[k]]
    return d


def esdf_from_occupancy(occ: np.ndarray, resolution: float) -> np.ndarray:
    """Compute ESDF (meters) from occ bool mask (True=obstacle/unknown).
    Based on two-pass 1D squared distance transform.
    """
    # f = 0 at obstacles; +inf elsewhere
    h, w = occ.shape
    inf = 1e15
    f = np.where(occ, 0.0, inf)

    # Transform along columns
    dt = np.empty_like(f)
    for x in range(w):
        dt[:, x] = edt_1d(f[:, x])
    # Then rows
    d2 = np.empty_like(dt)
    for y in range(h):
        d2[y, :] = edt_1d(dt[y, :])

    # sqrt and scale to meters
    dist_px = np.sqrt(d2)
    return (np.sqrt(d2).astype(np.float32)) * float(resolution)

