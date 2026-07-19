"""
SO(3) helpers for the GNSS/IMU ESKF — pure NumPy, matching pyquaternion's
(w, x, y, z) convention so we can drop-in replace pyquaternion on the hot
predict loop without changing downstream conventions.

Conventions:
  * Quaternion layout: q = [w, x, y, z]
  * Active rotation: R = quat_to_R(q) maps body vectors to the navigation frame.
      v_nav = R @ v_body
  * Attitude error: right-multiply (body-frame error),
      q_true = q_nom ⊗ zeta(δθ)
    where δθ ∈ R^3 is the body-frame rotation error (as in
    scripts/estimation/eskf_class.py:186-190).

All functions are scalar-safe (operate on single states, not batches); the hot
loop invokes them once per IMU step.
"""
import math
import numpy as np

__all__ = [
    'skew',
    'quat_to_R',
    'R_to_quat',
    'quat_mul',
    'quat_normalize',
    'expm_so3',
    'zeta',
    'Q_dtheta',
]


def skew(v):
    """3-vector → 3×3 skew-symmetric cross-product matrix.

    skew(v) @ a = v × a
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def quat_to_R(q):
    """Quaternion (w, x, y, z) → 3×3 rotation matrix (body → world).

    Matches pyquaternion.Quaternion.rotation_matrix for unit quaternions.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1.0 - 2.0*(y*y + z*z),  2.0*(x*y - w*z),        2.0*(x*z + w*y)      ],
        [2.0*(x*y + w*z),        1.0 - 2.0*(x*x + z*z),  2.0*(y*z - w*x)      ],
        [2.0*(x*z - w*y),        2.0*(y*z + w*x),        1.0 - 2.0*(x*x + y*y)],
    ])


def R_to_quat(R):
    """3×3 rotation matrix → quaternion (w, x, y, z).

    Shepperd's method — numerically stable across the whole rotation group.
    Returned quaternion has non-negative w (double-cover canonicalization).
    """
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 2.0 * math.sqrt(1.0 + tr)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0.0:
        q = -q
    return q


def quat_mul(q1, q2):
    """Hamilton product (w, x, y, z) ⊗ (w, x, y, z)."""
    q1 = np.asarray(q1, dtype=float).reshape(-1)
    q2 = np.asarray(q2, dtype=float).reshape(-1)
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_normalize(q):
    """Normalize a quaternion; canonicalize to w ≥ 0."""
    q = np.asarray(q, dtype=float).reshape(-1)
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / n
    if q[0] < 0.0:
        q = -q
    return q


def expm_so3(phi):
    """Rodrigues' formula: exp([phi]_×) for a 3-vector phi ∈ R^3 → 3×3 rotation."""
    phi = np.asarray(phi, dtype=float).reshape(-1)
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        # 2nd-order expansion to keep the Jacobian non-degenerate near zero
        K = skew(phi)
        return np.eye(3) + K + 0.5 * (K @ K)
    k = phi / theta
    K = skew(k)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def zeta(phi):
    """
    Rotation vector phi ∈ R^3 → unit quaternion (w, x, y, z) via exp on S^3.

        q = (cos(||phi||/2), sin(||phi||/2) * phi / ||phi||)

    Matches scripts/estimation/eskf_class.py:242-249 so the UWB ESKF's
    quaternion-injection convention is preserved bit-for-bit.
    """
    phi = np.asarray(phi, dtype=float).reshape(-1)
    n = np.linalg.norm(phi)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half = 0.5 * n
    s = math.sin(half) / n
    return np.array([math.cos(half), s*phi[0], s*phi[1], s*phi[2]])


def Q_dtheta(q):
    """
    Jacobian ∂q / ∂(δθ) at δθ = 0 with the convention
        q_true = q_nom ⊗ exp_quat(0.5 · [δθ]_×)
    Returns the 4×3 matrix that maps a body-frame error rotation δθ ∈ R^3
    into the 4-vector quaternion tangent.

    Lifted verbatim from scripts/estimation/eskf_class.py:221-226.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return 0.5 * np.array([
        [-x, -y, -z],
        [ w, -z,  y],
        [ z,  w, -x],
        [-y,  x,  w],
    ])
