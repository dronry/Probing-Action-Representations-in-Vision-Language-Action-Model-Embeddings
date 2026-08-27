"""
Sanity Check: Orientation Target Distribution
================================================
Before concluding orientation is fundamentally hard to decode, rule out
a much simpler explanation: pathological target distributions.

Two specific failure modes to check:

1. ANGLE WRAPAROUND: if roll/pitch/yaw are stored as raw angles (radians
   or degrees) and the demonstrations cross the +-pi boundary, R2 computed
   on the raw value will look artificially terrible even if the underlying
   physical orientation is well represented — e.g. 179 degrees and -179
   degrees are almost the same orientation but numerically ~358 apart.

2. NEAR-CONSTANT / LOW-VARIANCE TARGETS: if the wrist barely rotates during
   most of LIBERO-Spatial's tasks (plausible — many pick-place tasks keep
   a fairly fixed gripper orientation), R2 can look bad simply because
   there is very little true variance to explain, and small absolute
   errors get amplified by R2's normalization by variance. This wouldn't
   mean the representation is bad — it would mean R2 itself is a somewhat
   misleading metric for these dims.

This needs ONLY the already-saved actions array. No GPU, no re-streaming.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FEATURES_PATH = "/kaggle/input/datasets/tahak9/boyfla/features.npz"
SAVE_DIR = Path("/kaggle/working/probe_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ACTION_DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ORIENTATION_DIMS = ["roll", "pitch", "yaw"]


def load_actions(features_path=FEATURES_PATH):
    data = np.load(features_path)
    return data["actions"]


def check_wraparound(actions, dim_idx, dim_name):
    """
    Look for values clustered near the extremes of a plausible angle range
    AND values on both sides (near +max and near -max simultaneously),
    which is the signature of wraparound discontinuity in the raw values.
    """
    vals = actions[:, dim_idx]
    vmin, vmax = vals.min(), vals.max()
    vrange = vmax - vmin

    # Heuristic: check what fraction of mass sits in the outer 10% of the
    # observed range on EACH side. If both tails are populated, that's
    # consistent with (but not proof of) wraparound.
    lower_thresh = vmin + 0.1 * vrange
    upper_thresh = vmax - 0.1 * vrange
    frac_lower = np.mean(vals <= lower_thresh)
    frac_upper = np.mean(vals >= upper_thresh)

    # Also check: is the range suspiciously close to a full 2*pi (~6.28) or
    # pi (~3.14), which would suggest angle units are in play at all?
    near_2pi = abs(vrange - 2 * np.pi) < 0.5
    near_pi  = abs(vrange - np.pi) < 0.5

    print(f"\n  {dim_name}:")
    print(f"    Range: [{vmin:.4f}, {vmax:.4f}]  (span: {vrange:.4f})")
    print(f"    Mass in bottom 10% of range: {frac_lower*100:.1f}%")
    print(f"    Mass in top 10% of range:    {frac_upper*100:.1f}%")
    print(f"    Range ~ 2*pi (6.28)? {near_2pi}   Range ~ pi (3.14)? {near_pi}")

    both_tails_populated = frac_lower > 0.03 and frac_upper > 0.03
    if both_tails_populated and (near_2pi or near_pi):
        print(f"    ⚠️  SUSPICIOUS: both tails populated AND range matches an angle "
              f"period. Wraparound is PLAUSIBLE for {dim_name}.")
        return True
    else:
        print(f"    No strong wraparound signature detected for {dim_name}.")
        return False


def check_variance(actions, dim_idx, dim_name, all_dims_std):
    """
    Compare this dimension's std to the others. If it's much smaller,
    R2 may look bad simply because there's little to explain — small
    absolute prediction errors get amplified by R2's variance normalization.
    """
    vals = actions[:, dim_idx]
    std = vals.std()
    mean = vals.mean()
    relative_std = std / (np.mean(all_dims_std) + 1e-8)

    print(f"\n  {dim_name}:")
    print(f"    Mean: {mean:.4f}   Std: {std:.4f}")
    print(f"    Relative std (vs avg of all dims): {relative_std:.3f}x")

    if relative_std < 0.3:
        print(f"    ⚠️  LOW VARIANCE: {dim_name}'s std is much smaller than typical. "
              f"R2 may be a misleading metric here — the target barely moves.")
        return True
    else:
        print(f"    Variance looks comparable to other dimensions.")
        return False


def plot_distributions(actions, dim_names):
    n_dims = actions.shape[1]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    fig.suptitle("Action Dimension Distributions (raw, un-normalized)",
                  fontsize=13, fontweight="bold")

    for i in range(n_dims):
        ax = axes[i]
        vals = actions[:, i]
        ax.hist(vals, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
        ax.set_title(f"{dim_names[i]}\nstd={vals.std():.3f}, "
                      f"range=[{vals.min():.2f}, {vals.max():.2f}]", fontsize=10)
        ax.axvline(vals.mean(), color="red", linestyle="--", alpha=0.7, label="mean")
        is_orientation = dim_names[i] in ORIENTATION_DIMS
        if is_orientation:
            for spine in ax.spines.values():
                spine.set_edgecolor("#e67e22")
                spine.set_linewidth(2)

    for j in range(n_dims, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save_path = SAVE_DIR / "action_distributions.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nDistribution plot saved -> {save_path}")
    print("(orange borders = orientation dimensions)")


def main():
    print("=" * 60)
    print("SANITY CHECK: Orientation Target Distributions")
    print("=" * 60)

    actions = load_actions()
    n_dims = actions.shape[1]
    dim_names = ACTION_DIM_NAMES[:n_dims]

    print(f"\nLoaded {len(actions)} action samples, {n_dims} dimensions\n")

    all_stds = actions.std(axis=0)

    print("=" * 60)
    print("CHECK 1: Angle Wraparound")
    print("=" * 60)
    wraparound_flags = {}
    for i, name in enumerate(dim_names):
        if name in ORIENTATION_DIMS:
            wraparound_flags[name] = check_wraparound(actions, i, name)

    print("\n" + "=" * 60)
    print("CHECK 2: Low Variance / Near-Constant Targets")
    print("=" * 60)
    variance_flags = {}
    for i, name in enumerate(dim_names):
        variance_flags[name] = check_variance(actions, i, name, all_stds)

    plot_distributions(actions, dim_names)

    # ── Verdict ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    any_wraparound = any(wraparound_flags.values())
    orientation_low_var = [name for name in ORIENTATION_DIMS
                             if name in variance_flags and variance_flags[name]]

    if any_wraparound:
        flagged = [k for k, v in wraparound_flags.items() if v]
        print(f"\n⚠️  Wraparound is plausible for: {flagged}")
        print("   RECOMMENDATION: recompute R2 using sin/cos encoding of these")
        print("   angles (predict [sin(theta), cos(theta)] instead of theta directly)")
        print("   before drawing further conclusions about representation quality.")
    else:
        print("\n✅ No strong wraparound signature detected in any orientation dim.")

    if orientation_low_var:
        print(f"\n⚠️  Low variance detected for: {orientation_low_var}")
        print("   This means R2's poor score may partly reflect 'little to explain'")
        print("   rather than 'representation doesn't encode it'. Consider reporting")
        print("   raw MSE / degrees-of-error alongside R2 for these dims.")
    else:
        print("\n✅ Orientation dimensions have comparable variance to position/gripper.")

    if not any_wraparound and not orientation_low_var:
        print("\n" + "-" * 60)
        print("Neither pathology detected. The orientation R2 ceiling found in")
        print("Experiments 1a-1c appears to be a GENUINE representational property,")
        print("not a data-artifact. Safe to proceed to vision-token probing.")
        print("-" * 60)

    print("=" * 60)

    return wraparound_flags, variance_flags


if __name__ == "__main__":
    main()