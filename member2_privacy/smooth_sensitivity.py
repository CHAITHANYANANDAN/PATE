import argparse
import json
import numpy as np
from common import paths
from common.contracts import DataManifest, PrivacyReport
from . import accountant as acc
from .aggregate import vote_counts
from ._io import art_dir

BETA = 0.09   # the authors' default


def smooth_sensitivity(counts, gamma, beta=BETA, lambdas=acc.LAMBDAS):
    """beta-smooth sensitivity of the TOTAL moments alpha_total(lambda); shape (L,).
    Entries are inf for lambda where the monotonicity assumption does not hold."""
    c = -np.sort(-np.asarray(counts, float), axis=1)          # classes sorted by count, descending
    lam = np.asarray(lambdas, float)
    top, sec = c[:, 0], c[:, 1]
    ss = np.zeros((len(c), len(lam)))
    for k in range(int(c.max()) + 1):
        valid = top >= sec + k                                # authors: sensitivity 0 beyond this
        if not valid.any():
            break
        ck, ck1 = c.copy(), c.copy()
        ck[:, 0], ck[:, 1] = top - k, sec + k
        ck1[:, 0], ck1[:, 1] = top - k - 1, sec + k + 1
        ls = acc.alpha_dependent(ck1, gamma, lambdas) - acc.alpha_dependent(ck, gamma, lambdas)
        ls = np.where(valid[:, None], np.maximum(ls, 0.0), 0.0)
        ss = np.maximum(ss, np.exp(-beta * k) * ls)
    total = ss.sum(axis=0)
    total[~(0.5 * gamma * lam <= 1)] = np.inf
    return total


def release_epsilon(counts, scale, delta, beta=BETA, rng=None, lambdas=acc.LAMBDAS):
    """Noisy release of the data-dependent epsilon.
    Returns a dict with the un-noised epsilon, the noise scale, one noisy draw, and the privacy level
    eps_ss of the release itself (Cor. 2.4: eps_ss = 2 * beta * ln(1/delta))."""
    gamma = 1.0 / scale
    lam = np.asarray(lambdas, float)
    alpha = acc.alpha_dependent(counts, gamma, lambdas).sum(axis=0)
    eps_l = (alpha + np.log(1.0 / delta)) / lam
    ss = smooth_sensitivity(counts, gamma, beta, lambdas)
    usable = np.isfinite(ss)
    if not usable.any():
        raise ValueError("no lambda satisfies 0.5 * gamma * lambda <= 1; use a larger noise scale")
    i = int(np.argmin(np.where(usable, eps_l, np.inf)))
    eps_ss = 2.0 * beta * np.log(1.0 / delta)
    noise_scale = (2.0 / eps_ss) * ss / lam                    # per-lambda scale for epsilon_lambda
    noisy = float(eps_l[i] + np.random.default_rng(rng).laplace(0.0, noise_scale[i]))
    return dict(epsilon=float(eps_l[i]), lambda_used=int(lam[i]), noise_scale=float(noise_scale[i]),
                epsilon_noisy=noisy, eps_ss=float(eps_ss), beta=float(beta), delta=float(delta))


def release_for_run(run_id, beta=BETA, seed=0):
    """Read an artifact folder, release epsilon for the queries actually answered, write epsilon_release.json."""
    d = art_dir(run_id)
    m = DataManifest.model_validate_json((d / paths.MANIFEST).read_text())
    rep = PrivacyReport.model_validate_json((d / paths.PRIVACY_REPORT).read_text())
    votes = np.load(d / paths.TEACHER_VOTES)
    idx = np.load(d / paths.QUERY_IDX)
    out = release_epsilon(vote_counts(votes[:, idx], m.n_classes), rep.noise_scale, rep.delta, beta, seed)
    out["run_id"] = run_id
    (d / "epsilon_release.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="t250_b20_q500_s42")
    ap.add_argument("--beta", type=float, default=BETA)
    a = ap.parse_args()
    r = release_for_run(a.run_id, a.beta)
    print(f"{r['run_id']}: epsilon = {r['epsilon']:.3f} (lambda {r['lambda_used']}), "
          f"noise scale = {r['noise_scale']:.4f}, one noisy release = {r['epsilon_noisy']:.3f}, "
          f"release itself is ({r['eps_ss']:.2f}, {r['delta']:g})-DP")
