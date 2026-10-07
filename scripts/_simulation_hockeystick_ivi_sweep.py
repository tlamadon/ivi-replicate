r"""IVI runner on the simulation-hockeystick (paper old calibration) DGP
with per-epoch inner-VI tracking. Sibling of
`_simulation_psid_hockeystick_ivi_sweep.py` but on the loose-/medium-
innovation paper DGP (not the kink-fork HYBRID truth).

The simulation-hockeystick truth is parameterised by env overrides so
we can probe how the VI lock on log_beta moves with sigma_0 and
sigma_2 (the lr/init sweep series identified sigma_0 as the dominant
lock control; this script is the IVI follow-up at a fixed nearby
calibration).

theta_VI_obs is computed inline at the start of main() via a 20000-
epoch joint-normal h=64 VI fit on y_obs at lr=1e-2 + step-to-step
CRN, since the inlined-hardcode pattern used in the PSID-hockeystick
script does not apply to a DGP whose truth is user-parameterisable.

Env vars (with defaults):
  METHOD               picard_cold | picard_warm | anderson | vi_only |
                       mle_direct                            (picard_cold)
  ENCODER              jn | mean_field | struct_markov | tridiag_jn |
                       tjn                                   (jn)
                       Required when METHOD=vi_only; ignored when
                       METHOD=mle_direct.
  EMISSION             sinh | degenerate                     (sinh)
                       Required when METHOD=mle_direct (must be
                       degenerate); ignored otherwise.
  PICARD_ALPHA         damping (also Anderson beta)          (0.6)
                       Ignored when METHOD in {vi_only, mle_direct}.
  N_EPOCHS_INNER       inner-VI epoch budget                 (16000)
                       Ignored when METHOD in {vi_only, mle_direct}.
  N_ITERS_OUTER        outer iter count                      (10)
                       Ignored when METHOD in {vi_only, mle_direct}.
  N_EPOCHS_MLE         METHOD=mle_direct only: epochs of AdamW
                       on the closed-form prior log-likelihood
                       at z = y_obs                          (20000)
  M_MEM                Anderson memory                       (4)
  LOG_EVERY            snapshot frequency                    (200)
  LR                   inner-VI learning rate                (1e-2)
  FIX_NOISE            pre-draw eps once + reuse             (1)
                       Ignored when METHOD=mle_direct.
  NOISE_SEED           eps draw seed                         (12345)
  N_EPOCHS_VI_OBS      epochs for the theta_VI_obs pre-fit   (20000)
                       Ignored when METHOD=mle_direct.
  TJN_WARM_DECODER_FREEZE_EPOCHS
                       TJN cell only: epochs (from start of
                       TJN fit) during which decoder.log_sigma
                       and decoder.log_beta are pinned        (1000)
  N                    panel size                            (30000)
  T                    panel length. The canonical spec pins
                       T=6; the T=40 IVI-JN variant sets T=40.
                       When T!=6 the cell filename gains a
                       `_T<T>` suffix so it never collides with
                       the T=6 bundle.                            (6)
  OUT_DIR              output directory for the cell JSON. The
                       T=40 variant points this at
                       output/bundles/cells/_simulation_hockeystick_t40_ivi_sweep
                       so its bundle sits apart from the T=6
                       bundle and the T=6 exporter globs never
                       pick it up.
                       (output/bundles/cells/_simulation_hockeystick_ivi_sweep)

  Truth overrides (default = simulation-hockeystick.md paper spec):
  MU1                  (default 0.9)
  SIGMA0               (default -1.35)
  SIGMA2               (default 0.35)
  LOGBETA              (default 0.755)

Output:
  output/bundles/cells/_simulation_hockeystick_ivi_sweep/
    IVI:        <method>_alpha<a>[_m<m>]_ep<n>_lr<tag>[_crn][_sig0-<x>][_sig2-<x>].json
    vi_only:    vi_only_<encoder>_h64_ep<vi_obs_ep>_lr<tag>[_crn][_sig0-<x>][_sig2-<x>].json
    mle_direct: mle_direct_no_me_ep<mle_ep>_lr<tag>[_sig0-<x>][_sig2-<x>].json

Phase 3 (post-fit diagnostics) appends a `diagnostics` block to the JSON:
  elbo_at_vi, iwae_at_vi      : at theta_VI_obs (phase-1 encoder reused)
  elbo_at_ivi, iwae_at_ivi    : at theta_K (fresh joint-normal h=64 encoder
                                 frozen prior+decoder, 8000 epochs, lr=1e-3, CRN)
                                 *IVI methods only; null for METHOD=vi_only.*
  elbo_at_truth, iwae_at_truth: at theta_truth (same encoder recipe as IVI)
  smc_log_p_at_truth          : per-individual bootstrap PF at K=1000
"""
import os
import sys
import time
import json

import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softplus_sinh_posterior_comparison as M
from smc_truth_profile import _Bundle
from mlye.eval import BootstrapParticleFilter


_orig_build_encoder = M.build_encoder
def _wide_build_encoder(encoder_type, T, regularize=1e-3, hidden_dim=64):
    if encoder_type == 'tridiag_joint_normal':
        from mlye.models.encoders.base import JointNormalConfig
        return JointNormalConfig(dim=T, type='tridiag_joint_normal',
                                  regularize=regularize,
                                  hidden_dim=hidden_dim).build()
    return _orig_build_encoder(encoder_type, T, regularize=regularize,
                                hidden_dim=hidden_dim)
M.build_encoder = _wide_build_encoder


# Map spec's ENCODER label to mlye encoder_type strings.
ENCODER_LABEL_TO_TYPE = {
    'jn':            'joint_normal',
    'mean_field':    'normal_diagonal',
    'struct_markov': 'conditional_markov',
    'tridiag_jn':    'tridiag_joint_normal',
    'tjn':           'transformed_joint_normal',
}

# Map ENCODER label to the spec-declared filename stem (per the VI-only
# cells table in simulation-hockeystick.md). Mean-field is rendered
# without the underscore in filenames to match the spec.
ENCODER_LABEL_TO_FILENAME_TAG = {
    'jn':            'jn',
    'mean_field':    'meanfield',
    'struct_markov': 'struct_markov',
    'tridiag_jn':    'tridiag_jn',
    'tjn':           'tjn',
}


def build_full_model_with_encoder(encoder_label, T, device):
    """Build a FullModel with the chosen encoder family at hidden=64.

    For 'jn' / 'mean_field' / 'struct_markov' this is M.build_full_model
    composed with the monkey-patched M.build_encoder (which forces
    hidden_dim=64). For 'tjn' we instantiate TJN directly because
    FlowConfig does not currently expose hidden_dim. In every case
    `prior.z1_skew` is pinned at 0 (`requires_grad=False`) on the
    returned model per the 2026-06-15 spec.
    """
    encoder_type = ENCODER_LABEL_TO_TYPE[encoder_label]
    if encoder_label == 'tjn':
        from mlye.models.encoders.normal import TransformedJointNormalPosterior
        from mlye.models.full_model import FullModel
        encoder = TransformedJointNormalPosterior(
            dim=T, hidden=64, regularize=1e-3).to(device)
        prior, decoder = M.build_prior_decoder(T, device)
        model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)
    else:
        model = M.build_full_model(encoder_type, T, device)
    _pin_z1_skew(model)
    return model


# Truth: simulation-hockeystick.md paper old calibration; mu1, sigma0,
# sigma2, log_beta overridable via env vars.
_MU1     = float(os.environ.get('MU1',     0.9000))
_SIGMA0  = float(os.environ.get('SIGMA0', -1.3500))
_SIGMA2  = float(os.environ.get('SIGMA2',  0.3500))
_LOGBETA = float(os.environ.get('LOGBETA', 0.7550))
HOCKEY_TRUTH = dict(M.TRUTH)
HOCKEY_TRUTH['alpha0']        = -0.25
HOCKEY_TRUTH['log_alpha1']    = -2.3026     # alpha_1 = 0.1
HOCKEY_TRUTH['mu0']           =  0.0000
HOCKEY_TRUTH['mu1']           = _MU1
HOCKEY_TRUTH['mu2']           =  0.0000
HOCKEY_TRUTH['sigma0']        = _SIGMA0
HOCKEY_TRUTH['sigma1']        =  0.0000
HOCKEY_TRUTH['sigma2']        = _SIGMA2
HOCKEY_TRUTH['z1_log_std']    = -0.9040     # gamma_1 = 0.34
HOCKEY_TRUTH['z1_skew']       =  0.0000
HOCKEY_TRUTH['z1_log_tail']   = +0.1165     # tailweight = 1.124
HOCKEY_TRUTH['log_sigma_eps'] = -3.4112     # psi_1 = 0.033
HOCKEY_TRUTH['log_beta']      = _LOGBETA
M.TRUTH = HOCKEY_TRUTH


PARAM_NAMES = M.PARAM_NAMES


def theta_to_vec(theta):
    return np.array([theta[k] for k in PARAM_NAMES], dtype=np.float64)


def vec_to_theta(vec):
    return {k: float(vec[i]) for i, k in enumerate(PARAM_NAMES)}


def L2(theta, truth):
    return float(np.linalg.norm(theta_to_vec(theta) - theta_to_vec(truth)))


def _pin_z1_skew(model):
    """Pin `prior.z1_skew = 0` with `requires_grad=False`.

    Per simulation-hockeystick.md (2026-06-15): the DGP truth has
    z_1 skewness 0 (symmetric sinh-arcsinh), and the inference model
    pins z1_skew at 0 -- reducing free theta from 13 to 12.
    """
    with torch.no_grad():
        model.prior.z1_skew.zero_()
    model.prior.z1_skew.requires_grad_(False)


def _set_prior_decoder_from_dict(model, theta):
    with torch.no_grad():
        model.prior.b.fill_(theta['alpha0'])
        model.prior.log_gamma.fill_(theta['log_alpha1'])
        model.prior.net_mu.coeffs.copy_(torch.tensor(
            [theta['mu0'], theta['mu1'], theta['mu2']],
            dtype=model.prior.net_mu.coeffs.dtype,
            device=model.prior.net_mu.coeffs.device))
        model.prior.net_sigma.coeffs.copy_(torch.tensor(
            [theta['sigma0'], theta['sigma1'], theta['sigma2']],
            dtype=model.prior.net_sigma.coeffs.dtype,
            device=model.prior.net_sigma.coeffs.device))
        model.prior.z1_log_std.fill_(theta['z1_log_std'])
        # z1_skew is pinned at 0 in the inference model per spec
        # (2026-06-15); ignore any non-zero value in the input dict.
        model.prior.z1_skew.zero_()
        model.prior.z1_log_tail.fill_(theta['z1_log_tail'])
        model.decoder.log_sigma.fill_(theta['log_sigma_eps'])
        model.decoder.log_beta.fill_(theta['log_beta'])
    # Re-pin in case requires_grad was reset by a caller.
    model.prior.z1_skew.requires_grad_(False)


def fit_encoder_only_at_theta(theta, y, T, vi_seed, n_epochs, lr, device,
                                  n_retries=10, encoder_label='jn'):
    r"""Fresh encoder of the chosen family with prior+decoder frozen at `theta`.

    Used by Phase 3 to evaluate ELBO/IWAE at a given (theta_K or theta_truth)
    that the phase-1 / phase-2 encoders were not trained against. Step-to-step
    CRN is on (pre-drawn eps reused every epoch); the encoder reaches its
    fixed point in ~8000 epochs at lr=1e-3.

    For `encoder_label='tjn'`, the encoder is the same TJN $h=64$ class as
    Phase 1 / Phase 2. The TJN warm-start convention degenerates here because
    prior+decoder are frozen at `theta` throughout, so the "internal JN
    warm-start" of `estimators.md` \S2.1 would only train (and then discard)
    JN encoder weights without changing prior+decoder; we therefore train
    the TJN encoder directly from cold, matching the spec's noted
    simplification.
    """
    last_err = None
    for s in M._retry_seeds(vi_seed, n_retries):
        try:
            torch.manual_seed(s); np.random.seed(s)
            model = build_full_model_with_encoder(encoder_label, T, device)
            _set_prior_decoder_from_dict(model, theta)
            for p in model.prior.parameters():
                p.requires_grad_(False)
            for p in model.decoder.parameters():
                p.requires_grad_(False)
            seed_dim = model.encoder.get_seed_dim()
            torch.manual_seed(s + 1)
            eps_fixed_local = torch.randn((1, y.shape[0], seed_dim),
                                            dtype=torch.float32, device=device)
            opt = torch.optim.AdamW(
                [p for p in model.encoder.parameters() if p.requires_grad], lr=lr)
            t0 = time.time()
            nan_streak = 0
            for epoch in range(n_epochs):
                opt.zero_grad()
                loss = -model.elbo(y, ndraws=1, eps_all=eps_fixed_local)
                if not torch.isfinite(loss):
                    nan_streak += 1
                    if epoch < 200 and nan_streak >= 10:
                        raise M.NaNFailure(f"NaN at epoch {epoch}")
                    continue
                nan_streak = 0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 5.0)
                opt.step()
                if (epoch + 1) % 2000 == 0 or epoch == 0:
                    print(f"  [enc@theta/s{s}] ep {epoch+1:5d}: "
                          f"ELBO={-loss.item():+.3f}  "
                          f"({time.time()-t0:.1f}s)", flush=True)
            return model
        except M.NaNFailure as e:
            print(f"  [enc@theta/s{s}] NaN bail: {e}; retrying", flush=True)
            last_err = e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"encoder-only fit failed after {n_retries} retries "
                        f"(last err: {last_err})")


def fit_vi_inner(model, y, n_epochs, lr, log_every,
                  label='vi', warm_start=False, vi_seed=None,
                  n_retries=10, eps_fixed=None, encoder_label='jn',
                  theta_avg_epochs=0):
    """Cold- or warm-start a VI fit with the chosen encoder family.

    `encoder_label` matches the spec's ENCODER env-var labels (jn /
    mean_field / struct_markov / tjn) and is the cell's posterior
    family. Warm-start (picard_warm) reuses the passed-in `model` and
    ignores `encoder_label`.

    `theta_avg_epochs > 0` returns the tail-averaged theta: the mean
    of the 13-param vector over the last `theta_avg_epochs` stepped
    epochs, instead of the final iterate. Constant-lr AdamW ends in a
    limit cycle of amplitude O(lr) around its optimum even on the
    CRN-deterministic objective; averaging over the cycle removes the
    endpoint jitter that otherwise sets the floor of the IVI binding
    residual. The trace and the returned model are unchanged (the
    model's live parameters stay at the final iterate).
    """
    last_err = None
    seeds_to_try = M._retry_seeds(vi_seed, n_retries) if not warm_start else [vi_seed]
    for s in seeds_to_try:
        try:
            if not warm_start:
                torch.manual_seed(s); np.random.seed(s)
                model = build_full_model_with_encoder(
                    encoder_label, y.shape[1], y.device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
            trace = []
            t0 = time.time()
            nan_streak = 0
            prev_theta_vec = None
            last_grad_norm = float('nan')
            theta_avg_sum = None
            theta_avg_n = 0
            for epoch in range(n_epochs):
                opt.zero_grad()
                loss = -model.elbo(y, ndraws=1, eps_all=eps_fixed)
                if not torch.isfinite(loss):
                    nan_streak += 1
                    if epoch < 200 and nan_streak >= 10:
                        raise M.NaNFailure(f"NaN at epoch {epoch}")
                    continue
                nan_streak = 0
                loss.backward()
                grad_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_sq += float(p.grad.detach().pow(2).sum().item())
                last_grad_norm = float(np.sqrt(grad_sq))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                if theta_avg_epochs > 0 and epoch >= n_epochs - theta_avg_epochs:
                    vec = theta_to_vec(M.extract_params(model))
                    theta_avg_sum = (vec if theta_avg_sum is None
                                     else theta_avg_sum + vec)
                    theta_avg_n += 1
                if epoch == 0 or (epoch + 1) % log_every == 0 \
                        or epoch == n_epochs - 1:
                    theta = M.extract_params(model)
                    theta_vec = theta_to_vec(theta)
                    delta = (float(np.linalg.norm(theta_vec - prev_theta_vec))
                              if prev_theta_vec is not None else None)
                    prev_theta_vec = theta_vec
                    trace.append({
                        'epoch':            epoch + 1,
                        'elbo':             float(-loss.item()),
                        'theta':            theta,
                        'grad_norm':        last_grad_norm,
                        'param_delta_norm': delta,
                    })
                    if (epoch + 1) % (log_every * 10) == 0 \
                            or epoch == 0 or epoch == n_epochs - 1:
                        print(f"    [{label}/s{s}] ep {epoch+1:5d}  "
                              f"ELBO={trace[-1]['elbo']:+.3f}  "
                              f"log_b={theta['log_beta']:+.3f}  "
                              f"mu1={theta['mu1']:+.3f}  "
                              f"||g||={last_grad_norm:.3f}  "
                              f"||dt||={delta if delta is None else f'{delta:.4f}'}  "
                              f"({time.time()-t0:.1f}s)", flush=True)
            if theta_avg_n > 0:
                theta_ret = vec_to_theta(theta_avg_sum / theta_avg_n)
                print(f"    [{label}/s{s}] tail-averaged theta over "
                      f"{theta_avg_n} epochs", flush=True)
            else:
                theta_ret = M.extract_params(model)
            return theta_ret, trace, s, model
        except M.NaNFailure as e:
            print(f"    [{label}/s{s}] NaN bail: {e}; retrying", flush=True)
            last_err = e
            if not warm_start and torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"inner VI failed after {len(seeds_to_try)} seed retries "
                        f"(last err: {last_err})")


def fit_tjn_warm(y, n_epochs, freeze_epochs, lr, log_every, vi_seed,
                   eps_fixed, jn_theta):
    """TJN two-phase warm-start fit.

    Builds a TJN $h=64$ FullModel, copies prior+decoder from `jn_theta`
    (a JN-converged theta_VI dict), trains for `n_epochs` total. The
    first `freeze_epochs` epochs pin `decoder.log_sigma` and
    `decoder.log_beta`; afterwards both unfreeze. Returns the
    converged theta dict, the contiguous per-snapshot trace, the
    used seed offset, and the model.
    """
    T = y.shape[1]
    last_err = None
    for s in M._retry_seeds(vi_seed, 10):
        try:
            torch.manual_seed(s); np.random.seed(s)
            model = build_full_model_with_encoder('tjn', T, y.device)
            _set_prior_decoder_from_dict(model, jn_theta)
            # Freeze decoder for the warm phase.
            for p in model.decoder.parameters():
                p.requires_grad_(False)
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=lr)
            trace = []
            t0 = time.time()
            nan_streak = 0
            prev_theta_vec = None
            last_grad_norm = float('nan')
            thawed = False
            for epoch in range(n_epochs):
                if not thawed and epoch >= freeze_epochs:
                    # Thaw decoder and rebuild optimizer to include its params.
                    for p in model.decoder.parameters():
                        p.requires_grad_(True)
                    opt = torch.optim.AdamW(
                        [p for p in model.parameters() if p.requires_grad], lr=lr)
                    thawed = True
                    print(f"    [tjn/s{s}] ep {epoch+1}: decoder thawed",
                          flush=True)
                opt.zero_grad()
                loss = -model.elbo(y, ndraws=1, eps_all=eps_fixed)
                if not torch.isfinite(loss):
                    nan_streak += 1
                    if epoch < 200 and nan_streak >= 10:
                        raise M.NaNFailure(f"NaN at epoch {epoch}")
                    continue
                nan_streak = 0
                loss.backward()
                grad_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_sq += float(p.grad.detach().pow(2).sum().item())
                last_grad_norm = float(np.sqrt(grad_sq))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                if (epoch == 0 or (epoch + 1) % log_every == 0
                        or epoch == n_epochs - 1):
                    theta = M.extract_params(model)
                    theta_vec = theta_to_vec(theta)
                    delta = (float(np.linalg.norm(theta_vec - prev_theta_vec))
                              if prev_theta_vec is not None else None)
                    prev_theta_vec = theta_vec
                    trace.append({
                        'epoch':            epoch + 1,
                        'elbo':             float(-loss.item()),
                        'theta':            theta,
                        'grad_norm':        last_grad_norm,
                        'param_delta_norm': delta,
                        'decoder_frozen':   not thawed,
                    })
                    if ((epoch + 1) % (log_every * 10) == 0
                            or epoch == 0 or epoch == n_epochs - 1):
                        print(f"    [tjn/s{s}] ep {epoch+1:5d}  "
                              f"ELBO={trace[-1]['elbo']:+.3f}  "
                              f"log_b={theta['log_beta']:+.3f}  "
                              f"frozen={not thawed}  "
                              f"({time.time()-t0:.1f}s)", flush=True)
            return M.extract_params(model), trace, s, model
        except M.NaNFailure as e:
            print(f"    [tjn/s{s}] NaN bail: {e}; retrying", flush=True)
            last_err = e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"TJN warm-start failed after seed retries "
                        f"(last err: {last_err})")


def anderson_step(x_k, g_k, history, m, beta, reg=1e-8):
    hist = history[-m:] if len(history) > 0 else []
    m_k = len(hist)
    if m_k == 0:
        return x_k + beta * g_k, None
    xs = [h['x'] for h in hist] + [x_k]
    gs = [h['g'] for h in hist] + [g_k]
    Delta_X = np.column_stack([xs[i+1] - xs[i] for i in range(m_k)])
    Delta_G = np.column_stack([gs[i+1] - gs[i] for i in range(m_k)])
    AtA = Delta_G.T @ Delta_G + reg * np.eye(m_k)
    Atb = Delta_G.T @ g_k
    gamma = np.linalg.solve(AtA, Atb)
    x_next = x_k + beta * g_k - (Delta_X + beta * Delta_G) @ gamma
    return x_next, gamma


def main():
    method            = os.environ.get('METHOD', 'picard_cold')
    encoder_label     = os.environ.get('ENCODER', 'jn')
    emission          = os.environ.get('EMISSION', 'sinh')
    alpha             = float(os.environ.get('PICARD_ALPHA', 0.6))
    n_epochs          = int(os.environ.get('N_EPOCHS_INNER', 16000))
    n_iters           = int(os.environ.get('N_ITERS_OUTER', 10))
    m_mem             = int(os.environ.get('M_MEM', 4))
    log_every         = int(os.environ.get('LOG_EVERY', 200))
    lr                = float(os.environ.get('LR', 1e-2))
    fix_noise         = int(os.environ.get('FIX_NOISE', 1)) != 0
    noise_seed        = (int(os.environ.get('NOISE_SEED', 12345))
                          if fix_noise else None)
    n_epochs_vi_obs   = int(os.environ.get('N_EPOCHS_VI_OBS', 20000))
    n_epochs_encoder  = int(os.environ.get('N_EPOCHS_ENCODER', 8000))
    n_epochs_mle      = int(os.environ.get('N_EPOCHS_MLE', 20000))
    lr_encoder        = float(os.environ.get('LR_ENCODER', 1e-3))
    L_iwae            = int(os.environ.get('L_IWAE', 100))
    K_smc             = int(os.environ.get('K_SMC', 1000))
    N_override        = int(os.environ.get('N', 30000))
    T_override        = int(os.environ.get('T', 6))
    out_dir           = os.environ.get(
        'OUT_DIR', 'output/bundles/cells/_simulation_hockeystick_ivi_sweep')
    tjn_freeze_epochs = int(os.environ.get(
        'TJN_WARM_DECODER_FREEZE_EPOCHS', 1000))
    # Tail-average the inner theta over the last A stepped epochs
    # (0 = off, return the final iterate as before). Applied to both
    # the Phase 1 theta_VI_obs fit and every Phase 2 inner fit so the
    # two sides of the binding equation use the same estimator.
    theta_avg_epochs  = int(os.environ.get('THETA_AVG_EPOCHS', 0))

    assert method in ('picard_cold', 'picard_warm', 'anderson',
                       'vi_only', 'mle_direct'), \
        f"unknown METHOD: {method}"
    assert emission in ('sinh', 'degenerate'), \
        f"unknown EMISSION: {emission}"
    # Per spec (2026-06-22): mle_direct requires EMISSION=degenerate;
    # every other METHOD ignores EMISSION (defaults to sinh).
    if method == 'mle_direct':
        assert emission == 'degenerate', \
            "METHOD=mle_direct requires EMISSION=degenerate"
    else:
        # Force EMISSION back to sinh for IVI/VI cells; the env-var is
        # only meaningful for mle_direct.
        emission = 'sinh'
    is_vi_only = method == 'vi_only'
    is_mle_direct = method == 'mle_direct'
    if not is_mle_direct:
        assert encoder_label in ENCODER_LABEL_TO_TYPE, \
            f"unknown ENCODER: {encoder_label}"
    # Per spec (2026-06-15): Picard cells are JN-only; Anderson and
    # vi_only honour the env-passed ENCODER. The IVI posterior cells
    # use METHOD=anderson with ENCODER in {mean_field, struct_markov,
    # tjn}. METHOD=anderson with ENCODER=tridiag_jn is out of scope
    # for this spec; coerce to vi_only to make the cell self-consistent
    # rather than fail mid-run.
    if method in ('picard_cold', 'picard_warm'):
        encoder_label = 'jn'
    if method == 'anderson' and encoder_label == 'tridiag_jn':
        print(f"[warn] METHOD=anderson with ENCODER=tridiag_jn is out of "
              f"scope for this spec; coercing to METHOD=vi_only.", flush=True)
        method = 'vi_only'
        is_vi_only = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}", flush=True)
    print(f"DGP: simulation-hockeystick (paper old calib).", flush=True)
    print(f"  sigma_0 = {_SIGMA0:+.4f}  sigma_2 = {_SIGMA2:+.4f}  "
          f"mu_1 = {_MU1:+.4f}  log_beta = {_LOGBETA:+.4f}", flush=True)
    if is_mle_direct:
        print(f"method={method} emission={emission} "
              f"n_epochs_mle={n_epochs_mle} log_every={log_every} "
              f"lr={lr}", flush=True)
    else:
        print(f"method={method} encoder={encoder_label} alpha={alpha} "
              f"n_iters={n_iters} n_epochs_inner={n_epochs} m_mem={m_mem} "
              f"log_every={log_every} lr={lr} fix_noise={fix_noise} "
              f"noise_seed={noise_seed}", flush=True)
        print(f"n_epochs_vi_obs={n_epochs_vi_obs}", flush=True)
        if encoder_label == 'tjn':
            print(f"tjn_warm_decoder_freeze_epochs={tjn_freeze_epochs}",
                  flush=True)

    N, T = N_override, T_override
    OBS_SEED = 11
    vi_seed = OBS_SEED * 1000 + 7

    torch.manual_seed(OBS_SEED)
    z_noise = torch.randn(N, T, device=device)
    y_noise = torch.randn(N, T, device=device)

    # Simulate y_obs at truth.
    print(f"\n=== Phase 0: simulate y_obs (N={N}, T={T}) ===", flush=True)
    y_obs = M.simulate_from_truth(N, T, HOCKEY_TRUTH, OBS_SEED, device)
    print(f"  y_obs: std={y_obs.std().item():+.4f}  "
          f"std(diff)={torch.diff(y_obs, dim=1).std().item():+.4f}",
          flush=True)

    truth = HOCKEY_TRUTH
    truth_vec = theta_to_vec(truth)

    # Pre-draw step-to-step CRN epsilon. All four encoder families ship with
    # `get_seed_dim() == T`, so the eps tensor shape is shared across cells.
    # mle_direct cells have no encoder, so skip the eps tensor entirely.
    seed_dim = T
    if fix_noise and not is_mle_direct:
        torch.manual_seed(noise_seed)
        eps_fixed = torch.randn((1, N, seed_dim),
                                  dtype=torch.float32, device=device)
        print(f"\n[crn] eps_fixed pre-drawn at noise_seed={noise_seed}  "
              f"shape={tuple(eps_fixed.shape)}", flush=True)
    else:
        eps_fixed = None

    if is_mle_direct:
        # METHOD=mle_direct + EMISSION=degenerate path.
        # No encoder; the inference model is the prior alone, fit by direct
        # MLE on the closed-form prior log-likelihood at z = y_obs (per
        # models.md S3.5). No Phase 1 / Phase 2 -- a single MLE fit on the
        # 10 free prior + z_1 coordinates (psi_1, psi_2 do not exist in this
        # model class). prior.z1_skew is pinned at 0 (`requires_grad=False`)
        # to match the IVI cells' 12-coordinate convention; the
        # misspecification cell's free count drops further to 10.
        from mlye.models.priors import MarkovNormalConditionalPolyPrior
        print(f"\n=== Phase 0b: MLE direct (no measurement error) ===",
              flush=True)
        torch.manual_seed(vi_seed); np.random.seed(vi_seed)
        prior = MarkovNormalConditionalPolyPrior(
            nt=T, poly_degree=2, law_model='softplus',
            z1_distr='sinh', extra_heterogeneity=False).to(device)
        with torch.no_grad():
            prior.z1_skew.zero_()
        prior.z1_skew.requires_grad_(False)
        opt = torch.optim.AdamW(
            [p for p in prior.parameters() if p.requires_grad], lr=lr)
        mle_trace = []
        t0 = time.time()
        prev_theta_vec = None
        nan_streak = 0
        for epoch in range(n_epochs_mle):
            opt.zero_grad()
            loss = -prior.log_prob(y_obs).mean()
            if not torch.isfinite(loss):
                nan_streak += 1
                if epoch < 200 and nan_streak >= 10:
                    raise RuntimeError(
                        f"mle_direct NaN streak at epoch {epoch}")
                continue
            nan_streak = 0
            loss.backward()
            grad_sq = 0.0
            for p in prior.parameters():
                if p.grad is not None:
                    grad_sq += float(p.grad.detach().pow(2).sum().item())
            last_grad_norm = float(np.sqrt(grad_sq))
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 5.0)
            opt.step()
            if epoch == 0 or (epoch + 1) % log_every == 0 \
                    or epoch == n_epochs_mle - 1:
                # Read out the prior + z_1 params; decoder params are NaN.
                t = {
                    'alpha0':        float(prior.b.item()),
                    'log_alpha1':    float(prior.log_gamma.item()),
                    'mu0':           float(prior.net_mu.coeffs[0].item()),
                    'mu1':           float(prior.net_mu.coeffs[1].item()),
                    'mu2':           float(prior.net_mu.coeffs[2].item()),
                    'sigma0':        float(prior.net_sigma.coeffs[0].item()),
                    'sigma1':        float(prior.net_sigma.coeffs[1].item()),
                    'sigma2':        float(prior.net_sigma.coeffs[2].item()),
                    'z1_log_std':    float(prior.z1_log_std.item()),
                    'z1_skew':       float(prior.z1_skew.item()),
                    'z1_log_tail':   float(prior.z1_log_tail.item()),
                    'log_sigma_eps': float('nan'),
                    'log_beta':      float('nan'),
                }
                t_vec_prior = np.array(
                    [t[k] for k in PARAM_NAMES[:11]], dtype=np.float64)
                delta = (float(np.linalg.norm(t_vec_prior - prev_theta_vec))
                          if prev_theta_vec is not None else None)
                prev_theta_vec = t_vec_prior
                mle_trace.append({
                    'epoch':            epoch + 1,
                    'log_p':            float(-loss.item()),
                    'theta':            t,
                    'grad_norm':        last_grad_norm,
                    'param_delta_norm': delta,
                })
                if ((epoch + 1) % (log_every * 10) == 0
                        or epoch == 0 or epoch == n_epochs_mle - 1):
                    print(f"  [mle/{epoch+1:5d}]  log_p={-loss.item():+.3f}  "
                          f"mu1={t['mu1']:+.3f}  sigma0={t['sigma0']:+.3f}  "
                          f"sigma2={t['sigma2']:+.3f}  "
                          f"||g||={last_grad_norm:.3f}  "
                          f"({time.time()-t0:.1f}s)", flush=True)
        theta_MLE = mle_trace[-1]['theta']
        # L2 over the 10 free prior + z_1 coordinates (skip z1_skew which
        # is pinned at 0, and the two decoder coords which don't exist).
        free_keys = [k for k in PARAM_NAMES[:11] if k != 'z1_skew']
        L2_MLE = float(np.linalg.norm(
            np.array([theta_MLE[k] for k in free_keys], dtype=np.float64) -
            np.array([truth[k]     for k in free_keys], dtype=np.float64)))
        print(f"\ntheta_MLE: mu1={theta_MLE['mu1']:+.4f}  "
              f"sigma0={theta_MLE['sigma0']:+.4f}  "
              f"sigma2={theta_MLE['sigma2']:+.4f}  "
              f"L2(10 free prior+z1)={L2_MLE:.4f}", flush=True)
        prior_log_p_at_mle = float(-loss.item())

        # Closed-form prior log-likelihood at theta_truth on y_obs (same
        # degenerate-emission model; for diagnostic reference).
        torch.manual_seed(vi_seed + 1)
        prior_truth = MarkovNormalConditionalPolyPrior(
            nt=T, poly_degree=2, law_model='softplus',
            z1_distr='sinh', extra_heterogeneity=False).to(device)
        with torch.no_grad():
            prior_truth.b.fill_(truth['alpha0'])
            prior_truth.log_gamma.fill_(truth['log_alpha1'])
            prior_truth.net_mu.coeffs.copy_(torch.tensor(
                [truth['mu0'], truth['mu1'], truth['mu2']],
                dtype=prior_truth.net_mu.coeffs.dtype,
                device=prior_truth.net_mu.coeffs.device))
            prior_truth.net_sigma.coeffs.copy_(torch.tensor(
                [truth['sigma0'], truth['sigma1'], truth['sigma2']],
                dtype=prior_truth.net_sigma.coeffs.dtype,
                device=prior_truth.net_sigma.coeffs.device))
            prior_truth.z1_log_std.fill_(truth['z1_log_std'])
            prior_truth.z1_skew.zero_()
            prior_truth.z1_log_tail.fill_(truth['z1_log_tail'])
        with torch.no_grad():
            prior_log_p_at_truth = float(
                prior_truth.log_prob(y_obs).mean().item())
        print(f"  prior_log_p_at_mle  = {prior_log_p_at_mle:+.4f}", flush=True)
        print(f"  prior_log_p_at_truth= {prior_log_p_at_truth:+.4f}",
              flush=True)
        del prior, prior_truth
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # SMC log p_hat at truth (under the *correct* sinh emission --
        # comparable across cells per the schema invariant).
        print(f"\n[mle_direct/3d] SMC log p_hat at truth (K={K_smc}) ...",
              flush=True)
        model_truth = build_full_model_with_encoder('jn', T, device)
        with torch.no_grad():
            model_truth.prior.b.fill_(truth['alpha0'])
            model_truth.prior.log_gamma.fill_(truth['log_alpha1'])
            model_truth.prior.net_mu.coeffs.copy_(torch.tensor(
                [truth['mu0'], truth['mu1'], truth['mu2']],
                dtype=model_truth.prior.net_mu.coeffs.dtype,
                device=model_truth.prior.net_mu.coeffs.device))
            model_truth.prior.net_sigma.coeffs.copy_(torch.tensor(
                [truth['sigma0'], truth['sigma1'], truth['sigma2']],
                dtype=model_truth.prior.net_sigma.coeffs.dtype,
                device=model_truth.prior.net_sigma.coeffs.device))
            model_truth.prior.z1_log_std.fill_(truth['z1_log_std'])
            model_truth.prior.z1_skew.zero_()
            model_truth.prior.z1_log_tail.fill_(truth['z1_log_tail'])
            model_truth.decoder.log_sigma.fill_(truth['log_sigma_eps'])
            model_truth.decoder.log_beta.fill_(truth['log_beta'])
        bundle = _Bundle(prior=model_truth.prior, decoder=model_truth.decoder)
        pf = BootstrapParticleFilter(bundle, K=K_smc)
        smc_log_p_at_truth = float(pf.log_marginal(y_obs).mean().item())
        print(f"  smc_log_p_at_truth = {smc_log_p_at_truth:+.4f}", flush=True)
        del model_truth
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Output: mle_direct_no_me_ep<N>_lr<tag>[_sig0-<x>][_sig2-<x>].json
        lr_tag = os.environ.get('LR_TAG', f'{lr:.0e}')
        suffixes = [f'lr{lr_tag}']
        if abs(HOCKEY_TRUTH['sigma0'] - (-1.35)) > 1e-6:
            suffixes.append(f"sig0-{HOCKEY_TRUTH['sigma0']:.2f}")
        if abs(HOCKEY_TRUTH['sigma2'] - 0.35) > 1e-6:
            suffixes.append(f"sig2-{HOCKEY_TRUTH['sigma2']:.2f}")
        if T != 6:
            suffixes.append(f"T{T}")
        suffix = '_'.join(suffixes)
        os.makedirs(out_dir, exist_ok=True)
        fname = f"mle_direct_no_me_ep{n_epochs_mle}_{suffix}.json"
        fname = os.environ.get('OUT_NAME', '') or fname
        out_path = os.path.join(out_dir, fname)

        config = {
            'dgp':              'simulation-hockeystick',
            'method':           method,
            'encoder':          None,
            'emission':         emission,
            'alpha':            None,
            'm_mem':            None,
            'n_iters_outer':    None,
            'n_epochs_inner':   None,
            'n_epochs_vi_obs':  None,
            'n_epochs_encoder': None,
            'n_epochs_mle':     n_epochs_mle,
            'tjn_warm_decoder_freeze_epochs': None,
            'log_every':        log_every,
            'lr':               lr,
            'lr_encoder':       None,
            'fix_noise':        None,
            'noise_seed':       None,
            'L_iwae':           L_iwae,
            'K_smc':            K_smc,
            'N':                N, 'T': T,
            'obs_seed':         OBS_SEED,
            'vi_seed':          vi_seed,
            'restart_factor':   None,
        }
        out = {
            'config':                 config,
            'truth':                  {kk: float(v) for kk, v in
                                         HOCKEY_TRUTH.items()},
            'theta_VI_obs':           theta_MLE,
            'theta_VI_obs_trace':     None,
            'theta_VI_obs_jn_trace':  None,
            'mle_trace':              mle_trace,
            'L2_VI':                  L2_MLE,
            'trajectory':             [],
            'final_theta':            theta_MLE,
            'final_L2':               L2_MLE,
            'diagnostics': {
                'elbo_at_vi':          None,
                'iwae_at_vi':          None,
                'elbo_at_ivi':         None,
                'iwae_at_ivi':         None,
                'elbo_at_truth':       None,
                'iwae_at_truth':       None,
                'smc_log_p_at_truth':  smc_log_p_at_truth,
                'prior_log_p_at_mle':   prior_log_p_at_mle,
                'prior_log_p_at_truth': prior_log_p_at_truth,
            },
        }
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nDone. L2_MLE={L2_MLE:.4f}", flush=True)
        print(f"Wrote {out_path}", flush=True)
        return

    # === Phase 1: fit theta_VI_obs from y_obs ===
    # TJN cells run a two-phase warm-start: an internal JN h=64 fit
    # whose converged theta is loaded into the TJN model before the
    # TJN encoder + decoder train (with decoder frozen for the first
    # TJN_WARM_DECODER_FREEZE_EPOCHS epochs).
    print(f"\n=== Phase 1: fit theta_VI_obs ({n_epochs_vi_obs} ep, "
          f"encoder={encoder_label}) ===", flush=True)
    vi_obs_jn_trace = None
    if encoder_label == 'tjn':
        print(f"\n[tjn] internal JN warm-start ({n_epochs_vi_obs} ep) ...",
              flush=True)
        theta_VI_jn, jn_trace, _, _model_jn = fit_vi_inner(
            None, y_obs, n_epochs_vi_obs, lr=lr, log_every=log_every,
            label='VI_obs_jn', warm_start=False, vi_seed=vi_seed,
            eps_fixed=eps_fixed, encoder_label='jn')
        vi_obs_jn_trace = jn_trace
        del _model_jn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n[tjn] TJN fit ({n_epochs_vi_obs} ep, "
              f"freeze_decoder_first={tjn_freeze_epochs}) ...", flush=True)
        theta_VI_obs, vi_obs_trace, _, model_VI_obs = fit_tjn_warm(
            y_obs, n_epochs=n_epochs_vi_obs,
            freeze_epochs=tjn_freeze_epochs, lr=lr,
            log_every=log_every, vi_seed=vi_seed, eps_fixed=eps_fixed,
            jn_theta=theta_VI_jn)
    else:
        theta_VI_obs, vi_obs_trace, _, model_VI_obs = fit_vi_inner(
            None, y_obs, n_epochs_vi_obs, lr=lr, log_every=log_every,
            label='VI_obs', warm_start=False, vi_seed=vi_seed,
            eps_fixed=eps_fixed, encoder_label=encoder_label,
            theta_avg_epochs=theta_avg_epochs)
    theta_VI_obs_vec = theta_to_vec(theta_VI_obs)
    L2_VI = float(np.linalg.norm(theta_VI_obs_vec - truth_vec))
    print(f"\ntheta_VI_obs:  log_beta={theta_VI_obs['log_beta']:+.4f}  "
          f"mu1={theta_VI_obs['mu1']:+.4f}  "
          f"sigma0={theta_VI_obs['sigma0']:+.4f}  "
          f"L2={L2_VI:.4f}", flush=True)
    print(f"truth:         log_beta={truth['log_beta']:+.4f}  "
          f"mu1={truth['mu1']:+.4f}  "
          f"sigma0={truth['sigma0']:+.4f}", flush=True)

    # Output path.
    lr_tag = os.environ.get('LR_TAG', f'{lr:.0e}')
    suffixes = [f'lr{lr_tag}']
    if fix_noise:
        suffixes.append('crn')
    if abs(HOCKEY_TRUTH['sigma0'] - (-1.35)) > 1e-6:
        suffixes.append(f"sig0-{HOCKEY_TRUTH['sigma0']:.2f}")
    if abs(HOCKEY_TRUTH['sigma2'] - 0.35) > 1e-6:
        suffixes.append(f"sig2-{HOCKEY_TRUTH['sigma2']:.2f}")
    if T != 6:
        suffixes.append(f"T{T}")
    if theta_avg_epochs > 0:
        suffixes.append(f"avg{theta_avg_epochs}")
    suffix = '_'.join(suffixes)
    os.makedirs(out_dir, exist_ok=True)
    if is_vi_only:
        enc_tag = ENCODER_LABEL_TO_FILENAME_TAG[encoder_label]
        fname = f"vi_only_{enc_tag}_h64_ep{n_epochs_vi_obs}_{suffix}.json"
    elif method == 'anderson':
        if encoder_label == 'jn':
            # JN Anderson cell's filename is unchanged for backward
            # compatibility (analysis-hockeystick-ivi.md consumes this
            # exact path).
            fname = f"anderson_alpha{alpha:.2f}_m{m_mem}_ep{n_epochs}_{suffix}.json"
        else:
            enc_tag = ENCODER_LABEL_TO_FILENAME_TAG[encoder_label]
            fname = (f"anderson_{enc_tag}_h64_alpha{alpha:.2f}_m{m_mem}_"
                     f"ep{n_epochs}_{suffix}.json")
    else:
        fname = f"{method}_alpha{alpha:.2f}_ep{n_epochs}_{suffix}.json"
    # OUT_NAME overrides the filename. replicate.py uses it to regenerate the
    # historical `vi_<enc>_h64_...` cells (written by an earlier revision of
    # this script under METHOD=vi_only) that the summary bundle still lists.
    fname = os.environ.get('OUT_NAME', '') or fname
    out_path = os.path.join(out_dir, fname)

    RESTART_FACTOR = 1.5

    # Build the cell's static config block. Several fields are null for
    # vi_only cells; per spec they must be `null` (not 0 / NaN).
    config = {
        'dgp':              'simulation-hockeystick',
        'method':           method,
        'encoder':          encoder_label,
        'emission':         emission,
        'alpha':            None if is_vi_only else alpha,
        'm_mem':            m_mem if method == 'anderson' else None,
        'n_iters_outer':    None if is_vi_only else n_iters,
        'n_epochs_inner':   None if is_vi_only else n_epochs,
        'n_epochs_vi_obs':  n_epochs_vi_obs,
        'n_epochs_encoder': n_epochs_encoder,
        'n_epochs_mle':     None,
        'theta_avg_epochs': theta_avg_epochs if theta_avg_epochs > 0 else None,
        'tjn_warm_decoder_freeze_epochs': (
            tjn_freeze_epochs if encoder_label == 'tjn' else None),
        'log_every':        log_every,
        'lr':               lr,
        'lr_encoder':       lr_encoder,
        'fix_noise':        fix_noise,
        'noise_seed':       noise_seed,
        'L_iwae':           L_iwae,
        'K_smc':            K_smc,
        'N':                N, 'T': T,
        'obs_seed':         OBS_SEED,
        'vi_seed':          vi_seed,
        'restart_factor':   None if is_vi_only else RESTART_FACTOR,
    }
    base_out = {
        'config':                 config,
        'truth':                  {kk: float(v) for kk, v in HOCKEY_TRUTH.items()},
        'theta_VI_obs':           theta_VI_obs,
        'theta_VI_obs_trace':     vi_obs_trace,
        'theta_VI_obs_jn_trace':  vi_obs_jn_trace,
        'mle_trace':              None,
        'L2_VI':                  L2_VI,
    }

    if is_vi_only:
        # VI-only cells: skip Phase 2 entirely. Per spec (2026-06-22),
        # trajectory is the empty list and final_theta / final_L2 copy
        # the Phase 1 estimate so downstream consumers can read these
        # keys uniformly across cells.
        print(f"\n=== Phase 2 SKIPPED (METHOD=vi_only) ===", flush=True)
        out = {
            **base_out,
            'trajectory':  [],
            'final_theta': {kk: float(theta_VI_obs[kk]) for kk in PARAM_NAMES},
            'final_L2':    L2_VI,
        }
        theta_k_for_phase3 = None
    else:
        shared_model = None
        if method == 'picard_warm':
            torch.manual_seed(vi_seed); np.random.seed(vi_seed)
            shared_model = build_full_model_with_encoder('jn', T, device)
            print(f"\n[warm] built shared model (encoder reused across iters)",
                  flush=True)

        theta_k = dict(theta_VI_obs)
        history = []
        prev_g_norm = None

        trajectory = [{'iter': 0,
                        'theta_k':  {k: float(theta_k[k]) for k in PARAM_NAMES},
                        'L2_truth': L2_VI}]

        print(f"\n=== Phase 2: IVI outer loop ({n_iters} iters) ===",
              flush=True)
        L2_to_truth = L2_VI
        for k in range(1, n_iters + 1):
            iter_t0 = time.time()
            print(f"\n[iter {k}] simulate y_sim + inner VI "
                  f"({n_epochs} ep, method={method})...", flush=True)

            theta_t = torch.tensor([theta_k[name] for name in PARAM_NAMES],
                                    dtype=torch.float32, device=device)
            with torch.no_grad():
                y_sim = M.reparam_simulate(theta_t, z_noise, y_noise)

            if method == 'picard_warm':
                _set_prior_decoder_from_dict(shared_model, theta_k)
                theta_VI, trace, used_seed, _ = fit_vi_inner(
                    shared_model, y_sim, n_epochs, lr=lr, log_every=log_every,
                    label=f"warm/it{k}", warm_start=True, vi_seed=vi_seed,
                    eps_fixed=eps_fixed, encoder_label='jn',
                    theta_avg_epochs=theta_avg_epochs)
            elif encoder_label == 'tjn':
                # TJN inner fit: internal JN warm-start then TJN, per
                # the per-inner-fit warm-start convention of
                # estimators.md S2.1 (specs/simulation-hockeystick.md
                # "IVI posterior cells to run" / Phase 2).
                print(f"  [tjn/it{k}] internal JN warm-start "
                      f"({n_epochs} ep) ...", flush=True)
                theta_VI_jn_inner, _jn_trace_inner, _jn_used_seed, _model_jn = \
                    fit_vi_inner(
                        None, y_sim, n_epochs, lr=lr, log_every=log_every,
                        label=f"jn_warm/it{k}", warm_start=False,
                        vi_seed=vi_seed, eps_fixed=eps_fixed,
                        encoder_label='jn')
                del _model_jn
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  [tjn/it{k}] TJN fit ({n_epochs} ep, "
                      f"freeze_decoder_first={tjn_freeze_epochs}) ...",
                      flush=True)
                theta_VI, trace, used_seed, _ = fit_tjn_warm(
                    y_sim, n_epochs=n_epochs,
                    freeze_epochs=tjn_freeze_epochs, lr=lr,
                    log_every=log_every, vi_seed=vi_seed,
                    eps_fixed=eps_fixed, jn_theta=theta_VI_jn_inner)
            else:
                theta_VI, trace, used_seed, _ = fit_vi_inner(
                    None, y_sim, n_epochs, lr=lr, log_every=log_every,
                    label=f"{method[:6]}/it{k}", warm_start=False,
                    vi_seed=vi_seed, eps_fixed=eps_fixed,
                    encoder_label=encoder_label,
                    theta_avg_epochs=theta_avg_epochs)

            iter_wall = time.time() - iter_t0
            theta_VI_vec = theta_to_vec(theta_VI)
            x_k = theta_to_vec(theta_k)

            restart = False
            if method == 'anderson':
                g_k = theta_VI_obs_vec - theta_VI_vec
                g_norm = float(np.linalg.norm(g_k))
                if prev_g_norm is not None and g_norm > RESTART_FACTOR * prev_g_norm:
                    print(f"    [restart] ||g||={g_norm:.4f} "
                          f"(was {prev_g_norm:.4f})", flush=True)
                    history = []
                    restart = True
                x_next, gamma = anderson_step(
                    x_k, g_k, history, m=m_mem, beta=alpha)
                history.append({'x': x_k.copy(), 'g': g_k.copy()})
                prev_g_norm = g_norm
            else:
                x_next = x_k + alpha * (theta_VI_obs_vec - theta_VI_vec)
                g_k = theta_VI_obs_vec - theta_VI_vec
                g_norm = float(np.linalg.norm(g_k))
                gamma = None

            theta_next = vec_to_theta(x_next)
            L2_to_truth = float(np.linalg.norm(x_next - truth_vec))

            print(f"  inner VI final: log_beta={theta_VI['log_beta']:+.4f}  "
                  f"mu1={theta_VI['mu1']:+.4f}  "
                  f"sigma2={theta_VI['sigma2']:+.4f}", flush=True)
            print(f"  theta_{k}: log_beta={theta_next['log_beta']:+.4f}  "
                  f"mu1={theta_next['mu1']:+.4f}  "
                  f"L2={L2_to_truth:.4f}  ||g||={g_norm:.4f}  "
                  f"iter_wall={iter_wall:.1f}s", flush=True)

            trajectory.append({
                'iter':            k,
                'iter_wall_s':     iter_wall,
                'theta_k':         {kk: float(theta_next[kk]) for kk in PARAM_NAMES},
                'theta_VI':        {kk: float(theta_VI[kk]) for kk in PARAM_NAMES},
                'L2_truth':        L2_to_truth,
                'g_norm':          g_norm,
                'restart':         restart,
                'gamma':           gamma.tolist() if gamma is not None else None,
                'used_seed':       int(used_seed) if used_seed is not None else None,
                'inner_trace':     trace,
            })
            theta_k = theta_next

            out = {
                **base_out,
                'trajectory':  trajectory,
                'final_theta': {kk: float(theta_k[kk]) for kk in PARAM_NAMES},
                'final_L2':    L2_to_truth,
            }
            with open(out_path, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"  [cache write: {out_path}]", flush=True)

            if torch.cuda.is_available() and method != 'picard_warm':
                torch.cuda.empty_cache()
        theta_k_for_phase3 = theta_k

    print(f"\n=== Phase 3: post-fit diagnostics ===", flush=True)

    # 3a. ELBO + IWAE at theta_VI_obs using the phase-1 encoder.
    print(f"\n[3a] ELBO/IWAE at theta_VI_obs (phase-1 encoder, L={L_iwae}) ...",
          flush=True)
    diag_vi = model_VI_obs.elbo_diagnostics(y_obs, ndraws=L_iwae)
    elbo_at_vi = float(diag_vi['elbo'])
    iwae_at_vi = float(diag_vi['iwae'])
    print(f"  elbo_at_vi={elbo_at_vi:+.4f}  iwae_at_vi={iwae_at_vi:+.4f}  "
          f"gap={iwae_at_vi - elbo_at_vi:+.4f}", flush=True)
    del model_VI_obs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3b. ELBO + IWAE at theta_K with a fresh joint-normal h=64 encoder fit
    #     at (theta_K, y_obs). *IVI cells only*; vi_only cells emit null.
    if is_vi_only:
        elbo_at_ivi = None
        iwae_at_ivi = None
    else:
        print(f"\n[3b] fit encoder@theta_K + ELBO/IWAE "
              f"(JN h=64 encoder, {n_epochs_encoder} ep, "
              f"lr={lr_encoder}) ...", flush=True)
        # Phase 3 at-IVI: use a fresh JN h=64 encoder regardless of
        # the cell's ENCODER family. This keeps the elbo_at_ivi /
        # iwae_at_ivi diagnostic comparable across cells (same
        # encoder, varying theta_K) and avoids the cold-start TJN
        # encoder collapse when prior+decoder are frozen at theta_K
        # (the warm-start convention degenerates and the TJN encoder
        # NaN's). The cell's at-VI encoder reuses Phase 1 (the cell's
        # encoder) for elbo_at_vi / iwae_at_vi, so the cell's
        # encoder is still represented in the diagnostics.
        model_ivi = fit_encoder_only_at_theta(
            theta_k_for_phase3, y_obs, T, vi_seed,
            n_epochs=n_epochs_encoder, lr=lr_encoder, device=device,
            encoder_label='jn')
        diag_ivi = model_ivi.elbo_diagnostics(y_obs, ndraws=L_iwae)
        elbo_at_ivi = float(diag_ivi['elbo'])
        iwae_at_ivi = float(diag_ivi['iwae'])
        print(f"  elbo_at_ivi={elbo_at_ivi:+.4f}  "
              f"iwae_at_ivi={iwae_at_ivi:+.4f}  "
              f"gap={iwae_at_ivi - elbo_at_ivi:+.4f}", flush=True)
        del model_ivi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3c. ELBO + IWAE at theta_truth with a fresh joint-normal h=64 encoder.
    print(f"\n[3c] fit encoder@truth + ELBO/IWAE (8000 ep, lr=1e-3) ...",
          flush=True)
    model_truth = fit_encoder_only_at_theta(
        truth, y_obs, T, vi_seed, n_epochs=n_epochs_encoder,
        lr=lr_encoder, device=device)
    diag_truth = model_truth.elbo_diagnostics(y_obs, ndraws=L_iwae)
    elbo_at_truth = float(diag_truth['elbo'])
    iwae_at_truth = float(diag_truth['iwae'])
    print(f"  elbo_at_truth={elbo_at_truth:+.4f}  "
          f"iwae_at_truth={iwae_at_truth:+.4f}  "
          f"gap={iwae_at_truth - elbo_at_truth:+.4f}", flush=True)

    # 3d. SMC log p_hat at truth (K=1000), per-individual mean.
    print(f"\n[3d] SMC log p_hat at truth (K={K_smc}) ...", flush=True)
    bundle = _Bundle(prior=model_truth.prior, decoder=model_truth.decoder)
    pf = BootstrapParticleFilter(bundle, K=K_smc)
    smc_log_p_at_truth = float(pf.log_marginal(y_obs).mean().item())
    print(f"  smc_log_p_at_truth={smc_log_p_at_truth:+.4f}", flush=True)
    del model_truth
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out['diagnostics'] = {
        'elbo_at_vi':           elbo_at_vi,
        'iwae_at_vi':           iwae_at_vi,
        'elbo_at_ivi':          elbo_at_ivi,
        'iwae_at_ivi':          iwae_at_ivi,
        'elbo_at_truth':        elbo_at_truth,
        'iwae_at_truth':        iwae_at_truth,
        'smc_log_p_at_truth':   smc_log_p_at_truth,
        'prior_log_p_at_mle':   None,
        'prior_log_p_at_truth': None,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    if is_vi_only:
        print(f"\nDone. L2_VI={L2_VI:.4f}", flush=True)
    else:
        print(f"\nDone. Final L2={trajectory[-1]['L2_truth']:.4f}", flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    # Fix every RNG (python / numpy / torch CPU+CUDA) and request deterministic
    # kernels before any draw; the script's own explicit seeds still apply.
    from mlye.seeding import seed_everything, env_seed
    seed_everything(env_seed("OBS_SEED", 11))
    main()
