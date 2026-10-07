r"""Layer 1a fit script for the simulation-ar1n sweep.

Per `specs/compute-simulation-ar1n.md`. Six configurations on the
AR(1)+Normal canonical DGP (mu_1=0.9, sigma_0=-1.48 preset of
`mlye/simulator/simulator.py`; N=30000, T=6):

  Four VI-only configurations (METHOD=vi_only, ENCODER in
  {mean_field, tridiag, jn, struct_markov}):
    vi_meanfield_h32_ep10000_lr1e-2_crn.json    -- normal_diagonal encoder
    vi_tridiag_h32_ep10000_lr1e-2_crn.json      -- tridiag_joint_normal
    vi_jointnormal_h32_ep10000_lr1e-2_crn.json  -- joint_normal
    vi_struct_markov_h32_ep10000_lr1e-2_crn.json -- conditional_markov

  One IVI (damped-Picard, strict CRN, mean-field encoder fixed):
    picard_meanfield_alpha0.60_ep8000_lr1e-2_crn.json
    theta_{k+1} = (1 - alpha) theta_k + alpha b(theta_k),
    b(theta_k) = theta_VI_MF(y_sim(theta_k)).
    Strict CRN: same vi_seed every outer iter, same eps_fixed noise,
    AdamW reset to identical state per outer iter (rebuild model).

  One MLE (Model B, no measurement error, L-BFGS on the closed-form
  poly-2 chain log-likelihood):
    mle_noerror_lbfgs.json

Phase 3 (post-fit diagnostics) appends a `diagnostics` block to each
JSON. VI-only configs report elbo/iwae at vi + at truth and SMC log p
at truth. The Picard IVI config adds elbo/iwae at the IVI estimate
(theta_K). The MLE config reports logp_at_mle and logp_at_truth under
the closed-form Model B likelihood.

Env vars (with defaults):
  METHOD               vi_only | picard | mle_noerror      (vi_only)
  ENCODER              mean_field | tridiag | jn |         (mean_field)
                       struct_markov | tjn -- required for VI/picard;
                       ignored for mle_noerror. tjn = transformed
                       joint-normal, two-phase JN warm start (vi_only).
  TJN_WARM_DECODER_FREEZE_EPOCHS  decoder freeze for tjn       (500)
  PICARD_ALPHA         damping for Picard outer loop       (0.6)
  N_EPOCHS_INNER       inner-VI epoch budget (picard)      (8000)
  N_ITERS_OUTER        outer Picard iter count             (10)
  N_EPOCHS_VI_OBS      epochs for the theta_VI_obs pre-fit (10000)
  LOG_EVERY            snapshot frequency                  (100)
  LR                   inner-VI learning rate              (1e-2)
  LR_TAG               filename tag for LR                 (1e-2)
  FIX_NOISE            pre-draw eps once + reuse           (1)
  NOISE_SEED           eps draw seed                       (12345)
  OBS_SEED             y_obs simulation seed               (11)
  VI_SEED              encoder init seed                   (OBS_SEED*1000+7)

  Truth overrides (default = spec calibration):
    MU1 (0.9), SIGMA0 (-1.48), SIGMA_Z1 (-0.733),
    LOG_SIGMA_EPS (-1.470).

Output:
  output/bundles/cells/_simulation_ar1n_sweep/<cell_key>.json
"""
import os
import sys
import time
import json

import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ar1n_posterior_comparison as M
from smc_truth_profile import _Bundle
from mlye.eval import BootstrapParticleFilter


# Map spec's ENCODER label to mlye encoder_type strings.
ENCODER_LABEL_TO_TYPE = {
    'mean_field':    'normal_diagonal',
    'tridiag':       'tridiag_joint_normal',
    'jn':            'joint_normal',
    'struct_markov': 'conditional_markov',
    'tjn':           'transformed_joint_normal',
}

# Map ENCODER label to the spec-declared filename stem.
ENCODER_LABEL_TO_FILENAME_TAG = {
    'mean_field':    'meanfield',
    'tridiag':       'tridiag',
    'jn':            'jointnormal',
    'struct_markov': 'struct_markov',
    'tjn':           'tjn',
}

# Hidden width for this entry (all configurations use h=32).
HIDDEN_DIM = 32


def build_full_model_with_encoder(encoder_label, T, device):
    encoder_type = ENCODER_LABEL_TO_TYPE[encoder_label]
    encoder = M.build_encoder(encoder_type, T, hidden_dim=HIDDEN_DIM).to(device)
    from mlye.models.full_model import FullModel
    prior, decoder = M.build_prior_decoder(T, device)
    model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(device)
    M._apply_pinning_masks(model, device)
    return model


# Truth: AR(1) + Normal canonical calibration. Overridable via env.
_MU1            = float(os.environ.get('MU1',            M.MU1_DEFAULT))
_SIGMA0         = float(os.environ.get('SIGMA0',         M.SIGMA0_DEFAULT))
_SIGMA_Z1       = float(os.environ.get('SIGMA_Z1',       M.SIGMA_Z1_DEFAULT))
_LOG_SIGMA_EPS  = float(os.environ.get('LOG_SIGMA_EPS',  M.LOG_SIGMA_EPS_DEFAULT))

AR1N_TRUTH = dict(M.TRUTH)
AR1N_TRUTH['mu1']           = _MU1
AR1N_TRUTH['sigma0']        = _SIGMA0
AR1N_TRUTH['z1_log_std']    = _SIGMA_Z1
AR1N_TRUTH['log_sigma_eps'] = _LOG_SIGMA_EPS
M.TRUTH = AR1N_TRUTH

PARAM_NAMES = M.PARAM_NAMES


def theta_to_vec(theta):
    return np.array([theta[k] for k in PARAM_NAMES], dtype=np.float64)


def vec_to_theta(vec):
    return {k: float(vec[i]) for i, k in enumerate(PARAM_NAMES)}


def L2_free(theta, truth):
    """L2 distance over the 8 FREE parameters (PARAM_NAMES[FREE_INDICES]).

    Per spec the eight free coords are
    (mu0, mu1, mu2, sigma0, sigma1, sigma2, z1_log_std, log_sigma_eps);
    the pinned-shape coords (log_beta, theta_MA, z1_skew, z1_log_tail)
    are excluded.
    """
    a = theta_to_vec(theta)
    b = theta_to_vec(truth)
    return float(np.linalg.norm((a - b)[M.FREE_INDICES]))


def fit_encoder_only_at_theta(theta, y, T, vi_seed, n_epochs, lr, device,
                                  n_retries=10):
    """Fresh joint-normal h=32 encoder with prior+decoder frozen at `theta`.

    Used by Phase 3 to evaluate ELBO/IWAE at a given (theta_VI, theta_K,
    or theta_truth) under a *common* truth-side encoder so the
    identifiability ceiling is shared across configurations.
    """
    last_err = None
    for s in M._retry_seeds(vi_seed, n_retries):
        try:
            torch.manual_seed(s); np.random.seed(s)
            model = build_full_model_with_encoder('jn', T, device)
            M.set_prior_decoder_to_truth(model, theta)
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


def fit_vi_inner(y, n_epochs, lr, log_every,
                  label='vi', vi_seed=None,
                  n_retries=10, eps_fixed=None, encoder_label='mean_field'):
    """Cold-start a VI fit with the chosen encoder family.

    `encoder_label` matches the spec's ENCODER env-var labels
    (mean_field / tridiag / jn / struct_markov). Strict CRN: the model
    is rebuilt fresh under `torch.manual_seed(vi_seed)` on every call,
    so the encoder init AND the AdamW initial state are identical
    across calls (the optimiser has no carry-over state at
    construction time).
    """
    last_err = None
    seeds_to_try = M._retry_seeds(vi_seed, n_retries)
    for s in seeds_to_try:
        try:
            torch.manual_seed(s); np.random.seed(s)
            model = build_full_model_with_encoder(
                encoder_label, y.shape[1], y.device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
            trace = []
            t0 = time.time()
            nan_streak = 0
            prev_theta_vec = None
            last_grad_norm = float('nan')
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
                              f"mu1={theta['mu1']:+.3f}  "
                              f"sigma0={theta['sigma0']:+.3f}  "
                              f"||g||={last_grad_norm:.3f}  "
                              f"||dt||={delta if delta is None else f'{delta:.4f}'}  "
                              f"({time.time()-t0:.1f}s)", flush=True)
            return M.extract_params(model), trace, s, model
        except M.NaNFailure as e:
            print(f"    [{label}/s{s}] NaN bail: {e}; retrying", flush=True)
            last_err = e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"inner VI failed after {len(seeds_to_try)} seed retries "
                        f"(last err: {last_err})")


def fit_tjn_warm(y, n_epochs, freeze_epochs, lr, log_every, vi_seed,
                   eps_fixed, jn_theta):
    """TJN two-phase warm-start fit (ENCODER=tjn, VI-only cells).

    Restored from the afeed8d revision of this script, which produced
    vi_tjn_h32_ep10000_lr1e-3_crn.json in ar1n_parameter_summary.json.

    Builds a TJN h=32 FullModel, copies prior+decoder from `jn_theta`
    (a JN-converged theta_VI dict), trains for `n_epochs` total. The
    first `freeze_epochs` epochs pin `decoder.log_sigma` (`log_beta`
    is already non-trainable via `fix_beta=True` so the freeze of
    `decoder.log_beta` is automatic).
    """
    T = y.shape[1]
    last_err = None
    for s in M._retry_seeds(vi_seed, 10):
        try:
            torch.manual_seed(s); np.random.seed(s)
            model = build_full_model_with_encoder('tjn', T, y.device)
            M.set_prior_decoder_to_truth(model, jn_theta)
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
                    # Thaw the decoder, but `log_beta` stays
                    # non-trainable because `fix_beta=True` was set at
                    # construction time. The only param that actually
                    # unfreezes is `log_sigma`.
                    for p in model.decoder.parameters():
                        # Re-enable grad for log_sigma but not log_beta.
                        if p is model.decoder.log_beta:
                            continue
                        p.requires_grad_(True)
                    opt = torch.optim.AdamW(
                        [p for p in model.parameters() if p.requires_grad], lr=lr)
                    thawed = True
                    print(f"    [tjn/s{s}] ep {epoch+1}: decoder thawed "
                          f"(log_sigma)", flush=True)
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
                              f"mu1={theta['mu1']:+.3f}  "
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


def picard_step(x_k, g_k, alpha):
    """Damped IVI binding-equation iteration step:
        theta_{k+1} = theta_k + alpha * (theta_VI_obs - b(theta_k))
                    = theta_k + alpha * g_k
    where g_k = theta_VI_obs_vec - theta_VI_vec is the IVI binding-
    equation residual.

    **This is NOT the standard damped fixed-point iteration**
    `theta_{k+1} = (1-alpha) theta_k + alpha b(theta_k)`. Standard
    damped Picard solves for `b(theta) = theta` -- a fixed point that
    on this DGP lies at the mean-field amortization gap's natural
    attractor, NOT at truth. The IVI binding equation solves for the
    different equation
        b(theta) = theta_VI_obs
    whose fixed point under strict CRN coincides with theta_truth
    (because y_sim(theta_truth) = y_obs and so b(theta_truth) =
    theta_VI(y_obs) = theta_VI_obs).

    The corresponding update direction (`theta_VI_obs - b(theta_k)`)
    is matched verbatim to the hockeystick IVI precedent in
    `scripts/_simulation_hockeystick_ivi_sweep.py`.
    """
    return x_k + alpha * g_k


def anderson_step(x_k, g_k, history, m, beta, reg=1e-8):
    """Type-II Anderson m-memory acceleration with damped-Picard base.

    Parameters
    ----------
    x_k     current state (1D np.ndarray, length len(PARAM_NAMES))
    g_k     residual g_k = b(x_k) - x_k (same shape as x_k)
    history list of dicts {'x': ..., 'g': ...} from previous iters
            (we use only the last m entries)
    m       memory length
    beta    damping/mixing parameter (the alpha env var in the IVI
            literature; same convention as picard_step's alpha)
    reg     Tikhonov regulariser on the Anderson least-squares solve

    Returns
    -------
    x_next  the Anderson-accelerated next state
    gamma   the LS coefficient vector (length m_k) or None at iter 1

    The base Picard step (m_k == 0) is x_k + beta * g_k = (1-beta) x_k
    + beta b(x_k), matching picard_step at alpha=beta. With history,
    the Anderson update solves
        gamma = argmin || (Delta_G) gamma - g_k ||^2
    where Delta_G[:, i] = g_{i+1} - g_i across history+current. Then
        x_next = x_k + beta g_k - (Delta_X + beta Delta_G) gamma.
    Verbatim port of the hockeystick precedent in
    `scripts/_simulation_hockeystick_ivi_sweep.py::anderson_step`.
    """
    hist = history[-m:] if len(history) > 0 else []
    m_k = len(hist)
    if m_k == 0:
        return x_k + beta * g_k, None
    xs = [h['x'] for h in hist] + [x_k]
    gs = [h['g'] for h in hist] + [g_k]
    Delta_X = np.column_stack([xs[i + 1] - xs[i] for i in range(m_k)])
    Delta_G = np.column_stack([gs[i + 1] - gs[i] for i in range(m_k)])
    AtA = Delta_G.T @ Delta_G + reg * np.eye(m_k)
    Atb = Delta_G.T @ g_k
    gamma = np.linalg.solve(AtA, Atb)
    x_next = x_k + beta * g_k - (Delta_X + beta * Delta_G) @ gamma
    return x_next, gamma


def mle_noerror_lbfgs(y, truth, max_iter=200, tolerance_grad=1e-7,
                       tolerance_change=1e-9, history_size=10, device=None):
    """Model B closed-form MLE via L-BFGS over 7 free parameters.

    Per spec 'No-measurement-error MLE configuration to run': initialise
    at truth restricted to the seven free coords, minimise
    `-M.poly2_chain_logp(...).mean()`, record the loss trajectory, and
    return the converged theta + diagnostic logp values.

    Returns:
      final_theta: dict of the seven keys at the MLE optimum.
      loss_trace: list of per-iter NLL values.
      n_iter:     int (number of LBFGS iters actually used).
      logp_at_mle: float (per-individual mean log p at the MLE).
      logp_at_truth: float (per-individual mean log p at truth-7).
    """
    if device is None:
        device = y.device

    # Seven free parameters under Model B (log_sigma_eps is removed).
    FREE_KEYS = ['mu0', 'mu1', 'mu2', 'sigma0', 'sigma1', 'sigma2',
                  'z1_log_std']

    init_vals = [float(truth[k]) for k in FREE_KEYS]
    # Pack into a single leaf tensor for L-BFGS.
    theta = torch.tensor(init_vals, dtype=torch.float64, device=device,
                          requires_grad=True)

    # y_64 for closed-form chain logp (poly2_chain_logp uses y.dtype).
    y64 = y.to(dtype=torch.float64)

    # logp_at_truth: closed form, no optimisation needed.
    with torch.no_grad():
        truth_tensor = torch.tensor(init_vals, dtype=torch.float64,
                                      device=device)
        logp_at_truth = float(
            M.poly2_chain_logp(
                y64,
                truth_tensor[0], truth_tensor[1], truth_tensor[2],
                truth_tensor[3], truth_tensor[4], truth_tensor[5],
                truth_tensor[6],
            ).mean().item()
        )

    opt = torch.optim.LBFGS(
        [theta], max_iter=max_iter,
        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
        history_size=history_size, line_search_fn='strong_wolfe',
    )
    loss_trace = []

    def closure():
        opt.zero_grad()
        logp = M.poly2_chain_logp(
            y64, theta[0], theta[1], theta[2],
            theta[3], theta[4], theta[5], theta[6],
        )
        loss = -logp.mean()
        loss.backward()
        loss_trace.append(float(loss.item()))
        return loss

    t0 = time.time()
    opt.step(closure)
    print(f"  [mle/lbfgs] {len(loss_trace)} iters in {time.time()-t0:.2f}s, "
          f"loss={loss_trace[-1]:+.6f}", flush=True)

    with torch.no_grad():
        logp_at_mle = float(
            M.poly2_chain_logp(
                y64, theta[0], theta[1], theta[2],
                theta[3], theta[4], theta[5], theta[6],
            ).mean().item()
        )

    final_theta = {k: float(theta[i].detach().item())
                    for i, k in enumerate(FREE_KEYS)}

    return (final_theta, loss_trace, len(loss_trace),
            logp_at_mle, logp_at_truth)


def main():
    method            = os.environ.get('METHOD', 'vi_only')
    encoder_label     = os.environ.get('ENCODER', 'mean_field')
    alpha             = float(os.environ.get('PICARD_ALPHA', 0.6))
    n_epochs          = int(os.environ.get('N_EPOCHS_INNER', 8000))
    n_iters           = int(os.environ.get('N_ITERS_OUTER', 10))
    log_every         = int(os.environ.get('LOG_EVERY', 100))
    lr                = float(os.environ.get('LR', 1e-2))
    fix_noise         = int(os.environ.get('FIX_NOISE', 1)) != 0
    noise_seed        = (int(os.environ.get('NOISE_SEED', 12345))
                          if fix_noise else None)
    n_epochs_vi_obs   = int(os.environ.get('N_EPOCHS_VI_OBS', 10000))
    n_epochs_encoder  = int(os.environ.get('N_EPOCHS_ENCODER', 8000))
    lr_encoder        = float(os.environ.get('LR_ENCODER', 1e-3))
    L_iwae            = int(os.environ.get('L_IWAE', 100))
    K_smc             = int(os.environ.get('K_SMC', 1000))
    N_override        = int(os.environ.get('N', 30000))
    m_mem             = int(os.environ.get('M_MEM', 4))  # anderson only
    tjn_freeze_epochs = int(os.environ.get(
        'TJN_WARM_DECODER_FREEZE_EPOCHS', 500))           # tjn only

    assert method in ('vi_only', 'picard', 'anderson', 'mle_noerror'), \
        (f"unknown METHOD: {method} "
          f"(supported: vi_only, picard, anderson, mle_noerror)")
    if method != 'mle_noerror':
        assert encoder_label in ENCODER_LABEL_TO_TYPE, \
            f"unknown ENCODER: {encoder_label}"
    # For both METHOD=picard and METHOD=anderson the inner encoder family
    # is *always* mean-field per the spec / IVI comparison study; we
    # silently override any other ENCODER value here.
    if method in ('picard', 'anderson'):
        encoder_label = 'mean_field'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}", flush=True)
    print(f"DGP: simulation-ar1n (AR(1) + Normal canonical calibration).",
          flush=True)
    print(f"  mu_1 = {_MU1:+.4f}  sigma_0 = {_SIGMA0:+.4f}  "
          f"z1_log_std = {_SIGMA_Z1:+.4f}  "
          f"log_sigma_eps = {_LOG_SIGMA_EPS:+.4f}", flush=True)
    print(f"method={method} encoder={encoder_label} alpha={alpha} "
          f"n_iters={n_iters} n_epochs_inner={n_epochs} "
          f"log_every={log_every} lr={lr} fix_noise={fix_noise} "
          f"noise_seed={noise_seed}", flush=True)
    print(f"n_epochs_vi_obs={n_epochs_vi_obs}", flush=True)

    N, T = N_override, 6
    OBS_SEED = int(os.environ.get('OBS_SEED', 11))
    vi_seed = int(os.environ.get('VI_SEED', OBS_SEED * 1000 + 7))

    # CRN noise on the data side (used for both y_obs and the
    # Picard-loop reparam_simulate calls).
    torch.manual_seed(OBS_SEED)
    z_noise = torch.randn(N, T, device=device)
    y_noise = torch.randn(N, T, device=device)

    # Simulate y_obs at truth. NOTE: the helper draws its own internal
    # noise at the seed argument; identical noise to the (z_noise,
    # y_noise) tensors above by construction (same seed, same order).
    print(f"\n=== Phase 0: simulate y_obs (N={N}, T={T}) ===", flush=True)
    y_obs = M.simulate_from_truth(N, T, AR1N_TRUTH, OBS_SEED, device)
    print(f"  y_obs: std={y_obs.std().item():+.4f}  "
          f"std(diff)={torch.diff(y_obs, dim=1).std().item():+.4f}",
          flush=True)

    truth = AR1N_TRUTH
    truth_vec = theta_to_vec(truth)

    # Pre-draw step-to-step CRN epsilon for the inner-VI ELBO.
    # All four encoder families ship with `get_seed_dim() == T`.
    seed_dim = T
    if fix_noise:
        torch.manual_seed(noise_seed)
        eps_fixed = torch.randn((1, N, seed_dim),
                                  dtype=torch.float32, device=device)
        print(f"\n[crn] eps_fixed pre-drawn at noise_seed={noise_seed}  "
              f"shape={tuple(eps_fixed.shape)}", flush=True)
    else:
        eps_fixed = None

    # ------------------------------------------------------------------
    # Output path + config block.
    # ------------------------------------------------------------------
    lr_tag = os.environ.get('LR_TAG', f'{lr:.0e}')
    suffixes = [f'lr{lr_tag}']
    if fix_noise:
        suffixes.append('crn')
    suffix = '_'.join(suffixes)
    out_dir = "output/bundles/cells/_simulation_ar1n_sweep"
    os.makedirs(out_dir, exist_ok=True)
    if method == 'vi_only':
        enc_tag = ENCODER_LABEL_TO_FILENAME_TAG[encoder_label]
        fname = (f"vi_{enc_tag}_h{HIDDEN_DIM}_ep{n_epochs_vi_obs}_"
                  f"{suffix}.json")
    elif method == 'picard':
        enc_tag = ENCODER_LABEL_TO_FILENAME_TAG['mean_field']
        fname = (f"picard_{enc_tag}_alpha{alpha:.2f}_ep{n_epochs}_"
                  f"{suffix}.json")
    elif method == 'anderson':
        enc_tag = ENCODER_LABEL_TO_FILENAME_TAG['mean_field']
        fname = (f"anderson_{enc_tag}_alpha{alpha:.2f}_m{m_mem}_"
                  f"ep{n_epochs}_{suffix}.json")
    else:  # mle_noerror
        fname = "mle_noerror_lbfgs.json"
    # OUT_NAME overrides the filename; replicate.py --smoke uses it to keep the
    # canonical cell names while shrinking the epoch counts encoded in them.
    fname = os.environ.get('OUT_NAME', '') or fname
    out_path = os.path.join(out_dir, fname)

    config = {
        'dgp':              'simulation-ar1n',
        'method':           method,
        'encoder':          encoder_label if method != 'mle_noerror' else None,
        'hidden_dim':       HIDDEN_DIM if method != 'mle_noerror' else None,
        'alpha':            alpha if method in ('picard', 'anderson') else None,
        'm_mem':            m_mem if method == 'anderson' else None,
        'tjn_warm_decoder_freeze_epochs': (
            tjn_freeze_epochs if encoder_label == 'tjn' else None),
        'n_iters_outer':    n_iters if method in ('picard', 'anderson') else None,
        'n_epochs_inner':   n_epochs if method in ('picard', 'anderson') else None,
        'n_epochs_vi_obs':  (n_epochs_vi_obs
                              if method != 'mle_noerror' else None),
        'n_epochs_encoder': (n_epochs_encoder
                              if method != 'mle_noerror' else None),
        'log_every':        log_every,
        'lr':               lr if method != 'mle_noerror' else None,
        'lr_encoder':       lr_encoder if method != 'mle_noerror' else None,
        'fix_noise':        fix_noise,
        'noise_seed':       noise_seed,
        'L_iwae':           L_iwae if method != 'mle_noerror' else None,
        'K_smc':            K_smc if method != 'mle_noerror' else None,
        'N':                N, 'T': T,
        'obs_seed':         OBS_SEED,
        'vi_seed':          vi_seed if method != 'mle_noerror' else None,
        'free_indices':     M.FREE_INDICES,
        'free_names':       M.FREE_NAMES,
    }
    truth_block = {kk: float(v) for kk, v in AR1N_TRUTH.items()}

    # ------------------------------------------------------------------
    # MLE branch: skip Phase 1 + Phase 2 entirely.
    # ------------------------------------------------------------------
    if method == 'mle_noerror':
        print(f"\n=== Phase 1+2 SKIPPED (METHOD=mle_noerror) ===", flush=True)
        print(f"\n=== Phase 3: L-BFGS over Model B closed-form NLL ===",
              flush=True)
        t0_total = time.time()
        (final_theta, loss_trace, n_iter,
          logp_at_mle, logp_at_truth) = mle_noerror_lbfgs(
            y_obs, truth, max_iter=200,
            tolerance_grad=1e-7, tolerance_change=1e-9,
            history_size=10, device=device)
        # L2 distance over the 7 free coords.
        FREE_KEYS_B = ['mu0', 'mu1', 'mu2', 'sigma0', 'sigma1', 'sigma2',
                        'z1_log_std']
        a = np.array([final_theta[k] for k in FREE_KEYS_B], dtype=np.float64)
        b = np.array([truth[k] for k in FREE_KEYS_B], dtype=np.float64)
        final_L2 = float(np.linalg.norm(a - b))
        wall_s = time.time() - t0_total

        print(f"\n  logp_at_mle  ={logp_at_mle:+.6f}", flush=True)
        print(f"  logp_at_truth={logp_at_truth:+.6f}", flush=True)
        print(f"  gap (mle-truth) ={logp_at_mle - logp_at_truth:+.6f}",
              flush=True)
        print(f"  final_L2 (7-coord) = {final_L2:.4f}", flush=True)

        out = {
            'config':       config,
            'truth':        truth_block,
            'final_theta':  final_theta,
            'final_L2':     final_L2,
            'loss_trace':   loss_trace,
            'n_iter':       int(n_iter),
            'logp_at_mle':  logp_at_mle,
            'logp_at_truth': logp_at_truth,
            'diagnostics':  {
                'logp_at_mle':   logp_at_mle,
                'logp_at_truth': logp_at_truth,
            },
            'wall_s':       wall_s,
        }
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nDone. Wrote {out_path}", flush=True)
        return

    # ------------------------------------------------------------------
    # Phase 1: fit theta_VI_obs from y_obs (vi_only + picard branches).
    # ------------------------------------------------------------------
    t0_total = time.time()
    print(f"\n=== Phase 1: fit theta_VI_obs ({n_epochs_vi_obs} ep, "
          f"encoder={encoder_label}) ===", flush=True)
    if encoder_label == 'tjn':
        # Two-phase warm start: an internal JN h=32 fit whose converged
        # theta seeds the TJN model, decoder frozen for the first
        # TJN_WARM_DECODER_FREEZE_EPOCHS epochs.
        print(f"\n[tjn] internal JN warm-start ({n_epochs_vi_obs} ep) ...",
              flush=True)
        theta_VI_jn, _, _, _model_jn = fit_vi_inner(
            y_obs, n_epochs_vi_obs, lr=lr, log_every=log_every,
            label='VI_obs_jn', vi_seed=vi_seed,
            eps_fixed=eps_fixed, encoder_label='jn')
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
            y_obs, n_epochs_vi_obs, lr=lr, log_every=log_every,
            label='VI_obs', vi_seed=vi_seed,
            eps_fixed=eps_fixed, encoder_label=encoder_label)
    theta_VI_obs_vec = theta_to_vec(theta_VI_obs)
    L2_VI = L2_free(theta_VI_obs, truth)
    print(f"\ntheta_VI_obs:  mu1={theta_VI_obs['mu1']:+.4f}  "
          f"sigma0={theta_VI_obs['sigma0']:+.4f}  "
          f"z1_log_std={theta_VI_obs['z1_log_std']:+.4f}  "
          f"log_sigma_eps={theta_VI_obs['log_sigma_eps']:+.4f}  "
          f"L2_free={L2_VI:.4f}", flush=True)
    print(f"truth:         mu1={truth['mu1']:+.4f}  "
          f"sigma0={truth['sigma0']:+.4f}  "
          f"z1_log_std={truth['z1_log_std']:+.4f}  "
          f"log_sigma_eps={truth['log_sigma_eps']:+.4f}", flush=True)

    base_out = {
        'config':                 config,
        'truth':                  truth_block,
        'theta_VI_obs':           theta_VI_obs,
        'theta_VI_obs_trace':     vi_obs_trace,
        'L2_VI':                  L2_VI,
    }

    # ------------------------------------------------------------------
    # Phase 2: damped-Picard outer loop (picard only).
    # ------------------------------------------------------------------
    if method == 'vi_only':
        print(f"\n=== Phase 2 SKIPPED (METHOD=vi_only) ===", flush=True)
        out = {
            **base_out,
            'final_theta':  theta_VI_obs,
            'final_L2':     L2_VI,
        }
        theta_k_for_phase3 = None
    else:
        # METHOD=picard or anderson: IVI binding-equation outer loop with
        # strict CRN. The iteration solves for theta such that
        #     b(theta) = theta_VI_obs
        # where theta_VI_obs is the Phase-1 inner-VI fit on y_obs and
        # b(theta) = theta_VI(y_sim(theta)) is the binding map.
        #
        # Under strict CRN, y_sim(theta_truth) = y_obs by construction,
        # so b(theta_truth) = theta_VI(y_obs) = theta_VI_obs, i.e.
        # theta_truth IS the binding-equation fixed point. This is the
        # canonical IVI iteration matched to the hockeystick precedent;
        # see `picard_step` docstring for the (important) distinction
        # vs. standard damped Picard iteration on `b(theta) = theta`.
        #
        # CRN regime for the inner-VI map y -> theta_VI(y):
        #   * vi_seed is the SAME on every outer iter (not iter-derived);
        #     `fit_vi_inner` rebuilds the model under
        #     torch.manual_seed(vi_seed) every call, so the encoder is
        #     cold-initialised to identical weights every outer iter.
        #   * eps_fixed (the inner-VI reparam noise) is the same tensor
        #     drawn once at noise_seed; reused unchanged every iter.
        #   * AdamW is constructed fresh inside `fit_vi_inner` every
        #     call, so its momenta start at zero -- identical state at
        #     start of every outer iter.
        # The simulator side is also strict-CRN: y_sim_k = reparam_simulate(
        #   theta_k, z_noise, y_noise) reuses the fixed (z_noise,
        #   y_noise) tensors pre-drawn at OBS_SEED. Hence
        #   b(theta_k) = theta_VI(y_sim(theta_k)) is deterministic in
        #   theta_k.
        theta_k = dict(theta_VI_obs)
        L2_to_truth = L2_VI
        theta_VI_obs_vec = theta_to_vec(theta_VI_obs)
        trajectory = [{
            'iter':         0,
            'iter_wall_s':  0.0,
            'theta_k':      {kk: float(theta_k[kk]) for kk in PARAM_NAMES},
            'theta_VI':     None,
            'theta_kp1':    {kk: float(theta_k[kk]) for kk in PARAM_NAMES},
            'L2_truth':     L2_to_truth,
            'L2_step':      0.0,
            'inner_trace':  None,
            'g_norm':       None,
            'gamma':        None,
        }]

        # Anderson-only state.
        anderson_history = []  # list of {'x': np.ndarray, 'g': np.ndarray}

        if method == 'picard':
            phase2_banner = (f"\n=== Phase 2: damped-Picard outer loop "
                              f"(alpha={alpha}, {n_iters} iters, strict CRN) ===")
            iter_tag = 'pic'
        else:  # anderson
            phase2_banner = (f"\n=== Phase 2: Anderson m={m_mem} outer loop "
                              f"(alpha={alpha}, {n_iters} iters, strict CRN) ===")
            iter_tag = 'and'
        print(phase2_banner, flush=True)

        for k in range(1, n_iters + 1):
            iter_t0 = time.time()
            print(f"\n[iter {k}] simulate y_sim + inner VI "
                  f"({n_epochs} ep, mean-field cold, vi_seed={vi_seed}) ...",
                  flush=True)

            theta_t = torch.tensor([theta_k[name] for name in PARAM_NAMES],
                                    dtype=torch.float32, device=device)
            with torch.no_grad():
                y_sim = M.reparam_simulate(theta_t, z_noise, y_noise)

            theta_VI, trace, used_seed, _ = fit_vi_inner(
                y_sim, n_epochs, lr=lr, log_every=log_every,
                label=f"{iter_tag}/it{k}", vi_seed=vi_seed,
                eps_fixed=eps_fixed, encoder_label='mean_field')

            iter_wall = time.time() - iter_t0
            theta_VI_vec = theta_to_vec(theta_VI)
            x_k = theta_to_vec(theta_k)
            # IVI binding-equation residual: g_k = theta_VI_obs - b(theta_k).
            # NOT the standard fixed-point residual b(theta_k) - theta_k.
            # See picard_step docstring for the geometric distinction.
            g_k = theta_VI_obs_vec - theta_VI_vec
            g_norm = float(np.linalg.norm(g_k[M.FREE_INDICES]))

            if method == 'picard':
                x_next = picard_step(x_k, g_k, alpha)
                gamma_log = None
            else:  # anderson
                x_next, gamma_arr = anderson_step(
                    x_k, g_k, anderson_history, m=m_mem, beta=alpha)
                # Record history AFTER computing the step (anderson_step's
                # convention is to read history NOT including current x_k).
                anderson_history.append({'x': x_k.copy(), 'g': g_k.copy()})
                gamma_log = (None if gamma_arr is None
                              else [float(g) for g in gamma_arr])

            theta_next = vec_to_theta(x_next)
            L2_to_truth = L2_free(theta_next, truth)
            L2_step = float(np.linalg.norm(
                (x_next - x_k)[M.FREE_INDICES]))

            print(f"  inner VI: mu1={theta_VI['mu1']:+.4f}  "
                  f"sigma0={theta_VI['sigma0']:+.4f}  "
                  f"log_sigma_eps={theta_VI['log_sigma_eps']:+.4f}",
                  flush=True)
            print(f"  theta_{k}: mu1={theta_next['mu1']:+.4f}  "
                  f"sigma0={theta_next['sigma0']:+.4f}  "
                  f"L2_free={L2_to_truth:.4f}  L2_step={L2_step:.4f}  "
                  f"||g||={g_norm:.4f}  "
                  f"iter_wall={iter_wall:.1f}s", flush=True)

            trajectory.append({
                'iter':         k,
                'iter_wall_s':  iter_wall,
                'theta_k':      {kk: float(theta_k[kk]) for kk in PARAM_NAMES},
                'theta_VI':     {kk: float(theta_VI[kk]) for kk in PARAM_NAMES},
                'theta_kp1':    {kk: float(theta_next[kk]) for kk in PARAM_NAMES},
                'L2_truth':     L2_to_truth,
                'L2_step':      L2_step,
                'inner_trace':  trace,
                'g_norm':       g_norm,
                'gamma':        gamma_log,
            })
            theta_k = theta_next

            # Incremental write so a crash mid-Phase 2 doesn't lose work.
            out = {
                **base_out,
                'trajectory':   trajectory,
                'final_theta':  {kk: float(theta_k[kk]) for kk in PARAM_NAMES},
                'final_L2':     L2_to_truth,
            }
            with open(out_path, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"  [cache write: {out_path}]", flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        theta_k_for_phase3 = theta_k

    # ------------------------------------------------------------------
    # Phase 3: post-fit diagnostics.
    # ------------------------------------------------------------------
    print(f"\n=== Phase 3: post-fit diagnostics ===", flush=True)

    # 3a. ELBO + IWAE at theta_VI_obs using the phase-1 encoder.
    print(f"\n[3a] ELBO/IWAE at theta_VI_obs (phase-1 encoder, "
          f"L={L_iwae}) ...", flush=True)
    diag_vi = model_VI_obs.elbo_diagnostics(y_obs, ndraws=L_iwae)
    elbo_at_vi = float(diag_vi['elbo'])
    iwae_at_vi = float(diag_vi['iwae'])
    print(f"  elbo_at_vi={elbo_at_vi:+.4f}  iwae_at_vi={iwae_at_vi:+.4f}  "
          f"gap={iwae_at_vi - elbo_at_vi:+.4f}", flush=True)
    del model_VI_obs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3b. ELBO + IWAE at theta_K = final IVI estimate (picard only).
    if method == 'vi_only':
        elbo_at_ivi = None
        iwae_at_ivi = None
    else:
        print(f"\n[3b] fit encoder@theta_K + ELBO/IWAE "
              f"({n_epochs_encoder} ep, lr={lr_encoder}) ...", flush=True)
        model_ivi = fit_encoder_only_at_theta(
            theta_k_for_phase3, y_obs, T, vi_seed,
            n_epochs=n_epochs_encoder, lr=lr_encoder, device=device)
        diag_ivi = model_ivi.elbo_diagnostics(y_obs, ndraws=L_iwae)
        elbo_at_ivi = float(diag_ivi['elbo'])
        iwae_at_ivi = float(diag_ivi['iwae'])
        print(f"  elbo_at_ivi={elbo_at_ivi:+.4f}  "
              f"iwae_at_ivi={iwae_at_ivi:+.4f}  "
              f"gap={iwae_at_ivi - elbo_at_ivi:+.4f}", flush=True)
        del model_ivi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3c. ELBO + IWAE at theta_truth with a fresh joint-normal h=32 encoder.
    print(f"\n[3c] fit encoder@truth + ELBO/IWAE "
          f"({n_epochs_encoder} ep, lr={lr_encoder}) ...", flush=True)
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

    diagnostics = {
        'elbo_at_vi':         elbo_at_vi,
        'iwae_at_vi':         iwae_at_vi,
        'elbo_at_truth':      elbo_at_truth,
        'iwae_at_truth':      iwae_at_truth,
        'smc_log_p_at_truth': smc_log_p_at_truth,
    }
    if method in ('picard', 'anderson'):
        diagnostics['elbo_at_ivi'] = elbo_at_ivi
        diagnostics['iwae_at_ivi'] = iwae_at_ivi

    out['diagnostics'] = diagnostics
    out['wall_s'] = time.time() - t0_total
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    if method == 'vi_only':
        print(f"\nDone. L2_VI={L2_VI:.4f}", flush=True)
    else:
        print(f"\nDone. Final L2_free={trajectory[-1]['L2_truth']:.4f}",
              flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    # Fix every RNG (python / numpy / torch CPU+CUDA) and request deterministic
    # kernels before any draw; the script's own explicit seeds still apply.
    from mlye.seeding import seed_everything, env_seed
    seed_everything(env_seed("OBS_SEED", 11))
    main()
