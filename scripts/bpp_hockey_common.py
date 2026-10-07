"""Shared helpers for the BPP-hockey catalogue (specs/psid_bpp_hockey.md).

The main export is ``compute_iwae_ref_log_p``: a reference marginal
log-likelihood evaluator that **correctly accounts for MA(1)**.

Why a custom helper instead of the SMC bootstrap PF (the convention
used by the ABB sibling in psid_estimates.md)?

The bootstrap particle filter weights particles by the marginal
emission density ``p(y_t | z_t)``. Under MA(1) the conditional
``p(y_t | z_t, y_{1:t-1})`` is NOT marginal in ``t``: it depends on
the lagged residual ``u_{t-1} = y_{t-1} - z_{t-1} - theta * u_{t-2}``.
At ``theta != 0`` the bootstrap PF is mis-specified and its
estimate of ``log p(y; theta)`` is biased.

The IWAE bound, by contrast, uses ``decoder.log_likelihood(y, z)``
which is the **joint** emission density across t (full MA(1)
Cholesky in MASinhEmission). The bound is therefore correct under
MA(1); the only error is the IWAE Jensen gap, which is below
estimation noise at K=200 with a converged joint-normal encoder.

Behaviour:
- If an `encoder` is passed (VI / IVI entries), uses it verbatim.
- If `encoder is None` (SMC-EM / FIVO entries), builds a fresh
  joint-normal encoder and trains it with prior + decoder **frozen**
  for `train_epochs` iterations. The trained encoder is then used
  for the IWAE bound at the entry's converged theta.

This matches the spec's intent: the reference is ``log p(y; theta_hat)``
where ``theta_hat`` is the entry's own converged point estimate,
not a re-fit.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from mlye.models.encoders import JointNormalConfig
from mlye.models.full_model import FullModel
from mlye.eval import FilteringVariationalObjective

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smc_truth_profile import _Bundle  # noqa: E402
import psid_smc_em_fivo_sinhz1 as _base  # noqa: E402


def compute_iwae_ref_log_p(
    prior,
    decoder,
    y: torch.Tensor,
    encoder=None,
    *,
    K: int = 200,
    train_epochs: int = 4000,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    clip: float = 5.0,
    seed: int = 11,
    verbose: bool = True,
) -> tuple[float, dict]:
    """Compute IWAE-at-converged-theta as the reference log p(y).

    Args:
        prior: trained prior module (e.g. MarkovNormalConditionalPolyPrior).
        decoder: trained decoder module (e.g. MASinhEmission).
        y: (N, T) panel.
        encoder: optional. If provided, used directly (VI/IVI case). If None,
            a fresh joint-normal encoder is built and trained with prior +
            decoder frozen (SMC-EM / FIVO case).
        K: number of importance samples in the IWAE bound.
        train_epochs: epochs for the encoder-only training (used only when
            ``encoder is None``).
        lr, hidden_dim, clip, seed: encoder-only training hyperparameters.
        verbose: whether to print progress.

    Returns:
        (iwae_log_p, meta) — the IWAE bound (a scalar float) and a small
        metadata dict for the entry's JSON.
    """
    T = y.shape[1]
    meta: dict = {
        "K": K,
        "ndraws_iwae": K,
        "method": "iwae_at_converged_theta",
        "note": (
            "Reference log p(y; theta_hat). The bootstrap PF used by "
            "the ABB sibling is biased under MA(1) because the weights "
            "use the marginal emission density; we use IWAE instead, "
            "which evaluates decoder.log_likelihood (joint MA(1) "
            "Cholesky)."
        ),
    }

    if encoder is None:
        # SMC-EM / FIVO case: build + train a joint-normal encoder with
        # prior and decoder frozen at the entry's converged theta.
        prior_required = [p.requires_grad for p in prior.parameters()]
        decoder_required = [p.requires_grad for p in decoder.parameters()]
        for p in prior.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = False

        torch.manual_seed(seed)
        np.random.seed(seed)
        encoder = (
            JointNormalConfig(
                dim=T,
                type="joint_normal",
                regularize=1e-3,
                hidden_dim=hidden_dim,
            )
            .build()
            .to(y.device)
        )

        model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(y.device)
        enc_params = list(encoder.parameters())
        opt = torch.optim.AdamW(enc_params, lr=lr)

        if verbose:
            print(
                f"[iwae-ref] training joint-normal encoder "
                f"(prior+decoder frozen, hidden={hidden_dim}, "
                f"epochs={train_epochs}, lr={lr})",
                flush=True,
            )
        t0 = time.time()
        for epoch in range(train_epochs):
            opt.zero_grad()
            loss = -model.elbo(y, ndraws=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_params, clip)
            opt.step()
            if verbose and (epoch % 500 == 0 or epoch == train_epochs - 1):
                print(
                    f"  [iwae-ref] epoch {epoch:5d}  ELBO={-loss.item():.4f}",
                    flush=True,
                )
        if verbose:
            print(f"  [iwae-ref] encoder trained in {time.time()-t0:.1f}s",
                  flush=True)

        meta["encoder_built_from_scratch"] = True
        meta["encoder_hidden_dim"] = hidden_dim
        meta["encoder_train_epochs"] = train_epochs

        # Restore parameter grads (the entry's prior/decoder might still
        # be referenced elsewhere).
        for p, flag in zip(prior.parameters(), prior_required):
            p.requires_grad = flag
        for p, flag in zip(decoder.parameters(), decoder_required):
            p.requires_grad = flag
    else:
        model = FullModel(encoder=encoder, decoder=decoder, prior=prior).to(y.device)
        meta["encoder_built_from_scratch"] = False

    # IWAE bound at K samples.
    with torch.no_grad():
        iwae = float(model.elbo_iwae(y, ndraws=K).item())
    if verbose:
        print(f"[iwae-ref] IWAE log p(y) at K={K}: {iwae:.4f}", flush=True)

    meta["value"] = iwae
    return iwae, meta


def compute_bhhh_se_fivo(
    prior,
    decoder,
    y: torch.Tensor,
    K: int = 500,
    n_reps: int = 3,
    seed_base: int = 11,
):
    """BHHH SE via FIVO scores, with ``allow_unused=True`` to gracefully
    handle parameters the bound is flat in.

    The BPP-hockey catalogue leaves ``decoder.theta`` free in every
    entry, but the bootstrap-PF FIVO bound uses ``decoder.get_distribution()``
    (the *marginal* sinh-arcsinh density) — it does not depend on
    ``theta``. Stock ``base.compute_bhhh_se`` raises "differentiated
    Tensor not used in graph" in that case. This variant catches the
    ``None`` gradient and substitutes a zero score row, so ``theta``'s
    SE comes out 0 / pseudo-inverse-derived (flagging it as
    unidentified by FIVO scores). Every other parameter is identified
    normally.
    """
    bundle = _Bundle(prior=prior, decoder=decoder)
    fivo = FilteringVariationalObjective(bundle, K=K)
    params = [p for _, p in _base.named_param_list(prior, decoder)]
    N = y.shape[0]
    P = sum(p.numel() for p in params)
    device = y.device

    scores_avg = torch.zeros(N, P, device=device, dtype=torch.float64)
    t0 = time.time()
    for rep in range(n_reps):
        torch.manual_seed(seed_base * 1000 + rep)
        for i in range(N):
            y_i = y[i:i + 1]
            log_p_i = fivo.fivo_bound(y_i).mean()
            grads = torch.autograd.grad(
                log_p_i, params, retain_graph=False, allow_unused=True
            )
            parts = []
            for g, p in zip(grads, params):
                if g is None:
                    parts.append(torch.zeros(p.numel(), device=device,
                                             dtype=torch.float64))
                else:
                    parts.append(g.detach().flatten().double())
            s_i = torch.cat(parts)
            scores_avg[i] += s_i / n_reps
        print(f"  BHHH rep {rep+1}/{n_reps} done ({time.time()-t0:.1f}s)",
              flush=True)

    I_hat = scores_avg.T @ scores_avg
    I_hat_np = I_hat.cpu().numpy()
    cov = np.linalg.pinv(I_hat_np)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return se, I_hat_np
