r"""Headline JSON bundle for the BPP hetero-scale catalogue.

Entry: psid-bpp-hetero-scale-bundle
(see specs/compute-psid_bpp_hetero_scale.md).

Pure aggregation — no fitting. Collects into one self-contained JSON:

  fits.vi_mf / fits.vi_jn / fits.vi_sm   the three K=40 VI fits
  fits.ivi_jn                            the K=40 IVI-JN canonical run
                                          (Picard 0.6, 50 outer iters)
  bootstrap.unconditional                B=100 fresh-seed full-IVI
                                          bootstrap: replicates matrix,
                                          se, ci95, paired
                                          decomposition, E[g]=0 test —
                                          the REPORTABLE inference
  bootstrap.conditional                  common-seed sibling (se/ci95
                                          reference; no replicates
                                          matrix, see its own artefact)
  provenance                             source paths + run ids

Skips missing inputs with a warning so it can rerun any time an input
refreshes.

Output: output/bundles/bundle_psid_heteroscale.json
"""
import os
import json

OUT_DIR = "output/bundles"
OUT_JSON = os.path.join(OUT_DIR, "bundle_psid_heteroscale.json")

FIT_INPUTS = {
    'vi_mf': "output/bundles/cells/employment-bpp-hetero-scale-vi-mf/bpp_hetero_scale_vi_mf.json",
    'vi_jn': "output/bundles/cells/employment-bpp-hetero-scale-vi-jn/bpp_hetero_scale_vi_jn.json",
    'vi_sm': "output/bundles/cells/employment-bpp-hetero-scale-vi-sm/bpp_hetero_scale_vi_sm.json",
    'ivi_jn': ("output/bundles/cells/employment-bpp-hetero-scale-ivi-jn-k40-picard-a06-it50/"
               "bpp_hetero_scale_ivi_jn.json"),
}
BOOT_UNCOND = ("output/bundles/cells/_bpp_hetero_scale_ivi_bootstrap_k40_uncond/"
               "bootstrap_theta.json")
BOOT_COND = "output/bundles/cells/_bpp_hetero_scale_ivi_bootstrap_k40/bootstrap_theta.json"

RUN_IDS = {
    'ivi_jn': 'e38ea352',
    'bootstrap_conditional': ['823d93c3', '53ef9b9d'],
    'bootstrap_unconditional': ['84563d64', '34f42135'],
}


def _load(path):
    if not os.path.exists(path):
        print(f"[warn] missing input: {path} — skipped", flush=True)
        return None
    with open(path) as f:
        return json.load(f)


def _fit_block(doc, method_key):
    """Uniform per-fit block from a catalogue fit JSON."""
    est = doc['estimates'][method_key]
    train = doc['training'][method_key]
    block = {
        'raw': est.get('raw'),
        'derived': est.get('derived'),
        'iwae_log_p': est.get('iwae_log_p'),
        'training': {k: train.get(k) for k in
                     ('encoder', 'encoder_config', 'hidden_dim',
                      'n_epochs', 'lr', 'ndraws', 'seed_base',
                      'n_epochs_vi_obs', 'n_epochs_inner',
                      'n_iters_outer', 'picard_alpha', 'method',
                      'n_sim', 'noise_seed', 'vi_seed')
                     if k in train},
        'se_method': doc.get('se_method'),
    }
    # IVI carries an explicit theta vector + trajectory.
    if 'theta_K' in est:
        block['theta'] = est['theta_K']
        block['theta_VI_obs'] = est.get('theta_VI_obs')
        block['g_norm_trajectory'] = [r.get('g_norm')
                                      for r in est.get('trajectory', [])]
    return block


def main():
    bundle = {'fits': {}, 'bootstrap': {}, 'provenance': {
        'inputs': dict(FIT_INPUTS,
                       bootstrap_unconditional=BOOT_UNCOND,
                       bootstrap_conditional=BOOT_COND),
        'scripthut_runs': RUN_IDS,
        'spec_entry': 'psid-bpp-hetero-scale-bundle',
        'journal': 'journal/2026-07-10-bpp-ivi-bootstrap-standard-errors.md',
    }}

    data_block = None
    for tag, path in FIT_INPUTS.items():
        doc = _load(path)
        if doc is None:
            continue
        method_key = 'ivi' if tag == 'ivi_jn' else tag
        bundle['fits'][tag] = _fit_block(doc, method_key)
        if data_block is None:
            data_block = doc['data']
        # K=40 guard: every bundled fit must be a multi-draw fit.
        nd = bundle['fits'][tag]['training'].get('ndraws')
        if nd != 40:
            print(f"[warn] {tag}: ndraws={nd} (expected 40)", flush=True)
    bundle['data'] = data_block

    unc = _load(BOOT_UNCOND)
    if unc is not None:
        bundle['bootstrap']['unconditional'] = {
            'note': 'REPORTABLE inference: fresh sim seeds per replicate '
                    '(data + simulator path + K=40 integration + init '
                    'jitter). For transformations h(theta): evaluate h '
                    'row-wise on replicates, take percentiles.',
            'B': unc['meta']['B'],
            'param_names': unc['meta']['param_names'],
            'rep_indices': unc['rep_indices'],
            'replicates': unc['replicates'],
            'point_estimate': unc['point_estimate'],
            'boot_mean': unc['boot_mean'],
            'se': unc['se'],
            'ci95': unc['ci95'],
            'paired_decomposition': unc.get('paired_decomposition'),
            'final_g_test': unc.get('final_g_test'),
        }
    cond = _load(BOOT_COND)
    if cond is not None:
        bundle['bootstrap']['conditional'] = {
            'note': 'Common production seed across replicates (data-only '
                    'spread; sim noise + jitter cancel). Reference column.',
            'B': cond['meta']['B'],
            'se': cond['se'],
            'ci95': cond['ci95'],
            'final_g_test': cond.get('final_g_test'),
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(bundle, f, indent=1)
    print(f"wrote {OUT_JSON}", flush=True)
    print(f"  fits: {sorted(bundle['fits'].keys())}", flush=True)
    print(f"  bootstrap blocks: {sorted(bundle['bootstrap'].keys())}",
          flush=True)


if __name__ == "__main__":
    main()
