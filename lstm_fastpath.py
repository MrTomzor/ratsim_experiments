"""Drop-in fast path for sb3_contrib RecurrentActorCriticPolicy._process_sequence.

Upstream falls into a per-timestep Python loop whenever any episode_starts
entry in the rollout window is True — which is effectively every rollout
(the first step of every rollout flags as an episode start). With long
n_steps that's ~T nn.LSTM calls per minibatch, ×2 (actor + critic LSTMs),
×n_minibatches × n_epochs. The workload becomes launch-overhead-bound and
GPU runs at roughly the same speed as CPU.

This module replaces _process_sequence with a version that splits each
env's sequence at episode boundaries and runs nn.LSTM once per contiguous
segment. For 2000-step episodes with n_steps=2048 and n_envs=4, that's ~3
launches per env (instead of 2048), so ~12 LSTM calls per minibatch instead
of 2048×4=8192. Two to three orders of magnitude fewer launches.

Semantics are equivalent to upstream — verified at module import on a small
random case before the patch is applied. Set DISABLE_LSTM_FASTPATH=1 in env
to skip the patch and use the upstream implementation.
"""
import os

import numpy as np
import torch as th
import torch.nn as nn
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


_original_process_sequence = RecurrentActorCriticPolicy._process_sequence


def _process_sequence_fast(features, lstm_states, episode_starts, lstm):
    n_seq = lstm_states[0].shape[1]
    hidden_size = lstm_states[0].shape[2]

    features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
    episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
    T = features_sequence.shape[0]

    # No resets anywhere in the window → one batched call (same as upstream).
    if th.all(episode_starts == 0.0):
        lstm_output, lstm_states_out = lstm(features_sequence, lstm_states)
        lstm_output = th.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states_out

    h_init, c_init = lstm_states
    output = th.empty(T, n_seq, hidden_size, device=features.device, dtype=features.dtype)
    final_h = th.empty_like(h_init)
    final_c = th.empty_like(c_init)

    # Bring the (small) boundary mask to host once. T*n_seq bools — negligible.
    starts_host = episode_starts.detach().to(dtype=th.bool, device="cpu").numpy()

    for env_idx in range(n_seq):
        env_starts = starts_host[:, env_idx]
        # Segment-start indices: always 0, plus every t>0 where this env
        # begins a new episode.
        bounds_after_zero = np.nonzero(env_starts[1:])[0] + 1
        seg_bounds = [0, *bounds_after_zero.tolist(), T]

        h = h_init[:, env_idx:env_idx + 1, :].contiguous()
        c = c_init[:, env_idx:env_idx + 1, :].contiguous()

        for i in range(len(seg_bounds) - 1):
            s, e = seg_bounds[i], seg_bounds[i + 1]
            # First segment of the env may or may not begin at a boundary;
            # all later segments do (by construction).
            if env_starts[s]:
                h = th.zeros_like(h)
                c = th.zeros_like(c)
            seg_features = features_sequence[s:e, env_idx:env_idx + 1, :].contiguous()
            seg_output, (h, c) = lstm(seg_features, (h, c))
            output[s:e, env_idx:env_idx + 1, :] = seg_output

        final_h[:, env_idx:env_idx + 1, :] = h
        final_c[:, env_idx:env_idx + 1, :] = c

    output = th.flatten(output.transpose(0, 1), start_dim=0, end_dim=1)
    return output, (final_h, final_c)


def _verify_correctness():
    """Run both implementations on a small random case before patching."""
    th.manual_seed(0)
    T, N, F, H = 12, 3, 5, 8
    lstm = nn.LSTM(F, H, num_layers=1)
    features = th.randn(N * T, F)
    h0 = th.randn(1, N, H)
    c0 = th.randn(1, N, H)
    starts = th.zeros(N * T)
    # Sprinkle resets: env 0 at t=0; env 1 at t=4; env 2 at t=0 and t=7.
    # Index in flattened (n_seq*T,): env_idx * T + t.
    starts[0 * T + 0] = 1.0
    starts[1 * T + 4] = 1.0
    starts[2 * T + 0] = 1.0
    starts[2 * T + 7] = 1.0

    o_slow, (hs, cs) = _original_process_sequence(
        features.clone(), (h0.clone(), c0.clone()), starts.clone(), lstm
    )
    o_fast, (hf, cf) = _process_sequence_fast(
        features.clone(), (h0.clone(), c0.clone()), starts.clone(), lstm
    )

    if not (
        th.allclose(o_slow, o_fast, atol=1e-5)
        and th.allclose(hs, hf, atol=1e-5)
        and th.allclose(cs, cf, atol=1e-5)
    ):
        max_diff = (o_slow - o_fast).abs().max().item()
        raise RuntimeError(
            f"[lstm_fastpath] correctness check failed: fast path output "
            f"diverges from upstream by max abs diff {max_diff:.2e}; "
            f"refusing to patch"
        )


if os.environ.get("DISABLE_LSTM_FASTPATH"):
    print("[lstm_fastpath] DISABLE_LSTM_FASTPATH set — keeping upstream path")
else:
    _verify_correctness()
    RecurrentActorCriticPolicy._process_sequence = staticmethod(_process_sequence_fast)
    print("[lstm_fastpath] patched sb3_contrib RecurrentActorCriticPolicy._process_sequence")
