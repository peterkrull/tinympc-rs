# tinympc-rs
Rust port of [TinyMPC](https://github.com/TinyMPC/TinyMPC) with support for `no_std` environments

## Desired future modifications
- Only define trajectory on subset of (transformed) states. Do not penalize states which should be considered "free variables".
- Allow for defining both control and prediction horizon. This can give a more stable trajectory towards end of control horizon.
- Incorporate integral action by solving for $\Delta u_k$ rather than $u_k$. This involves augmenting the state space system.