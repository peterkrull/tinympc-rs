# tinympc-rs
Rust port of [TinyMPC](https://github.com/TinyMPC/TinyMPC) with support for `no_std` environments

## Desired future modifications
- Only define trajectory on subset of (transformed) states, i.e. $z = Cx$. This will not penalize states which should be considered "free variables" and will make the trajectory matrix more readable.
- Allow for defining both control horizon $H_c$ and prediction horizon $H_p$ where $H_c < H_p$ and u_{k} = u_{k+1} for $H_c < k < H_p - 1$. This can give a more stable trajectory towards end of control horizon.
- Incorporate integral action by solving for $\Delta u_k$ rather than $u_k$. This involves augmenting the state space system and presenting $u_k = u_{k-1} + \Delta u_k$ to the user.