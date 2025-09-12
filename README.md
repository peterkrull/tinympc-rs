# tinympc-rs
Rust port of [TinyMPC](https://github.com/TinyMPC/TinyMPC) with support for `no_std` environments

## Showcase

Given some discrete time state space model $x_{k+1} = A x_k + B u_k$ with $N_x$ states and $N_u$ inputs, we may write this in code using consts and macros (from `nalgebra`) for convenience.

```rust
use nalgebra as na;

// Number of states and inputs
const NX: usize = .. ;
const NU: usize = .. ;

// System dynamics
const A: na::SMatrix<f32, NX, NX> = na::matrix![..];
const B: na::SMatrix<f32, NX, NU> = na::vector![..];
```

For LQR and MPC we typically have a cost matrix associated with the state and input deviating from their references. For simplicity these are defined using vectors representing the diagonal of such matrices.


```rust
// State and input error cost vectors
const Q: na::SVector<f32, NX> = na::vector![..];
const R: na::SVector<f32, NU> = na::vector![..];
```

We define the prediction horizon length $H_x$ and control horizon length $H_u$ where $H_x > H_u$. We also choose an ADMM penalty parameter $\rho$ (`rho`), and choose a caching strategy. Here we use the const-sized `LookupCache` with 5 elements.

```rust
// Set the prediction and control horizon length
const HX: usize = .. ;
const HU: usize = .. ;

// The ADMM penalty parameter. Does not have to be const
let rho = 2.0;

// Define shorthands for cache and mpc types
type Cache = LookupCache<f32, NX, NU, 5>;
type Mpc = TinyMpc<f32, Cache, NX, NU, HX, HU>;

// Run all precomputations, and unwrap on error if
// something went wrong or was misconfigured.
let mut mpc = Mpc::new(A, B, Q, R, rho).unwrap();

// Run a maximum of --vv-- iterations per solve
mpc.config.max_iter = 15;
```

The power of MPC comes from its ability to handle constraints. Constraints in `tinympc-rs` are flexible and composable thanks to rusts Trait system:
1. A single constraint holds dual/slack variables and a "projector"
2. Projectors implement the `Project` trait and push points into feasible regions
3. Multiple projectors can be combined using tuples or arrays

```rust
// The constraints are defined by so called "projectors"
// which are just types that are able to push the state
// or inputs back into the feasible domain.
let x_projector = (
    Sphere {
        center: ..,
        radius: ..,
    },
    Box {
        upper: ..,
        lower: ..,
    }
);

// Or we can just use a single projector
let u_projector = Box {
    upper: ..,
    lower: ..,
};

// The constraint() method automatically creates and manages
// the dual and slack variables needed by the optimizer
// internally. These are arraya since we can also provide
// multiple individual constraints.
let mut x_con = [x_projector.constraint()];
let mut u_con = [u_projector.constraint()];
```

With everything set up, we just need to run it! For this example we want our states to track a reference trajectory defined by the matrix `x_ref` below. We also have the option to provide a reference to the inputs. Now, all of the magic (math and optimizations) lies under the hood of that call to `solve()`.

```rust
loop {
    // Recalculate/shift the reference one step forward
    let xref: na::SMatrix<f32, NX, HX>; = //..

    // Get our state for this run
    system.next_sampling_time().await;
    let x_now = system.read_state();

    // Feed the initial condition, constraints and reference
    // into the MPC and run `solve()` to start converging
    // towards the optimal solution!
    let (reason, u_now) = mpc
        .initial_condition(xnow)
        .u_constraints(u_con.as_mut())
        .x_constraints(x_con.as_mut())
        .x_reference(x_ref.as_view())
        .solve();

    // Use the actuation command
    system.set_output(u_now);
}
```
