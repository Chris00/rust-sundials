Rust Sundials
=============

[Sundials][] is a s̲u̲ite of n̲onlinear and d̲i̲fferential/a̲l̲gebraic
equation s̲olvers.


# Example

The following code solves the equation ∂ₜu = f(t,u) where f is the
function (t,u) ↦ 1 using Adams' method.

```rust
use sundials::CVode;
let f = |t, u: &[f64; 1], du: &mut [f64; 1]| { *du = [1.] };
let ode = CVode::adams(f, 0., &[0.])?;
let mut u1 = [f64::NAN];
ode.solve(1., &mut u1);
```

Then `u[0]` contains the solution u at time t=1 and `u[1]` the
speed ∂ₜu(1).


[Sundials]: https://computing.llnl.gov/projects/sundials
