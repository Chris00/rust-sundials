Rust Sundials
=============

[Sundials][] is a s̲u̲ite of n̲onlinear and d̲i̲fferential/a̲l̲gebraic
equation s̲olvers.


# Example

The following code solves the equation ∂ₜu = f(t,u) where f is the
function (t,u) ↦ 1 using Adams' method.

```rust
use sundials::CVode;
let mut ode = CVode::adams(0., &[0.], |t, u, du| *du = [1.])?;
let (u1, _) = ode.solution(1.);
assert_eq!(u1[0], 1.);
```

Then `u[0]` contains the solution u at time t=1 and `u[1]` the
speed ∂ₜu(1).


[Sundials]: https://computing.llnl.gov/projects/sundials
