use sundials::{context, cvode::{CVode, Solver}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ode = CVode::adams(0., &[0.],
        |_t, _u, du| *du = [1.])
        .build(context!()?)?;
    let mut u = [f64::NAN];
    let (t, st) = ode.step(1., &mut u);
    println!("t = {t:e}, u = {u:?}, status: {st:?}");
    assert_eq!(u[0], t);
    let st = ode.solve(1., &mut u);
    println!("t = 1., u = {u:?}, status: {st:?}");
    assert_eq!(u[0], 1.);
    Ok(())
}
