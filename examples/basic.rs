use sundials::{context, cvode::CVode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = context!().unwrap();
    let mut ode = CVode::adams(ctx, 0., &[0.],
        |_t, _u, du| *du = [1.]).build()?;
    let mut u = [f64::NAN];
    let (t, st) = ode.step(1., &mut u);
    println!("t = {t:e}, u = {u:?}, status: {st:?}");
    assert_eq!(u[0], t);
    let st = ode.solve(1., &mut u);
    println!("t = 1., u = {u:?}, status: {st:?}");
    assert_eq!(u[0], 1.);
    Ok(())
}
