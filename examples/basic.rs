use sundials::{context, cvode::CVode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = context!().unwrap();
    let mut ode = CVode::adams(ctx, 0., &[0.],
        |_t, _u, du| *du = [1.])?;
    let mut u1 = [f64::NAN];
    ode.solve(1., &mut u1);
    assert_eq!(u1[0], 1.);
    Ok(())
}
