use sundials::{context, cvode::{CVode, CVStatus}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ode = CVode::adams(0., &[0.],
        |_t, _u, du| *du = [1.])
        .build(context!()?)?;
    let mut u = [f64::NAN];
    let st = ode.solve(1., &mut u);
    assert_eq!(st, CVStatus::Ok);
    assert_eq!(u[0], 1.);

    let ctx = ode.into_context();
    let u0 = vec![0.];
    let mut ode = CVode::adams(0., &u0,
        |_t, _u, du| du[0] = 1.)
        .build(ctx)?;
    let mut u = vec![f64::NAN];
    let st = ode.solve(1., &mut u);
    assert_eq!(st, CVStatus::Ok);
    assert_eq!(u[0], 1.);

    Ok(())
}
