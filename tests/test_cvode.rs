use sundials::{context, cvode::{CVode, CVStatus}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = context!().unwrap();
    let mut ode = CVode::adams(ctx, 0., &[0.],
        |_t, _u, du| *du = [1.]).build()?;
    let mut u = [f64::NAN];
    let st = ode.solve(1., &mut u);
    assert_eq!(st, CVStatus::Ok);
    assert_eq!(u[0], 1.);

    let ctx = ode.into_context();
    let mut ode = CVode::adams(ctx, 0., &vec![0.],
        |_t, _u, du| du[0] = 1.)
        .build()?;
    let mut u = vec![f64::NAN];
    let st = ode.solve(1., &mut u);
    assert_eq!(st, CVStatus::Ok);
    assert_eq!(u[0], 1.);

    Ok(())
}
