use sundials::CVode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ode = CVode::adams(0., &[0.],
                               |_t, _u, du| *du = [1.])?;
    let mut u1 = [f64::NAN];
    ode.solve(1., &mut u1);
    assert_eq!(u1[0], 1.);
    Ok(())
}
