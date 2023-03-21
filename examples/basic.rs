use sundials::CVode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let f = |_t, _u: &[f64; 1], du: &mut [f64; 1]| {
        *du = [1.]
    };
    let mut ode = CVode::adams(f, 0., &[0.])?;
    let mut u1 = [f64::NAN];
    ode.solve(1., &mut u1);
    assert_eq!(u1[0], 1.);
    Ok(())
}
