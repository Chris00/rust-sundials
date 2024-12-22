use std::time::Instant;
use sundials::{context, cvode::{CVode, Solver}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let now = Instant::now();
    for _ in 0 .. 100_000 {
        let mut ode = CVode::adams(0., &[0.],
            |_t, _u, du| *du = [1.])
            .build(context!()?)?;
        let mut u = [f64::NAN];
        ode.solve(1., &mut u);
        // let mut ode = CVode::adams(0., &(0., 0.),
        //     |_t, _u, du| *du = (1., 0.))
        //     .build(context!()?)?;
        // let mut u = (f64::NAN, 0.);
        // ode.solve(1., &mut u);
    }
    println!("Elapsed: {}", now.elapsed().as_secs_f64());
    Ok(())
}
