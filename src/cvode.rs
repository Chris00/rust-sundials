//! `Cvode` is a solver for stiff and nonstiff ODE systems ẏ = f(t,y)
//! with root detection and projection on a constraint manifold.
//! Based on Adams and BDF methods.
//!
//! # Example
//!
//! ```
//! use sundials::{context, cvode::CVode};
//! let mut ode = CVode::adams(0., &[0.], |t, u, du| *du = [1.])
//!     .build(context!(P1)?)?;
//! let (u1, _) = ode.cauchy(0., &[0.], 1.);
//! assert_eq!(u1[0], 1.);
//! # Ok::<(), sundials::Error>(())
//! ```

use std::{
    borrow::Borrow,
    ffi::{c_int, c_void, c_long},
    marker::PhantomData, ptr,
};
use sundials_sys::*;
use super::{
    Context,
    Error,
    vector::Vector,
    matrix::Matrix,
    linear_solver::LinSolver,
};

/// Configuration of a CVode solver.
#[derive(Clone)]
pub struct CVodeConf<'a, V, F, G, const M: usize>
where V: Vector {
    name: &'static str,
    lmm: c_int,
    t0: f64,
    y0: &'a V,
    f: F,
    g: G,
    rtol: f64,
    atol: f64,
    tstop: Option<f64>,
    maxord: Option<u8>,
    mxsteps: Option<i64>,
    max_hnil_warns: i32,
}

type RootFn<V, const M: usize> = fn(f64, &V, &mut [f64; M]);

impl<'a, V, F> CVodeConf<'a, V, F, (), 0>
where
    V: Vector,
    F: FnMut(f64, &V, &mut V),
{
    fn dummy_g(_: f64, _: &V, _: &mut [f64; 0]) {}

    fn new(
        name: &'static str, lmm: c_int,
        t0: f64, y0: &'a V, f: F,
    ) -> CVodeConf<'a, V, F, RootFn<V, 0>, 0> {
        CVodeConf {
            name, lmm,
            t0, y0,
            f,  g: Self::dummy_g,
            rtol: 1e-6,
            atol: 1e-12,
            tstop: None,
            maxord: None,
            mxsteps: None,
            // https://sundials.readthedocs.io/en/latest/cvode/Usage/index.html#c.CVodeSetMaxHnilWarns
            max_hnil_warns: 10, // Default
        }
    }
}

impl<'a, V, F, G, const M: usize> CVodeConf<'a, V, F, G, M>
where
    V: Vector,
    F: FnMut(f64, &V, &mut V),
    G: FnMut(f64, &V, &mut [f64; M]),
{
    /// Set the relative tolerance.
    pub fn rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol.max(0.);
        self
    }

    /// Set the absolute tolerance.
    pub fn atol(mut self, atol: f64) -> Self {
        self.atol = atol.max(0.);
        self
    }

    /// Set the maximum order of the method.  The default is 5 for
    /// [`CVode::bdf`] and 12 for [`CVode::adams`].
    pub fn maxord(mut self, o: u8) -> Self {
        self.maxord = Some(o);
        self
    }

    /// Specify the maximum number of steps to be taken by the solver
    /// in its attempt to reach the next output time.  Default: 500.
    // FIXME: make sure "mxstep steps taken before reaching tout" does
    // not abort the program.
    pub fn mxsteps(mut self, n: usize) -> Self {
        let n =
            if n <= c_long::MAX as usize { n as _ } else { c_long::MAX };
        self.mxsteps = Some(n);
        self
    }

    /// Specifies the value `tstop` of the independent variable t past
    /// which the solution is not to proceed.  Return `false` if
    /// `tstop` is not beyond the current t value.
    // Using [`f64::NAN`] disables the stop time.
    // (Requires version 6.5.1)
    pub fn set_tstop(mut self, tstop: f64) -> Self {
        self.tstop = Some(tstop);
        self
    }

    /// Specifies the maximum number of messages issued by the solver
    /// warning that t + h = t on the next internal step.
    pub fn max_hnil_warns(mut self, n: i32) -> Self {
        self.max_hnil_warns = n;
        self
    }

    /// Specifies that the roots of a set of functions gᵢ(t, y),
    /// 0 ≤ `i` < `N` (given by `g`(t,y, [g₁,...,gₙ])) are to be
    /// found while the IVP is being solved.
    ///
    pub fn root<const MG: usize, R>(self, g: R) -> CVodeConf<'a, V, F, R, MG>
    where R: FnMut(f64, &V, &mut [f64; MG]) {
        // It would be natural to want `[f64; Mg]` to be replaced by
        // `V`.  However, the C function (see `cvroot1`) passed to
        // Sundials will only provide `*mut f64` and not a `N_Vector`.
        // Since we do not know how this was allocated (it is not an
        // N_Vector), no other interface can be provided.
        CVodeConf {
            name: self.name,  lmm: self.lmm,
            t0: self.t0,  y0: self.y0,
            f: self.f,  g,
            rtol: self.rtol,
            atol: self.atol,
            tstop: self.tstop,
            maxord: self.maxord,
            mxsteps: self.mxsteps,
            max_hnil_warns: self.max_hnil_warns,
        }
    }

    /// Build the ODE solver with the [`Context`] `ctx`.
    pub fn build<Ctx: Context>(
        self, ctx: Ctx,
    ) -> Result<CVode<Ctx, V, F, G>, super::Error> {
        // All configuration errors are reported by this function,
        // which is easier for the user.
        let cvode_mem = unsafe {
            CVodeMem(CVodeCreate(self.lmm, ctx.as_ptr())) };
        if cvode_mem.0.is_null() {
            return Err(Error::Failure{
                name: self.name,
                msg: "Allocation of the ODE structure failed." })
        }
        // let n = V::len(y0);
        // SAFETY: Once `y0` has been passed to `CVodeInit`, it is
        // copied to internal structures and thus can be freed.
        let y0 = self.y0.borrow();
        let y0 =
            match unsafe { V::as_nvector(y0, ctx.as_ptr()) } {
                Some(y0) => y0,
                None => panic!("The context of y0 is not the same as the \
                    context of the CVode solver."),
            };
        let r = unsafe { CVodeInit(
            cvode_mem.0,
            Some(Self::cvrhs),
            self.t0,
            V::as_ptr(&y0) as *mut _) };
        if r == CV_MEM_FAIL {
            let msg = "a memory allocation request has failed";
            return Err(Error::Failure{name: self.name, msg})
        }
        if r == CV_ILL_INPUT {
            let msg = "An input argument has an illegal value";
            return Err(Error::Failure{name: self.name, msg})
        }
        // Set default tolerances (otherwise the solver will complain).
        unsafe { CVodeSStolerances(
            cvode_mem.0, self.rtol, self.atol); }
        // Set the default linear solver to one that does not require
        // the `…nvgetarraypointer` on vectors (FIXME: configurable)
        let linsolver = unsafe { LinSolver::spgmr(
            self.name,
            ctx.as_ptr(),
            V::as_ptr(&y0) as *mut _)? };
        let r = unsafe { CVodeSetLinearSolver(
            cvode_mem.0, linsolver.as_ptr(), ptr::null_mut()) };
        if r != CVLS_SUCCESS as i32 {
            return Err(Error::Failure {
                name: self.name,
                msg: "could not attach linear solver"
            })
        }
        unsafe { CVodeSStolerances(
            cvode_mem.0, self.rtol, self.atol); }
        if let Some(maxord) = self.maxord {
            unsafe { CVodeSetMaxOrd(
                cvode_mem.0,
                maxord as _); }
        }
        if let Some(mxsteps) = self.mxsteps {
            unsafe { CVodeSetMaxNumSteps(
                cvode_mem.0,
                mxsteps) };
        }
        if let Some(tstop) = self.tstop {
            if tstop.is_nan() {
                // unsafe { CVodeClearStopTime(self.cvode_mem.0); }
                ()
            } else {
                let ret = unsafe { CVodeSetStopTime(
                    cvode_mem.0,
                    tstop) };
                if ret == CV_ILL_INPUT {
                    // FIXME: should not happen in this configuration
                    // as it is fixed ahead of execution.
                    let msg = "The 'tstop' time is not is not beyond \
                        the current time value.";
                    return Err(Error::Failure { name: self.name, msg });
                }
            }
        }
        unsafe { CVodeSetMaxHnilWarns(cvode_mem.0, self.max_hnil_warns) };
        let mut rootsfound;
        if M > 0 {
            let r = unsafe  {
                CVodeRootInit(cvode_mem.0, M as _,
                    Some(Self::cvroot1)) };
            if r == CV_MEM_FAIL {
                panic!("Sundials::cvode::CVode::root: memory allocation \
                    failed.");
            }
            rootsfound = Vec::with_capacity(M);
            rootsfound.resize(M, 0);
        } else {
            rootsfound = vec![];
        }
        Ok(CVode {
            ctx,  cvode_mem,
            t0: self.t0,  vec: PhantomData,
            _matrix: None,  _linsolver: Some(linsolver),
            rootsfound,
            user_data: UserData { f: self.f, g: self.g }
        })
    }

    /// Callback for the right-hand side of the equation.
    extern "C" fn cvrhs(t: f64, nvy: N_Vector, nvdy: N_Vector,
                        user_data: *mut c_void
    ) -> c_int {
        // Protect against unwinding in C code.
        match std::panic::catch_unwind(|| {
            // Get f from user_data, whatever the type of g.
            let y = V::from_nvector(nvy);
            let dy = V::from_nvector_mut(nvdy);
            let u = unsafe { &mut *(user_data as *mut UserData<F, G>) };
            (u.f)(t, y, dy);
        }) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("sundials::CVode: right-hand side function \
                           panicked: {:?}", e);
                std::process::abort() // Doesn't unwind
            }
        }
    }

    /// Callback for the root-finding callback for `M` functions,
    /// where `M` is known at compile time.
    extern "C" fn cvroot1(
        t: f64, y: N_Vector, gout: *mut f64, user_data: *mut c_void) -> c_int {
        // Protect against unwinding in C code.
        match std::panic::catch_unwind(|| {
            let u = unsafe { &mut *(user_data as *mut UserData<F, G>) };
            let out = unsafe { &mut *(gout as *mut [f64; M]) };
            (u.g)(t, V::from_nvector(y), out);
        }) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("sundials::CVode: function passed to .root() \
                           panicked: {:?}", e);
                std::process::abort() // Doesn't unwind
            }
       }
    }
}

// Implement the Drop trait only on the pointer to be able to move
// values out of the structure `CVode`.
#[derive(Debug)]
struct CVodeMem(*mut c_void);

impl Drop for CVodeMem {
    fn drop(&mut self) { unsafe { CVodeFree(&mut self.0) } }
}

/// Return value of [`CVode::solve`] and [`CVode::step`].
#[derive(Debug, PartialEq)]
pub enum CVStatus {
    Ok,
    /// Succeeded by reaching the stopping point specified through
    /// [`CVode::set_tstop`].
    Tstop(f64),
    Root(f64, Vec<bool>),
    IllInput,
    /// The initial time `t0` and the output time `t` are too close to
    /// each other and the user did not specify an initial step size.
    TooClose,
    /// The solver took [`CVode::mxsteps`] internal steps but could
    /// not reach the final time.
    TooMuchWork,
    /// The solver could not satisfy the accuracy demanded by the user
    /// for some internal step.
    TooMuchAcc,
    ErrFailure,
    ConvFailure,
}

/// Solver for stiff and nonstiff initial value problems for ODE systems.
///
/// The generic parameters are as follows: `Ctx` is the type of the
/// context, `V` is the type of vectors, `F` the type of the function
/// being used as right-hand side of the ODE, and `G` is the type of
/// the functions (if any) of which we want to seek roots.
pub struct CVode<Ctx, V, F, G>
where V: Vector {
    // One must take ownership of the context because it can only be
    // used in a single ODE solver.
    ctx: Ctx,
    cvode_mem: CVodeMem,
    t0: f64,
    vec: PhantomData<V>,
    // We hold `Matrix` and `LinSolver` so they are freed when `CVode`
    // is dropped.
    _matrix: Option<Matrix>,
    _linsolver: Option<LinSolver>,
    rootsfound: Vec<c_int>, // cache, with len() == number of eq
    user_data: UserData<F, G>,
}

// The user-data may be updated according to the type of `G`.
// However, we must ensure that `f: F` is always extracted in the same
// way because `cvrhs` may only be passed during initialization.
#[repr(C)]
struct UserData<F, G> {
    f: F, // Right-hand side of the equation
    g: G, // Function whose roots we want to compute (if any)
}

impl<V, F> CVode<(), V, F, ()>
where V: Vector + Sized,
      F: FnMut(f64, &V, &mut V)
{
    /// Solver using the Adams linear multistep method.  Recommended
    /// for non-stiff problems.
    // The fixed-point solver is recommended for nonstiff problems.
    pub fn adams<'a>(
        t0: f64, y0: &'a V, f: F
    ) -> CVodeConf<'a, V, F, RootFn<V, 0>, 0> {
        CVodeConf::new("CVode::adams", CV_ADAMS, t0, y0, f)
    }

    /// Solver using the BDF linear multistep method.  Recommended for
    /// stiff problems.
    // The default Newton iteration is recommended for stiff problems,
    pub fn bdf<'a>(
        t0: f64, y0: &'a V, f: F
    ) -> CVodeConf<'a, V, F, RootFn<V, 0>, 0> {
        CVodeConf::new("CVode::bdf", CV_BDF, t0, y0, f)
    }
}

impl<Ctx, V, F, G> CVode<Ctx, V, F, G>
where Ctx: Context,
      V: Vector + Sized,
      F: FnMut(f64, &V, &mut V)
{
    /// Return a reference to the [`Context`] the CVode solver was
    /// built with.
    pub fn context(&self) -> &Ctx {
        &self.ctx
    }

    /// Consumes CVode and return the [`Context`] it was built with.
    pub fn into_context(self) -> Ctx {
        self.ctx
    }

    /// Set the user data for CVode.  Since the closures are on the
    /// stack, their location changes.  One must let Sundials know
    /// about the new locations before launching a solver.
    fn update_user_data(&mut self) {
        let ptr = &self.user_data as *const _;
        let ret = unsafe { CVodeSetUserData(
            self.cvode_mem.0,
            ptr as *mut c_void) };
        debug_assert_eq!(ret, 0);
    }

    /// Set `y` to the solution at time `t`.
    ///
    /// If this function returns [`CVStatus::IllInput`], it means that
    /// - `t` was not monotonic w.r.t. previous calls.
    /// - A component of the error weight vector became zero during
    ///   internal time-stepping.
    /// - A root of one of the root functions was found both at a point `t`
    ///   and also very near `t`.
    pub fn solve(&mut self, t: f64, y: &mut V) -> CVStatus {
        Self::integrate(self, t, y, CV_NORMAL).1
    }

    /// Same as [`CVode::solve`] but only perform one time step in the
    /// direction of `t`.
    pub fn step(&mut self, t: f64, y: &mut V) -> (f64, CVStatus) {
        Self::integrate(self, t, y, CV_ONE_STEP)
    }

    fn integrate(
        &mut self, t: f64, y: &mut V, itask: c_int
    ) -> (f64, CVStatus) {
        self.update_user_data();
        // Safety: `yout` does not escape this function and so will
        // not outlive `self.ctx`.
        //let n = y.len();
        let yout =
            match unsafe { V::as_mut_nvector(y, self.ctx.as_ptr()) }{
                Some(yout) => yout,
                None => panic!("The context of the output vector y is not \
                    the same as the context of CVode."),
            };
        let mut tret = self.t0;
        let r = unsafe { CVode(
            self.cvode_mem.0,
            t,
            V::as_mut_ptr(&yout),
            &mut tret, itask) };
        let status = match r {
            CV_SUCCESS => CVStatus::Ok,
            CV_TSTOP_RETURN => CVStatus::Tstop(tret),
            CV_ROOT_RETURN => {
                let ret = unsafe { CVodeGetRootInfo(
                    self.cvode_mem.0,
                    self.rootsfound.as_mut_ptr()) };
                debug_assert_eq!(ret, CV_SUCCESS);
                let z: c_int = 0;
                let roots = self.rootsfound.iter().map(|g| g != &z).collect();
                CVStatus::Root(tret, roots)
            }
            CV_MEM_NULL | CV_NO_MALLOC => unreachable!(),
            CV_ILL_INPUT => CVStatus::IllInput,
            CV_TOO_CLOSE => CVStatus::TooClose,
            CV_TOO_MUCH_WORK => CVStatus::TooMuchWork,
            CV_TOO_MUCH_ACC => CVStatus::TooMuchAcc,
            CV_ERR_FAILURE => CVStatus::ErrFailure,
            CV_CONV_FAILURE => CVStatus::ConvFailure,
            CV_LINIT_FAIL => panic!("The linear solver interface’s \
                initialization function failed."),
            CV_LSETUP_FAIL => panic!("The linear solver interface’s setup \
                function failed in an unrecoverable manner."),
            CV_LSOLVE_FAIL => panic!("The linear solver interface’s solve \
                function failed in an unrecoverable manner."),
            CV_CONSTR_FAIL => panic!("The inequality constraints were \
                violated and the solver was unable to recover."),
            CV_RHSFUNC_FAIL => panic!("The right-hand side function failed \
                in an unrecoverable manner."),
            CV_REPTD_RHSFUNC_ERR => panic!("Convergence test failures \
                occurred too many times due to repeated recoverable errors \
                in the right-hand side function."),
            CV_UNREC_RHSFUNC_ERR => panic!("The right-hand function had a \
                recoverable error, but no recovery was possible."),
            CV_RTFUNC_FAIL => panic!("The root function failed"),
            _ => panic!("sundials::CVode: unexpected return code {}", r),
        };
        (tret, status)
    }

    /// Return the solution with initial conditions (`t0`, `y0`) at
    /// time `t`.  This is a convenience function.
    pub fn cauchy(&mut self, t0: f64, y0: &V, t: f64) -> (V, CVStatus) {
        // FIXME: should we check the dim of `y0`?
        let mut y = y0.clone();
        // Avoid CVStatus::TooClose
        if t == t0 {
            return (y, CVStatus::Ok)
        }
        let y0 = match unsafe { V::as_nvector(y0, self.ctx.as_ptr()) } {
            Some(y0) => y0,
            None => panic!("The context of `y0` differs from the one \
                of the ODE solver."),
        };
        // Reinitialize to allow any time `t`, even if not monotonic
        // w.r.t. previous calls.
        let ret = unsafe {
            CVodeReInit(self.cvode_mem.0, t0, V::as_ptr(&y0) as *mut _)
        };
        if ret != CV_SUCCESS {
            panic!("CVodeReInit returned code {ret}.  Please report.");
        }
        let cv = self.solve(t, &mut y);
        (y, cv)
    }
}



#[cfg(test)]
mod tests {
    use crate::{context, Error, cvode::{CVode, CVStatus}};

    #[test]
    fn cvode_zero_time_step() {
        let ctx = context!(P0).unwrap();
        let mut ode = CVode::adams(0., &[0.],
            |_,_, du| *du = [1.])
            .build(ctx).unwrap();
        let mut u1 = [f64::NAN];
        let cv = ode.solve(0., &mut u1);
        assert_eq!(cv, CVStatus::TooClose);
    }

    #[test]
    fn cvode_solution() {
        let mut ode = CVode::adams(0., &[0.],
            |_,_, du| *du = [1.])
            .build(context!().unwrap()).unwrap();
        assert_eq!(ode.cauchy(0., &[0.], 1.).0, [1.]);
    }

    #[test]
    fn cvode_exp() {
        let mut ode = CVode::adams(0., &[1.],
            |_,u, du| *du = *u)
            .build(context!().unwrap()).unwrap();
        let mut u1 = [f64::NAN];
        ode.solve(1., &mut u1);
        assert_eq_tol!(u1[0], 1f64.exp(), 1e-5);
    }

    #[test]
    fn cvode_sin() {
        let mut ode = CVode::adams(0., &[0., 1.], |_, u, du| {
            *du = [u[1], -u[0]]
        })
            .mxsteps(500)
            .build(context!().unwrap()).unwrap();
        let mut u1 = [f64::NAN, f64::NAN];
        ode.solve(1., &mut u1);
        assert_eq_tol!(u1[0], 1f64.sin(), 1e-5);
    }

    #[test]
    fn cvode_nonmonotonic_time() {
        let ctx = context!().unwrap();
        let mut ode = CVode::adams(0., &[1.],
            |_, _, du| *du = [1.]).build(ctx).unwrap();
        let mut u = [f64::NAN];
        assert_eq!(ode.solve(1., &mut u), CVStatus::Ok);
        assert_eq!(ode.solve(0., &mut u), CVStatus::IllInput);
    }

    #[test]
    fn cvode_move() {
        let ctx = context!().unwrap();
        let init = [0.];
        let ode = move || {
            CVode::adams(0., &init, |_,_, du| *du = [1.])
                .build(ctx).unwrap()
        };
        let (u, st) = ode().cauchy(0., &init, 1.);
        assert_eq!(u, [1.]);
        assert_eq!(st, CVStatus::Ok)
    }

    #[test]
    fn cvode_clone_config() -> Result<(), Error> {
        let init = [1., 2.];
        let ode =
            CVode::adams(0., &init, |_,_, du: &mut [_;2]| {
                *du = [1., 1.]
            });
        let ctx = context!(P1)?;
        assert_eq!(ode.clone().build(ctx)?.cauchy(0., &init, 1.).0, [2., 3.]);
        let ctx = context!(P2)?;
        let (u, cv) = ode.clone()
            .root(|_, &u, z| *z = [u[0] - 2.])
            .build(ctx)?
            .cauchy(0., &init, 2.);
        assert!(matches!(cv, CVStatus::Root(_,_)));
        assert_eq!(u, [2., 3.]);
        let ctx = context!(P3)?;
        assert_eq!(ode.build(ctx)?.cauchy(0., &init, 2.).0, [3., 4.]);
        Ok(())
    }

    #[test]
    fn cvode_with_param() {
        let ctx = context!().unwrap();
        let c = 1.;
        let mut ode = CVode::adams(0., &[0.],
            |_,_, du| *du = [c])
            .build(ctx).unwrap();
        let mut u1 = [f64::NAN];
        let ret = ode.solve(1., &mut u1);
        assert_eq!(ret, CVStatus::Ok);
        assert_eq_tol!(u1[0], c, 1e-5);
    }

    #[test]
    fn cvode_with_root() {
        let ctx = context!().unwrap();
        let mut u = [f64::NAN; 2];
        let r = CVode::adams(0., &[0., 1.],  |_, u, du: &mut [_;2]| {
            *du = [u[1], -2.]
        })
            .root(|_,u, r| *r = [u[0], u[0] - 100.])
            .build(ctx).unwrap()
            .solve(2., &mut u); // Time is past the root
        match r {
            CVStatus::Root(t, roots) => {
                assert_eq!(roots, vec![true, false]);
                assert_eq_tol!(t, 1., 1e-12);
                assert_eq_tol!(u[0], 0., 1e-12);
                assert_eq_tol!(u[1], -1., 1e-12);
            }
            _ => panic!("`Root` expected")
        }
    }

    #[test]
    fn compatible_with_eyre() -> eyre::Result<()> {
        let ctx = context!().unwrap();
        let _ = CVode::adams(0., &[1.], |t, y, dy: &mut [_;1]| {
            *dy = [t * y[0]] })
            .build(ctx)?;
        Ok(())
    }
}
