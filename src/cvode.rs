//! `Cvode` is a solver for stiff and nonstiff ODE systems ẏ = f(t,y)
//! with root detection and projection on a constraint manifold.
//! Based on Adams and BDF methods.

use std::{
    ffi::{c_int, c_void, c_long},
    pin::Pin,
    ptr,
};
use sundials_sys::*;
use super::{
    Context,
    Error,
    vector::Vector,
    Matrix,
    LinSolver
};

// Implement the Drop trait only on the pointer to be able to move
// values out of the structure `CVode`.
#[derive(Debug)]
struct CVodeMem(*mut c_void);

impl Drop for CVodeMem {
    fn drop(&mut self) { unsafe { CVodeFree(&mut self.0) } }
}

/// Solver for stiff and nonstiff initial value problems for ODE systems.
///
/// The generic parameters are as follows: `Ctx` is the type of the
/// context, `V` is the type of vectors, `F` the type of the function
/// being used as right-hand side of the ODE, and `G` is the type of
/// the functions (if any) of which we want to seek roots.
pub struct CVode<'a, Ctx, V, F, G>
where V: Vector {
    // One must take ownership of the context because it can only be
    // used in a single ODE solver.
    ctx: Ctx,
    cvode_mem: CVodeMem,
    t0: f64,
    // FIXME: after used in CVodeInit, the initial vector is copied to
    // the internal structures.  No need to keep it.
    y0: V::NVectorRef<'a>,
    rtol: f64,
    atol: f64,
    matrix: Matrix,
    linsolver: LinSolver,
    rootsfound: Vec<c_int>, // cache, with len() == number of eq
    user_data: Pin<Box<UserData<F, G>>>,
}

// The user-data may be updated according to the type of `G`.
// However, we must ensure that `f: F` is always extracted in the same
// way because `cvrhs` may only be passed during initialization.
#[repr(C)]
struct UserData<F, G> {
    f: F, // Right-hand side of the equation
    g: G, // Function whose roots we want to compute (if any)
}

impl<'a, Ctx, V, F, G> CVode<'a, Ctx, V, F, G>
where Ctx: Context,
      V: Vector
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

    fn new_with_fn<G1>(ctx: Ctx, cvode_mem: CVodeMem, t0: f64,
               y0: V::NVectorRef<'a>,
               rtol: f64, atol: f64, matrix: Matrix, linsolver: LinSolver,
               f: F, g: G1, ng: usize
    ) -> CVode<'a, Ctx, V, F, G1> {
        // FIXME: One can move `f` and `g` and set the user data
        // before calling solving functions.
        let user_data = Box::pin(UserData { f, g });
        let user_data_ptr =
            user_data.as_ref().get_ref() as *const _ as *mut c_void;
        unsafe { CVodeSetUserData(cvode_mem.0, user_data_ptr) };
        let mut rootsfound = Vec::with_capacity(ng);
        rootsfound.resize(ng, 0);
        CVode {
            ctx, cvode_mem,
            t0, y0,
            rtol, atol, matrix,  linsolver, rootsfound, user_data,
        }
    }
}

impl<'a, Ctx, V, F> CVode<'a, Ctx, V, F, ()>
where Ctx: Context,
      V: Vector + Sized,
      F: FnMut(f64, V::View<'_>, V::ViewMut<'_>)
{
    /// Callback for the right-hand side of the equation.
    extern "C" fn cvrhs(t: f64, nvy: N_Vector, nvdy: N_Vector,
                        user_data: *mut c_void
    ) -> c_int {
        // Protect against unwinding in C code.
        match std::panic::catch_unwind(|| {
            // Get f from user_data, whatever the type of g.
            let y = V::from_nvector(nvy);
            let dy = V::from_nvector_mut(nvdy);
            let u = unsafe { &mut *(user_data as *mut UserData<F, ()>) };
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

    /// Initialize a CVode with method `llm`.
    #[inline]
    fn init(
        name: &'static str, lmm: c_int,
        // Take the context by move so only one solver can have it at
        // a given time.
        ctx: Ctx, t0: f64, y0: &'a V, f: F
        // FIXME: Once y0 has been passed to CVodeInit, it is copied
        // to interval structures and can be freed.
    ) -> Result<Self, Error> {
        let cvode_mem = unsafe {
            CVodeMem(CVodeCreate(lmm, ctx.as_ptr())) };
        if cvode_mem.0.is_null() {
            return Err(Error::Fail{name, msg: "Allocation failed"})
        }
        let n = y0.len();
        // Safety: `y0` will not outlive `ctx`: they are both in CVode
        // and `y0` cannot not be moved out of this structure.
        let y0 = unsafe { y0.to_nvector(ctx.as_ptr()) };
        let r = unsafe { CVodeInit(
            cvode_mem.0,
            Some(Self::cvrhs),
            t0,
            V::as_ptr(&y0) as *mut _) };
        if r == CV_MEM_FAIL {
            let msg = "a memory allocation request has failed";
            return Err(Error::Fail{name, msg})
        }
        if r == CV_ILL_INPUT {
            let msg = "An input argument has an illegal value";
            return Err(Error::Fail{name, msg})
        }
        let rtol = 1e-6;
        let atol = 1e-12;
        // Set default tolerances (otherwise the solver will complain).
        unsafe { CVodeSStolerances(
            cvode_mem.0, rtol, atol); }
        // Set the default linear solver (FIXME: configurable)
        let mat = Matrix::new(name, &ctx, n, n)?;
        let linsolver = unsafe { LinSolver::new(
            name, ctx.as_ptr(), V::as_ptr(&y0) as *mut _, &mat)? };
        let r = unsafe {
            CVodeSetLinearSolver(cvode_mem.0, linsolver.0, mat.0) };
        if r != CVLS_SUCCESS as i32 {
            return Err(Error::Fail{name, msg: "could not attach linear solver"})
        }
        Ok(Self::new_with_fn(
            ctx, cvode_mem, t0, y0,
            rtol, atol, mat, linsolver,
            f, (), 0))
    }

    /// Solver using the Adams linear multistep method.  Recommended
    /// for non-stiff problems.
    // The fixed-point solver is recommended for nonstiff problems.
    pub fn adams(ctx: Ctx, t0: f64, y0: &'a V, f: F) -> Result<Self, Error> {
        Self::init("CVode::adams", CV_ADAMS, ctx, t0, y0, f)
    }

    /// Solver using the BDF linear multistep method.  Recommended for
    /// stiff problems.
    // The default Newton iteration is recommended for stiff problems,
    pub fn bdf(ctx: Ctx, t0: f64, y0: &'a V, f: F) -> Result<Self, Error> {
        Self::init("CVode::bdf", CV_BDF, ctx, t0, y0, f)
    }
}

impl<'a, Ctx, V, F, G> CVode<'a, Ctx, V, F, G>
where V: Vector {
    pub fn rtol(self, rtol: f64) -> Self {
        unsafe { CVodeSStolerances(self.cvode_mem.0, rtol, self.atol); }
        self
    }

    pub fn atol(self, atol: f64) -> Self {
        unsafe { CVodeSStolerances(self.cvode_mem.0, self.rtol, atol); }
        self
    }

    pub fn maxord(self, o: u8) -> Self {
        unsafe { CVodeSetMaxOrd(self.cvode_mem.0, o as _); }
        self
    }

    /// Specify the maximum number of steps to be taken by the solver
    /// in its attempt to reach the next output time.  Default: 500.
    // FIXME: make sure "mxstep steps taken before reaching tout" does
    // not abort the program.
    pub fn mxsteps(self, n: usize) -> Self {
        let n =
            if n <= c_long::MAX as usize { n as _ } else { c_long::MAX };
        unsafe { CVodeSetMaxNumSteps(self.cvode_mem.0, n) };
        self
    }

    /// Specifies the maximum number of messages issued by the solver
    /// warning that t + h = t on the next internal step.
    pub fn max_hnil_warns(self, n: usize) -> Self {
        unsafe { CVodeSetMaxHnilWarns(self.cvode_mem.0, n as _) };
        self
    }
}

/// # Root-finding capabilities
impl<'a, Ctx, V, F, G> CVode<'a, Ctx, V, F, G>
where Ctx: Context,
      V: Vector,
      F: FnMut(f64, &V, &mut V) + Unpin,
      G: Unpin
{
    /// Callback for the root-finding callback for `N` functions,
    /// where `N` is known at compile time.
    extern "C" fn cvroot1<const N: usize, G1>(
        t: f64, y: N_Vector, gout: *mut f64, user_data: *mut c_void) -> c_int
    where G1: FnMut(f64, V::View<'_>, &mut [f64; N]) {
        // Protect against unwinding in C code.
        match std::panic::catch_unwind(|| {
            let u = unsafe { &mut *(user_data as *mut UserData<F, G1>) };
            let out = unsafe { &mut *(gout as *mut [f64; N]) };
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

    /// Specifies that the roots of a set of functions gᵢ(t, y),
    /// 0 ≤ `i` < `N` (given by `g`(t,y, [g₁,...,gₙ])) are to be
    /// found while the IVP is being solved.
    ///
    pub fn root<const M: usize, G1>(self, g: G1) -> CVode<'a, Ctx, V, F, G1>
    where G1: FnMut(f64, V::View<'_>, &mut [f64; M]) {
        // FIXME: Do we want a second (because it will not work when V
        // = [f64;N] since the number of equations is usually not the
        // same as the dimension of the problem) function accepting
        // V::ViewMut and a dim as second parameter?  In this case,
        // one must wrap `g` to check for the dimension of the
        // returned vector at each invocation.
        let r = unsafe  {
            CVodeRootInit(self.cvode_mem.0, M as _,
                          Some(Self::cvroot1::<M, G1>)) };
        if r == CV_MEM_FAIL {
            panic!("Sundials::CVode::root: memory allocation failed.");
        }
        let u = *Pin::into_inner(self.user_data);
        Self::new_with_fn(
            self.ctx,
            self.cvode_mem, self.t0, self.y0,
            self.rtol, self.atol,
            self.matrix, self.linsolver, u.f, g, M)
    }
}


/// Return value of [`CVode::solve`] and [`CVode::step`].
#[derive(Debug, PartialEq)]
pub enum CV {
    Ok,
    Root(f64, Vec<bool>),
    ErrFailure,
    ConvFailure,
}

/// # Solving the IVP
impl<'a, Ctx, V, F, G> CVode<'a, Ctx, V, F, G>
where Ctx: Context,
      V: Vector {
    // FIXME: for arrays, it would be more convenient to return the
    // array.  For types that are not on the stack (i.e. not Copy),
    // taking it as an additional parameter is better.
    pub fn solve(&mut self, t: f64, y: &mut V) -> CV {
        Self::integrate(self, t, y, CV_NORMAL)
    }

    pub fn step(&mut self, t: f64, y: &mut V) -> CV {
        Self::integrate(self, t, y, CV_ONE_STEP)
    }

    fn integrate(&mut self, t: f64, y: &mut V, itask: c_int) -> CV {
        // Safety: `yout` does not escape this function and so will
        // not outlive `self.ctx`.
        let n = y.len();
        let yout = unsafe { V::to_nvector_mut(y, self.ctx.as_ptr()) };
        if t == self.t0 {
            unsafe { ptr::copy_nonoverlapping(
                N_VGetArrayPointer(V::as_ptr(&self.y0) as *mut _),
                N_VGetArrayPointer(V::as_mut_ptr(&yout)),
                n) };
            return CV::Ok;
        }
        let mut t1 = self.t0;
        let r = unsafe { CVode(
            self.cvode_mem.0,
            t,
            V::as_mut_ptr(&yout),
            &mut t1, itask) };
        match r {
            CV_SUCCESS => CV::Ok,
            //CV_TSTOP_RETURN => ,
            CV_ROOT_RETURN => {
                let ret = unsafe { CVodeGetRootInfo(
                    self.cvode_mem.0,
                    self.rootsfound.as_mut_ptr()) };
                debug_assert_eq!(ret, CV_SUCCESS);
                let z: c_int = 0;
                let roots = self.rootsfound.iter().map(|g| g != &z).collect();
                CV::Root(t1, roots)
            }
            CV_MEM_NULL | CV_NO_MALLOC => unreachable!(),
            CV_ILL_INPUT => panic!("CV_ILL_INPUT"),
            CV_TOO_MUCH_WORK => panic!("Too much work"),
            CV_TOO_MUCH_ACC => panic!("Could not satisfy desired accuracy"),
            CV_ERR_FAILURE => CV::ErrFailure,
            CV_CONV_FAILURE => CV::ConvFailure,
            CV_LINIT_FAIL => panic!("CV_LINIT_FAIL"),
            CV_LSETUP_FAIL => panic!("CV_LSETUP_FAIL"),
            CV_LSOLVE_FAIL => panic!("CV_LSOLVE_FAIL"),
            CV_RTFUNC_FAIL => panic!("The root function failed"),
            _ => panic!("sundials::CVode: unexpected return code {}", r),
        }
    }

}

impl<'a, const N: usize, Ctx, F, G> CVode<'a, Ctx, [f64; N], F, G>
where Ctx: Context {
    /// Return the solution at time `t`.
    // FIXME: provide it for any type `V` — which must implement a
    // creation function.
    pub fn solution(&mut self, t: f64) -> ([f64; N], CV) {
        let mut y = [f64::NAN; N];
        let cv = self.solve(t, &mut y);
        (y, cv)
    }
}
