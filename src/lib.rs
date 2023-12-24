//! A high-level interface to Sundials.
//!
//! [Sundials][] is a s̲u̲ite of n̲onlinear and d̲i̲fferential/a̲l̲gebraic
//! equation s̲olvers.
//!
//!
//! # Example
//!
//! The following code solves the equation ∂ₜu = f(t,u) where f is the
//! function (t,u) ↦ 1 using Adams' method.
//!
//! ```
//! use sundials::CVode;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut ode = CVode::adams(0., &[0.], |t, u, du| *du = [1.])?;
//! let (u1, _) = ode.solution(1.);
//! assert_eq!(u1[0], 1.);
//! # Ok(()) }
//! ```
//!
//!
//!
//! [Sundials]: https://computing.llnl.gov/projects/sundials

use std::{ffi::c_void,
          marker::PhantomData,
          fmt::{self, Debug, Display, Formatter},
          ops::Drop,
          os::raw::{c_int, c_long},
          boxed::Box,
          pin::Pin,
          ptr};
use sundials_sys::*;

////////////////////////////////////////////////////////////////////////
//
// Error

#[derive(Debug)]
pub enum Error {
    /// The function or method failed with the attached message
    /// (without further details).
    Fail { name: &'static str, msg: &'static str },
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Error::Fail { name, msg} => {
                if msg.is_empty() {
                    write!(f, "The function {} failed.", name)
                } else {
                    write!(f, "The function {} failed with message: {}.",
                           name, msg)
                }
            }
        }
    }
}

impl std::error::Error for Error {}

////////////////////////////////////////////////////////////////////////
//
// SUNContext

/// Context is an object associated with the thread of execution.
pub struct Context(
    SUNContext,
);

impl Drop for Context {
    fn drop(&mut self) {
        // FIXME: Make sure the remark about MPI is followed (when
        // this library allows MPI)
        // https://sundials.readthedocs.io/en/latest/sundials/SUNContext_link.html#c.SUNContext_Free
        unsafe { SUNContext_Free(self.0 as *mut _); }
    }
}

impl Context {
    unsafe fn with_communicator(
        comm: *mut std::os::raw::c_void
    ) -> Result<Self, Error> {
        let mut ctx: SUNContext = ptr::null_mut();
        if unsafe { SUNContext_Create(comm, &mut ctx as *mut _) } < 0 {
            return Err(Error::Fail { name: "Context::new",
                                     msg: "Failed to create a context" })
        }
        Ok(Context(ctx))
    }

    fn new() -> Result<Self, Error> {
        unsafe { Self::with_communicator(ptr::null_mut()) }
    }

    #[cfg(feature = "mpi")]
    fn with_mpi(conn: impl mpi::topology::Communicator) -> Self {
        // https://crates.io/crates/mpi-sys https://crates.io/crates/mpi
        todo!()
        let comm = ptr::null_mut();
        unsafe { Self::with_communicator(comm) }
    }
}

////////////////////////////////////////////////////////////////////////
//
// NVector

// AsRef<[f64]> does not even seem to ne needed??  Beware of
// constraints though (see Sundials/ML paper).
pub unsafe trait NVector: AsRef<[f64]> + Clone {
    /// Type to store the table of vector operations (and any other
    /// information needed to perform cheap conversions).
    type Ops;

    fn ops() -> Self::Ops;

    // FIXME: Isn't the *mut sufficient?  That will make impossible to
    // borrow several times which is good.
    fn from_nvector<'a>(nv: N_Vector) -> &'a Self;
    fn from_nvector_mut<'a>(nv: N_Vector) -> &'a mut Self;

    // FIXME: should depend on Self::Ops
    /// Share data.
    fn to_nvector(v: &Self, ctx: &Context) -> N_Vector;
    fn to_nvector_mut(v: &mut Self, ctx: &Context) -> N_Vector;

    fn len(&self) -> usize;
}

unsafe impl<const N: usize> NVector for [f64; N] {
    type Ops = PhantomData<()>;
    fn ops() -> Self::Ops { PhantomData }

    #[inline]
    fn from_nvector<'a>(nv: N_Vector) -> &'a Self {
        unsafe { &*(N_VGetArrayPointer(nv) as *const [f64; N]) }
    }

    #[inline]
    fn from_nvector_mut<'a>(nv: N_Vector) -> &'a mut Self {
        unsafe { &mut *(N_VGetArrayPointer(nv) as *mut [f64; N]) }
    }

    #[inline]
    fn to_nvector(v: &Self, ctx: &Context) -> N_Vector {
        unsafe { N_VMake_Serial(N.try_into().unwrap(),
                                v.as_ptr() as *mut _,
                                ctx.0) }
    }

    #[inline]
    fn to_nvector_mut(v: &mut Self, ctx: &Context) -> N_Vector {
        unsafe { N_VMake_Serial(N.try_into().unwrap(),
                                v.as_mut_ptr(),
                                ctx.0) }
    }

    #[inline]
    fn len(&self) -> usize { N }
}

////////////////////////////////////////////////////////////////////////
//
// CVode

// Implement the Drop trait only on the pointer to be able to move
// values out of the structure `CVode`.
#[derive(Debug)]
struct CVodeMem(*mut c_void);

impl Drop for CVodeMem {
    fn drop(&mut self) { unsafe { CVodeFree(&mut self.0) } }
}

/// N_Vector whose "content" field is shared with a Rust value.  Thus
/// freeing it will only free the "ops" field and the structure itself.
struct SharedNVector(N_Vector);

impl Drop for SharedNVector {
    fn drop(&mut self) { unsafe { N_VFreeEmpty(self.0) } }
}

struct Matrix(SUNMatrix);

impl Drop for Matrix {
    fn drop(&mut self) { unsafe { SUNMatDestroy(self.0) } }
}

struct LinSolver(SUNLinearSolver);

impl Drop for LinSolver {
    // FIXME: handle possible returned error?
    // https://sundials.readthedocs.io/en/latest/sunlinsol/SUNLinSol_API_link.html?highlight=SUNLinSolFree#c.SUNLinSolFree
    fn drop(&mut self) { unsafe { SUNLinSolFree(self.0); } }
}

/// Solver for stiff and nonstiff initial value problems for ODE systems.
///
/// The generic parameters are as follows; `V` is the type of vectors,
/// `F` the type of the function being used as right-hand side of the
/// ODE, and `G` is the type of the functions (if any) of which we
/// want to seek roots.
pub struct CVode<'a, V, F, G> {
    // 'a for C vectors held by the solver
    marker: PhantomData<&'a V>,
    ctx: Context,
    cvode_mem: CVodeMem,
    t0: f64,
    y0: SharedNVector,
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

impl<'a, V, F, G> CVode<'a, V, F, G> {
    #[inline]
    fn new<G1>(ctx: Context,
        cvode_mem: CVodeMem, t0: f64, y0: SharedNVector,
        rtol: f64,  atol: f64, matrix: Matrix, linsolver: LinSolver,
        f: F, g: G1, ng: usize
    ) -> CVode<'a, V, F, G1> {
        let user_data = Box::pin(UserData { f, g });
        let user_data_ptr =
            user_data.as_ref().get_ref() as *const _ as *mut c_void;
        unsafe { CVodeSetUserData(cvode_mem.0, user_data_ptr) };
        let mut rootsfound = Vec::with_capacity(ng);
        rootsfound.resize(ng, 0);
        CVode { marker: PhantomData,
                ctx,
                cvode_mem,  t0, y0, rtol, atol,
                matrix,  linsolver,
                rootsfound, user_data }
    }
}

impl<'a, V, F> CVode<'a, V, F, ()>
where V: NVector,
      F: FnMut(f64, &V, &mut V)
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
    fn init(name: &'static str, lmm: c_int,
        t0: f64, y0: &'a V, f: F
    ) -> Result<Self, Error> {
        let ctx = Context::new()?;
        // FIXME: who will reclaim the N_Vector from y0?  Need to wrap it to
        // enable a `Drop`.
        let cvode_mem = unsafe {
            CVodeMem(CVodeCreate(lmm, ctx.0)) };
        if cvode_mem.0.is_null() {
            return Err(Error::Fail{name, msg: "Allocation failed"})
        }
        let nvy0 = V::to_nvector(y0, &ctx);
        let r = unsafe { CVodeInit(cvode_mem.0, Some(Self::cvrhs), t0, nvy0) };
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
                unsafe { CVodeSStolerances(cvode_mem.0, rtol, atol); }
        // Set the default linear solver (FIXME: configurable)
                let mat = unsafe {
                    SUNDenseMatrix(y0.len() as _, y0.len() as _, ctx.0) };
        if mat.is_null() {
            return Err(Error::Fail{name, msg: "matrix allocation failed"})
        }
        let linsolver = unsafe { SUNLinSol_Dense(nvy0, mat, ctx.0) };
        if linsolver.is_null() {
            return Err(Error::Fail{
                name, msg: "linear solver  allocation failed"})
        }
        let r = unsafe {
            CVodeSetLinearSolver(cvode_mem.0, linsolver, mat) };
        if r != CVLS_SUCCESS as i32 {
            return Err(Error::Fail{name, msg: "could not attach linear solver"})
        }
        Ok(Self::new(ctx, cvode_mem, t0, SharedNVector(nvy0),
                     rtol, atol, Matrix(mat), LinSolver(linsolver),
                     f, (), 0))
    }

    /// Solver using the Adams linear multistep method.  Recommended
    /// for non-stiff problems.
    // The fixed-point solver is recommended for nonstiff problems.
    pub fn adams(t0: f64, y0: &'a V, f: F) -> Result<Self, Error> {
        Self::init("CVode::adams", CV_ADAMS, t0, y0, f)
    }

    /// Solver using the BDF linear multistep method.  Recommended for
    /// stiff problems.
    // The default Newton iteration is recommended for stiff problems,
    pub fn bdf(t0: f64, y0: &'a V, f: F) -> Result<Self, Error> {
        Self::init("CVode::bdf", CV_BDF, t0, y0, f)
    }
}

impl<'a, V, F, G> CVode<'a, V, F, G>
where V: NVector {
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
impl<'a, V, F, G> CVode<'a, V, F, G>
where V: NVector,
      F: FnMut(f64, &V, &mut V) + Unpin,
      G: Unpin {

    /// Callback for the root-finding callback for `N` functions,
    /// where `N` is known at compile time.
    extern "C" fn cvroot1<const N: usize, G1>(
        t: f64, y: N_Vector, gout: *mut f64, user_data: *mut c_void) -> c_int
    where G1: FnMut(f64, &V, &mut [f64; N]) {
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
    pub fn root<const N: usize, G1>(self, g: G1) -> CVode<'a, V, F, G1>
    where G1: FnMut(f64, &V, &mut [f64; N]) {
        let r = unsafe  {
            CVodeRootInit(self.cvode_mem.0, N as _,
                          Some(Self::cvroot1::<N, G1>)) };
        if r == CV_MEM_FAIL {
            panic!("Sundials::CVode::root: memory allocation failed.");
        }
        let u = *Pin::into_inner(self.user_data);
        Self::new(self.ctx,
                  self.cvode_mem, self.t0, self.y0, self.rtol, self.atol,
                  self.matrix, self.linsolver, u.f, g, N)
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
impl<'a, V, F, G> CVode<'a, V, F, G>
where V: NVector {
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
        let yout = SharedNVector(V::to_nvector_mut(y, &self.ctx));
        if t == self.t0 {
            unsafe { ptr::copy_nonoverlapping(N_VGetArrayPointer(self.y0.0),
                                              N_VGetArrayPointer(yout.0),
                                              y.len()) };
            return CV::Ok;
        }
        let mut t1 = self.t0;
        let r = unsafe { CVode(self.cvode_mem.0, t, yout.0, &mut t1, itask) };
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
            CV_MEM_NULL | CV_NO_MALLOC | CV_ILL_INPUT => unreachable!(),
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

impl<'a, const N: usize, F, G> CVode<'a, [f64; N], F, G> {
    /// Return the solution at time `t`.
    // FIXME: provide it for any type `V` — which must implement a
    // creation function.
    pub fn solution(&mut self, t: f64) -> ([f64; N], CV) {
        let mut y = [f64::NAN; N];
        let cv = self.solve(t, &mut y);
        (y, cv)
    }
}

/// ODE systems, including sensitivity analysis capabilities (forward
/// and adjoint).
pub struct CVodes {}

/// Initial value ODE problems with additive Runge-Kutta methods,
/// including support for IMEX methods.
pub struct ARKODE {}

/// Initial value problems for DAE systems.
pub struct IDA {}

/// DAE systems, including sensitivity analysis capabilities (forward
/// and adjoint).
pub struct IDAS {}

/// Nonlinear algebraic systems.
pub struct KINSOL {}



#[cfg(test)]
mod tests {
    use crate::{CVode, CV};

    /// Check that `$left` and `$right` are the same up to an absolute
    /// error of `$tol`.
    macro_rules! assert_eq_tol {
        ($left: expr, $right: expr, $tol: expr) => {
            let left = $left;
            let right = $right;
            let tol = $tol;
            if !((left - right).abs() <= tol) {
                panic!("assertion failed: |left - right| ≤ tol, where\n\
                        - left:  {}\n\
                        - right: {}\n\
                        - tol: {}", left, right, tol);
            }
        }
    }

    #[test]
    fn cvode_zero_time_step() {
        let mut ode = CVode::adams(0., &[0.], |_,_, du| *du = [1.]).unwrap();
        let mut u1 = [f64::NAN];
        ode.solve(0., &mut u1); // make sure one can use initial time
        assert_eq!(u1, [0.]);
    }

    #[test]
    fn cvode_exp() {
        let mut ode = CVode::adams(0., &[1.], |_,u, du| *du = *u).unwrap();
        let mut u1 = [f64::NAN];
        ode.solve(1., &mut u1);
        assert_eq_tol!(u1[0], 1f64.exp(), 1e-5);
    }

    #[test]
    fn cvode_sin() {
        let ode = CVode::adams(0., &[0., 1.],
                               |_, u, du| *du = [u[1], -u[0]]).unwrap();
        let mut u1 = [f64::NAN, f64::NAN];
        ode.mxsteps(500).solve(1., &mut u1);
        assert_eq_tol!(u1[0], 1f64.sin(), 1e-5);
    }

    #[test]
    fn cvode_with_param() {
        let c = 1.;
        let mut ode = CVode::adams(0., &[0.],
                                   |_,_, du| *du = [c]).unwrap();
        let mut u1 = [f64::NAN];
        ode.solve(1., &mut u1);
        assert_eq_tol!(u1[0], c, 1e-5);
    }

    #[test]
    fn cvode_with_root() {
        let mut u = [f64::NAN; 2];
        let r = CVode::adams(0., &[0., 1.],
                             |_, u, du| *du = [u[1], -2.]).unwrap()
            .root(|_,u, r| *r = [u[0], u[0] - 100.])
            .solve(2., &mut u); // Time is past the root
        match r {
            CV::Root(t, roots) => {
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
        let _ = CVode::adams(0., &[1.], |t, y, dy| *dy = [t * y[0]])?;
        Ok(())
    }
}
