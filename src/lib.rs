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
//! use sundials::{context, CVode};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let ctx = context!()?;
//! let mut ode = CVode::adams(ctx, 0., &[0.], |t, u, du| *du = [1.])?;
//! let (u1, _) = ode.solution(1.);
//! assert_eq!(u1[0], 1.);
//! # Ok(()) }
//! ```
//!
//!
//!
//! [Sundials]: https://computing.llnl.gov/projects/sundials

use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::Drop,
    ptr
};
use sundials_sys::*;

#[cfg(test)]
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

mod arkode;
mod cvode;
mod cvodes;
mod ida;
mod idas;
mod kinsol;

mod vector;

pub use cvode::CVode;

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

// According to the [SUNContext][] documentation,
// - all objects in a simulation (solver, vectors,...) must share the
//   same context;
// - a context can only be used in a single simulation at a time.  It
//   can be reused in a different simulation provided all objects (the
//   ode solver of course be also all vectors,...) have been
//   destroyed.
//
// [SUNContext]: https://sundials.readthedocs.io/en/latest/sundials/SUNContext_link.html

// In order to give a static insurance that all objects share the same
// context, each context must define a new type with which all objects
// will be "tagged".  Then type checking will ensure coherence.  Since
// functions have given types, the new type must be created by a
// macro.  The fact that the new type is an acceptable Context is
// defined by the trait [`Context`].  For the guarantees to hold, the
// type can only "generate" a single Context.
//
// These static guarantees are nice but do not cover more dynamic
// situations.  Thus a type with dynamic checks is also defined.

/// The trait `Context` indicates that `self` is a suitable context to
/// be linked to all objects of a simulation.
pub trait Context {
    #[doc(hidden)]
    /// Return the `SUNContext` pointer.
    fn as_ptr(&self) -> SUNContext;

    #[doc(hidden)]
    /// Check whether two contexts are the same (if that cannot be
    /// guaranteed by the type system).
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_ptr() == other.as_ptr()
    }
}

/// Return a Sundials context (or an [`Error`]) with a new type, given
/// by the provided name, which will be a type parameter attached to
/// all objects in a Sundials simulation.
///
/// # Examples
///
/// ```
/// use sundials::context;
/// let ctx = context!(P1)?;
/// # Ok::<(), sundials::Error>(())
/// ```
///
/// If you do not provide a name, the default will be `Ctx` will be
/// used.  Unless you deal with a single context, we do not
/// recommended to use several times the same name however because the
/// compiler will distinguish them but you will have harder time to
/// know which is which.  For example, the last line of the following
/// code
///
/// ```compile_fail
/// use sundials::context;
/// let ctx1 = context!().unwrap();
/// let ctx2 = context!().unwrap();
/// [ctx1, ctx2]
/// ```
///
/// will be rejected with an error like "expected `main::Ctx`, found a
/// different `main::Ctx`".
#[macro_export]
macro_rules! context {
    () => { context!(Ctx) };
    ($name: ident) => {
        // Block so the type is local (thus cannot be instantiated
        // from outside).
        {
            use $crate::__BoxedContext;

            /// A [Sundials context][::sundials::context].
            #[allow(non_camel_case_types)]
            // This is not clone()able to ensure that the ODE solver
            // taking possession of the context is the only one that
            // does so.
            struct $name {
                ctx: __BoxedContext,
            }
            impl $crate::Context for $name {
                #[inline]
                fn as_ptr(&self) -> ::sundials_sys::SUNContext {
                    self.ctx.as_ptr()
                }

                #[inline]
                fn eq(&self, _: &Self) -> bool {
                    // The main point of this macro is to define a new
                    // type so that objects linked with different
                    // contexts will have different types.
                    true
                }
            }
            unsafe { __BoxedContext::new()
                .map(|ctx| $name { ctx }) }
        }
    };
    // FIXME: Add a third case when a MPI communicator is needed.
}

#[doc(hidden)]
/// A wrapper around `SUNContext` that ensures it is memory managed by Rust.
pub struct __BoxedContext(SUNContext);

impl Drop for __BoxedContext {
    fn drop(&mut self) {
        // FIXME: Make sure the remark about MPI is followed (when
        // this library allows MPI)
        // https://sundials.readthedocs.io/en/latest/sundials/SUNContext_link.html#c.SUNContext_Free
        unsafe { SUNContext_Free(self.0 as *mut _); }
    }
}

impl __BoxedContext {
    #[doc(hidden)]
    /// Return a raw pointer to a Sundials context.
    pub fn as_ptr(&self) -> SUNContext {
        self.0
    }

    unsafe fn with_communicator(
        comm: *mut std::os::raw::c_void
    ) -> Result<Self, Error> {
        let mut ctx: SUNContext = ptr::null_mut();
        if unsafe { SUNContext_Create(comm, &mut ctx as *mut _) } < 0 {
            return Err(Error::Fail { name: "Context::new",
                                     msg: "Failed to create a context" })
        }
        Ok(Self(ctx))
    }

    #[doc(hidden)]
    /// Returns a new `SUNContext` wrapped so that it will be freed
    /// when dropped.
    pub unsafe fn new() -> Result<Self, Error> {
        unsafe { Self::with_communicator(ptr::null_mut()) }
    }

    #[cfg(feature = "mpi")]
    fn with_mpi(conn: impl mpi::topology::Communicator) -> Self {
        // https://crates.io/crates/mpi-sys https://crates.io/crates/mpi
        todo!();
        let comm = ptr::null_mut();
        unsafe { Self::with_communicator(comm) }
    }
}

/// Type for Sundials contexts that are not distinguished by a type.
/// Unlike those generated using [`context`], they can be put in a
/// homogeneous container (such as an [array][::std::array] or a
/// [`Vec`]) but the fact that all objects in a simulation share the
/// same context will be dynamically checked (leading to panics if
/// this is not satisfied).
// FIXME: Can we use `dyn Context` vectors instead of this?
pub struct DynContext {
    ctx: __BoxedContext,
    //mpi: Option<()>, // FIXME: MPI communicator
}

impl Context for DynContext {
    #[inline]
    fn as_ptr(&self) -> SUNContext {
        self.ctx.0
    }
}


////////////////////////////////////////////////////////////////////////
//
// Matrix

struct Matrix(SUNMatrix);

impl Drop for Matrix {
    fn drop(&mut self) { unsafe { SUNMatDestroy(self.0) } }
}

impl Matrix {
    fn new(
        name: &'static str, ctx: &impl Context, m: usize, n: usize,
    ) -> Result<Self, Error> {
        let mat = unsafe {
            SUNDenseMatrix(m as _, n as _, ctx.as_ptr()) };
        if mat.is_null() {
            Err(Error::Fail{name, msg: "matrix allocation failed"})
        } else {
            Ok(Matrix(mat))
        }
    }
}


struct LinSolver(SUNLinearSolver);

impl Drop for LinSolver {
    // FIXME: handle possible returned error?
    // https://sundials.readthedocs.io/en/latest/sunlinsol/SUNLinSol_API_link.html?highlight=SUNLinSolFree#c.SUNLinSolFree
    fn drop(&mut self) { unsafe { SUNLinSolFree(self.0); } }
}

impl LinSolver {
    /// Return a new linear solver.
    ///
    /// # Safety
    /// The return value must not outlive `ctx`.
    unsafe fn new(
        name: &'static str, ctx: SUNContext, vec: N_Vector, mat: &Matrix,
    ) -> Result<Self, Error> {
        let linsolver = unsafe {
            SUNLinSol_Dense(vec, mat.0, ctx) };
        if linsolver.is_null() {
            Err(Error::Fail{ name, msg: "linear solver  allocation failed"})
        } else {
            Ok(LinSolver(linsolver))
        }
    }
}

