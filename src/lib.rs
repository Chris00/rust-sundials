//! A high-level interface to Sundials.
//!
//! [Sundials][] is a s̲u̲ite of n̲onlinear and d̲i̲fferential/a̲l̲gebraic
//! equation s̲olvers.
//!
//!
//! # Examples
//!
//! The following code solves the equation ∂ₜu = f(t,u) where f is the
//! function (t,u) ↦ 1 using Adams' method.
//!
//! ```
//! use sundials::{context, cvode::{CVode, Solver}};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut ode = CVode::adams(0., &0., |t, u, du| *du = 1.)
//!     .build(context!()?)?;
//! let (u1, _) = ode.cauchy(0., &0., 1.);
//! assert_eq!(u1, 1.);
//! # Ok(()) }
//! ```
//!
//! Here is a second example using [`array`]s to specify the equation
//! ∂ₜ(u₀, u₁) = (-u₁, u₀).
//!
//! ```
//! use std::f64::consts::PI;
//! use sundials::{context, cvode::{CVode, Solver}};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut ode = CVode::adams(0., &[1., 0.], |t, &[u0, u1], du| {
//!         *du = [-u1, u0]
//!     })
//!     .build(context!(Rot)?)?;
//! let mut u = [f64::NAN; 2];
//! ode.solve(0.5 * PI, &mut u);
//! assert!(u[0].abs() < 1e-6);
//! assert!((u[1] - 1.).abs() < 1e-6);
//! # Ok(()) }
//! ```
//!
//! [Sundials]: https://computing.llnl.gov/projects/sundials

#![cfg_attr(feature = "nightly", feature(allocator_api))]

use std::ptr;
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

#[cfg(feature = "arkode")]
mod arkode;
#[cfg(feature = "cvode")]
pub mod cvode;
#[cfg(feature = "cvodes")]
mod cvodes;
#[cfg(feature = "ida")]
mod ida;
#[cfg(feature = "idas")]
mod idas;
#[cfg(feature = "kinsol")]
mod kinsol;

pub mod vector;
pub mod matrix;
pub mod linear_solver;

////////////////////////////////////////////////////////////////////////
//
// Error

/// Errors for this library.
#[derive(Debug, Clone, Copy,  PartialEq)]
pub enum Error {
    /// Failed to allocate memory.
    AllocFailure,
    /// Incorrect inputs.
    // FIXME: we may want to ensure it can never be returned using the
    // type system.
    IllInput,
    /// The initial time `t0` and the output time `t` are too close to
    /// each other and the user did not specify an initial step size.
    TooClose,
    /// The solver took `mxsteps` internal steps but could not reach the
    /// final time.
    TooMuchWork,
    /// The solver could not satisfy the accuracy demanded by the user
    /// for some internal step.
    TooMuchAcc,
    /// Error test failures occurred too many times during one
    /// internal time step, or the minimum step size was reached.
    Failure,
    /// Convergence test failures occurred too many times during one
    /// internal time step, or the minimum step size was reached.
    ConvFailure,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Error::*;
        match self {
            AllocFailure => writeln!(f, "Sundials could not allocate memory."),
            IllInput => writeln!(f, "Sundials: incorrect inputs"),
            TooClose =>
                writeln!(f, "Sundials: the output time is too close to \
                    the initial time."),
            TooMuchWork =>
                writeln!(f, "Sundials: could not reach final time after \
                    `mxsteps`"),
            TooMuchAcc =>
                writeln!(f, "Sundials: could not satisfy the demanded \
                    accuracy."),
            Failure => writeln!(f, "Sundials: too many error test failures."),
            ConvFailure =>
                writeln!(f, "Sundials: too many convergence test failures."),
        }
    }
}

impl std::error::Error for Error {}

////////////////////////////////////////////////////////////////////////
//
// Context

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
        // There are two cases when we want to skip this dynamic check.
        // 1. When the object is auto-converted from Rust (in this
        //    case using the provided context). See e.g. the train
        //    `Vector` in `vector` and its implementation for `[f64;N]`.
        // 2. When the object comes from Sundials (so already has a
        //    context attached to it) but that context is reflected in
        //    the type, so the type system already guarantees the
        //    equality of contexts.  See the macro `context` below.
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
                fn as_ptr(&self) -> $crate::__SUNContext {
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
/// Make `SUNContext` visible for the `context` macro to work
pub use sundials_sys::SUNContext as __SUNContext;

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
        comm: SUNComm
    ) -> Result<Self, Error> {
        let mut ctx: SUNContext = ptr::null_mut();
        if unsafe { SUNContext_Create(comm, &mut ctx as *mut _) } < 0 {
            return Err(Error::AllocFailure)
        }
        Ok(Self(ctx))
    }

    #[doc(hidden)]
    /// Returns a new `SUNContext` wrapped so that it will be freed
    /// when dropped.
    pub unsafe fn new() -> Result<Self, Error> {
        unsafe { Self::with_communicator(sundials_sys::comm_no_mpi()) }
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


#[cfg(doctest)]
doc_comment::doctest!("../README.md");
