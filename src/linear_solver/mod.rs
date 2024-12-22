//! Linear solvers.

use sundials_sys::*;
use super::Error;

pub struct LinSolver(SUNLinearSolver);

impl Drop for LinSolver {
    // FIXME: handle possible returned error?
    // https://sundials.readthedocs.io/en/latest/sunlinsol/SUNLinSol_API_link.html?highlight=SUNLinSolFree#c.SUNLinSolFree
    fn drop(&mut self) {
        unsafe { SUNLinSolFree(self.0); }
    }
}

impl LinSolver {
    pub(crate) fn as_ptr(&self) -> SUNLinearSolver {
        self.0
    }

    /// Return a new linear solver.
    ///
    /// # Safety
    /// The return value must not outlive `ctx`.
    pub(crate) unsafe fn spgmr(
        name: &'static str, ctx: SUNContext,
        vec: N_Vector,
    ) -> Result<Self, Error> {
        let linsolver = unsafe {
            SUNLinSol_SPGMR(vec, SUN_PREC_NONE as _, 30, ctx) };
        if linsolver.is_null() {
            Err(Error::Failure {
                name,
                msg: "linear solver  allocation failed"
            })
        } else {
            Ok(LinSolver(linsolver))
        }
    }
}
