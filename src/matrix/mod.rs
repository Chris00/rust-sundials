//! Matrices.

use sundials_sys::*;
use super::{Context, Error};

pub struct Matrix(SUNMatrix);

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe { SUNMatDestroy(self.0) }
    }
}

impl Matrix {
    #[allow(dead_code)]
    pub(crate) fn dense(
        name: &'static str, ctx: &impl Context, m: usize, n: usize,
    ) -> Result<Self, Error> {
        let mat = unsafe {
            SUNDenseMatrix(m as _, n as _, ctx.as_ptr()) };
        if mat.is_null() {
            Err(Error::Failure{name, msg: "matrix allocation failed"})
        } else {
            Ok(Matrix(mat))
        }
    }
}
