//! Sundials serial N_Vectors.

use sundials_sys::*;
use crate::Context;


pub struct Vector<Ctx> {
    ctx: Ctx,
    nv: N_Vector,
}

impl<Ctx: Context> Vector<Ctx> {
    pub fn context(&self) -> &Ctx { &self.ctx }

    pub fn into_context(self) -> Ctx { self.ctx }

    pub fn new(ctx: Ctx, len: usize) -> Self {
        let nv = unsafe { N_VNew_Serial(
            len as _, ctx.as_ptr()) };
        Self { ctx, nv }
    }

    pub fn len(&self) -> usize {
        unsafe { N_VGetLength_Serial(self.nv) as _ }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
