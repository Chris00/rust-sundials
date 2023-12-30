//! Vectors

use std::{mem, slice, marker::PhantomData};
use sundials_sys::*;
use super::Context;

/// Trait implemented by types that are considered vectors by this
/// library.
pub unsafe trait Vector: Clone {
    /// Rust view to a `N_Vector` owned by Sundials.
    type View<'a>;
    type ViewMut<'a>;

    fn from_nvector<'a>(nv: N_Vector) -> Self::View<'a>;
    fn from_nvector_mut<'a>(nv: N_Vector) -> Self::ViewMut<'a>;

    /// Reference to a N_Vector.
    type NVectorRef<'a>;
    type NVectorMut<'a>;

    /// Convert `self` to a reference to a `N_Vector` reference.  If
    /// `self` already possesses a [`Context`], this *must* check that
    /// both contexts are the same and return `None` otherwise.
    ///
    /// # Safety
    /// The return value must not outlive `ctx` (and neither `self` but
    /// this is tracked by the lifetime).
    unsafe fn to_nvector(
        &self, ctx: SUNContext) -> Option<Self::NVectorRef<'_>>;

    /// Convert `self` to a reference to a mutable `N_Vector`
    /// reference.  If `self` already possesses a [`Context`], this
    /// *must* check that both contexts are the same and return `None`
    /// otherwise.
    ///
    /// # Safety
    /// The return value must not outlive `ctx` (and neither `self` but
    /// this is tracked by the lifetime).
    unsafe fn to_nvector_mut(
        &mut self, ctx: SUNContext) -> Option<Self::NVectorMut<'_>>;

    /// Cheap projection on a read-only `N_Vector` pointer.
    ///
    /// # Safety
    /// The return value must not outlive `v`.  Moreover, as long as
    /// the returned value is in use, `v` must not be moved.
    fn as_ptr(v: &Self::NVectorRef<'_>) -> *const _generic_N_Vector;

    /// Cheap projection on a mutable `N_Vector` pointer.
    ///
    /// # Safety
    /// The return value must not outlive `v`.  Moreover, as long as
    /// the returned value is in use, `v` must not be moved.
    fn as_mut_ptr(v: &Self::NVectorMut<'_>) -> N_Vector;

    /// Length of the vector.
    fn len(&self) -> usize;
}

/// N_Vector serial wrapper whose "content" field is shared with a
/// Rust value.  This is used internally to convert Rust values to
/// appropriate input/outputs for Sundials routines.
pub struct SharedSerial<V> {
    pub(crate) nv: N_Vector,
    marker: PhantomData<V>, // Lifetime of the Rust vector
}

impl<V> Drop for SharedSerial<V> {
    fn drop(&mut self) {
        // `N_VDestroy_Serial` will only free the data values if
        // `content->own_data` is true — and we are careful to create
        // them with non-owned data.
        unsafe { N_VDestroy_Serial(self.nv) };
    }
}

unsafe impl<const N: usize> Vector for [f64; N] {
    type View<'a> = &'a [f64; N];

    #[inline]
    fn from_nvector<'a>(nv: N_Vector) -> &'a Self {
        unsafe {
            let ptr = N_VGetArrayPointer_Serial(nv);
            // Assume that the requirements are the same as
            // `std::slice::from_raw_parts`.  Check alignment.
            debug_assert_eq!(0,
                (ptr as usize).rem_euclid(mem::align_of::<f64>()));
            &*(ptr as *const [f64; N])
        }
    }

    type ViewMut<'a> = &'a mut [f64; N];

    #[inline]
    fn from_nvector_mut<'a>(nv: N_Vector) -> &'a mut Self {
        unsafe {
            let ptr = N_VGetArrayPointer_Serial(nv);
            debug_assert_eq!(0,
                (ptr as usize).rem_euclid(mem::align_of::<f64>()));
            &mut *(ptr as *mut [f64; N])
        }
    }

    type NVectorRef<'a> = SharedSerial<&'a [f64; N]>;

    #[inline]
    unsafe fn to_nvector(
        &self, ctx: SUNContext
    ) -> Option<Self::NVectorRef<'_>> {
        // `N_VMake_Serial` set `content->own_data` to false, so it
        // will not be freed by the drop trait.
        let nv = unsafe {
            N_VMake_Serial(
                N.try_into().unwrap(),
                self.as_ptr() as *mut _,
                ctx) };
            Some(SharedSerial { nv, marker: PhantomData })
    }

    type NVectorMut<'a> = SharedSerial<&'a mut [f64; N]>;

    #[inline]
    unsafe fn to_nvector_mut(
        &mut self, ctx: SUNContext
    ) -> Option<Self::NVectorMut<'_>> {
        let nv = unsafe {
            N_VMake_Serial(
                N.try_into().unwrap(),
                self.as_mut_ptr(),
                ctx) };
            Some(SharedSerial { nv, marker: PhantomData })
    }

    #[inline]
    fn as_ptr(v: &Self::NVectorRef<'_>) -> *const _generic_N_Vector {
        // FIXME: Do we need to update the data pointer or the fact
        // that `self` is borrowed means its address cannot change?
        v.nv
    }

    #[inline]
    fn as_mut_ptr(v: &Self::NVectorMut<'_>) -> N_Vector {
        v.nv
    }

    #[inline]
    fn len(&self) -> usize { N }
}

/// Types implementing this trait will be acceptable vector types for
/// this library.
pub trait AsVector: AsRef<[f64]> + AsMut<[f64]> + Clone + 'static {}
// This trait is not automatic because we want to treat `[f64; N]`
// in a special way.

impl AsVector for Vec<f64> {}

unsafe impl<V> Vector for V
where V: AsVector {
    // It is not possible to reconstruct `V` (e.g. `Vec<f64>`) from
    // `N_Vector` because we do not own the `N_Vector` that Sundials
    // gives us.  Also it was likely provided by a different allocator.
    type View<'a> = &'a [f64];

    #[inline]
    fn from_nvector<'a>(nv: N_Vector) -> &'a [f64] {
        unsafe {
            let ptr = N_VGetArrayPointer_Serial(nv);
            // Check alignment requirements of `std::slice::from_raw_parts`.
            debug_assert_eq!(0,
                (ptr as usize).rem_euclid(mem::align_of::<f64>()));
            let n = N_VGetLength_Serial(nv);
            slice::from_raw_parts(ptr, n as _)
        }
    }

    type ViewMut<'a> = &'a mut [f64];

    #[inline]
    fn from_nvector_mut<'a>(nv: N_Vector) -> &'a mut [f64] {
        unsafe {
            let ptr = N_VGetArrayPointer_Serial(nv);
            // Check alignment requirements of `slice::from_raw_parts_mut`.
            debug_assert_eq!(0,
                (ptr as usize).rem_euclid(mem::align_of::<f64>()));
            let n = N_VGetLength_Serial(nv);
            slice::from_raw_parts_mut(ptr, n as _)
        }
    }

    type NVectorRef<'a> = SharedSerial<&'a V>;

    #[inline]
    unsafe fn to_nvector(
        &self, ctx: SUNContext
    ) -> Option<Self::NVectorRef<'_>> {
        let nv = unsafe {
            N_VMake_Serial(
                self.len().try_into().unwrap(),
                self.as_ref().as_ptr() as *mut _,
                ctx) };
        Some(SharedSerial { nv, marker: PhantomData })
    }

    type NVectorMut<'a> = SharedSerial<&'a mut V>;

    #[inline]
    unsafe fn to_nvector_mut(
        &mut self, ctx: SUNContext
    ) -> Option<Self::NVectorMut<'_>> {
        let nv = unsafe {
            N_VMake_Serial(
                self.len().try_into().unwrap(),
                self.as_mut().as_ptr() as *mut _,
                ctx) };
        Some(SharedSerial { nv, marker: PhantomData })
    }

    #[inline]
    fn as_ptr(v: &Self::NVectorRef<'_>) -> *const _generic_N_Vector {
        // FIXME: update the data pointer?
        v.nv as *const _generic_N_Vector
    }

    #[inline]
    fn as_mut_ptr(v: &Self::NVectorMut<'_>) -> N_Vector {
        v.nv
    }

    #[inline]
    fn len(&self) -> usize { self.as_ref().len() }
}



/// Trait implemented by all vector types provided by Sundials.
pub trait SunVector: Clone {
    /// Context of the vector.
    type Ctx: Context;

    // FIXME: Needed?
    // fn context(&self) -> &Context;

    // Must take ownership of `self` as a context can only be used in
    // a single ODE solver.
    // fn into_context(self) -> Context;

}

