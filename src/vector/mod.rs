//! Vectors

use sundials_sys::*;

pub mod custom;
pub mod serial;

/// Trait implemented by types that are considered vectors by this
/// library.
///
/// # Safety
/// You should likely not need to implement this trait.  Its functions
/// are only meant to be called to interact with Sundials.
pub unsafe trait Vector: Clone {
    /// Length of the vector.
    ///
    /// It is a function and not a method to avoid conflicting with
    /// methods with the same name defined on `v`.
    fn len(v: &Self) -> usize;

    fn from_nvector<'a>(nv: N_Vector) -> &'a Self;

    /// # Safety
    /// This function is only called on N_Vectors `nv` owned by
    /// Sundials.  Thus it will not violate the fact that mutable
    /// references are exclusive.
    fn from_nvector_mut<'a>(nv: N_Vector) -> &'a mut Self;

    /// Wrapper of a N_Vector borrowing `self`.
    type NVectorRef<'a>;
    /// Wrapper of a N_Vector mutably borrowing `self`.
    type NVectorMut<'a>;

    /// Return a wrapper of a `N_Vector` that references `self`.  If
    /// `self` already possesses a [`Context`][crate::Context], this
    /// *must* check that both contexts are the same and return `None`
    /// otherwise.
    ///
    /// # Safety
    /// The return value must not outlive `ctx` (and neither `self` but
    /// this is tracked by the lifetime).
    unsafe fn as_nvector(
        v: &Self, ctx: SUNContext) -> Option<Self::NVectorRef<'_>>;

    /// Return a wrapper of a `N_Vector` that can mutate `self`.  If
    /// `self` already possesses a [`Context`][crate::Context], this
    /// *must* check that both contexts are the same and return `None`
    /// otherwise.
    ///
    /// # Safety
    /// The return value must not outlive `ctx` (and neither `self` but
    /// this is tracked by the lifetime).
    unsafe fn as_mut_nvector(
        v: &mut Self, ctx: SUNContext) -> Option<Self::NVectorMut<'_>>;

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
}
