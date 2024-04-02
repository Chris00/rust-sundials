//! Generic vector trait and its implementations.

use std::ffi::c_void;
use sundials_sys::*;

pub mod serial;

/// Trait implemented by types that are considered vectors by this
/// library.  This trait is implemented by all rust values that
/// satisfy [`NVectorOps`] as well as Sundials vector types (see the
/// sub-modules).  In particular `[f64; N]`, `Vec<f64>` and `f64` and
/// `ndarray::Array1<f64>` (activating the feature "ndarray") can be
/// used as vectors out of the box.  Moreover, if `T1`, `T2`,
/// `T3`,... are types considered as vectors (more specifically, they
/// implement [`NVectorOps`]), then `(T1, T2)`,
/// `(T1, T2, T3)`,... also do.
///
/// # Safety
/// It is very unlikely that you need to implement this trait.  Its
/// functions are only meant to be called to interact with Sundials.
pub unsafe trait Vector: Clone {
    /// Length of the vector.
    ///
    /// It is a function and not a method to avoid conflicting with
    /// methods with the same name defined on `v`.
    fn len(v: &Self) -> usize;

    /// # Safety
    /// This function is only called on N_Vectors `nv` owned by
    /// Sundials.
    fn from_nvector<'a>(nv: N_Vector) -> &'a Self;

    /// # Safety
    /// This function is only called on N_Vectors `nv` owned by
    /// Sundials.  Thus it will not violate the fact that mutable
    /// references are exclusive.
    fn from_nvector_mut<'a>(nv: N_Vector) -> &'a mut Self;

    /// Return a wrapper of a `N_Vector` that references `self`.  If
    /// `self` already possesses a [`Context`][crate::Context], this
    /// *must* check that both contexts are the same and return `None`
    /// otherwise.
    ///
    /// # Safety
    /// The return value must not outlive `v` and `ctx`.  Moreover `v`
    /// should not move while the return value is in use.
    unsafe fn as_nvector(
        v: &Self, ctx: SUNContext) -> Option<*const _generic_N_Vector>;

    /// Return a wrapper of a `N_Vector` that can mutate `self`.  If
    /// `self` already possesses a [`Context`][crate::Context], this
    /// *must* check that both contexts are the same and return `None`
    /// otherwise.
    ///
    /// # Safety
    /// The return value must not outlive `v` and `ctx`.  Moreover `v`
    /// should not move while the return value is in use.
    unsafe fn as_mut_nvector(
        v: &mut Self, ctx: SUNContext) -> Option<N_Vector>;
}


/// Operations that Rust values must support to be used as vectors for
/// this library.
///
/// These operations are declared as associate functions to reduce the
/// possible name clash with methods.
pub trait NVectorOps: Clone {
    /// Length of the vector.
    fn len(x: &Self) -> usize;

    /// Sets all components of `z` to `c`: ∀i, zᵢ = c.
    fn const_assign(z: &mut Self, c: f64);

    /// Set each component of `z` to the absolute value of the
    /// corresponding component in `x`: ∀i, zᵢ = |xᵢ|.
    fn abs_assign(z: &mut Self, x: &Self);

    /// Set `z` to the component-wise product of `x` and `y`:
    /// ∀i, zᵢ = xᵢ yᵢ.
    fn mul_assign(z: &mut Self, x: &Self, y: &Self);

    /// Set each component of the `z` to the inverse of the
    /// corresponding component of `x`: ∀i, zᵢ = 1/xᵢ.
    fn inv_assign(z: &mut Self, x: &Self);

    /// Replace each component of the `z` by its inverse: ∀i, zᵢ = 1/zᵢ.
    fn inv(z: &mut Self);

    /// Set `z` to the component-wise ratio of `x` and `y`:
    /// ∀i, zᵢ = xᵢ/yᵢ.  The yᵢ may not be tested for 0 values.
    fn div_assign(z: &mut Self, x: &Self, y: &Self);

    /// Set `z` to the component-wise ratio of `z` and `y`:
    /// ∀i, zᵢ = zᵢ/yᵢ.  The yᵢ may not be tested for 0 values.
    fn div(z: &mut Self, y: &Self);

    /// Set `z` to the component-wise ratio of `x` and `z`:
    /// ∀i, zᵢ = xᵢ/zᵢ.  The zᵢ may not be tested for 0 values.
    fn inv_mul(z: &mut Self, x: &Self);

    /// Set `z` to the scaling of `x` by the factor `c`: ∀i, zᵢ = cxᵢ.
    fn scale_assign(z: &mut Self, c: f64, x: &Self);

    /// Scale `z` by the factor `c`: ∀i, zᵢ = czᵢ.
    fn scale(z: &mut Self, c: f64);

    /// Set each component of `z` to the sum of the corresponding
    /// component in `x` and `b`: ∀i, zᵢ = xᵢ + b.
    fn add_const_assign(z: &mut Self, x: &Self, b: f64);

    /// Add `b` to each component of `z`: ∀i, zᵢ = zᵢ + b.
    fn add_const(z: &mut Self, b: f64);

    /// Performs the operation `z = ax + by`.
    fn linear_sum_assign(z: &mut Self, a: f64, x: &Self, b: f64, y: &Self);

    /// Performs the operation `z = az + by`.
    fn linear_sum(z: &mut Self, a: f64, b: f64, y: &Self);

    /// Return the dot-product of `x` and `y`: ∑ xᵢ yᵢ.
    fn dot(x: &Self, y: &Self) -> f64;

    /// Returns the value of the ℓ^∞ norm of `x`: maxᵢ |xᵢ|.
    fn max_norm(x: &Self) -> f64;

    /// Returns the weighted root-mean-square norm of `x` with
    /// (positive) weight vector `nw`: √(∑ (xᵢ wᵢ)² / n)
    /// where n is the length of `x` and `w`.
    fn wrms_norm(x: &Self, w: &Self) -> f64;

    /// Returns the weighted root mean square norm of `x` with weight
    /// vector `w` built using only the elements of `x` corresponding
    /// to positive elements of the `id`: √(∑ (xᵢ wᵢ H(idᵢ))² / n)
    /// where H(α) = 1 if α > 0 and H(α) = 0 if α ≤ 0
    /// and n is the length of `x`, `w` and `id`.
    fn wrms_norm_mask(x: &Self, w: &Self, id: &Self) -> f64;

    /// Returns the smallest element of the `x`: minᵢ xᵢ.
    fn min(x: &Self) -> f64;

    /// Returns the weighted Euclidean norm of `x` with weight vector
    /// `w`: √(∑ (xᵢ wᵢ)²).
    fn wl2_norm(x: &Self, w: &Self) -> f64;

    /// Returns the ℓ¹ norm of `x`: ∑ |xᵢ|.
    fn l1_norm(x: &Self) -> f64;

    /// Compare the components of `x` to the scalar `c` and set `z`
    /// such that ∀i, zᵢ = 1.0 if |xᵢ| ≥ `c` and zᵢ = 0.0 otherwise.
    fn compare_assign(z: &mut Self, c: f64, x: &Self);

    /// Sets the components of `z` to be the inverses of the
    /// components of `x`, with prior testing for zero values:
    /// ∀i, zᵢ = 1/xᵢ.  This routine returns `true` if all components
    /// of x are nonzero (successful inversion) and returns `false`
    /// otherwise.
    fn inv_test_assign(z: &mut Self, x: &Self) -> bool;

    /// Performs the following constraint tests based on the values in cᵢ:
    /// - xᵢ > 0 if cᵢ = 2,
    /// - xᵢ ≥ 0 if cᵢ = 1,
    /// - xᵢ < 0 if cᵢ = -2,
    /// - xᵢ ≤ 0 if cᵢ = -1.
    /// There is no constraint on xᵢ if cᵢ = 0.  This routine returns
    /// `false` if any element failed the constraint test and `true`
    /// if all passed.  It also sets a mask vector `m`, with elements
    /// equal to 1.0 where the constraint test failed, and 0.0 where
    /// the test passed.  This routine is used only for constraint
    /// checking.
    fn constr_mask_assign(m: &mut Self, c: &Self, x: &Self) -> bool;

    /// Returns the minimum of the quotients obtained by termwise
    /// dividing the elements of `num` by the elements in `denom`:
    /// minᵢ numᵢ/denomᵢ.  Zero elements in `denom` must be skipped.
    /// If no such quotients are found, then `f64::MAX` must be returned.
    fn min_quotient(num: &Self, denom: &Self) -> f64;
}

//////////////////////////////////////////////////////////////////////
//
// Implement `Vector` for types supporting `NVectorOps`.

extern "C" fn nvgetvectorid_rust(_: N_Vector) -> N_Vector_ID {
    // See vendor/include/sundials/sundials_nvector.h
    N_Vector_ID_SUNDIALS_NVEC_CUSTOM
}

trait Ops {
    const OPS: _generic_N_Vector_Ops;
}

impl<T: NVectorOps> Ops for T {
    const OPS: _generic_N_Vector_Ops = {
        /// Return the Rust value stored in the N_Vector.
        ///
        /// # Safety
        /// This box must be leaked before it is dropped because the
        /// data belongs to `nv` if internal to Sundials or to another
        /// Rust value if it is shared.
        #[inline]
        unsafe fn mut_of_nvector<'a, T>(nv: N_Vector) -> &'a mut T {
            // FIXME: When `nv` content is shared with another Rust
            // value, there will be temporarily be two Rust values
            // pointing to the same data.  Make sure it is fine as
            // this one is very local.
            ((*nv).content as *mut T).as_mut().unwrap()
        }

        #[inline]
        unsafe fn ref_of_nvector<'a, T>(nv: N_Vector) -> &'a T {
            mut_of_nvector(nv)
        }

        /// Creates a new N_Vector of the same type as an existing vector
        /// `nv` and sets the ops field.  It does not copy the vector, but
        /// rather allocates storage for the new vector.
        #[cfg(feature = "nightly")]
        unsafe extern "C" fn nvclone_rust<T: NVectorOps>(
            nw: N_Vector
        ) -> N_Vector {
            let w = ref_of_nvector::<T>(nw);
            // Rust memory cannot be uninitialized, thus clone.
            let v = w.clone();
            //// Do not go through the C malloc for this
            let sunctx = (*nw).sunctx;
            let ops = (*(*nw).ops).clone();
            let v = _generic_N_Vector {
                sunctx,
                ops: Box::into_raw(Box::new(ops)),
                content: Box::into_raw(Box::new(v)) as *mut c_void,
            };
            // The `new_in` using the `System` allocator should be
            // compatible with the C code.
            Box::into_raw(Box::new_in(v, std::alloc::System))
        }

        #[cfg(not(feature = "nightly"))]
        unsafe extern "C" fn nvclone_rust<T: NVectorOps>(
            nw: N_Vector
        ) -> N_Vector {
            let w = ref_of_nvector::<T>(nw);
            // Rust memory cannot be uninitialized, thus clone.
            let v = w.clone();
            //// Sundials functions — slow.
            // let nv = N_VNewEmpty((*nw).sunctx);
            // if N_VCopyOps(nw, nv) != 0 {
            //     return std::ptr::null_mut()
            // }
            // (*nv).content = Box::into_raw(Box::new(v)) as *mut c_void;
            // nv

            //// libc version — safe as Sundials uses malloc.
            let sunctx = (*nw).sunctx;
            let nv = libc::malloc(
                std::mem::size_of::<_generic_N_Vector>()) as N_Vector;
            (*nv).sunctx = sunctx;
            let n = std::mem::size_of::<_generic_N_Vector_Ops>();
            let ops = libc::malloc(n);
            libc::memcpy(ops, (*nw).ops as *mut c_void, n);
            (*nv).ops = ops as N_Vector_Ops;
            (*nv).content = Box::into_raw(Box::new(v)) as *mut c_void;
            nv
        }

        /// Destroys the N_Vector `nv` and frees memory allocated for
        /// its internal data.
        unsafe extern "C" fn nvdestroy_rust<T: NVectorOps>(nv: N_Vector) {
            // This is for N_Vectors managed by Sundials.  Rust `Shared`
            // values will not call N_Vector operations.
            let v = Box::from_raw((*nv).content as *mut T);
            drop(v);
            N_VFreeEmpty(nv);
        }

        /// Returns storage requirements for the N_Vector `nv`:
        /// - `lrw` contains the number of realtype words;
        /// - `liw` contains the number of integer words.
        /// This function is advisory only.
        unsafe extern "C" fn nvspace_rust<T: NVectorOps>(
            nv: N_Vector, lrw: *mut sunindextype, liw: *mut sunindextype
        ) {
            let v = ref_of_nvector::<T>(nv);
            let n = T::len(v);
            *lrw = n as sunindextype;
            *liw = 1;
        }

        /// Returns the global length (number of “active” entries) in the
        /// N_Vector `nv`.
        unsafe extern "C" fn nvgetlength_rust<T: NVectorOps>(
            nv: N_Vector
        ) -> sunindextype {
            let v = ref_of_nvector::<T>(nv);
            let n = T::len(v);
            n as sunindextype
        }

        /// Performs the operation `z = ax + by`.
        unsafe extern "C" fn nvlinearsum_rust<T: NVectorOps>(
            a: realtype, nx: N_Vector, b: realtype, ny: N_Vector, nz: N_Vector
        ) {
            let z = mut_of_nvector(nz);
            if nz == nx && nz == ny { // z = (a + b) z
                T::scale(z, a+b);
                return
            }
            if nz == nx { // ≠ ny
                let y = ref_of_nvector(ny);
                T::linear_sum(z, a, b, y);
                return
            }
            if nz == ny { // ≠ nx
                let x = ref_of_nvector(nx);
                T::linear_sum(z, b, a, x);
                return
            }
            let x = ref_of_nvector(nx);
            let y = ref_of_nvector(ny);
            T::linear_sum_assign(z, a, x, b, y);
        }

        /// Sets all components of `nz` to realtype `c`.
        unsafe extern "C" fn nvconst_rust<T: NVectorOps>(
            c: realtype, nz: N_Vector
        ) {
            let z = mut_of_nvector::<T>(nz);
            T::const_assign(z, c);
        }

        /// Sets `nz` to be the component-wise product of the inputs `nx`
        /// and `ny`: ∀i, zᵢ = xᵢ yᵢ.
        unsafe extern "C" fn nvprod_rust<T: NVectorOps>(
            nx: N_Vector, ny: N_Vector, nz: N_Vector
        ) {
            assert_ne!(nz, nx);
            assert_ne!(nz, ny);
            let x = ref_of_nvector(nx);
            let y = ref_of_nvector(ny);
            let z = mut_of_nvector(nz);
            T::mul_assign(z, x, y);
        }

        /// Sets the `nz` to be the component-wise ratio of the inputs
        /// `nx` and `ny`: ∀i, zᵢ = xᵢ/yᵢ.  The yᵢ may not be tested
        /// for 0 values.  This function should only be called with a
        /// y that is guaranteed to have all nonzero components.
        unsafe extern "C" fn nvdiv_rust<T: NVectorOps>(
            nx: N_Vector, ny: N_Vector, nz: N_Vector
        ) {
            let z = mut_of_nvector(nz);
            if nz == nx {
                if nz == ny {
                    T::const_assign(z, 1.);
                    return
                } else {
                    let y = ref_of_nvector(ny);
                    T::div(z, y);
                    return
                }
            }
            if nz == ny {
                let x = ref_of_nvector(nx);
                T::inv_mul(z, x);
                return
            }
            let x = ref_of_nvector(nx);
            let y = ref_of_nvector(ny);
            T::div_assign(z, x, y);
        }

        /// Scales the `nx` by the scalar `c` and returns the result
        /// in `z`: ∀i, zᵢ = cxᵢ.
        unsafe extern "C" fn nvscale_rust<T: NVectorOps>(
            c: f64, nx: N_Vector, nz: N_Vector
        ) {
            let z = mut_of_nvector(nz);
            if nz == nx {
                T::scale(z, c);
            } else {
                let x = ref_of_nvector(nx);
                T::scale_assign(z, c, x);
            }
        }

        /// Sets the components of the `nz` to be the absolute values
        /// of the components of the `nx`: ∀i, zᵢ = |xᵢ|.
        unsafe extern "C" fn nvabs_rust<T: NVectorOps>(
            nx: N_Vector, nz: N_Vector
        ) {
            assert_ne!(nz, nx);
            let x = ref_of_nvector(nx);
            let z = mut_of_nvector(nz);
            T::abs_assign(z, x);
        }

        /// Sets the components of the `nz` to be the inverses of the
        /// components of `nx`: ∀i, zᵢ = 1/xᵢ.
        unsafe extern "C" fn nvinv_rust<T: NVectorOps>(
            nx: N_Vector, nz: N_Vector
        ) {
            let z = mut_of_nvector(nz);
            if nz == nx {
                T::inv(z);
            } else {
                let x = ref_of_nvector(nx);
                T::inv_assign(z, x);
            }
        }

        /// Adds the scalar `b` to all components of `nx` and returns
        /// the result in `nz`: ∀i, zᵢ = xᵢ + b.
        unsafe extern "C" fn nvaddconst_rust<T: NVectorOps>(
            nx: N_Vector, b: f64, nz: N_Vector
        ) {
            let z = mut_of_nvector(nz);
            if nz == nx {
                T::add_const(z, b);
            } else {
                let x = ref_of_nvector(nx);
                T::add_const_assign(z, x, b);
            }
        }

        /// Returns the value of the dot-product of `nx` and `ny`: ∑ xᵢ yᵢ.
        unsafe extern "C" fn nvdotprod_rust<T: NVectorOps>(
            nx: N_Vector, ny: N_Vector
        ) -> realtype {
            let x = ref_of_nvector(nx);
            let y = ref_of_nvector(ny);
            T::dot(&x, &y)
        }

        /// Returns the value of the ℓ^∞ norm of `nx`: maxᵢ |xᵢ|.
        unsafe extern "C" fn nvmaxnorm_rust<T: NVectorOps>(
            nx: N_Vector
        ) -> realtype {
            let x = ref_of_nvector(nx);
            T::max_norm(x)
        }

        /// Returns the weighted root-mean-square norm of `nx` with
        /// (positive) realtype weight vector `nw`: √(∑ (xᵢ wᵢ)² / n).
        unsafe extern "C" fn nvwrmsnorm_rust<T: NVectorOps>(
            nx: N_Vector, nw: N_Vector
        ) -> realtype {
            let x = ref_of_nvector(nx);
            let w = ref_of_nvector(nw);
            T::wrms_norm(&x, &w)
        }

        /// Returns the weighted root mean square norm of `nx` with
        /// weight vector `nw` built using only the elements of `nx`
        /// corresponding to positive elements of the `nid`:
        /// √(∑ (xᵢ wᵢ H(idᵢ))² / n) where H(α) = 1 if α > 0 and
        /// H(α) = 0 if α ≤ 0.
        unsafe extern "C" fn nvwrmsnormmask_rust<T: NVectorOps>(
            nx: N_Vector, nw: N_Vector, nid: N_Vector
        ) -> realtype {
            let x = ref_of_nvector(nx);
            let w = ref_of_nvector(nw);
            let id = ref_of_nvector(nid);
            T::wrms_norm_mask(x, w, id)
        }

        /// Returns the smallest element of the `nx`: minᵢ xᵢ.
        unsafe extern "C" fn nvmin_rust<T: NVectorOps>(
            nx: N_Vector
        ) -> realtype {
            let x = ref_of_nvector(nx);
            T::min(x)
        }

        /// Returns the weighted Euclidean norm of `nx` with realtype
        /// weight vector `nw`: √(∑ (xᵢ wᵢ)²).
        unsafe extern "C" fn nvwl2norm_rust<T: NVectorOps>(
            nx: N_Vector, nw: N_Vector
        ) -> realtype {
            let x = ref_of_nvector(nx);
            let w = ref_of_nvector(nw);
            T::wl2_norm(x, w)
        }

        /// Returns the ℓ¹ norm of `nx`: ∑ |xᵢ|.
        unsafe extern "C" fn nvl1norm_rust<T: NVectorOps>(
            nx: N_Vector
        ) -> realtype {
            let x = ref_of_nvector(nx);
            T::l1_norm(x)
        }

        /// Compares the components of `nx` to the realtype scalar `c`
        /// and returns a `nz` such that ∀i, zᵢ = 1.0 if |xᵢ| ≥ `c`
        /// and zᵢ = 0.0 otherwise.
        unsafe extern "C" fn nvcompare_rust<T: NVectorOps>(
            c: f64, nx: N_Vector, nz: N_Vector
        ) {
            assert_ne!(nz, nx);
            let x = ref_of_nvector(nx);
            let z = mut_of_nvector(nz);
            T::compare_assign(z, c, x)
        }

        /// Sets the components of `nz` to be the inverses of the
        /// components of `nx`, with prior testing for zero values:
        /// ∀i, zᵢ = 1/xᵢ.  This routine returns a boolean assigned to
        /// `SUNTRUE` if all components of x are nonzero (successful
        /// inversion) and returns `SUNFALSE` otherwise.
        unsafe extern "C" fn nvinvtest_rust<T: NVectorOps>(
            nx: N_Vector, nz: N_Vector
        ) -> i32 {
            assert_ne!(nz, nx);
            let x = ref_of_nvector(nx);
            let z = mut_of_nvector(nz);
            let b = T::inv_test_assign(z, x);
            (if b { SUNTRUE } else { SUNFALSE }) as _
        }

        /// Performs the following constraint tests based on the values in cᵢ:
        /// - xᵢ > 0 if cᵢ = 2,
        /// - xᵢ ≥ 0 if cᵢ = 1,
        /// - xᵢ < 0 if cᵢ = -2,
        /// - xᵢ ≤ 0 if cᵢ = -1.
        /// There is no constraint on xᵢ if cᵢ = 0.  This routine
        /// returns a boolean assigned to `SUNFALSE` if any element
        /// failed the constraint test and assigned to `SUNTRUE` if
        /// all passed.  It also sets a mask vector `nm`, with
        /// elements equal to 1.0 where the constraint test failed,
        /// and 0.0 where the test passed.  This routine is used only
        /// for constraint checking.
        unsafe extern "C" fn nvconstrmask_rust<T: NVectorOps>(
            nc: N_Vector, nx: N_Vector, nm: N_Vector
        ) -> i32 {
            assert_ne!(nm, nc);
            assert_ne!(nm, nx);
            let c = ref_of_nvector(nc);
            let x = ref_of_nvector(nx);
            let m = mut_of_nvector(nm);
            let b = T::constr_mask_assign(m, c, x);
            (if b { SUNTRUE } else { SUNFALSE }) as _
        }

        /// This routine returns the minimum of the quotients obtained
        /// by termwise dividing the elements of n = `nnum` by the
        /// elements in d = `ndenom`: minᵢ nᵢ/dᵢ.
        unsafe extern "C" fn nvminquotient_rust<T: NVectorOps>(
            nnum: N_Vector, ndenom: N_Vector
        ) -> f64 {
            let num = ref_of_nvector(nnum);
            let denom = ref_of_nvector(ndenom);
            T::min_quotient(num, denom)
        }

        _generic_N_Vector_Ops {
            nvgetvectorid: Some(nvgetvectorid_rust),
            nvclone: Some(nvclone_rust::<T>),
            // Contrarily to "nvclone", "nvcloneempty" does not
            // allocate the storage for the vector data.  This is only
            // used in some linear solvers (that custom vectors will
            // not support) in combination with `N_VSetArrayPointer`.
            nvcloneempty: None,
            nvdestroy: Some(nvdestroy_rust::<T>),
            nvspace: Some(nvspace_rust::<T>),
            // `nvgetarraypointer` and `nvsetarraypointer` are used in
            // some linear solvers, thus not on values managed by Rust.
            nvgetarraypointer: None,
            nvgetdevicearraypointer: None,
            nvsetarraypointer: None,
            nvgetcommunicator: None,
            nvgetlength: Some(nvgetlength_rust::<T>),
            nvgetlocallength: Some(nvgetlength_rust::<T>),
            // Standard vector operations
            nvlinearsum: Some(nvlinearsum_rust::<T>),
            nvconst: Some(nvconst_rust::<T>),
            nvprod: Some(nvprod_rust::<T>),
            nvdiv: Some(nvdiv_rust::<T>),
            nvscale: Some(nvscale_rust::<T>),
            nvabs: Some(nvabs_rust::<T>),
            nvinv: Some(nvinv_rust::<T>),
            nvaddconst: Some(nvaddconst_rust::<T>),
            nvdotprod: Some(nvdotprod_rust::<T>),
            nvmaxnorm: Some(nvmaxnorm_rust::<T>),
            nvwrmsnorm: Some(nvwrmsnorm_rust::<T>),
            nvwrmsnormmask: Some(nvwrmsnormmask_rust::<T>),
            nvmin: Some(nvmin_rust::<T>),
            nvwl2norm: Some(nvwl2norm_rust::<T>),
            nvl1norm: Some(nvl1norm_rust::<T>),
            nvcompare: Some(nvcompare_rust::<T>),
            nvinvtest: Some(nvinvtest_rust::<T>),
            nvconstrmask: Some(nvconstrmask_rust::<T>),
            nvminquotient: Some(nvminquotient_rust::<T>),
            // OPTIONAL operations.
            // These operations provide default implementations that
            // may be overridden.
            // OPTIONAL fused vector operations
            nvlinearcombination: None,
            nvscaleaddmulti: None,
            nvdotprodmulti: None,
            // OPTIONAL vector array operations
            nvlinearsumvectorarray: None,
            nvscalevectorarray: None,
            nvconstvectorarray: None,
            nvwrmsnormvectorarray: None,
            nvwrmsnormmaskvectorarray: None,
            nvscaleaddmultivectorarray: None,
            nvlinearcombinationvectorarray: None,
            // OPTIONAL operations with no default implementation.
            // Local reduction kernels (no parallel communication)
            nvdotprodlocal: None,
            nvmaxnormlocal: None,
            nvminlocal: None,
            nvl1normlocal: None,
            nvinvtestlocal: None,
            nvconstrmasklocal: None,
            nvminquotientlocal: None,
            nvwsqrsumlocal: None,
            nvwsqrsummasklocal: None,
            // Single buffer reduction operations
            nvdotprodmultilocal: None,
            nvdotprodmultiallreduce: None,
            // XBraid interface operations
            nvbufsize: None,
            nvbufpack: None,
            nvbufunpack: None,
            // Debugging functions (called when
            // SUNDIALS_DEBUG_PRINTVEC is defined).
            nvprint: None,
            nvprintfile: None,
        }
    };
}

#[inline]
unsafe fn new_nvector<T: NVectorOps + Ops>(ctx: SUNContext) -> N_Vector {
    let nv = N_VNewEmpty(ctx);
    if nv.is_null() {
        panic!("sundials::vector::custom::Shared::new: \
                Could not allocate new N_Vector.");
    }
    (*nv).ops = Box::into_raw(Box::new(T::OPS));
    nv
}

unsafe impl<T: NVectorOps> Vector for T {
    fn len(v: &Self) -> usize { T::len(v) }

    fn from_nvector<'a>(nv: N_Vector) -> &'a Self {
        unsafe {
            ((*nv).content as *const Self).as_ref().unwrap()
        }
    }

    fn from_nvector_mut<'a>(nv: N_Vector) -> &'a mut Self {
        unsafe {
            ((*nv).content as *mut Self).as_mut().unwrap()
        }
    }

    #[inline]
    unsafe fn as_nvector(
        v: &Self, ctx: SUNContext
    ) -> Option<*const _generic_N_Vector> {
        // See https://sundials.readthedocs.io/en/latest/nvectors/NVector_API_link.html#implementing-a-custom-nvector
        let nv = new_nvector::<T>(ctx);
        (*nv).content = v as *const T as *mut c_void;
        Some(nv)
    }

    #[inline]
    unsafe fn as_mut_nvector(
        v: &mut Self, ctx: SUNContext
    ) -> Option<N_Vector> {
        let nv = new_nvector::<T>(ctx);
        (*nv).content = v as *mut T as *mut c_void;
        Some(nv)
    }
}


////////////////////////////////////////////////////////////////////////
//
// Implementation for Arrays and Vec

// Implementation of Sundials operations on slices following
// vendor/src/nvector/serial/nvector_serial.c but using iterators in
// order to have index free loops.

/// z = x + y
fn add_assign_serial<'a>(
    z: impl Iterator<Item = &'a mut f64>,
    x: impl Iterator<Item = &'a f64>,
    y: impl Iterator<Item = &'a f64>,
) {
    for ((&x, &y), z) in x.zip(y).zip(z) {
        *z = x + y;
    }
}

/// z = x - y
fn sub_assign_serial<'a>(
    z: impl Iterator<Item = &'a mut f64>,
    x: impl Iterator<Item = &'a f64>,
    y: impl Iterator<Item = &'a f64>,
) {
    for ((&x, &y), z) in x.zip(y).zip(z) {
        *z = x - y;
    }
}

/// z = ax + by
fn linear_sum_serial<'a>(
    z: impl Iterator<Item = &'a mut f64>,
    a: f64, x: impl Iterator<Item = &'a f64>,
    b: f64, y: impl Iterator<Item = &'a f64>,
) {
    if a == 1. && b == 1. {
        add_assign_serial(z, x, y);
        return;
    }
    if a == 1. && b == -1. {
        sub_assign_serial(z, x, y);
        return;
    }
    if a == -1. && b == 1. {
        sub_assign_serial(z, y, x);
        return;
    }

    for ((&x, &y), z) in x.zip(y).zip(z) {
        *z = a * x + b * y;
    }
}

macro_rules! nvector_ops_for_iter { () => {
    #[inline]
    fn linear_sum_assign(z: &mut Self, a: f64, x: &Self, b: f64, y: &Self) {
        linear_sum_serial(z.iter_mut(), a, x.iter(), b, y.iter())
    }

    #[inline]
    fn linear_sum(z: &mut Self, a: f64, b: f64, y: &Self) {
        // FIXME: Distinguish more cases such as b = ±1?
        if a == 1. {
            for (z, &y) in z.iter_mut().zip(y) {
                *z += b * y
            }
        } else {
            for (z, &y) in z.iter_mut().zip(y) {
                *z = a * *z + b * y
            }
        }
    }

    #[inline]
    fn const_assign(z: &mut Self, c: f64) {
        for z in z {
            *z = c;
        }
    }

    #[inline]
    fn mul_assign(z: &mut Self, x: &Self, y: &Self) {
        for ((&x, &y), z) in x.iter().zip(y).zip(z) {
            *z = x * y;
        }
    }

    #[inline]
    fn div_assign(z: &mut Self, x: &Self, y: &Self) {
        for ((&x, &y), z) in x.iter().zip(y).zip(z) {
            *z = x / y;
        }
    }

    #[inline]
    fn div(z: &mut Self, y: &Self) {
        for (z, &y) in z.iter_mut().zip(y) {
            *z /= y;
        }
    }

    #[inline]
    fn inv_mul(z: &mut Self, x: &Self) {
        for (z, &x) in z.iter_mut().zip(x) {
            *z = x / *z;
        }
    }

    #[inline]
    fn scale_assign(z: &mut Self, c: f64, x: &Self) {
        // FIXME: want to distinguish the cases c == 1. and c == -1. ?
        for (&x, z) in x.iter().zip(z) {
            *z = c * x;
        }
    }

    #[inline]
    fn scale(z: &mut Self, c: f64) {
        for z in z {
            *z *= c;
        }
    }

    #[inline]
    fn abs_assign(z: &mut Self, x: &Self) {
        for (x, z) in x.iter().zip(z) {
            *z = x.abs();
        }
    }

    #[inline]
    fn inv_assign(z: &mut Self, x: &Self) {
        for (&x, z) in x.iter().zip(z) {
            *z = 1. / x;
        }
    }

    #[inline]
    fn inv(z: &mut Self) {
        for z in z { *z = 1. / *z }
    }

    #[inline]
    fn add_const_assign(z: &mut Self, x: &Self, b: f64) {
        for (&x, z) in x.iter().zip(z) {
            *z = x + b;
        }
    }

    #[inline]
    fn add_const(z: &mut Self, b: f64) {
        for z in z {
            *z += b;
        }
    }

    #[inline]
    fn dot(x: &Self, y: &Self) -> f64 {
        let mut sum = 0.;
        for (&x, &y) in x.iter().zip(y) {
            sum += x * y
        }
        sum
    }

    #[inline]
    fn max_norm(x: &Self) -> f64 {
        let mut m = 0.;  // Zero length ⇒ vector space is {0}
        for x in x {
            let x_abs = x.abs();
            if x_abs > m { m = x_abs }
        }
        m
    }

    #[inline]
    fn wrms_norm(x: &Self, w: &Self) -> f64 {
        debug_assert_eq!(x.len(), w.len());
        // FIXME: handle better possible over/underflow.
        let mut sum = 0.;
        for (&x, &w) in x.iter().zip(w) {
            sum += (x * w).powi(2);
        }
        (sum / x.len() as f64).sqrt()
    }

    #[inline]
    fn wrms_norm_mask(x: &Self, w: &Self, id: &Self) -> f64 {
        debug_assert_eq!(x.len(), w.len());
        debug_assert_eq!(x.len(), id.len());
        // FIXME: handle better possible over/underflow.
        let mut sum = 0.;
        for ((&x, &w), &id) in x.iter().zip(w).zip(id) {
            if id > 0. {
                sum += (x * w).powi(2);
            }
        }
        (sum / x.len() as f64).sqrt()
    }

    #[inline]
    fn min(x: &Self) -> f64 {
        if x.len() == 0 {
            return 0.; // Zero dim space is {0}
        }
        let mut x = x.iter();
        let mut m = *x.next().unwrap();
        for &x in x {
            if x < m { m = x }
        }
        m
    }

    #[inline]
    fn wl2_norm(x: &Self, w: &Self) -> f64 {
        debug_assert_eq!(x.len(), w.len());
        // FIXME: handle better possible over/underflow.
        let mut sum = 0.;
        for (&x, &w) in x.iter().zip(w) {
            sum += (x * w).powi(2);
        }
        sum.sqrt()
    }

    #[inline]
    fn l1_norm(x: &Self) -> f64 {
        let mut sum = 0.;
        for x in x {
            sum += x.abs()
        }
        sum
    }

    #[inline]
    fn compare_assign(z: &mut Self, c: f64, x: &Self) {
        debug_assert_eq!(x.len(), z.len());
        for (x, z) in x.iter().zip(z) {
            if x.abs() >= c {
                *z = 1.;
            } else {
                *z = 0.;
            }
        }
    }

    #[inline]
    fn inv_test_assign(z: &mut Self, x: &Self) -> bool {
        let mut all_nonzero = true;
        for (&x, z) in x.iter().zip(z) {
            all_nonzero = all_nonzero && x != 0.;
            *z = 1. / x;
        }
        all_nonzero
    }

    #[inline]
    fn constr_mask_assign(m: &mut Self, c: &Self, x: &Self) -> bool {
        let mut all_pass = true;
        for ((&c, &x), m) in c.iter().zip(x).zip(m) {
            let c_abs = c.abs();
            let x_c = x * c;
            if (c_abs > 1.5 && x_c <= 0.) || (c_abs > 0.5 && x_c < 0.) {
                all_pass = false;
                *m = 1.;
            } else {
                *m = 0.;
            }
        }
        all_pass
    }

    #[inline]
    fn min_quotient(num: &Self, denom: &Self) -> f64 {
        let mut m = f64::MAX;
        for (&n, &d) in num.iter().zip(denom) {
            if d == 0. { continue }
            let q = n / d;
            if q < m { m = q }
        }
        m
    }
}}

impl<const N: usize> NVectorOps for [f64; N] {
    #[inline]
    fn len(_: &Self) -> usize { N }

    nvector_ops_for_iter!();
}

impl NVectorOps for Vec<f64> {
    #[inline]
    fn len(x: &Self) -> usize { Vec::len(x) }

    nvector_ops_for_iter!();
}

////////////////////////////////////////////////////////////////////////
//
// Implementation for f64
//
// The main purpose if this is not to ease the treatment of 1D ODEs
// but to enable products of `f64` and other types implementing
// `NVectorOps` (see below).

impl NVectorOps for f64 {
    #[inline]
    fn len(_: &Self) -> usize { 1 }

    #[inline]
    fn const_assign(z: &mut Self, c: f64) { *z = c }

    #[inline]
    fn abs_assign(z: &mut Self, x: &Self) { *z = x.abs() }

    #[inline]
    fn mul_assign(z: &mut Self, x: &Self, y: &Self) { *z = x * y }

    #[inline]
    fn inv_assign(z: &mut Self, x: &Self) { *z = 1. / x }

    #[inline]
    fn inv(z: &mut Self) { *z = 1. / *z }

    #[inline]
    fn div_assign(z: &mut Self, x: &Self, y: &Self) { *z = x / y }

    #[inline]
    fn div(z: &mut Self, y: &Self) { *z = *z / y }

    #[inline]
    fn inv_mul(z: &mut Self, x: &Self) { *z = x / *z }

    #[inline]
    fn scale_assign(z: &mut Self, c: f64, x: &Self) { *z = c * x }

    #[inline]
    fn scale(z: &mut Self, c: f64) { *z = c * *z }

    #[inline]
    fn add_const_assign(z: &mut Self, x: &Self, b: f64) { *z = x + b }

    #[inline]
    fn add_const(z: &mut Self, b: f64) { *z = *z + b }

    #[inline]
    fn linear_sum_assign(z: &mut Self, a: f64, x: &Self, b: f64, y: &Self) {
        *z = a * x + b * y
    }

    #[inline]
    fn linear_sum(z: &mut Self, a: f64, b: f64, y: &Self) {
        *z = a * *z + b * y
    }

    #[inline]
    fn dot(x: &Self, y: &Self) -> f64 { x * y}

    #[inline]
    fn max_norm(x: &Self) -> f64 { x.abs() }

    #[inline]
    fn wrms_norm(x: &Self, w: &Self) -> f64 { (x * w).abs() }

    #[inline]
    fn wrms_norm_mask(x: &Self, w: &Self, id: &Self) -> f64 {
        if *id > 0. { (x * w).abs() } else { 0. }
    }

    #[inline]
    fn min(x: &Self) -> f64 { *x }

    #[inline]
    fn wl2_norm(x: &Self, w: &Self) -> f64 { (x * w).abs() }

    #[inline]
    fn l1_norm(x: &Self) -> f64 { x.abs() }

    #[inline]
    fn compare_assign(z: &mut Self, c: f64, x: &Self) {
        if x.abs() >= c { *z = 1. } else { *z = 0. }
    }

    #[inline]
    fn inv_test_assign(z: &mut Self, x: &Self) -> bool {
        if *x != 0. {
            *z = 1. / x;
            true
        } else {
            false
        }
    }

    #[inline]
    fn constr_mask_assign(m: &mut Self, c: &Self, x: &Self) -> bool {
        let test = if *c == 2. {
            *x > 0.
        } else if *c == 1. {
            *x >= 0.
        } else if *c == -2. {
            *x < 0.
        } else if *c == -1. {
            *x <= 0.
        } else { // no constraint
            true
        };
        *m = if test { 0. } else { 1. };
        test
    }

    #[inline]
    fn min_quotient(num: &Self, denom: &Self) -> f64 {
        if *denom == 0. { f64::MAX } else { num / denom }
    }
}

////////////////////////////////////////////////////////////////////////
//
// Implementation for tuples

macro_rules! impl_tuple {
    ($x1: ident, $y1: ident, $z1: ident, $t1: ident, $ty1: ident,
        $($x: ident, $y: ident, $z: ident, $t: ident, $ty: ident),*) =>
{
    impl<$ty1, $($ty),*> NVectorOps for ($ty1, $($ty),*)
    where $ty1: NVectorOps, $($ty: NVectorOps),* {
        #[inline]
        fn len(($x1, $($x),*): &Self) -> usize {
            $ty1::len($x1) $(+ $ty::len($x))*
        }

        #[inline]
        fn const_assign(($z1, $($z),*): &mut Self, c: f64) {
            $ty1::const_assign($z1, c);
            $($ty::const_assign($z, c);)*
        }

        #[inline]
        fn abs_assign(($z1, $($z),*): &mut Self, ($x1, $($x),*): &Self) {
            $ty1::abs_assign($z1, $x1);
            $($ty::abs_assign($z, $x);)*
        }

        #[inline]
        fn mul_assign(
            ($z1, $($z),*): &mut Self,
            ($x1, $($x),*): &Self,
            ($y1, $($y),*): &Self
        ) {
            $ty1::mul_assign($z1, $x1, $y1);
            $($ty::mul_assign($z, $x, $y);)*
        }

        #[inline]
        fn inv_assign(($z1, $($z),*): &mut Self, ($x1, $($x),*): &Self) {
            $ty1::inv_assign($z1, $x1);
            $($ty::inv_assign($z, $x);)*
        }

        #[inline]
        fn inv(($z1, $($z),*): &mut Self) {
            $ty1::inv($z1);
            $($ty::inv($z);)*
        }

        #[inline]
        fn div_assign(
            ($z1, $($z),*): &mut Self,
            ($x1, $($x),*): &Self,
            ($y1, $($y),*): &Self
        ) {
            $ty1::div_assign($z1, $x1, $y1);
            $($ty::div_assign($z, $x, $y);)*

        }

        #[inline]
        fn div(($z1, $($z),*): &mut Self, ($y1, $($y),*): &Self) {
            $ty1::div($z1, $y1);
            $($ty::div($z, $y);)*
        }

        #[inline]
        fn inv_mul(($z1, $($z),*): &mut Self, ($x1, $($x),*): &Self) {
            $ty1::inv_mul($z1, $x1);
            $($ty::inv_mul($z, $x);)*
        }

        #[inline]
        fn scale_assign(
            ($z1, $($z),*): &mut Self,
            c: f64,
            ($x1, $($x),*): &Self
        ) {
            $ty1::scale_assign($z1, c, $x1);
            $($ty::scale_assign($z, c, $x);)*
        }

        #[inline]
        fn scale(($z1, $($z),*): &mut Self, c: f64) {
            $ty1::scale($z1, c);
            $($ty::scale($z, c);)*
        }

        #[inline]
        fn add_const_assign(
            ($z1, $($z),*): &mut Self,
            ($x1, $($x),*): &Self,
            b: f64
        ) {
            $ty1::add_const_assign($z1, $x1, b);
            $($ty::add_const_assign($z, $x, b);)*
        }

        #[inline]
        fn add_const(($z1, $($z),*): &mut Self, b: f64) {
            $ty1::add_const($z1, b);
            $($ty::add_const($z, b);)*
        }

        #[inline]
        fn linear_sum_assign(
            ($z1, $($z),*): &mut Self,
            a: f64,  ($x1, $($x),*): &Self,
            b: f64,  ($y1, $($y),*): &Self
        ) {
            $ty1::linear_sum_assign($z1, a, $x1, b, $y1);
            $($ty::linear_sum_assign($z, a, $x, b, $y);)*
        }

        #[inline]
        fn linear_sum(
            ($z1, $($z),*): &mut Self,
            a: f64, b: f64,
            ($y1, $($y),*): &Self
        ) {
            $ty1::linear_sum($z1, a, b, $y1);
            $($ty::linear_sum($z, a, b, $y);)*
        }

        #[inline]
        fn dot(($x1, $($x),*): &Self, ($y1, $($y),*): &Self) -> f64 {
            $ty1::dot($x1, $y1) $(+ $ty::dot($x, $y))*
        }

        #[inline]
        fn max_norm(($x1, $($x),*): &Self) -> f64 {
            $ty1::max_norm($x1)
            $(.max($ty::max_norm($x)))*
        }

        #[inline]
        fn wrms_norm(
            ($x1, $($x),*): &Self,
            ($y1, $($y),*): &Self, // (w1, w2,...)
        ) -> f64 {
            let $t1 = $ty1::len($x1);
            $(let $t = $ty::len($x);)*
            let n = ($t1 $(+ $t)*) as f64;
            // FIXME: avoid possible overflow in intermediate computations.
            ($ty1::wrms_norm($x1, $y1).powi(2) * ($t1 as f64/ n)
                $(+ $ty::wrms_norm($x, $y).powi(2) * ($t as f64/ n))*).sqrt()
        }

        #[inline]
        fn wrms_norm_mask(
            ($x1, $($x),*): &Self,
            ($y1, $($y),*): &Self, // (w1, w2,...)
            ($z1, $($z),*): &Self, // (id1, id2,...)
        ) -> f64 {
            let $t1 = $ty1::len($x1);
            $(let $t = $ty::len($x);)*
            let n = ($t1 $(+ $t)*) as f64;
            // FIXME: avoid possible overflow in intermediate computations.
            ($ty1::wrms_norm_mask($x1, $y1, $z1).powi(2) * $t1 as f64 / n
                $(+ $ty::wrms_norm_mask($x, $y, $z).powi(2)
                    * $t as f64 / n)*).sqrt()
        }

        #[inline]
        fn min(($x1, $($x),*): &Self) -> f64 {
            $ty1::min($x1) $(.min($ty::min($x)))*
        }

        #[inline]
        fn wl2_norm(
            ($x1, $($x),*): &Self,
            ($y1, $($y),*): &Self // (w1, w2,...)
        ) -> f64 {
            // FIXME: Better implementation based on hypot impl?
            $ty1::wl2_norm($x1, $y1)
            $(.hypot($ty::wl2_norm($x, $y)))*
        }

        #[inline]
        fn l1_norm(($x1, $($x),*): &Self) -> f64 {
            $ty1::l1_norm($x1) $(+ $ty::l1_norm($x))*
        }

        #[inline]
        fn compare_assign(
            ($z1, $($z),*): &mut Self,
            c: f64,
            ($x1, $($x),*): &Self
        ) {
            $ty1::compare_assign($z1, c, $x1);
            $($ty::compare_assign($z, c, $x);)*
        }

        #[inline]
        fn inv_test_assign(
            ($z1, $($z),*): &mut Self,
            ($x1, $($x),*): &Self,
        ) -> bool {
            // Beware that the assignation must occur even if some test fails.
            let $t1 = $ty1::inv_test_assign($z1, $x1);
            $(let $t = $ty::inv_test_assign($z, $x);)*
            $t1 $(&& $t)*
        }

        #[inline]
        fn constr_mask_assign(
            ($y1, $($y),*): &mut Self, // (m1, m2,...)
            ($z1, $($z),*): &Self, // (c1, c2,...)
            ($x1, $($x),*): &Self,
        ) -> bool {
            // Beware that the assignation must occur even if some test fails.
            let $t1 = $ty1::constr_mask_assign($y1, $z1, $x1);
            $(let $t = $ty::constr_mask_assign($y, $z, $x);)*
            $t1 $(&& $t)*
        }

        #[inline]
        fn min_quotient(
            ($x1, $($x),*): &Self, // (num1, num2,...)
            ($y1, $($y),*): &Self, // (denom1, denom2,...)
        ) -> f64 {
            $ty1::min_quotient($x1, $y1)
            $(.min($ty::min_quotient($x, $y)))*
        }
    }
}}

impl_tuple!(x1, y1, z1, t1, T1, x2, y2, z2, t2, T2);
impl_tuple!(x1, y1, z1, t1, T1, x2, y2, z2, t2, T2, x3, y3, z3, t3, T3);
impl_tuple!(x1, y1, z1, t1, T1, x2, y2, z2, t2, T2, x3, y3, z3, t3, T3,
    x4, y4, z4, t4, T4);
impl_tuple!(x1, y1, z1, t1, T1, x2, y2, z2, t2, T2, x3, y3, z3, t3, T3,
    x4, y4, z4, t4, T4, x5, y5, z5, t5, T5);
impl_tuple!(x1, y1, z1, t1, T1, x2, y2, z2, t2, T2, x3, y3, z3, t3, T3,
    x4, y4, z4, t4, T4, x5, y5, z5, t5, T5, x6, y6, z6, t6, T6);
impl_tuple!(x1, y1, z1, t1, T1, x2, y2, z2, t2, T2, x3, y3, z3, t3, T3,
    x4, y4, z4, t4, T4, x5, y5, z5, t5, T5, x6, y6, z6, t6, T6,
    x7, y7, z7, t7, T7);
impl_tuple!(x1, y1, z1, t1, T1, x2, y2, z2, t2, T2, x3, y3, z3, t3, T3,
    x4, y4, z4, t4, T4, x5, y5, z5, t5, T5, x6, y6, z6, t6, T6,
    x7, y7, z7, t7, T7, x8, y8, z8, t8, T8);
impl_tuple!(x1, y1, z1, t1, T1, x2, y2, z2, t2, T2, x3, y3, z3, t3, T3,
    x4, y4, z4, t4, T4, x5, y5, z5, t5, T5, x6, y6, z6, t6, T6,
    x7, y7, z7, t7, T7, x8, y8, z8, t8, T8, x9, y9, z9, t9, T9);
impl_tuple!(x1, y1, z1, t1, T1, x2, y2, z2, t2, T2, x3, y3, z3, t3, T3,
    x4, y4, z4, t4, T4, x5, y5, z5, t5, T5, x6, y6, z6, t6, T6,
    x7, y7, z7, t7, T7, x8, y8, z8, t8, T8, x9, y9, z9, t9, T9,
    x10, y10, z10, t10, T10);
impl_tuple!(x1, y1, z1, t1, T1, x2, y2, z2, t2, T2, x3, y3, z3, t3, T3,
    x4, y4, z4, t4, T4, x5, y5, z5, t5, T5, x6, y6, z6, t6, T6,
    x7, y7, z7, t7, T7, x8, y8, z8, t8, T8, x9, y9, z9, t9, T9,
    x10, y10, z10, t10, T10, x11, y11, z11, t11, T11);
impl_tuple!(x1, y1, z1, t1, T1, x2, y2, z2, t2, T2, x3, y3, z3, t3, T3,
    x4, y4, z4, t4, T4, x5, y5, z5, t5, T5, x6, y6, z6, t6, T6,
    x7, y7, z7, t7, T7, x8, y8, z8, t8, T8, x9, y9, z9, t9, T9,
    x10, y10, z10, t10, T10, x11, y11, z11, t11, T11, x12, y12, z12, t12, T12);
impl_tuple!(x1, y1, z1, t1, T1, x2, y2, z2, t2, T2, x3, y3, z3, t3, T3,
    x4, y4, z4, t4, T4, x5, y5, z5, t5, T5, x6, y6, z6, t6, T6,
    x7, y7, z7, t7, T7, x8, y8, z8, t8, T8, x9, y9, z9, t9, T9,
    x10, y10, z10, t10, T10, x11, y11, z11, t11, T11, x12, y12, z12, t12, T12,
    x13, y13, z13, t13, T13);



////////////////////////////////////////////////////////////////////////
//
// Implementation for ndarray

#[cfg(feature = "ndarray")]
impl NVectorOps for ndarray::Array1<f64> {
    fn len(v: &Self) -> usize { ndarray::Array1::len(v) }

    nvector_ops_for_iter!();
}
