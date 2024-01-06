//! Custom N_Vector that hold any Rust value (that possesses some
//! operations).

use std::{ffi::c_void, marker::PhantomData};
use sundials_sys::*;

/// Operations that Rust values must support to be converted to
/// Sundials N_Vector.
pub trait NVectorOps: Clone {
    /// Length of the vector.
    fn len(&self) -> usize;

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

/// Wrapper around a [`N_Vector`] that holds a reference to a Rust
/// value.
///
/// # Safety
///
/// This shared wrapper is only valid as long as the Rust value does
/// not move.
pub struct Shared<V> {
    // The N_Vector `content` will point to the Rust value.
    nv: N_Vector,
    marker: PhantomData<V>,
}

impl<V> Drop for Shared<V> {
    fn drop(&mut self) {
        // Since this is a shared N_Vector, we only drop the N_Vector
        // structure. See vendor/src/sundials/sundials_nvector.c
        unsafe { N_VFreeEmpty(self.nv) }
    }
}

extern "C" fn nvgetvectorid_rust(_: N_Vector) -> N_Vector_ID {
    // See vendor/include/sundials/sundials_nvector.h
    N_Vector_ID_SUNDIALS_NVEC_CUSTOM
}

// Beware that the operations attached to the N_Vector will not only
// be for `Shared` ones but also for clones that will live inside
// Sundials solvers,...
impl<'a, V> Shared<&'a mut V>
where V: NVectorOps + 'a {
    /// Return the Rust value stored in the N_Vector.
    ///
    /// # Safety
    /// This box must be leaked before it is dropped because the data
    /// belongs to `nv` if internal to Sundials or to another Rust
    /// value if it is shared.
    #[inline]
    unsafe fn mut_of_nvector(nv: N_Vector) -> &'a mut V {
        // FIXME: When `nv` content is shared with another Rust value,
        // there will be temporarily be two Rust values pointing to the
        // same data.  Make sure it is fine as this one is very local.
        ((*nv).content as *mut V).as_mut().unwrap()
    }

    #[inline]
    unsafe fn ref_of_nvector(nv: N_Vector) -> &'a V {
        Self::mut_of_nvector(nv)
    }

    /// Creates a new N_Vector of the same type as an existing vector
    /// `nv` and sets the ops field.  It does not copy the vector, but
    /// rather allocates storage for the new vector.
    #[cfg(feature = "nightly")]
    unsafe extern "C" fn nvclone_rust(nw: N_Vector) -> N_Vector {
        let w = Self::ref_of_nvector(nw);
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
    unsafe extern "C" fn nvclone_rust(nw: N_Vector) -> N_Vector {
        let w = Self::ref_of_nvector(nw);
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
        let nv = libc::malloc(std::mem::size_of::<_generic_N_Vector>())
            as N_Vector;
        (*nv).sunctx = sunctx;
        let n = std::mem::size_of::<_generic_N_Vector_Ops>();
        let ops = libc::malloc(n);
        libc::memcpy(ops, (*nw).ops as *mut c_void, n);
        (*nv).ops = ops as N_Vector_Ops;
        (*nv).content = Box::into_raw(Box::new(v)) as *mut c_void;
        nv
    }

    /// Destroys the N_Vector `nv` and frees memory allocated for its
    /// internal data.
    unsafe extern "C" fn nvdestroy_rust(nv: N_Vector) {
        // This is for N_Vectors managed by Sundials.  Rust `Shared`
        // values will not call N_Vector operations.
        let v = Box::from_raw((*nv).content as *mut V);
        drop(v);
        N_VFreeEmpty(nv);
    }

    /// Returns storage requirements for the N_Vector `nv`:
    /// - `lrw` contains the number of realtype words;
    /// - `liw` contains the number of integer words.
    /// This function is advisory only.
    unsafe extern "C" fn nvspace_rust(
        nv: N_Vector, lrw: *mut sunindextype, liw: *mut sunindextype) {
            let v = Self::ref_of_nvector(nv);
            let n = v.len();
            *lrw = n as sunindextype;
            *liw = 1;
        }

    /// Returns the global length (number of “active” entries) in the
    /// N_Vector `nv`.
    unsafe extern "C" fn nvgetlength_rust(nv: N_Vector) -> sunindextype {
        let v = Self::ref_of_nvector(nv);
        let n = v.len();
        n as sunindextype
    }

    /// Performs the operation `z = ax + by`.
    unsafe extern "C" fn nvlinearsum_rust(
        a: realtype, nx: N_Vector, b: realtype, ny: N_Vector, nz: N_Vector
    ) {
        let z = Self::mut_of_nvector(nz);
        if nz == nx && nz == ny { // z = (a + b) z
            V::scale(z, a+b);
            return
        }
        if nz == nx { // ≠ ny
            let y = Self::ref_of_nvector(ny);
            V::linear_sum(z, a, b, y);
            return
        }
        if nz == ny { // ≠ nx
            let x = Self::ref_of_nvector(nx);
            V::linear_sum(z, b, a, x);
            return
        }
        let x = Self::ref_of_nvector(nx);
        let y = Self::ref_of_nvector(ny);
        V::linear_sum_assign(z, a, x, b, y);
    }

    /// Sets all components of `nz` to realtype `c`.
    unsafe extern "C" fn nvconst_rust(c: realtype, nz: N_Vector) {
        let z = Self::mut_of_nvector(nz);
        V::const_assign(z, c);
    }

    /// Sets `nz` to be the component-wise product of the inputs `nx`
    /// and `ny`: ∀i, zᵢ = xᵢ yᵢ.
    unsafe extern "C" fn nvprod_rust(
        nx: N_Vector, ny: N_Vector, nz: N_Vector
    ) {
        assert_ne!(nz, nx);
        assert_ne!(nz, ny);
        let x = Self::ref_of_nvector(nx);
        let y = Self::ref_of_nvector(ny);
        let z = Self::mut_of_nvector(nz);
        V::mul_assign(z, x, y);
    }

    /// Sets the `nz` to be the component-wise ratio of the inputs
    /// `nx` and `ny`: ∀i, zᵢ = xᵢ/yᵢ.  The yᵢ may not be tested for 0
    /// values.  This function should only be called with a y that is
    /// guaranteed to have all nonzero components.
    unsafe extern "C" fn nvdiv_rust(
        nx: N_Vector, ny: N_Vector, nz: N_Vector
    ) {
        let z = Self::mut_of_nvector(nz);
        if nz == nx {
            if nz == ny {
                V::const_assign(z, 1.);
                return
            } else {
                let y = Self::ref_of_nvector(ny);
                V::div(z, y);
                return
            }
        }
        if nz == ny {
            let x = Self::ref_of_nvector(nx);
            V::inv_mul(z, x);
            return
        }
        let x = Self::ref_of_nvector(nx);
        let y = Self::ref_of_nvector(ny);
        V::div_assign(z, x, y);
    }

    /// Scales the `nx` by the scalar `c` and returns the result in
    /// `z`: ∀i, zᵢ = cxᵢ.
    unsafe extern "C" fn nvscale_rust(c: f64, nx: N_Vector, nz: N_Vector) {
        let z = Self::mut_of_nvector(nz);
        if nz == nx {
            V::scale(z, c);
        } else {
            let x = Self::ref_of_nvector(nx);
            V::scale_assign(z, c, x);
        }
    }

    /// Sets the components of the `nz` to be the absolute values of
    /// the components of the `nx`: ∀i, zᵢ = |xᵢ|.
    unsafe extern "C" fn nvabs_rust(nx: N_Vector, nz: N_Vector) {
        assert_ne!(nz, nx);
        let x = Self::ref_of_nvector(nx);
        let z = Self::mut_of_nvector(nz);
        V::abs_assign(z, x);
    }

    /// Sets the components of the `nz` to be the inverses of the
    /// components of `nx`: ∀i, zᵢ = 1/xᵢ.
    unsafe extern "C" fn nvinv_rust(nx: N_Vector, nz: N_Vector) {
        let z = Self::mut_of_nvector(nz);
        if nz == nx {
            V::inv(z);
        } else {
            let x = Self::ref_of_nvector(nx);
            V::inv_assign(z, x);
        }
    }

    /// Adds the scalar `b` to all components of `nx` and returns the
    /// result in `nz`: ∀i, zᵢ = xᵢ + b.
    unsafe extern "C" fn nvaddconst_rust(nx: N_Vector, b: f64, nz: N_Vector) {
        let z = Self::mut_of_nvector(nz);
        if nz == nx {
            V::add_const(z, b);
        } else {
            let x = Self::ref_of_nvector(nx);
            V::add_const_assign(z, x, b);
        }
    }

    /// Returns the value of the dot-product of `nx` and `ny`: ∑ xᵢ yᵢ.
    unsafe extern "C" fn nvdotprod_rust(
        nx: N_Vector, ny: N_Vector
    ) -> realtype {
        let x = Self::ref_of_nvector(nx);
        let y = Self::ref_of_nvector(ny);
        V::dot(&x, &y)
    }

    /// Returns the value of the ℓ^∞ norm of `nx`: maxᵢ |xᵢ|.
    unsafe extern "C" fn nvmaxnorm_rust(nx: N_Vector) -> realtype {
        let x = Self::ref_of_nvector(nx);
        V::max_norm(x)
    }

    /// Returns the weighted root-mean-square norm of `nx` with
    /// (positive) realtype weight vector `nw`: √(∑ (xᵢ wᵢ)² / n).
    unsafe extern "C" fn nvwrmsnorm_rust(
        nx: N_Vector, nw: N_Vector
    ) -> realtype {
        let x = Self::ref_of_nvector(nx);
        let w = Self::ref_of_nvector(nw);
        V::wrms_norm(&x, &w)
    }

    /// Returns the weighted root mean square norm of `nx` with weight
    /// vector `nw` built using only the elements of `nx` corresponding
    /// to positive elements of the `nid`: √(∑ (xᵢ wᵢ H(idᵢ))² / n)
    /// where H(α) = 1 if α > 0 and H(α) = 0 if α ≤ 0.
    unsafe extern "C" fn nvwrmsnormmask_rust(
        nx: N_Vector, nw: N_Vector, nid: N_Vector
    ) -> realtype {
        let x = Self::ref_of_nvector(nx);
        let w = Self::ref_of_nvector(nw);
        let id = Self::ref_of_nvector(nid);
        V::wrms_norm_mask(x, w, id)
    }

    /// Returns the smallest element of the `nx`: minᵢ xᵢ.
    unsafe extern "C" fn nvmin_rust(nx: N_Vector) -> realtype {
        let x = Self::ref_of_nvector(nx);
        V::min(x)
    }

    /// Returns the weighted Euclidean norm of `nx` with realtype
    /// weight vector `nw`: √(∑ (xᵢ wᵢ)²).
    unsafe extern "C" fn nvwl2norm_rust(
        nx: N_Vector, nw: N_Vector
    ) -> realtype {
        let x = Self::ref_of_nvector(nx);
        let w = Self::ref_of_nvector(nw);
        V::wl2_norm(x, w)
    }

    /// Returns the ℓ¹ norm of `nx`: ∑ |xᵢ|.
    unsafe extern "C" fn nvl1norm_rust(nx: N_Vector) -> realtype {
        let x = Self::ref_of_nvector(nx);
        V::l1_norm(x)
    }

    /// Compares the components of `nx` to the realtype scalar `c` and
    /// returns a `nz` such that ∀i, zᵢ = 1.0 if |xᵢ| ≥ `c` and zᵢ = 0.0
    /// otherwise.
    unsafe extern "C" fn nvcompare_rust(c: f64, nx: N_Vector, nz: N_Vector) {
        assert_ne!(nz, nx);
        let x = Self::ref_of_nvector(nx);
        let z = Self::mut_of_nvector(nz);
        V::compare_assign(z, c, x)
    }

    /// Sets the components of `nz` to be the inverses of the
    /// components of `nx`, with prior testing for zero values:
    /// ∀i, zᵢ = 1/xᵢ.  This routine returns a boolean assigned to
    /// `SUNTRUE` if all components of x are nonzero (successful
    /// inversion) and returns `SUNFALSE` otherwise.
    unsafe extern "C" fn nvinvtest_rust(nx: N_Vector, nz: N_Vector) -> i32 {
        assert_ne!(nz, nx);
        let x = Self::ref_of_nvector(nx);
        let z = Self::mut_of_nvector(nz);
        let b = V::inv_test_assign(z, x);
        (if b { SUNTRUE } else { SUNFALSE }) as _
    }

    /// Performs the following constraint tests based on the values in cᵢ:
    /// - xᵢ > 0 if cᵢ = 2,
    /// - xᵢ ≥ 0 if cᵢ = 1,
    /// - xᵢ < 0 if cᵢ = -2,
    /// - xᵢ ≤ 0 if cᵢ = -1.
    /// There is no constraint on xᵢ if cᵢ = 0.  This routine returns a
    /// boolean assigned to `SUNFALSE` if any element failed the
    /// constraint test and assigned to `SUNTRUE` if all passed.  It
    /// also sets a mask vector `nm`, with elements equal to 1.0 where
    /// the constraint test failed, and 0.0 where the test
    /// passed.  This routine is used only for constraint checking.
    unsafe extern "C" fn nvconstrmask_rust(
        nc: N_Vector, nx: N_Vector, nm: N_Vector
    ) -> i32 {
        assert_ne!(nm, nc);
        assert_ne!(nm, nx);
        let c = Self::ref_of_nvector(nc);
        let x = Self::ref_of_nvector(nx);
        let m = Self::mut_of_nvector(nm);
        let b = V::constr_mask_assign(m, c, x);
        (if b { SUNTRUE } else { SUNFALSE }) as _
        }

    /// This routine returns the minimum of the quotients obtained by
    /// termwise dividing the elements of n = `nnum` by the elements
    /// in d = `ndenom`: minᵢ nᵢ/dᵢ.
    unsafe extern "C" fn nvminquotient_rust(
        nnum: N_Vector, ndenom: N_Vector
    ) -> f64 {
        let num = Self::ref_of_nvector(nnum);
        let denom = Self::ref_of_nvector(ndenom);
        V::min_quotient(num, denom)
    }

    #[inline]
    fn ops() -> _generic_N_Vector_Ops {
        _generic_N_Vector_Ops {
            nvgetvectorid: Some(nvgetvectorid_rust),
            nvclone: Some(Self::nvclone_rust),
            // Contrarily to "nvclone", "nvcloneempty" does not
            // allocate the storage for the vector data.  This is only
            // used in some linear solvers (that custom vectors will
            // not support) in combination with `N_VSetArrayPointer`.
            nvcloneempty: None,
            nvdestroy: Some(Self::nvdestroy_rust),
            nvspace: Some(Self::nvspace_rust),
            nvgetarraypointer: None,
            nvgetdevicearraypointer: None,
            nvsetarraypointer: None,
            nvgetcommunicator: None,
            nvgetlength: Some(Self::nvgetlength_rust),
            nvgetlocallength: Some(Self::nvgetlength_rust),
            // Standard vector operations
            nvlinearsum: Some(Self::nvlinearsum_rust),
            nvconst: Some(Self::nvconst_rust),
            nvprod: Some(Self::nvprod_rust),
            nvdiv: Some(Self::nvdiv_rust),
            nvscale: Some(Self::nvscale_rust),
            nvabs: Some(Self::nvabs_rust),
            nvinv: Some(Self::nvinv_rust),
            nvaddconst: Some(Self::nvaddconst_rust),
            nvdotprod: Some(Self::nvdotprod_rust),
            nvmaxnorm: Some(Self::nvmaxnorm_rust),
            nvwrmsnorm: Some(Self::nvwrmsnorm_rust),
            nvwrmsnormmask: Some(Self::nvwrmsnormmask_rust),
            nvmin: Some(Self::nvmin_rust),
            nvwl2norm: Some(Self::nvwl2norm_rust),
            nvl1norm: Some(Self::nvl1norm_rust),
            nvcompare: Some(Self::nvcompare_rust),
            nvinvtest: Some(Self::nvinvtest_rust),
            nvconstrmask: Some(Self::nvconstrmask_rust),
            nvminquotient: Some(Self::nvminquotient_rust),
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
    }

    #[inline]
    unsafe fn new_nvector(ctx: SUNContext) -> N_Vector {
        let nv = N_VNewEmpty(ctx);
        if nv.is_null() {
            panic!("sundials::vector::custom::Shared::new: \
                Could not allocate new N_Vector.");
        }
        let ops = Self::ops();
        (*nv).ops = Box::into_raw(Box::new(ops));
        nv
    }

    /// Return a new shared [`N_Vector`] built with context `ctx`.  It
    /// is required that `V` implements the trait `NVector`.  This is
    /// the case for [`[f64; N]`][std::array] and
    /// [`Vec<f64>`][std::vec::Vec].
    ///
    /// # Safety
    /// If used in Sundials functions, make sure the returned value
    /// does not outlive `ctx`.
    pub unsafe fn new_mut(v: &'a mut V, ctx: SUNContext) -> Self {
        // See https://sundials.readthedocs.io/en/latest/nvectors/NVector_API_link.html#implementing-a-custom-nvector
        let nv = Self::new_nvector(ctx);
        (*nv).content = v as *mut V as *mut c_void;
        Shared { nv , marker: PhantomData }
    }

    pub unsafe fn new_ref(v: &'a V, ctx: SUNContext) -> Shared<&'a V> {
        let nv = Self::new_nvector(ctx);
        (*nv).content = v as *const V as *mut c_void;
        Shared { nv , marker: PhantomData }
    }
}


impl<'a, V> Shared<&'a mut V>
where V: NVectorOps + 'a + AsRef<[f64]> + AsMut<[f64]> {
    // FIXME: These functions are used in some linear solvers, thus
    // not on values managed by Rust (except for the temporary
    // conversion in these functions).
    unsafe extern "C" fn nvgetarraypointer_rust(nv: N_Vector) -> *mut realtype {
        let v = Self::ref_of_nvector(nv);
        let ptr: &[f64] = v.as_ref();
        ptr.as_ptr() as *mut f64
    }

    unsafe extern "C" fn nvsetarraypointer_rust(
        data: *mut realtype, nv: N_Vector
    ) {
        // FIXME: Are we sure of the origin of data?  If it is not
        // aligned or allocated with a different allocator than
        // Rust's, we cannot support this.
        let v = Self::mut_of_nvector(nv);
        let ptr: &mut [f64] = v.as_mut();
        // FIXME: How is the original `nv` data reclaimed?
        *ptr.as_mut_ptr() = *data;
    }

    /// Same as `ops` but set the `…nvgetarraypointer` functions.
    #[inline]
    fn ops_slice() -> _generic_N_Vector_Ops {
        let mut ops = Self::ops();
        // FIXME:
        //ops.nvcloneempty = Some(...);
        ops.nvgetarraypointer = Some(Self::nvgetarraypointer_rust);
        ops.nvsetarraypointer = Some(Self::nvsetarraypointer_rust);
        ops
    }

    #[inline]
    unsafe fn new_nvector_slice(ctx: SUNContext) -> N_Vector {
        let nv = N_VNewEmpty(ctx);
        if nv.is_null() {
            panic!("sundials::vector::custom::Shared::new: \
                Could not allocate new N_Vector.");
        }
        let ops = Self::ops_slice();
        (*nv).ops = Box::into_raw(Box::new(ops));
        nv
    }

    pub unsafe fn new_mut_slice(v: &'a mut V, ctx: SUNContext) -> Self {
        let nv = Self::new_nvector_slice(ctx);
        (*nv).content = v as *mut V as *mut c_void;
        Shared { nv , marker: PhantomData }
    }

    pub unsafe fn new_ref_slice(v: &'a V, ctx: SUNContext) -> Shared<&'a V> {
        let nv = Self::new_nvector_slice(ctx);
        (*nv).content = v as *const V as *mut c_void;
        Shared { nv , marker: PhantomData }
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
    fn len(&self) -> usize { N }

    nvector_ops_for_iter!();
}

unsafe impl<const N: usize> super::Vector for [f64; N] {
    fn len(_: &Self) -> usize { N }

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

    type NVectorRef<'a> = Shared<&'a [f64; N]>;
    type NVectorMut<'a> = Shared<&'a mut [f64; N]>;

    #[inline]
    unsafe fn as_nvector(
        v: &Self, ctx: SUNContext) -> Option<Self::NVectorRef<'_>> {
            Some(Shared::new_ref(v, ctx))
    }

    #[inline]
    unsafe fn as_mut_nvector(
        v: &mut Self, ctx: SUNContext) -> Option<Self::NVectorMut<'_>> {
            Some(Shared::new_mut(v, ctx))
        }

    fn as_ptr(v: &Self::NVectorRef<'_>) -> *const _generic_N_Vector {
        v.nv
    }

    fn as_mut_ptr(v: &Self::NVectorMut<'_>) -> N_Vector {
        v.nv
    }
}


impl NVectorOps for Vec<f64> {
    #[inline]
    fn len(&self) -> usize { Vec::len(&self) }

    nvector_ops_for_iter!();
}

unsafe impl super::Vector for Vec<f64> {
    fn len(v: &Self) -> usize { Vec::len(v) }

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

    type NVectorRef<'a> = Shared<&'a Vec<f64>>;
    type NVectorMut<'a> = Shared<&'a mut Vec<f64>>;

    #[inline]
    unsafe fn as_nvector(
        v: &Self, ctx: SUNContext) -> Option<Self::NVectorRef<'_>> {
            Some(Shared::new_ref(v, ctx))
    }

    #[inline]
    unsafe fn as_mut_nvector(
        v: &mut Self, ctx: SUNContext) -> Option<Self::NVectorMut<'_>> {
            Some(Shared::new_mut(v, ctx))
        }

    fn as_ptr(v: &Self::NVectorRef<'_>) -> *const _generic_N_Vector {
        v.nv
    }

    fn as_mut_ptr(v: &Self::NVectorMut<'_>) -> N_Vector {
        v.nv
    }
}
