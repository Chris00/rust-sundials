//! Custom N_Vector that hold any Rust value (that possesses some
//! operations).

use std::{ffi::c_void, marker::PhantomData};
use sundials_sys::*;

use crate::Context;

/// Operations that Rust values must support to be converted to
/// Sundials N_Vector.
pub trait NVector: Clone {
    /// Length of the vector.
    fn len(&self) -> usize;

    /// Return a new empty vector.
    fn new() -> Self;

    /// Performs the operation `z = ax + by`.
    fn linear_sum(a: f64, x: &Self, b: f64, y: &Self, z: &mut Self);

    /// Sets all components of `z` to `c`: ∀i, zᵢ = c.
    fn set_const(c: f64, z: &mut Self);

    /// Set `z` to the component-wise product of `x` and `y`:
    /// ∀i, zᵢ = xᵢ yᵢ.
    fn set_prod(x: &Self, y: &Self, z: &mut Self);

    /// Set `z` to the component-wise ratio of `x` and `y`:
    /// ∀i, zᵢ = xᵢ/yᵢ.  The yᵢ may not be tested for 0 values.
    fn set_div(x: &Self, y: &Self, z: &mut Self);

    /// Set `z` to the scaling of `x` by the factor `c`: ∀i, zᵢ = cxᵢ.
    fn set_scale(c: f64, x: &Self, z: &mut Self);

    /// Set each component of `z` to the absolute value of the
    /// corresponding component in `x`: ∀i, zᵢ = |xᵢ|.
    fn set_abs(x: &Self, z: &mut Self);

    /// Set each component of the `z` to the inverse of the
    /// corresponding component of `x`: ∀i, zᵢ = 1/xᵢ.
    fn set_inv(x: &Self, z: &mut Self);

    /// Set each component of `z` to the sum of the corresponding
    /// component in `x` and `b`: ∀i, zᵢ = xᵢ + b.
    fn set_add_const(x: &Self, b: f64, z: &mut Self);

    /// Return the dot-product of `x` and `y`: ∑ xᵢ yᵢ.
    fn dot_prod(x: &Self, y: &Self) -> f64;

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

    /// Compare the components of `x` to the realtype scalar `c`
    /// and returns a `z` such that ∀i, zᵢ = 1.0 if |xᵢ| ≥ `c` and
    /// zᵢ = 0.0 otherwise.
    fn set_compare(c: f64, x: &Self, z: &mut Self);

    /// Sets the components of `z` to be the inverses of the
    /// components of `x`, with prior testing for zero values:
    /// ∀i, zᵢ = 1/xᵢ.  This routine returns `true` if all components
    /// of x are nonzero (successful inversion) and returns `false`
    /// otherwise.
    fn set_inv_test(x: &Self, z: &mut Self) -> bool;

    /// Performs the following constraint tests based on the values in cᵢ:
    /// - xᵢ > 0 if cᵢ = 2,
    /// - xᵢ ≥ 0 if cᵢ = 1,
    /// - xᵢ < 0 if cᵢ = -2,
    /// - xᵢ ≤ 0 if cᵢ = -1.
    /// There is no constraint on if cᵢ = 0.  This routine returns
    /// `false` if any element failed the constraint test and `true`
    /// if all passed.  It also sets a mask vector `m`, with elements
    /// equal to 1.0 where the constraint test failed, and 0.0 where
    /// the test passed.  This routine is used only for constraint
    /// checking.
    fn set_constr_mask(c: &Self, x: &Self, m: &mut Self) -> bool;

    /// Returns the minimum of the quotients obtained by termwise
    /// dividing the elements of `num` by the elements in `denom`:
    /// minᵢ numᵢ/denomᵢ.
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
impl<V> Shared<V>
where V: NVector {
    /// Return the Rust value stored in the N_Vector.
    ///
    /// # Safety
    /// This box must be leaked before it is dropped because the data
    /// belongs to `nv` if internal to Sundials or to another Rust
    /// value if it is shared.
    unsafe fn box_of_nvector(nv: N_Vector) -> Box<V> {
        // FIXME: When `nv` content is shared with another Rust value,
        // there will be temporarily be two Rust values pointing to the
        // same data.  Make sure it is fine.
        Box::from_raw((*nv).content as *mut V)
    }

    /// Creates a new N_Vector of the same type as an existing vector
    /// `nv` and sets the ops field.  It does not copy the vector, but
    /// rather allocates storage for the new vector.
    unsafe extern "C" fn nvclone_rust(nv: N_Vector) -> N_Vector {
        let v = Self::box_of_nvector(nv);
        let w = v.clone();
        // Rust memory cannot be uninitialized, thus clone.
        Box::leak(v);
        let nw = N_VNewEmpty((*nv).sunctx);
        let ret = N_VCopyOps(nw, nv);
        debug_assert_eq!(ret, 0);
        (*nw).content = Box::into_raw(w) as *mut c_void;
        nw
    }

    /// Creates a new N_Vector of the same type as an existing vector
    /// w and sets the ops field. It does not allocate storage for the
    /// new vector’s data.
    unsafe extern "C" fn nvcloneempty_rust(nv: N_Vector) -> N_Vector {
        let nw = N_VNewEmpty((*nv).sunctx);
        let ret = N_VCopyOps(nw, nv);
        debug_assert_eq!(ret, 0);
        (*nw).content = Box::into_raw(Box::new(V::new())) as *mut c_void;
        nw
    }

    /// Destroys the N_Vector `nv` and frees memory allocated for its
    /// internal data.
    unsafe extern "C" fn nvdestroy_rust(nv: N_Vector) {
        // This is for N_Vectors managed by Sundials.  Rust `Shared`
        // values will not call N_Vector operations.
        let v = Self::box_of_nvector(nv);
        drop(v);
        N_VFreeEmpty(nv);
    }

    /// Returns storage requirements for the N_Vector `nv`:
    /// - `lrw` contains the number of realtype words;
    /// - `liw` contains the number of integer words.
    /// This function is advisory only.
    unsafe extern "C" fn nvspace_rust(
        nv: N_Vector, lrw: *mut sunindextype, liw: *mut sunindextype) {
            let v = Self::box_of_nvector(nv);
            let n = v.len();
            Box::leak(v);
            *lrw = n as sunindextype;
            *liw = 1;
        }

    extern "C" fn nvgetarraypointer_rust(nv: N_Vector) -> *mut realtype {
        todo!()
    }

    extern "C" fn nvsetarraypointer_rust(data: *mut realtype, nv: N_Vector) {
        // Are we sure of the origin of data?  If it is not aligned or
        // allocated with a different allocator than Rust's, we cannot
        // support this.
        todo!()
    }

    /// Returns the global length (number of “active” entries) in the
    /// N_Vector `nv`.
    unsafe extern "C" fn nvgetlength_rust(nv: N_Vector) -> sunindextype {
        let v = Self::box_of_nvector(nv);
        let n = v.len();
        Box::leak(v);
        n as sunindextype
    }

    /// Performs the operation `z = ax + by`.
    unsafe extern "C" fn nvlinearsum_rust(
        a: realtype, nx: N_Vector, b: realtype, ny: N_Vector, nz: N_Vector) {
            // FIXME: in C, it is possible that nz == nx,...  Do we
            // have to care about that?  Likely because it violates
            // the borrow checker rules.  (May be fine for the code we
            // write below but not for *any* user code.)
            let x = Self::box_of_nvector(nx);
            let y = Self::box_of_nvector(ny);
            let mut z = Self::box_of_nvector(nz);
            V::linear_sum(a, &x, b, &y, &mut z);
            Box::leak(x);
            Box::leak(y);
            Box::leak(z);
        }

    /// Sets all components of `nz` to realtype `c`.
    unsafe extern "C" fn nvconst_rust(c: realtype, nz: N_Vector) {
        let mut z = Self::box_of_nvector(nz);
        V::set_const(c, &mut z);
        Box::leak(z);
    }

    /// Sets `nz` to be the component-wise product of the inputs `nx`
    /// and `ny`: ∀i, zᵢ = xᵢ yᵢ.
    unsafe extern "C" fn nvprod_rust(
        nx: N_Vector, ny: N_Vector, nz: N_Vector
    ) {
        let x = Self::box_of_nvector(nx);
        let y = Self::box_of_nvector(ny);
        let mut z = Self::box_of_nvector(nz);
        V::set_prod(&x, &y, &mut z);
        Box::leak(x);
        Box::leak(y);
        Box::leak(z);
    }

    /// Sets the `nz` to be the component-wise ratio of the inputs
    /// `nx` and `ny`: ∀i, zᵢ = xᵢ/yᵢ.  The yᵢ may not be tested for 0
    /// values.  This function should only be called with a y that is
    /// guaranteed to have all nonzero components.
    unsafe extern "C" fn nvdiv_rust(
        nx: N_Vector, ny: N_Vector, nz: N_Vector
    ) {
        let x = Self::box_of_nvector(nx);
        let y = Self::box_of_nvector(ny);
        let mut z = Self::box_of_nvector(nz);
        V::set_div(&x, &y, &mut z);
        Box::leak(x);
        Box::leak(y);
        Box::leak(z);
    }

    /// Scales the `nx` by the scalar `c` and returns the result in
    /// `z`: ∀i, zᵢ = cxᵢ.
    unsafe extern "C" fn nvscale_rust(c: f64, nx: N_Vector, nz: N_Vector) {
        let x = Self::box_of_nvector(nx);
        let mut z = Self::box_of_nvector(nz);
        V::set_scale(c, &x, &mut z);
        Box::leak(x);
        Box::leak(z);
    }

    /// Sets the components of the `nz` to be the absolute values of
    /// the components of the `nx`: ∀i, zᵢ = |xᵢ|.
    unsafe extern "C" fn nvabs_rust(nx: N_Vector, nz: N_Vector) {
        let x = Self::box_of_nvector(nx);
        let mut z = Self::box_of_nvector(nz);
        V::set_abs(&x, &mut z);
        Box::leak(x);
        Box::leak(z);
    }

    /// Sets the components of the `nz` to be the inverses of the
    /// components of `nx`: ∀i, zᵢ = 1/xᵢ.
    unsafe extern "C" fn nvinv_rust(nx: N_Vector, nz: N_Vector) {
        let x = Self::box_of_nvector(nx);
        let mut z = Self::box_of_nvector(nz);
        V::set_inv(&x, &mut z);
        Box::leak(x);
        Box::leak(z);
    }

    /// Adds the scalar `b` to all components of `nx` and returns the
    /// result in `nz`: ∀i, zᵢ = xᵢ + b.
    unsafe extern "C" fn nvaddconst_rust(nx: N_Vector, b: f64, nz: N_Vector) {
        let x = Self::box_of_nvector(nx);
        let mut z = Self::box_of_nvector(nz);
        V::set_add_const(&x, b, &mut z);
        Box::leak(x);
        Box::leak(z);
    }

    /// Returns the value of the dot-product of `nx` and `ny`: ∑ xᵢ yᵢ.
    unsafe extern "C" fn nvdotprod_rust(
        nx: N_Vector, ny: N_Vector
    ) -> realtype {
        let x = Self::box_of_nvector(nx);
        let y = Self::box_of_nvector(ny);
        let dot = V::dot_prod(&x, &y);
        Box::leak(x);
        Box::leak(y);
        dot
    }

    /// Returns the value of the ℓ^∞ norm of `nx`: maxᵢ |xᵢ|.
    unsafe extern "C" fn nvmaxnorm_rust(nx: N_Vector) -> realtype {
        let x = Self::box_of_nvector(nx);
        let norm = V::max_norm(&x);
        Box::leak(x);
        norm
    }

    /// Returns the weighted root-mean-square norm of `nx` with
    /// (positive) realtype weight vector `nw`: √(∑ (xᵢ wᵢ)² / n).
    unsafe extern "C" fn nvwrmsnorm_rust(
        nx: N_Vector, nw: N_Vector
    ) -> realtype {
        let x = Self::box_of_nvector(nx);
        let w = Self::box_of_nvector(nw);
        let norm = V::wrms_norm(&x, &w);
        Box::leak(x);
        Box::leak(w);
        norm
    }

    /// Returns the weighted root mean square norm of `nx` with weight
    /// vector `nw` built using only the elements of `nx` corresponding
    /// to positive elements of the `nid`: √(∑ (xᵢ wᵢ H(idᵢ))² / n)
    /// where H(α) = 1 if α > 0 and H(α) = 0 if α ≤ 0.
    unsafe extern "C" fn nvwrmsnormmask_rust(
        nx: N_Vector, nw: N_Vector, nid: N_Vector
    ) -> realtype {
        let x = Self::box_of_nvector(nx);
        let w = Self::box_of_nvector(nw);
        let id = Self::box_of_nvector(nid);
        let norm = V::wrms_norm_mask(&x, &w, &id);
        Box::leak(x);
        Box::leak(w);
        Box::leak(id);
        norm
        }

    /// Returns the smallest element of the `nx`: minᵢ xᵢ.
    unsafe extern "C" fn nvmin_rust(nx: N_Vector) -> realtype {
        let x = Self::box_of_nvector(nx);
        let m = V::min(&x);
        Box::leak(x);
        m
    }

    /// Returns the weighted Euclidean norm of `nx` with realtype
    /// weight vector `nw`: √(∑ (xᵢ wᵢ)²).
    unsafe extern "C" fn nvwl2norm_rust(
        nx: N_Vector, nw: N_Vector
    ) -> realtype {
        let x = Self::box_of_nvector(nx);
        let w = Self::box_of_nvector(nw);
        let norm = V::wl2_norm(&x, &w);
        Box::leak(x);
        Box::leak(w);
        norm
    }

    /// Returns the ℓ¹ norm of `nx`: ∑ |xᵢ|.
    unsafe extern "C" fn nvl1norm_rust(nx: N_Vector) -> realtype {
        let x = Self::box_of_nvector(nx);
        let m = V::l1_norm(&x);
        Box::leak(x);
        m
    }

    /// Compares the components of `nx` to the realtype scalar `c` and
    /// returns a `nz` such that ∀i, zᵢ = 1.0 if |xᵢ| ≥ `c` and zᵢ = 0.0
    /// otherwise.
    unsafe extern "C" fn nvcompare_rust(c: f64, nx: N_Vector, nz: N_Vector) {
        let x = Self::box_of_nvector(nx);
        let mut z = Self::box_of_nvector(nz);
        V::set_compare(c, &x, &mut z);
        Box::leak(x);
        Box::leak(z);
    }

    /// Sets the components of `nz` to be the inverses of the
    /// components of `nx`, with prior testing for zero values:
    /// ∀i, zᵢ = 1/xᵢ.  This routine returns a boolean assigned to
    /// `SUNTRUE` if all components of x are nonzero (successful
    /// inversion) and returns `SUNFALSE` otherwise.
    unsafe extern "C" fn nvinvtest_rust(nx: N_Vector, nz: N_Vector) -> i32 {
        let x = Self::box_of_nvector(nx);
        let mut z = Self::box_of_nvector(nz);
        let b = V::set_inv_test(&x, &mut z);
        Box::leak(x);
        Box::leak(z);
        (if b { SUNTRUE } else { SUNFALSE }) as _
    }

    /// Performs the following constraint tests based on the values in cᵢ:
    /// - xᵢ > 0 if cᵢ = 2,
    /// - xᵢ ≥ 0 if cᵢ = 1,
    /// - xᵢ < 0 if cᵢ = -2,
    /// - xᵢ ≤ 0 if cᵢ = -1.
    /// There is no constraint on if cᵢ = 0.  This routine returns a
    /// boolean assigned to `SUNFALSE` if any element failed the
    /// constraint test and assigned to `SUNTRUE` if all passed.  It
    /// also sets a mask vector `nm`, with elements equal to 1.0 where
    /// the constraint test failed, and 0.0 where the test
    /// passed.  This routine is used only for constraint checking.
    unsafe extern "C" fn nvconstrmask_rust(
        nc: N_Vector, nx: N_Vector, nm: N_Vector
    ) -> i32 {
        let c = Self::box_of_nvector(nc);
        let x = Self::box_of_nvector(nx);
        let mut m = Self::box_of_nvector(nm);
        let b = V::set_constr_mask(&c, &x, &mut m);
        Box::leak(c);
        Box::leak(x);
        Box::leak(m);
        (if b { SUNTRUE } else { SUNFALSE }) as _
        }

    /// This routine returns the minimum of the quotients obtained by
    /// termwise dividing the elements of n = `nnum` by the elements
    /// in d = `ndenom`: minᵢ nᵢ/dᵢ.
    unsafe extern "C" fn nvminquotient_rust(
        nnum: N_Vector, ndenom: N_Vector
    ) -> f64 {
        let num = Self::box_of_nvector(nnum);
        let denom = Self::box_of_nvector(ndenom);
        let m = V::min_quotient(&num, &denom);
        Box::leak(num);
        Box::leak(denom);
        m
    }

    /// Return a new shared [`N_Vector`] built with context `ctx`.  It
    /// is required that `V` implements the trait `NVector`.  This is
    /// the case for [`[f64; N]`][std::array] and
    /// [`Vec<f64>`][std::vec::Vec].
    pub fn new(v: V, ctx: impl Context) -> Self {
        let nv = unsafe { N_VNewEmpty(ctx.as_ptr()) };
        if nv.is_null() {
            panic!("sundials::vector::custom::Shared::new: \
                Could not allocate new N_Vector.");
        }
        let ops = _generic_N_Vector_Ops {
            nvgetvectorid: Some(nvgetvectorid_rust),
            nvclone: Some(Self::nvclone_rust),
            nvcloneempty: Some(Self::nvcloneempty_rust),
            nvdestroy: Some(Self::nvdestroy_rust),
            nvspace: Some(Self::nvspace_rust),
            nvgetarraypointer: Some(Self::nvgetarraypointer_rust),
            nvgetdevicearraypointer: None,
            nvsetarraypointer: Some(Self::nvsetarraypointer_rust),
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
        };
        unsafe {
            (*nv).ops = Box::leak(Box::new(ops)) as *mut _;
        }
        Shared { nv , marker: PhantomData }
    }
}
