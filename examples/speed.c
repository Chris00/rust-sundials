#include <assert.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <cvode/cvode.h>

int cvrhs(double t, N_Vector u, N_Vector du, void* user_data) {
  /* double *u_data = N_VGetArrayPointer(u); */
  double *du_data = N_VGetArrayPointer(du);
  du_data[0] = 1.;
  return 0;
}

int main() {
  for (int i = 0; i < 100000; i++) {
    SUNContext ctx;
    void *cvode_mem;
    double t0, t, *y0_data, *u_data;
    N_Vector y0, u;
    SUNLinearSolver linsol;
    int ret;
    
    if (SUNContext_Create(NULL, &ctx) < 0) {
      printf("SUNContext_Create");
      return 1;
    };
    cvode_mem = CVodeCreate(CV_ADAMS, ctx);    
    t0 = 0.;
    y0 = N_VNew_Serial(1, ctx);
    if (y0 == NULL) {
      printf("N_VNew_Serial");
      return 1;
    }
    y0_data = N_VGetArrayPointer(y0);
    y0_data[0] = 0.;
    ret = CVodeInit(cvode_mem, cvrhs, t0, y0);
    if (ret != CV_SUCCESS) {
      printf("CVodeInit");
      return 1;
    }
    CVodeSStolerances(cvode_mem, 1e-6, 1e-12);

    /* Because the dimension is small, a direct solver is more efficient. */
    /* SUNMatrix mat = SUNDenseMatrix(1, 1, ctx); */
    /* linsol = SUNLinSol_Dense(y0, mat, ctx); */
    /* ret = CVodeSetLinearSolver(cvode_mem, linsol, mat); */

    /* Iterative solver (default in Rust) */
    linsol = SUNLinSol_SPGMR(y0, SUN_PREC_NONE, 30, ctx);
    ret = CVodeSetLinearSolver(cvode_mem, linsol, NULL);

    if (ret != CV_SUCCESS) {
      printf("CVodeSetLinearSolver");
      return 1;
    }
    CVodeSetMaxHnilWarns(cvode_mem, 10);

    u = N_VNew_Serial(1, ctx);
    if (u == NULL) {
      printf("N_VNew_Serial: u");
      return 1;
    }
    int status = CVode(cvode_mem, 1., u, &t, CV_NORMAL);
    assert(status == CV_SUCCESS);
    u_data = N_VGetArrayPointer(u);
    /* printf("%f\n", u_data[0]); */

    CVodeFree(&cvode_mem);
    N_VDestroy(y0);
    N_VDestroy(u);
    /* SUNMatDestroy(mat); */
    SUNLinSolFree(linsol);
    SUNContext_Free(&ctx);
  }
  return 0;
}


/* Local Variables: */
/* compile-command: "gcc -O3 speed.c -o ../target/speed_c -lsundials_generic -lsundials_nvecserial -lsundials_cvode -lsundials_sunmatrixdense -lsundials_sunlinsoldense -lsundials_sunlinsolspgmr" */
/* End: */
