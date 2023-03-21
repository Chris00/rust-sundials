use sundials_sys as s;

fn main() {
    println!("cargo:rustc-cfg=sundials_major=\"{}\"",
             s::SUNDIALS_VERSION_MAJOR);
    println!("cargo:rustc-cfg=sundials_minor=\"{}\"",
             s::SUNDIALS_VERSION_MINOR);
    if s::SUNDIALS_VERSION_MAJOR >= 6 {
        println!("cargo:rustc-cfg=has_context")
    }
}
