use sundials_sys::*;
use std::any::TypeId;

fn main() {
    // Write a custom error message
    if TypeId::of::<realtype>() != TypeId::of::<f64>() {
        println!("cargo::error=Sundials must be compiled with f64 precision.");
    }
}
