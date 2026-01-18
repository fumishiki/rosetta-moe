//! Build script for CUDA kernels

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");

    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_include = format!("{}/include", cuda_path);
    let cuda_lib = format!("{}/lib64", cuda_path);

    // Check if CUDA is available
    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let has_cuda = std::path::Path::new(&nvcc).exists();

    if !has_cuda {
        println!("cargo:warning=CUDA not found, building stub library");
        build_stub();
        return;
    }

    // Compile CUDA kernels
    let kernel_dir = PathBuf::from("kernels");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let kernels = [
        "elementwise",
        "softmax",
        "rmsnorm",
        "gemm",
        "rope",
        "attention",
        "loss",
        "optimizer",
    ];

    let mut objects = Vec::new();

    for kernel in &kernels {
        let src = kernel_dir.join(format!("{}.cu", kernel));
        let obj = out_dir.join(format!("{}.o", kernel));

        let status = std::process::Command::new(&nvcc)
            .args([
                "-c",
                "-O3",
                "--use_fast_math",
                "-Xcompiler", "-fPIC",
                "-arch=sm_80",  // Ampere (A100, RTX 30xx)
                "-gencode=arch=compute_70,code=sm_70",  // Volta (V100)
                "-gencode=arch=compute_75,code=sm_75",  // Turing (RTX 20xx)
                "-gencode=arch=compute_80,code=sm_80",  // Ampere
                "-gencode=arch=compute_86,code=sm_86",  // Ampere (RTX 30xx consumer)
                "-gencode=arch=compute_89,code=sm_89",  // Ada (RTX 40xx)
                "-gencode=arch=compute_90,code=sm_90",  // Hopper (H100)
                "-I", kernel_dir.to_str().unwrap(),
                "-o", obj.to_str().unwrap(),
                src.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to run nvcc");

        if !status.success() {
            panic!("Failed to compile {}", kernel);
        }

        objects.push(obj);
    }

    // Link into static library
    let lib_path = out_dir.join("libnn_cuda_kernels.a");
    let mut ar_cmd = std::process::Command::new("ar");
    ar_cmd.arg("rcs").arg(&lib_path);
    for obj in &objects {
        ar_cmd.arg(obj);
    }
    let status = ar_cmd.status().expect("Failed to run ar");
    if !status.success() {
        panic!("Failed to create static library");
    }

    // Link instructions
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=nn_cuda_kernels");
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    // Include path for Rust
    println!("cargo:include={}", cuda_include);
}

/// Build a stub library when CUDA is not available
fn build_stub() {
    cc::Build::new()
        .file("src/stub.c")
        .compile("nn_cuda_kernels");
}
