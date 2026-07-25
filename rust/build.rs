// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

fn main() {
    #[cfg(feature = "metal")]
    {
        cc::Build::new()
            .file("metal-bridge/metal_bridge.m")
            .flag("-fobjc-arc")
            .compile("metal_bridge");

        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
    }
}
