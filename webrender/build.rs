/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::io::prelude::*;
use std::fs::{canonicalize, read_dir, File};
use std::process::{self, Command, Stdio};

#[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
const SHADER_VERSION: &'static str = "#version 150\n";

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
const SHADER_VERSION: &'static str = "#version 300 es\n";

#[cfg(any(target_os = "windows"))]
const SHADER_VERSION: &'static str = "";

const SUPPORTED_SHADERS: &'static [&'static str] = &["ps_rectangle."];

fn write_shaders(glsl_files: Vec<PathBuf>, shader_file_path: &Path) {
    let mut shader_file = File::create(shader_file_path).unwrap();

    write!(shader_file, "/// AUTO GENERATED BY build.rs\n\n").unwrap();
    write!(shader_file, "use std::collections::HashMap;\n").unwrap();
    write!(shader_file, "lazy_static! {{\n").unwrap();
    write!(shader_file, "  pub static ref SHADERS: HashMap<&'static str, &'static str> = {{\n").unwrap();
    write!(shader_file, "    let mut h = HashMap::with_capacity({});\n", glsl_files.len()).unwrap();
    for glsl in glsl_files {
        let shader_name = glsl.file_name().unwrap().to_str().unwrap();
        // strip .glsl
        let shader_name = shader_name.replace(".glsl", "");
        let full_path = canonicalize(&glsl).unwrap();
        let full_name = full_path.as_os_str().to_str().unwrap();
        // if someone is building on a network share, I'm sorry.
        let full_name = full_name.replace("\\\\?\\", "");
        let full_name = full_name.replace("\\", "/");
        write!(shader_file, "    h.insert(\"{}\", include_str!(\"{}\"));\n",
               shader_name, full_name).unwrap();
    }
    write!(shader_file, "    h\n").unwrap(); 
    write!(shader_file, "  }};\n").unwrap(); 
    write!(shader_file, "}}\n").unwrap(); 
}

fn create_shaders(glsl_files: Vec<PathBuf>, out_dir: String) -> Vec<String> {
    fn gen_shaders(glsl_files: Vec<PathBuf>) -> HashMap<String, String> {
        let mut shaders: HashMap<String, String> = HashMap::with_capacity(glsl_files.len());
        for glsl in glsl_files {
            let shader_name = glsl.file_name().unwrap().to_str().unwrap();
            // strip .glsl
            let shader_name = shader_name.replace(".glsl", "");
            let full_path = canonicalize(&glsl).unwrap();
            let full_name = full_path.as_os_str().to_str().unwrap();
            // if someone is building on a network share, I'm sorry.
            let full_name = full_name.replace("\\\\?\\", "");
            let full_name = full_name.replace("\\", "/");
            shaders.insert(shader_name, full_name);
        }
        shaders
    }

    fn get_shader_source(shader_file: &String) -> String {
        let shared_file_path = Path::new(shader_file);
        let mut shader_source_file = File::open(shared_file_path).unwrap();
        let mut s = String::new();
        shader_source_file.read_to_string(&mut s).unwrap();
        s
    }

    let shaders = &gen_shaders(glsl_files);
    let shared_src = shaders.get("shared").unwrap();
    let prim_shared_src = shaders.get("prim_shared").unwrap();
    let clip_shared_src = shaders.get("clip_shared").unwrap();
    let mut file_name_vector = vec![];

    for (filename, file_source) in shaders {
        let is_prim = filename.starts_with("ps_");
        let is_cache = filename.starts_with("cs_");
        let is_clip_cache = filename.starts_with("cs_clip");
        let is_vert = filename.ends_with(".vs");
        let is_frag = filename.ends_with(".fs");
        let is_ps_rect = filename.starts_with("ps_rectangle");
        let is_ps_text_run = filename.starts_with("ps_text_run");
        let is_ps_blend = filename.starts_with("ps_blend");
        let is_ps_hw_composite = filename.starts_with("ps_hardware_composite");
        let is_ps_composite = filename.starts_with("ps_composite");
        let is_ps_split_composite = filename.starts_with("ps_split_composite");
        let use_dither  = filename.starts_with("ps_gradient") ||
                          filename.starts_with("ps_angle_gradient") ||
                          filename.starts_with("ps_radial_gradient");
        let is_ps_yuv = filename.starts_with("ps_yuv");
        // The shader must be primitive or clip (only one of them)
        // and it must be fragment or vertex shader (only one of them), else we skip it.
        if !(is_prim ^ is_cache) || !(is_vert ^ is_frag) {
            continue;
        }

        let base_filename = filename.splitn(2, '.').next().unwrap();
        let mut shader_prefix =
            format!("{}\n// Base shader: {}\n#define WR_MAX_VERTEX_TEXTURE_WIDTH {}\n",
                    SHADER_VERSION, base_filename, 1024);

        if is_vert {
            shader_prefix.push_str("#define WR_VERTEX_SHADER\n");
        } else {
            shader_prefix.push_str("#define WR_FRAGMENT_SHADER\n");
        }

        let mut build_configs = vec!["#define WR_FEATURE_TRANSFORM\n"];
        if is_prim {
            // the transform feature may be disabled for the prim shaders
            build_configs.push("// WR_FEATURE_TRANSFORM disabled\n");
        }

        if is_ps_rect {
            build_configs.push("#define WR_FEATURE_TRANSFORM\n#define WR_FEATURE_CLIP\n");
            build_configs.push("// WR_FEATURE_TRANSFORM disabled\n#define WR_FEATURE_CLIP\n");
        }

        if is_ps_text_run {
            build_configs.push("#define WR_FEATURE_TRANSFORM\n#define WR_FEATURE_SUBPIXEL_AA\n");
            build_configs.push("// WR_FEATURE_TRANSFORM disabled\n#define WR_FEATURE_SUBPIXEL_AA\n");
        }

        if use_dither {
            build_configs.push("#define WR_FEATURE_TRANSFORM\n#define WR_FEATURE_DITHERING\n");
            build_configs.push("// WR_FEATURE_TRANSFORM disabled\n#define WR_FEATURE_DITHERING\n");
        }

        if is_ps_yuv {
            build_configs = vec!["// WR_FEATURE_TRANSFORM disabled\n#define WR_FEATURE_NV12\n"];
            build_configs.push("// WR_FEATURE_TRANSFORM disabled\n");
            build_configs.push("// WR_FEATURE_TRANSFORM disabled\n#define WR_FEATURE_INTERLEAVED_Y_CB_CR\n");
            build_configs.push("// WR_FEATURE_TRANSFORM disabled\n#define WR_FEATURE_NV12\n#define WR_FEATURE_YUV_REC709\n");
            build_configs.push("// WR_FEATURE_TRANSFORM disabled\n#define WR_FEATURE_YUV_REC709\n");
            build_configs.push("// WR_FEATURE_TRANSFORM disabled\n#define WR_FEATURE_INTERLEAVED_Y_CB_CR\n#define WR_FEATURE_YUV_REC709\n");
            build_configs.push("#define WR_FEATURE_TRANSFORM\n#define WR_FEATURE_NV12\n");
            build_configs.push("#define WR_FEATURE_TRANSFORM\n");
            build_configs.push("#define WR_FEATURE_TRANSFORM\n#define WR_FEATURE_INTERLEAVED_Y_CB_CR\n");
            build_configs.push("#define WR_FEATURE_TRANSFORM\n#define WR_FEATURE_NV12\n#define WR_FEATURE_YUV_REC709\n");
            build_configs.push("#define WR_FEATURE_TRANSFORM\n#define WR_FEATURE_YUV_REC709\n");
            build_configs.push("#define WR_FEATURE_TRANSFORM\n#define WR_FEATURE_INTERLEAVED_Y_CB_CR\n#define WR_FEATURE_YUV_REC709\n");

        }

        for (iter, config_prefix) in build_configs.iter().enumerate() {
            let mut shader_source = String::new();
            shader_source.push_str(shader_prefix.as_str());
            shader_source.push_str(config_prefix);
            shader_source.push_str(&get_shader_source(&shared_src));
            shader_source.push_str(&get_shader_source(&prim_shared_src));
            if is_clip_cache {
                shader_source.push_str(&get_shader_source(&clip_shared_src));
            }
            if let Some(optional_src) = shaders.get(base_filename) {
                shader_source.push_str(&get_shader_source(&optional_src));
            }
            shader_source.push_str(&get_shader_source(&file_source));
            let mut file_name = String::from(base_filename);
            if !is_ps_yuv {
            // The following cases are possible:
            // 0: Default, transfrom feature is enabled.
            //    Except for ps_blend, ps_hw_composite, ps_composite and ps_split_composite shaders.
            // 1: If the shader is prim shader, and the transform feature is disabled.
            //    This is the default case for ps_blend, ps_hw_composite, ps_composite and ps_split_composite shaders.
            // 2: If the shader is the `ps_rectangle`/`ps_text_run`/`gradient` shader
            //    and the `clip`/`subpixel AA`/`dither`, transfrom features are enabled.
            // 3: If the shader is the `ps_rectangle`/`ps_text_run`/`gradient` shader
            //    and the `clip`/`subpixel AA`/`dither` feature is enabled but the the transfrom feature is disabled.
                match iter {
                    0 => {
                        if is_prim && !(is_ps_blend || is_ps_hw_composite || is_ps_composite || is_ps_split_composite) {
                            file_name.push_str("_transform");
                        }
                    },
                    1 => {},
                    2 => {
                        if is_ps_rect {
                            file_name.push_str("_clip_transform");
                        }
                        if is_ps_text_run {
                            file_name.push_str("_subpixel_transform");
                        }
                        if use_dither {
                            file_name.push_str("_dither_transform");
                        }
                    },
                    3 => {
                        if is_ps_rect {
                            file_name.push_str("_clip");
                        }
                        if is_ps_text_run {
                            file_name.push_str("_subpixel");
                        }
                        if use_dither {
                            file_name.push_str("_dither");
                        }
                    },
                    _ => unreachable!(),
                }
            } else {
                match iter {
                    0 => file_name.push_str("_nv12_601"),
                    1 => file_name.push_str("_planar_601"),
                    2 => file_name.push_str("_interleaved_601"),
                    3 => file_name.push_str("_nv12_709"),
                    4 => file_name.push_str("_planar_709"),
                    5 => file_name.push_str("_interleaved_709"),
                    6 => file_name.push_str("_nv12_601_transform"),
                    7 => file_name.push_str("_planar_601_transform"),
                    8 => file_name.push_str("_interleaved_601_transform"),
                    9 => file_name.push_str("_nv12_709_transform"),
                    10 => file_name.push_str("_planar_709_transform"),
                    11 => file_name.push_str("_interleaved_709_transform"),
                    _ => unreachable!(),
                }
            }
            if is_vert {
                file_name.push_str(".vert");
            } else {
                file_name.push_str(".frag");
            }
            let file_path = Path::new(&out_dir).join(&file_name);
            let mut file = File::create(&file_path).unwrap();
            write!(file, "{}", shader_source).unwrap();
            file_name_vector.push(file_name);
        }
    }
    return file_name_vector;
}

#[cfg(any(target_os = "windows"))]
fn compile_fx_files(file_name_vector: Vec<String>, out_dir: String) {
    for mut file_name in file_name_vector {
        let is_vert = file_name.ends_with(".vert");
        let supported = SUPPORTED_SHADERS.iter().any(|s| file_name.contains(s));
        let file_path = Path::new(&out_dir).join(&file_name);
        if cfg!(target_os = "windows") && supported {
            file_name.push_str(".fx");
            let fx_file_path = Path::new(&out_dir).join(&file_name);
            //let sdk_path = env::var("DXSDK_DIR").ok().expect("Please set the DXSDK_DIR enviroment variable");
            //let sdk_path = PaŁth::new(&sdk_path);
            let pf_path = env::var("ProgramFiles(x86)").ok().expect("Please set the ProgramFiles(x86) enviroment variable");
            let pf_path = Path::new(&pf_path);
            let format = if is_vert {
                "vs_5_0"
            } else {
                "ps_5_0"
            };
            let mut command = Command::new(pf_path.join("Windows Kits").join("8.1").join("bin").join("x64").join("fxc.exe").to_str().unwrap());
            command.arg("/T");
            command.arg(format);
            command.arg("/Fo");
            command.arg(&fx_file_path);
            command.arg(&file_path);
            println!("{:?}", command);
            if command.stdout(Stdio::inherit())
                      .stderr(Stdio::inherit())
                      .status().unwrap().code().unwrap() != 0
            {
                println!("Error while executing fxc");
                process::exit(1)
            }
        }
    }
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap_or("out".to_owned());

    let shaders_file = Path::new(&out_dir).join("shaders.rs");
    let mut glsl_files = vec![];

    println!("cargo:rerun-if-changed=res");
    let res_dir = Path::new("res");
    for entry in read_dir(res_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if entry.file_name().to_str().unwrap().ends_with(".glsl") {
            println!("cargo:rerun-if-changed={}", path.display());
            glsl_files.push(path.to_owned());
        }
    }

    write_shaders(glsl_files.clone(), &shaders_file);
    let file_name_vector = create_shaders(glsl_files.clone(), out_dir.clone());
    #[cfg(any(target_os = "windows"))]
    compile_fx_files(file_name_vector, out_dir);
}
