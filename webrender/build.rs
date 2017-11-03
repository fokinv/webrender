/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::collections::HashMap;
use std::env;
use std::fs::{canonicalize, read_dir, File};
use std::io::prelude::*;
use std::path::{Path, PathBuf};

#[cfg(all(target_os = "windows", feature="dx11"))]
use std::process::{self, Command, Stdio};


#[cfg(all(target_os = "windows", feature="dx11"))]
const DX11_DEFINE: &str = "#define WR_DX11\n";
const SHADER_IMPORT: &str = "#include ";
const SHADER_KIND_FRAGMENT: &str = "#define WR_FRAGMENT_SHADER\n";
const SHADER_KIND_VERTEX: &str = "#define WR_VERTEX_SHADER\n";
const SHADER_PREFIX: &str = "#define WR_MAX_VERTEX_TEXTURE_WIDTH 1024\n";
#[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
const SHADER_VERSION: &'static str = "#version 150\n";
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
const SHADER_VERSION: &'static str = "#version 300 es\n";

const CACHE_FEATURES: &[&str] = &[""];
const CLIP_FEATURES: &[&str] = &["TRANSFORM"];
const PRIM_DITHER_FEATURES: &[&str] = &["", "TRANSFORM", "DITHERING", "DITHERING,TRANSFORM"];
const PRIM_FEATURES: &[&str] = &["", "TRANSFORM"];

struct Shader {
    name: &'static str,
    source_name: &'static str,
    features: &'static [&'static str],
}

const SHADERS: &[Shader] = &[
    // Clip mask shaders
    Shader {
        name: "cs_clip_rectangle",
        source_name: "cs_clip_rectangle",
        features: CLIP_FEATURES,
    },
    Shader {
        name: "cs_clip_image",
        source_name: "cs_clip_image",
        features: CLIP_FEATURES,
    },
    Shader {
        name: "cs_clip_border",
        source_name: "cs_clip_border",
        features: CLIP_FEATURES,
    },
    // Cache shaders
    Shader {
        name: "cs_blur_a8",
        source_name: "cs_blur",
        features: &["ALPHA_TARGET"],
    },
    Shader {
        name: "cs_blur_rgba8",
        source_name: "cs_blur",
        features: &["COLOR_TARGET"],
    },
    Shader {
        name: "cs_text_run",
        source_name: "cs_text_run",
        features: CACHE_FEATURES,
    },
    Shader {
        name: "cs_line",
        source_name: "ps_line",
        features: &["CACHE"],
    },
    // Prim shaders
    Shader {
        name: "ps_line",
        source_name: "ps_line",
        features: &["", "TRANSFORM"],
    },
    Shader {
        name: "ps_border_corner",
        source_name: "ps_border_corner",
        features: PRIM_FEATURES,
    },
    Shader {
        name: "ps_border_edge",
        source_name: "ps_border_edge",
        features: PRIM_FEATURES,
    },
    Shader {
        name: "ps_gradient",
        source_name: "ps_gradient",
        features: PRIM_DITHER_FEATURES,
    },
    Shader {
        name: "ps_angle_gradient",
        source_name: "ps_angle_gradient",
        features: PRIM_DITHER_FEATURES,
    },
    Shader {
        name: "ps_radial_gradient",
        source_name: "ps_radial_gradient",
        features: PRIM_DITHER_FEATURES,
    },
    Shader {
        name: "ps_blend",
        source_name: "ps_blend",
        features: &[""],
    },
    Shader {
        name: "ps_composite",
        source_name: "ps_composite",
        features: &[""],
    },
    Shader {
        name: "ps_hardware_composite",
        source_name: "ps_hardware_composite",
        features: &[""],
    },
    Shader {
        name: "ps_split_composite",
        source_name: "ps_split_composite",
        features: &[""],
    },
    Shader {
        name: "ps_image",
        source_name: "ps_image",
        features: PRIM_FEATURES,
    },
    Shader {
        name: "ps_yuv_image",
        source_name: "ps_yuv_image",
        features: &["NV12",                      "",                     "INTERLEAVED_Y_CB_CR",
                    "NV12,YUV_REC709",           "YUV_REC709",           "INTERLEAVED_Y_CB_CR,YUV_REC709",
                    "NV12,TRANSFORM",            "TRANSFORM",            "INTERLEAVED_Y_CB_CR,TRANSFORM",
                    "NV12,YUV_REC709,TRANSFORM", "YUV_REC709,TRANSFORM", "INTERLEAVED_Y_CB_CR,YUV_REC709,TRANSFORM"],
    },
    Shader {
        name: "ps_text_run",
        source_name: "ps_text_run",
        features: PRIM_FEATURES,
    },
    Shader {
        name: "ps_rectangle",
        source_name: "ps_rectangle",
        features: &["", "TRANSFORM", "CLIP", "CLIP,TRANSFORM"],
    },
    // Brush shaders
    Shader {
        name: "brush_mask",
        source_name: "brush_mask",
        features: &[""],
    },
    Shader {
        name: "brush_image",
        source_name: "brush_image",
        features: &["ALPHA_TARGET","ALPHA_TARGET,ALPHA_PASS","COLOR_TARGET","COLOR_TARGET,ALPHA_PASS"],
    },
    Shader {
        name: "debug_color",
        source_name: "debug_color",
        features: &[""],
    },
    Shader {
        name: "debug_font",
        source_name: "debug_font",
        features: &[""],
    },
];

fn write_shaders(glsl_files: Vec<PathBuf>, shader_file_path: &Path) -> HashMap<String, String> {
    let mut shader_file = File::create(shader_file_path).unwrap();
    let mut shader_map: HashMap<String, String> = HashMap::with_capacity(glsl_files.len());

    write!(shader_file, "/// AUTO GENERATED BY build.rs\n\n").unwrap();
    write!(shader_file, "use std::collections::HashMap;\n").unwrap();
    write!(shader_file, "lazy_static! {{\n").unwrap();
    write!(shader_file,"  pub static ref SHADERS: HashMap<&'static str, &'static str> = {{\n").unwrap();
    write!(shader_file, "    let mut h = HashMap::new();\n").unwrap();
    for glsl in glsl_files {
        let shader_name = glsl.file_name().unwrap().to_str().unwrap();
        // strip .glsl
        let shader_name = shader_name.replace(".glsl", "");
        let full_path = canonicalize(&glsl).unwrap();
        let full_name = full_path.as_os_str().to_str().unwrap();
        // if someone is building on a network share, I'm sorry.
        let full_name = full_name.replace("\\\\?\\", "");
        let full_name = full_name.replace("\\", "/");
        shader_map.insert(shader_name.clone(), full_name.clone());
        write!(
            shader_file,
            "    h.insert(\"{}\", include_str!(\"{}\"));\n",
            shader_name,
            full_name
        ).unwrap();
    }
    write!(shader_file, "    h\n").unwrap();
    write!(shader_file, "  }};\n").unwrap();
    write!(shader_file, "}}\n").unwrap();
    shader_map
}

fn create_shaders(out_dir: String, shaders: &HashMap<String, String>) -> Vec<String> {
    fn get_shader_source(shader_name: &str, shaders: &HashMap<String, String>) -> Option<String> {
        if let Some(shader_file) = shaders.get(shader_name) {
            let shader_file_path = Path::new(shader_file);
            if let Ok(mut shader_source_file) = File::open(shader_file_path) {
                let mut source = String::new();
                shader_source_file.read_to_string(&mut source).unwrap();
                Some(source)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn parse_shader_source(source: &str, shaders: &HashMap<String, String>, output: &mut String) {
        for line in source.lines() {
            if line.starts_with(SHADER_IMPORT) {
                let imports = line[SHADER_IMPORT.len()..].split(",");
                // For each import, get the source, and recurse.
                for import in imports {
                    if let Some(include) = get_shader_source(import, shaders) {
                        parse_shader_source(&include, shaders, output);
                    }
                }
            } else {
                output.push_str(line);
                output.push_str("\n");
            }
        }
    }

    pub fn build_shader_strings(base_filename: &str, features: &str, shaders: &HashMap<String, String>) -> (String, String) {
        // Construct a list of strings to be passed to the shader compiler.
        let mut vs_source = String::new();
        let mut fs_source = String::new();

        #[cfg(not(feature = "dx11"))]
        vs_source.push_str(SHADER_VERSION);
        #[cfg(not(feature = "dx11"))]
        fs_source.push_str(SHADER_VERSION);

        // Define a constant depending on whether we are compiling VS or FS.
        vs_source.push_str(SHADER_KIND_VERTEX);
        fs_source.push_str(SHADER_KIND_FRAGMENT);

        // Add any defines that were passed by the caller.
        vs_source.push_str(features);
        fs_source.push_str(features);

        // Parse the main .glsl file, including any imports
        // and append them to the list of sources.
        let mut shared_result = String::new();
        if let Some(shared_source) = get_shader_source(base_filename, shaders) {
            parse_shader_source(&shared_source, shaders, &mut shared_result);
        }

        //vs_source.push_str(SHADER_LINE_MARKER);
        vs_source.push_str(&shared_result);
        //fs_source.push_str(SHADER_LINE_MARKER);
        fs_source.push_str(&shared_result);

        (vs_source, fs_source)
    }

    let mut file_names = Vec::new();
    for shader in SHADERS {
        for config in shader.features {
            let mut features = String::new();

            features.push_str(SHADER_PREFIX);
            #[cfg(all(target_os = "windows", feature="dx11"))]
            features.push_str(DX11_DEFINE);
            features.push_str(format!("//Source: {}.glsl\n", shader.source_name).as_str());

            let mut file_name_postfix = String::new();
            for feature in config.split(",") {
                if !feature.is_empty() {
                    features.push_str(&format!("#define WR_FEATURE_{}\n", feature));
                    if shader.name == shader.source_name {
                        file_name_postfix.push_str(&format!("_{}", feature.to_lowercase().as_str()));
                    }
                }
            }
            let (mut vs_source, mut fs_source) = build_shader_strings(shader.source_name, &features, shaders);

            let mut filename = String::from(shader.name);
            filename.push_str(file_name_postfix.as_str());
            let (mut vs_name, mut fs_name) = (filename.clone(), filename);
            vs_name.push_str(".vert");
            fs_name.push_str(".frag");
            println!("vs_name = {}, shader.name = {}", vs_name, shader.name);
            let (vs_file_path, fs_file_path) = (Path::new(&out_dir).join(vs_name.clone()), Path::new(&out_dir).join(fs_name.clone()));
            let (mut vs_file, mut fs_file) = (File::create(vs_file_path).unwrap(), File::create(fs_file_path).unwrap());
            write!(vs_file, "{}", vs_source).unwrap();
            write!(fs_file, "{}", fs_source).unwrap();
            file_names.push(vs_name);
            file_names.push(fs_name);
        }
    }
    file_names
}

#[cfg(all(target_os = "windows", feature="dx11"))]
fn compile_fx_files(file_names: Vec<String>, out_dir: String) {
    for mut file_name in file_names {
        //TODO: Remove SUPPORTED_SHADERS when all shader conversion is done.
        if file_name.contains("ps_clear")
           || !(file_name.contains("ps_") || file_name.contains("cs_") || file_name.starts_with("debug_")) {
            continue;
        }
        let is_vert = file_name.ends_with(".vert");
        if !is_vert && !file_name.ends_with(".frag") {
            continue;
        }
        let file_path = Path::new(&out_dir).join(&file_name);
        file_name.push_str(".fx");
        let fx_file_path = Path::new(&out_dir).join(&file_name);
        let pf_path = env::var("ProgramFiles(x86)").ok().expect("Please set the ProgramFiles(x86) enviroment variable");
        let pf_path = Path::new(&pf_path);
        let format = if is_vert {
            "vs_5_0"
        } else {
            "ps_5_0"
        };
        let mut command = Command::new(pf_path.join("Windows Kits").join("8.1").join("bin").join("x64").join("fxc.exe").to_str().unwrap());
        command.arg("/Zi"); // Debug info
        command.arg("/T");
        command.arg(format);
        command.arg("/Fo");
        command.arg(&fx_file_path);
        command.arg(&file_path);
        println!("{:?}", command);
        if command.stdout(Stdio::inherit()).stderr(Stdio::inherit()).status().unwrap().code().unwrap() != 0
        {
            println!("Error while executing fxc");
            process::exit(1)
        }
    }
}

#[cfg(feature = "vulkan")]
fn compile_spirv(file_name_vector: Vec<String>, out_dir: String) {
    for mut file_name in file_name_vector {
        let file_path = Path::new(&out_dir).join(&file_name);
        file_name.push_str(".spv");
        let spirv_file_path = Path::new(&out_dir).join(&file_name);
        #[cfg(any(target_os = "android", target_os = "linux"))]
        let mut command = Command::new(Path::new("./tools/glslangValidator"));
        #[cfg(target_os = "windows")]
        let mut command = Command::new(Path::new("./tools/glslangValidator.exe"));
        command.arg("-V");
        command.arg("-o");
        command.arg(&spirv_file_path);
        command.arg(&file_path);
        println!("{:?}", command);
        if command.stdout(Stdio::inherit())
                  .stderr(Stdio::inherit())
                  .status().unwrap().code().unwrap() != 0
        {
            println!("Error while compiling spirv");
            process::exit(1)
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

    // Sort the file list so that the shaders.rs file is filled
    // deterministically.
    glsl_files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    let shader_map = write_shaders(glsl_files, &shaders_file);
    let _file_names = create_shaders(out_dir.clone(), &shader_map);

    #[cfg(all(target_os = "windows", feature = "dx11"))]
    compile_fx_files(_file_names, out_dir);

    #[cfg(feature = "vulkan")]
    compile_spirv(_file_names, out_dir);

    //write_shaders(glsl_files, &shaders_file);
}
