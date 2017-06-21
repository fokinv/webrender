/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::Transform3D;
use fnv::FnvHasher;
use internal_types::{PackedVertex, RenderTargetMode, TextureSampler, DEFAULT_TEXTURE};
use internal_types::{BlurAttribute, ClipAttribute, VertexAttribute};
use internal_types::{DebugFontVertex, DebugColorVertex};
//use notify::{self, Watcher};
use super::shader_source;
use std::collections::HashMap;
use std::fs::File;
use std::hash::BuildHasherDefault;
use std::io::Read;
use std::iter::repeat;
use std::mem;
use std::ops::Add;
use std::path::PathBuf;
use std::ptr;
use std::rc::Rc;
//use std::sync::mpsc::{channel, Sender};
//use std::thread;
use webrender_traits::{ColorF, ImageFormat};
use webrender_traits::{DeviceIntPoint, DeviceIntRect, DeviceIntSize, DeviceUintSize};

use euclid::Matrix4D;

use rand::Rng;
use std;
use gfx;
use gfx::memory::Typed;
use gfx::Factory;
use gfx::texture::Kind;
use gfx::traits::FactoryExt;
use gfx::format::{DepthStencil as DepthFormat, Rgba8 as ColorFormat};
use gfx_device_gl as device_gl;
use gfx_device_gl::{Resources as R, CommandBuffer as CB};
use gfx::CombinedError;
use gfx::format::{Formatted, R8, Rgba8, Rgba32F, Srgba8, SurfaceTyped, TextureChannel, TextureSurface, Unorm};
use gfx::format::{R8_G8_B8_A8, R32_G32_B32_A32, Srgb};
use glutin;
use pipelines::{primitive, ClipProgram, Position, PrimitiveInstances, Program};
use prim_store::GRADIENT_DATA_SIZE;
use tiling::{CacheClipInstance, PrimitiveInstance};
use renderer::{BlendMode, DITHER_ID, DUMMY_A8_ID, DUMMY_RGBA8_ID, MAX_VERTEX_TEXTURE_WIDTH};
use webrender_traits::DeviceUintRect;

pub type A8 = (R8, Unorm);
pub const VECS_PER_DATA_16: usize = 1;
pub const VECS_PER_DATA_32: usize = 2;
pub const VECS_PER_DATA_64: usize = 4;
pub const VECS_PER_DATA_128: usize = 8;
pub const VECS_PER_GRADIENT_DATA: usize = 520;
pub const VECS_PER_LAYER: usize = 9;
pub const VECS_PER_PRIM_GEOM: usize = 2;
pub const VECS_PER_RENDER_TASK: usize = 3;
pub const VECS_PER_RESOURCE_RECTS: usize = 1;
pub const VECS_PER_SPLIT_GEOM: usize = 3;
pub const LAYER_TEXTURE_WIDTH: usize = 1017;
pub const RENDER_TASK_TEXTURE_WIDTH: usize = 1023;
pub const TEXTURE_HEIGTH: usize = 8;
pub const DEVICE_PIXEL_RATIO: f32 = 1.0;
pub const MAX_INSTANCE_COUNT: usize = 2000;

pub const A_STRIDE: usize = 1;
pub const RG_STRIDE: usize = 2;
pub const RGB_STRIDE: usize = 3;
pub const RGBA_STRIDE: usize = 4;
pub const FIRST_UNRESERVED_ID: u32 = DITHER_ID + 1;

#[derive(Clone, Debug, PartialEq)]
pub struct Texture<R, T> where R: gfx::Resources,
                               T: gfx::format::TextureFormat {
    /// Pixel storage for texture.
    pub surface: gfx::handle::Texture<R, T::Surface>,
    /// Sampler for texture.
    pub sampler: gfx::handle::Sampler<R>,
    /// View used by shader.
    pub view: gfx::handle::ShaderResourceView<R, T::View>,
    /// Filtering mode
    pub filter: TextureFilter,
    /// ImageFormat
    pub format: ImageFormat,
    /// Render Target mode
    pub mode: RenderTargetMode,
}

impl<R, T> Texture<R, T> where R: gfx::Resources, T: gfx::format::TextureFormat {

    pub fn empty<F>(factory: &mut F, size: [usize; 2], filter_method: TextureFilter, texture_kind: TextureTarget) -> Result<Texture<R, T>, CombinedError>
        where F: gfx::Factory<R>
    {
        Texture::create(factory, None, size, filter_method, texture_kind)
    }

    pub fn create<F>(factory: &mut F,
                     data: Option<&[&[u8]]>,
                     size: [usize; 2],
                     filter: TextureFilter,
                     texture_kind: TextureTarget,
    ) -> Result<Texture<R, T>, CombinedError>
        where F: gfx::Factory<R>
    {
        let (width, height) = (size[0] as u16, size[1] as u16);
        let tex_kind = match texture_kind {
            TextureTarget::Array => Kind::D2Array(width, height, 1, gfx::texture::AaMode::Single),
            _ => Kind::D2(width, height, gfx::texture::AaMode::Single),
        };

        let filter_method = match filter {
            TextureFilter::Nearest => gfx::texture::FilterMethod::Scale,
            TextureFilter::Linear => gfx::texture::FilterMethod::Bilinear,
        };

        let mut sampler_info = gfx::texture::SamplerInfo::new(
            filter_method,
            gfx::texture::WrapMode::Clamp
        );
        sampler_info.wrap_mode = (gfx::texture::WrapMode::Clamp, gfx::texture::WrapMode::Clamp, gfx::texture::WrapMode::Tile);

        let (surface, view, format) = {
            use gfx::{format, texture};
            use gfx::memory::{Usage, SHADER_RESOURCE};

            let surface = <T::Surface as format::SurfaceTyped>::get_surface_type();
            let desc = texture::Info {
                kind: tex_kind,
                levels: 1,
                format: surface,
                bind: SHADER_RESOURCE,
                usage: Usage::Dynamic,
            };
            let cty = <T::Channel as format::ChannelTyped>::get_channel_type();
            let raw = try!(factory.create_texture_raw(desc, Some(cty), data));
            let levels = (0, raw.get_info().levels - 1);
            let tex = Typed::new(raw);
            let view = try!(factory.view_texture_as_shader_resource::<T>(
                &tex, levels, format::Swizzle::new()
            ));
            let format = match surface {
                gfx::format::SurfaceType::R8 => ImageFormat::A8,
                gfx::format::SurfaceType::R8_G8_B8_A8 | gfx::format::SurfaceType::B8_G8_R8_A8 => ImageFormat::BGRA8,
                gfx::format::SurfaceType::R32_G32_B32_A32 => ImageFormat::RGBAF32,
                _ => unimplemented!(),
            };
            (tex, view, format)
        };

        let sampler = factory.create_sampler(sampler_info);

        Ok(Texture {
            surface: surface,
            sampler: sampler,
            view: view,
            filter: filter,
            format: format,
            mode: RenderTargetMode::None,
        })
    }

    #[inline(always)]
    pub fn get_size(&self) -> (usize, usize) {
        let (w, h, _, _) = self.surface.get_info().kind.get_dimensions();
        (w as usize, h as usize)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd)]
pub struct FrameId(usize);

impl FrameId {
    pub fn new(value: usize) -> FrameId {
        FrameId(value)
    }
}

impl Add<usize> for FrameId {
    type Output = FrameId;

    fn add(self, other: usize) -> FrameId {
        FrameId(self.0 + other)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TextureTarget {
    Default,
    Array,
    External,
    Rect,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TextureFilter {
    Nearest,
    Linear,
}

impl TextureId {
    pub fn new(name: u32, _: TextureTarget) -> TextureId {
        TextureId {
            name: name,
        }
    }

    pub fn invalid() -> TextureId {
        TextureId {
            name: 0,
        }
    }

    pub fn invalid_a8() -> TextureId {
        TextureId {
            name: 1,
        }
    }

    pub fn is_valid(&self) -> bool { !(*self == TextureId::invalid() || *self == TextureId::invalid_a8()) }
    pub fn is_dummy(&self) -> bool { self.name == DUMMY_A8_ID || self.name == DUMMY_RGBA8_ID }
    pub fn is_dither(&self) -> bool { self.name == DITHER_ID }
    pub fn is_skipable(&self) -> bool { !(self.is_valid()) || self.is_dummy() || self.is_dither() }
}

#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Copy, Clone)]
pub struct TextureId {
    name: u32,
}

#[derive(Debug)]
pub struct TextureData {
    pub id: TextureId,
    pub data: Vec<u8>,
    pub stride: usize,
    pub pitch: usize,
}

#[derive(Clone, Debug)]
pub enum ShaderError {
    Compilation(String, String), // name, error mssage
    Link(String), // error message
}

pub struct Device {
    pub device: device_gl::Device,
    pub factory: device_gl::Factory,
    pub encoder: gfx::Encoder<R,CB>,
    pub textures: HashMap<TextureId, TextureData>,
    pub color0: Texture<R, Srgba8>,
    pub color1: Texture<R, Rgba8>,
    pub color2: Texture<R, Rgba8>,
    pub dither: Texture<R, A8>,
    pub cache_a8: Texture<R, Rgba8>,
    pub cache_rgba8: Texture<R, Rgba8>,
    pub data16: Texture<R, Rgba32F>,
    pub data32: Texture<R, Rgba32F>,
    pub data64: Texture<R, Rgba32F>,
    pub data128: Texture<R, Rgba32F>,
    pub gradient_data: Texture<R, Srgba8>,
    pub layers: Texture<R, Rgba32F>,
    pub prim_geo: Texture<R, Rgba32F>,
    pub render_tasks: Texture<R, Rgba32F>,
    pub resource_rects: Texture<R, Rgba32F>,
    pub split_geo: Texture<R, Rgba32F>,
    pub max_texture_size: u32,
    pub main_color: gfx::handle::RenderTargetView<R, ColorFormat>,
    pub main_depth: gfx::handle::DepthStencilView<R, DepthFormat>,
    pub vertex_buffer: gfx::handle::Buffer<R, Position>,
    pub slice: gfx::Slice<R>,
    pub rtv: gfx::handle::RenderTargetView<R, ColorFormat>,
    pub dsv: gfx::handle::DepthStencilView<R, DepthFormat>,
}

impl Device {
    pub fn new(window: &glutin::Window) -> Device {
        let (device, mut factory, main_color, main_depth) = init_existing::<ColorFormat, DepthFormat>(window);
        /*println!("Vendor: {:?}", device.get_info().platform_name.vendor);
        println!("Renderer: {:?}", device.get_info().platform_name.renderer);
        println!("Version: {:?}", device.get_info().version);
        println!("Shading Language: {:?}", device.get_info().shading_language);*/
        let encoder = factory.create_command_buffer().into();
        let max_texture_size = MAX_VERTEX_TEXTURE_WIDTH as u32;

        let (x0, y0, x1, y1) = (0.0, 0.0, 1.0, 1.0);
        let quad_indices: &[u16] = &[ 0, 1, 2, 2, 1, 3 ];
        let quad_vertices = [
            Position::new([x0, y0]),
            Position::new([x1, y0]),
            Position::new([x0, y1]),
            Position::new([x1, y1]),
        ];

        let (vertex_buffer, mut slice) = factory.create_vertex_buffer_with_slice(&quad_vertices, quad_indices);
        slice.instances = Some((MAX_INSTANCE_COUNT as u32, 0));

        let (w, h, _, _) = main_color.get_dimensions();
        let texture_size = [std::cmp::max(MAX_VERTEX_TEXTURE_WIDTH, h as usize), std::cmp::max(MAX_VERTEX_TEXTURE_WIDTH, w as usize)];
        let color0 = Texture::empty(&mut factory, texture_size, TextureFilter::Linear, TextureTarget::Default).unwrap();
        let color1 = Texture::empty(&mut factory, texture_size, TextureFilter::Linear, TextureTarget::Default).unwrap();
        let color2 = Texture::empty(&mut factory, texture_size, TextureFilter::Linear, TextureTarget::Default).unwrap();
        let dither = Texture::empty(&mut factory, [8, 8], TextureFilter::Nearest, TextureTarget::Default).unwrap();
        let cache_a8 = Texture::empty(&mut factory, texture_size, TextureFilter::Linear, TextureTarget::Array).unwrap();
        let cache_rgba8 = Texture::empty(&mut factory, texture_size, TextureFilter::Linear, TextureTarget::Array).unwrap();

        // TODO define some maximum boundaries for texture height
        let data16_tex = Texture::empty(&mut factory, [MAX_VERTEX_TEXTURE_WIDTH, TEXTURE_HEIGTH * 4], TextureFilter::Nearest, TextureTarget::Default).unwrap();
        let data32_tex = Texture::empty(&mut factory, [MAX_VERTEX_TEXTURE_WIDTH, TEXTURE_HEIGTH], TextureFilter::Nearest, TextureTarget::Default).unwrap();
        let data64_tex = Texture::empty(&mut factory, [MAX_VERTEX_TEXTURE_WIDTH, TEXTURE_HEIGTH], TextureFilter::Nearest, TextureTarget::Default).unwrap();
        let data128_tex = Texture::empty(&mut factory, [MAX_VERTEX_TEXTURE_WIDTH, TEXTURE_HEIGTH * 4], TextureFilter::Nearest, TextureTarget::Default).unwrap();
        let gradient_data = Texture::empty(&mut factory, [2* GRADIENT_DATA_SIZE, TEXTURE_HEIGTH * 10], TextureFilter::Nearest, TextureTarget::Default).unwrap();
        let layers_tex = Texture::empty(&mut factory, [LAYER_TEXTURE_WIDTH, 64], TextureFilter::Nearest, TextureTarget::Default).unwrap();
        let prim_geo_tex = Texture::empty(&mut factory, [MAX_VERTEX_TEXTURE_WIDTH, TEXTURE_HEIGTH], TextureFilter::Nearest, TextureTarget::Default).unwrap();
        let render_tasks_tex = Texture::empty(&mut factory, [RENDER_TASK_TEXTURE_WIDTH, TEXTURE_HEIGTH], TextureFilter::Nearest, TextureTarget::Default).unwrap();
        let resource_rects = Texture::empty(&mut factory, [MAX_VERTEX_TEXTURE_WIDTH, TEXTURE_HEIGTH * 2], TextureFilter::Nearest, TextureTarget::Default).unwrap();
        let split_geo_tex = Texture::empty(&mut factory, [MAX_VERTEX_TEXTURE_WIDTH, TEXTURE_HEIGTH * 2], TextureFilter::Nearest, TextureTarget::Default).unwrap();

        let mut textures = HashMap::new();
        let (w, h) = color0.get_size();
        let invalid_id = TextureId::invalid();
        textures.insert(invalid_id, TextureData { id: invalid_id, data: vec![0u8; w * h * RGBA_STRIDE], stride: RGBA_STRIDE, pitch: w * RGBA_STRIDE });
        let invalid_a8_id = TextureId::invalid_a8();
        textures.insert(invalid_a8_id, TextureData { id: invalid_a8_id, data: vec![0u8; w * h * A_STRIDE], stride: A_STRIDE, pitch: w * A_STRIDE });
        let dummy_rgba8_id = TextureId { name: DUMMY_RGBA8_ID };
        textures.insert(dummy_rgba8_id, TextureData { id: dummy_rgba8_id, data: vec![0u8; w * h * RGBA_STRIDE], stride: RGBA_STRIDE, pitch: w * RGBA_STRIDE });
        let dummy_a8_id = TextureId { name: DUMMY_A8_ID };
        textures.insert(dummy_a8_id, TextureData { id: dummy_a8_id, data: vec![0u8; w * h * RGBA_STRIDE], stride: RGBA_STRIDE, pitch: w * RGBA_STRIDE });
        let dither_id = TextureId { name: DITHER_ID };
        let dither_matrix = vec![
            00, 48, 12, 60, 03, 51, 15, 63,
            32, 16, 44, 28, 35, 19, 47, 31,
            08, 56, 04, 52, 11, 59, 07, 55,
            40, 24, 36, 20, 43, 27, 39, 23,
            02, 50, 14, 62, 01, 49, 13, 61,
            34, 18, 46, 30, 33, 17, 45, 29,
            10, 58, 06, 54, 09, 57, 05, 53,
            42, 26, 38, 22, 41, 25, 37, 21
        ];
        textures.insert(dither_id, TextureData { id: dither_id, data: dither_matrix, stride: A_STRIDE, pitch: 8 * RGBA_STRIDE });

        let (rtv, dsv) = create_main_targets::<ColorFormat, DepthFormat>(&mut factory, w as u16, h as u16).unwrap();

        Device {
            device: device,
            factory: factory,
            encoder: encoder,
            textures: textures,
            color0: color0,
            color1: color1,
            color2: color2,
            dither: dither,
            cache_a8: cache_a8,
            cache_rgba8: cache_rgba8,
            data16: data16_tex,
            data32: data32_tex,
            data64: data64_tex,
            data128: data128_tex,
            gradient_data: gradient_data,
            layers: layers_tex,
            prim_geo: prim_geo_tex,
            render_tasks: render_tasks_tex,
            resource_rects: resource_rects,
            split_geo: split_geo_tex,
            max_texture_size: max_texture_size,
            main_color: main_color,
            main_depth: main_depth,
            vertex_buffer: vertex_buffer,
            slice: slice,
            rtv: rtv,
            dsv: dsv,
        }
    }

    pub fn read_pixels(&mut self, rect: DeviceUintRect, output: &mut [u8]) {
        // TODO add bgra flag
        self.encoder.flush(&mut self.device);
        let tex = self.main_color.raw().get_texture();
        let tex_info = tex.get_info().to_raw_image_info(gfx::format::ChannelType::Unorm, 0);
        let (w, h, _, _) = self.main_color.get_dimensions();
        let buf = self.factory.create_buffer::<u8>(w as usize * h as usize * RGBA_STRIDE,
                                                   gfx::buffer::Role::Vertex,
                                                   gfx::memory::Usage::Download,
                                                   gfx::TRANSFER_DST).unwrap();
        self.encoder.copy_texture_to_buffer_raw(tex, None, tex_info, buf.raw(), 0).unwrap();
        self.encoder.flush(&mut self.device);
        {
            let reader = self.factory.read_mapping(&buf).unwrap();
            let data = &*reader;
            for j in 0..rect.size.height as usize {
                for i in 0..rect.size.width as usize {
                    let offset = i * RGBA_STRIDE + j * rect.size.width as usize * RGBA_STRIDE;
                    let src = &data[(j + rect.origin.y as usize) * w as usize * RGBA_STRIDE + (i + rect.origin.x as usize) * RGBA_STRIDE ..];
                    output[offset + 0] = src[0];
                    output[offset + 1] = src[1];
                    output[offset + 2] = src[2];
                    output[offset + 3] = src[3];
                }
            }
        }
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    fn generate_texture_id(&mut self) -> TextureId {
        use rand::OsRng;

        let mut rng = OsRng::new().unwrap();
        let mut texture_id = TextureId::invalid();
        while self.textures.contains_key(&texture_id) {
            texture_id.name = rng.gen_range(FIRST_UNRESERVED_ID, u32::max_value());
        }
        texture_id
    }

    pub fn create_texture_id(&mut self,
                             _target: TextureTarget,
                             format: ImageFormat) -> TextureId {
        let (w, h) = self.color0.get_size();
        let texture_id = self.generate_texture_id();
        let stride = match format {
            ImageFormat::A8 => A_STRIDE,
            ImageFormat::BGRA8 => RGBA_STRIDE,
            ImageFormat::RG8 => RG_STRIDE,
            ImageFormat::RGB8 => RGB_STRIDE,
            _ => unimplemented!(),
        };
        let texture_data = vec![0u8; w * h * stride];
        assert!(!self.textures.contains_key(&texture_id));
        self.textures.insert(texture_id, TextureData {id: texture_id, data: texture_data, stride: stride, pitch: w * stride });
        texture_id
    }

    pub fn init_texture(&mut self,
                        texture_id: TextureId,
                        _width: u32,
                        _height: u32,
                        format: ImageFormat,
                        _filter: TextureFilter,
                        _mode: RenderTargetMode,
                        pixels: Option<&[u8]>) {
        //println!("init_texture texture_id={:?} _width={:?} _height={:?} format={:?} _filter={:?} _mode={:?}", texture_id, _width, _height, format, _filter, _mode);
        let texture = self.textures.get_mut(&texture_id).expect("Didn't find texture!");
        //println!("init_texture texture.stride={:?} texture.data.len={:?}", texture.stride, texture.data.len());
        let stride = match format {
            ImageFormat::A8 => A_STRIDE,
            ImageFormat::BGRA8 => RGBA_STRIDE,
            ImageFormat::RG8 => RG_STRIDE,
            ImageFormat::RGB8 => RGB_STRIDE,
            _ => unimplemented!(),
        };
        if stride != texture.stride {
            texture.stride = stride;
            texture.data.clear();
        }
        let actual_pixels = match pixels {
            Some(data) => data.to_vec(),
            None => {
                let (w, h) = self.color0.get_size();
                vec![0u8; w * h * texture.stride]
            },
        };
        assert!(texture.data.len() == actual_pixels.len());
        mem::replace(&mut texture.data, actual_pixels);
    }

    pub fn update_texture(&mut self,
                          texture_id: TextureId,
                          x0: u32,
                          y0: u32,
                          width: u32,
                          height: u32,
                          stride: Option<u32>,
                          data: &[u8]) {
        //println!("update {:?} x0={:?} y0={:?} width={:?} height={:?} stride={:?} size={:?}", texture_id, x0, y0, width, height, stride, data.len());
        let texture = self.textures.get_mut(&texture_id).expect("Didn't find texture!");
        assert!(!(texture.data.len() < data.len()));
        //println!("\ttex.stride={:?} tex.pitch={:?} tex.size={:?}", texture.stride, texture.pitch, texture.data.len());
        //let (w, _) = self.color0.get_size();
        let row_length = match stride {
            Some(value) => value as usize / texture.stride,
            None => width as usize,
        };
        // Take the stride into account for all rows, except the last one.
        let data_pitch = row_length * texture.stride;
        let len = data_pitch * (height - 1) as usize + width as usize * texture.stride;
        //let len = std::cmp::min(texture.stride * row_length * (height - 1) as usize
        //                        + width as usize * texture.stride, data.len());
        //println!("\tcomputed len={:?}, height*pitch={:?} data_size={:?}", len, height as usize * data_pitch, data.len());
        //println!("\ttarget_width={:?}, row_length={:?}", w, row_length);
        let data = &data[0 .. len];
        Device::update_texture_data(texture, x0 as usize, y0 as usize, width as usize, height as usize, data_pitch, data);
    }

    pub fn resize_texture(&mut self,
                          texture_id: TextureId,
                          new_width: u32,
                          new_height: u32,
                          format: ImageFormat,
                          _filter: TextureFilter,
                          _mode: RenderTargetMode) {
        let texture = self.textures.get_mut(&texture_id).expect("Didn't find texture!");
        let stride = match format {
            ImageFormat::A8 => A_STRIDE,
            ImageFormat::BGRA8 => RGBA_STRIDE,
            ImageFormat::RG8 => RG_STRIDE,
            ImageFormat::RGB8 => RGB_STRIDE,
            _ => unimplemented!(),
        };
        if stride != texture.stride {
            texture.stride = stride;
            texture.data.clear();
        }
        let new_len = new_width as usize * new_height as usize * texture.stride;
        texture.data.resize(new_len, 0u8);
    }

    pub fn deinit_texture(&mut self, texture_id: TextureId) {
        let texture = self.textures.get_mut(&texture_id).expect("Didn't find texture!");
        let (w, h) = self.color0.get_size();
        let data = vec![0u8; w * h * texture.stride];
        assert!(texture.data.len() == data.len());
        mem::replace(&mut texture.data, data.to_vec());
    }

    fn update_texture_data(texture: &mut TextureData,
        x_offset: usize, y_offset: usize,
        width: usize, height: usize,
        data_pitch: usize, new_data: &[u8]
    ) {
        assert_eq!(data_pitch * (height-1) + width * texture.stride, new_data.len());
        for j in 0..height {
            if texture.stride != RGBA_STRIDE {
                //fast path
                let dst_offset = x_offset*texture.stride + (j+y_offset)*texture.pitch;
                let src = &new_data[j * data_pitch ..];
                texture.data[dst_offset .. dst_offset + width*texture.stride].copy_from_slice(&src[.. width*texture.stride]);
                continue;
            }
            for i in 0..width {
                let offset = (i + x_offset)*texture.stride + (j+y_offset)*texture.pitch;
                let src = &new_data[j * data_pitch + i * texture.stride ..];
                assert!(offset + 3 < texture.data.len()); // optimization
                // convert from BGRA
                texture.data[offset + 0] = src[2];
                texture.data[offset + 1] = src[1];
                texture.data[offset + 2] = src[0];
                texture.data[offset + 3] = src[3];
            }
        }
    }

    pub fn bind_texture(&mut self,
                        sampler: TextureSampler,
                        texture_id: TextureId) {
        if texture_id.is_skipable() {
            return;
        }
        let texture = match self.textures.get(&texture_id) {
            Some(data) => data,
            None => {
                println!("Didn't find texture! {}", texture_id.name);
                return;
            }
        };
        println!("bind_texture {:?} {:?} {:?} {:?}", texture_id, sampler, texture.stride, texture.data.len());
        println!("texture.data={:?}", &texture.data[0..64]);
        match sampler {
            TextureSampler::Color0 => Device::update_texture_surface(&mut self.encoder, &self.color0, texture.data.as_slice(), RGBA_STRIDE),
            TextureSampler::Color1 => Device::update_texture_surface(&mut self.encoder, &self.color1, texture.data.as_slice(), RGBA_STRIDE),
            TextureSampler::Color2 => Device::update_texture_surface(&mut self.encoder, &self.color2, texture.data.as_slice(), RGBA_STRIDE),
            TextureSampler::CacheA8 => Device::update_texture_surface(&mut self.encoder, &self.cache_a8, texture.data.as_slice(), RGBA_STRIDE),
            TextureSampler::CacheRGBA8 => Device::update_texture_surface(&mut self.encoder, &self.cache_rgba8, texture.data.as_slice(), RGBA_STRIDE),
            TextureSampler::Dither => Device::update_texture_surface(&mut self.encoder, &self.dither, texture.data.as_slice(), A_STRIDE),
            _ => println!("There are only 5 samplers supported. {:?}", sampler),
        }
    }

    pub fn bind_yuv_texture(&mut self,
                            sampler: TextureSampler,
                            texture_id: TextureId) {
        let texture = match self.textures.get_mut(&texture_id) {
            Some(data) => data,
            None => {
                println!("Didn't find texture! {}", texture_id.name);
                return;
            }
        };

        let (w, h) = self.color0.get_size();
        let new_data = Device::convert_data_to_rgba8(w, h, texture.data.as_slice(), texture.stride);
        match sampler {
            TextureSampler::Color0 => Device::update_texture_surface(&mut self.encoder, &self.color0, new_data.as_slice(), RGBA_STRIDE),
            TextureSampler::Color1 => Device::update_texture_surface(&mut self.encoder, &self.color1, new_data.as_slice(), RGBA_STRIDE),
            TextureSampler::Color2 => Device::update_texture_surface(&mut self.encoder, &self.color2, new_data.as_slice(), RGBA_STRIDE),
            _ => println!("The yuv image shouldn't use this sampler: {:?}", sampler),
        }
    }

    fn convert_data_to_rgba8(width: usize, height: usize, data: &[u8], orig_stride: usize) -> Vec<u8> {
        let mut new_data = vec![0u8; width * height * RGBA_STRIDE];
        for s in 0..orig_stride {
            for h in 0..height {
                for w in 0..width {
                    new_data[s+(w*RGBA_STRIDE)+h*width*RGBA_STRIDE] = data[s+(w*orig_stride)+h*width*orig_stride];
                }
            }
        }
        return new_data;
    }

    pub fn update_sampler_f32(&mut self,
                              sampler: TextureSampler,
                              data: &[f32]) {
        match sampler {
            TextureSampler::Layers => Device::update_texture_surface(&mut self.encoder, &self.layers, data, RGBA_STRIDE),
            TextureSampler::RenderTasks => Device::update_texture_surface(&mut self.encoder, &self.render_tasks, data, RGBA_STRIDE),
            //TextureSampler::Geometry => Device::update_texture_surface(&mut self.encoder, &self.prim_geo, data, RGBA_STRIDE),
            //TextureSampler::SplitGeometry => Device::update_texture_surface(&mut self.encoder, &self.split_geo, data, RGBA_STRIDE),
            //TextureSampler::Data16 => Device::update_texture_surface(&mut self.encoder, &self.data16, data, RGBA_STRIDE),
            //TextureSampler::Data32 => Device::update_texture_surface(&mut self.encoder, &self.data32, data, RGBA_STRIDE),
            //TextureSampler::Data64 => Device::update_texture_surface(&mut self.encoder, &self.data64, data, RGBA_STRIDE),
            //TextureSampler::Data128 => Device::update_texture_surface(&mut self.encoder, &self.data128, data, RGBA_STRIDE),
            //TextureSampler::ResourceRects => Device::update_texture_surface(&mut self.encoder, &self.resource_rects, data, RGBA_STRIDE),
            _ => println!("{:?} sampler is not supported", sampler),
        }
    }

    pub fn update_sampler_u8(&mut self,
                             sampler: TextureSampler,
                             data: &[u8]) {
        match sampler {
            //TextureSampler::Gradients => Device::update_texture_surface(&mut self.encoder, &self.gradient_data, data, RGBA_STRIDE),
            _ => println!("{:?} sampler is not supported", sampler),
        }
    }

    pub fn clear_target(&mut self, color: Option<[f32; 4]>, depth: Option<f32>) {
        if let Some(color) = color {
            self.encoder.clear(&self.main_color,
                               //Srgba gamma correction
                               [color[0].powf(2.2),
                                color[1].powf(2.2),
                                color[2].powf(2.2),
                                color[3].powf(2.2)]);
        }

        if let Some(depth) = depth {
            self.encoder.clear_depth(&self.main_depth, depth);
        }
    }

    pub fn flush(&mut self) {
        self.encoder.flush(&mut self.device);
    }

    pub fn create_program(&mut self, vert_src: &[u8], frag_src: &[u8]) -> Program {
        let upload = self.factory.create_upload_buffer(MAX_INSTANCE_COUNT).unwrap();
        {
            let mut writer = self.factory.write_mapping(&upload).unwrap();
            for i in 0..MAX_INSTANCE_COUNT {
                writer[i] = PrimitiveInstances::new();
            }
        }

        let instances = self.factory.create_buffer(MAX_INSTANCE_COUNT,
                                                   gfx::buffer::Role::Vertex,
                                                   gfx::memory::Usage::Data,
                                                   gfx::TRANSFER_DST).unwrap();

        let data = primitive::Data {
            transform: [[0f32; 4]; 4],
            device_pixel_ratio: DEVICE_PIXEL_RATIO,
            vbuf: self.vertex_buffer.clone(),
            ibuf: instances,
            color0: (self.color0.clone().view, self.color0.clone().sampler),
            color1: (self.color1.clone().view, self.color1.clone().sampler),
            color2: (self.color2.clone().view, self.color2.clone().sampler),
            dither: (self.dither.clone().view, self.dither.clone().sampler),
            cache_a8: (self.cache_a8.clone().view, self.cache_a8.clone().sampler),
            cache_rgba8: (self.cache_rgba8.clone().view, self.cache_rgba8.clone().sampler),
            data16: (self.data16.clone().view, self.data16.clone().sampler),
            data32: (self.data32.clone().view, self.data32.clone().sampler),
            data64: (self.data64.clone().view, self.data64.clone().sampler),
            data128: (self.data128.clone().view, self.data128.clone().sampler),
            gradients: (self.gradient_data.clone().view, self.gradient_data.clone().sampler),
            layers: (self.layers.clone().view, self.layers.clone().sampler),
            prim_geometry: (self.prim_geo.clone().view, self.prim_geo.clone().sampler),
            render_tasks: (self.render_tasks.clone().view, self.render_tasks.clone().sampler),
            resource_rects: (self.resource_rects.clone().view, self.resource_rects.clone().sampler),
            split_geometry: (self.split_geo.clone().view, self.split_geo.clone().sampler),
            out_color: self.main_color.raw().clone(),
            out_depth: self.main_depth.clone(),
            blend_value: [0.0, 0.0, 0.0, 0.0]
        };
        let psos = self.create_prim_psos(vert_src, frag_src);
        Program::new(data, psos, self.slice.clone(), upload)
    }

    pub fn draw(&mut self,
                program: &mut Program,
                proj: &Matrix4D<f32>,
                instances: &[PrimitiveInstance],
                blendmode: &BlendMode,
                enable_depth_write: bool) {
        program.data.transform = proj.to_row_arrays();

        {
            let mut writer = self.factory.write_mapping(&program.upload).unwrap();
            for (i, inst) in instances.iter().enumerate() {
                writer[i].update(inst);
            }
        }

        {
            program.slice.instances = Some((instances.len() as u32, 0));
        }

        if let &BlendMode::Subpixel(ref color) = blendmode {
            program.data.blend_value = [color.r, color.g, color.b, color.a];
        }

        self.encoder.copy_buffer(&program.upload, &program.data.ibuf, 0, 0, program.upload.len()).unwrap();
        self.encoder.draw(&program.slice, &program.get_pso(blendmode, enable_depth_write), &program.data);
        //self.encoder.flush(&mut self.device);
    }

    pub fn draw_clip(&mut self,
                     program: &mut ClipProgram,
                     proj: &Matrix4D<f32>,
                     instances: &[CacheClipInstance],
                     blendmode: &BlendMode/*,
                     texture_id: TextureId*/) {
        program.data.transform = proj.to_row_arrays();
        let (w, h) = self.color0.get_size();

        //let (tex, resource, target) = self.factory.create_render_target::<ColorFormat>(w as u16, h as u16).unwrap();
        //let (tex2, resource2, target2) = self.factory.create_depth_stencil::<DepthFormat>(w as u16, h as u16).unwrap();
        //let mut text_data = self.textures.get_mut(&texture_id).unwrap();
        //println!("text_data.id={:?} text_data.stride={:?}, text_data.pitch={:?}", text_data.id, text_data.stride, text_data.pitch);
        //println!("text_data.data={:?}", &text_data.data[0..64]);

        //let new_data = Device::convert_data_to_rgba8(w, h, text_data.data.as_slice(), text_data.stride);


        //program.data.out_color = self.rtv.raw().clone();
        //program.data.out_depth = self.dsv.clone();
        {
            let mut writer = self.factory.write_mapping(&program.upload).unwrap();
            for (i, inst) in instances.iter().enumerate() {
                writer[i].update(inst);
            }
        }

        {
            program.slice.instances = Some((instances.len() as u32, 0));
        }

        self.encoder.flush(&mut self.device);
        self.encoder.copy_buffer(&program.upload, &program.data.ibuf, 0, 0, program.upload.len()).unwrap();
        self.encoder.flush(&mut self.device);
        self.encoder.draw(&program.slice, &program.get_pso(blendmode), &program.data);
        self.encoder.flush(&mut self.device);

        /*{
            let mut output = vec![255u8; w as usize * h as usize * RGBA_STRIDE];
            let res_tex = program.data.out_color.get_texture();
            println!("res_tex={:?}", res_tex);
            let res_tex_info = res_tex.get_info().to_raw_image_info(gfx::format::ChannelType::Unorm, 0);
            println!("res_tex_info={:?}", res_tex_info);
            let (w, h, _, _) = program.data.out_color.get_dimensions();
            let buf = self.factory.create_buffer::<u8>(w as usize * h as usize * RGBA_STRIDE,
                                                       gfx::buffer::Role::Vertex,
                                                       gfx::memory::Usage::Download,
                                                       gfx::TRANSFER_DST).unwrap();
            self.encoder.copy_texture_to_buffer_raw(res_tex, None, res_tex_info, buf.raw(), 0).unwrap();
            self.encoder.flush(&mut self.device);
            {
                let reader = self.factory.read_mapping(&buf).unwrap();
                for x in 0..w as usize * h as usize * RGBA_STRIDE {
                    if (reader[x] != 0) {
                        println!("reader[{:?}]={:?}", x, reader[x]);
                    }
                }
                let data = &*reader;
                for j in 0..h as usize {
                    for i in 0..w as usize {
                        let offset = i * RGBA_STRIDE + j * w as usize * RGBA_STRIDE;
                        let src = &data[j * w as usize * RGBA_STRIDE + i * RGBA_STRIDE ..];
                        output[offset + 0] = src[0];
                        output[offset + 1] = src[1];
                        output[offset + 2] = src[2];
                        output[offset + 3] = src[3];
                        for x in 0..4 {
                            if (src[x] != 0) {
                                println!("src[{:?}, {:?},{:?}]={:?}", x, i, j, src[x]);
                            }
                        }
                    }
                }
            }
            //println!("output={:?}", output);
            //println!("text_data={:?}", text_data.data);
            let mut i = 0;
            for x in 0..text_data.data.len() {
                text_data.data[x] = output[i];
                i = i+RGBA_STRIDE;
            }
        }*/
        //program.data.out_color = self.main_color.raw().clone();
        //program.data.out_depth = self.main_depth.clone();
        //println!("text_data.data={:?}", &text_data.data[0..255]);
    }

    pub fn update_texture_surface<S, F, T>(encoder: &mut gfx::Encoder<R,CB>,
                                           texture: &Texture<R, F>,
                                           memory: &[T],
                                           stride: usize)
    where S: SurfaceTyped + TextureSurface,
          S::DataType: Copy,
          F: Formatted<Surface=S>,
          F::Channel: TextureChannel,
          T: Default + Clone + gfx::traits::Pod {
        let (width, height) = texture.get_size();
        let resized_data = Device::convert_sampler_data(memory, (width * height * stride) as usize);
        let img_info = gfx::texture::ImageInfoCommon {
            xoffset: 0,
            yoffset: 0,
            zoffset: 0,
            width: width as u16,
            height: height as u16,
            depth: 0,
            format: (),
            mipmap: 0,
        };

        let data = gfx::memory::cast_slice(resized_data.as_slice());
        encoder.update_texture::<_, F>(&texture.surface, None, img_info, data).unwrap();
    }

    fn convert_sampler_data<T: Default + Clone>(data: &[T], max_size: usize) -> Vec<T> {
        let mut data = data.to_vec();
        let len = data.len();
        if len < max_size {
            data.extend_from_slice(&vec![T::default(); max_size - len]);
        }
        assert!(data.len() == max_size);
        data
    }
}

fn create_main_targets<T: gfx::format::RenderFormat + gfx::format::TextureFormat,
                       S: gfx::format::DepthFormat + gfx::format::TextureFormat>
                       (factory: &mut device_gl::Factory, width: gfx::texture::Size, height: gfx::texture::Size)
                        -> Result<(gfx::handle::RenderTargetView<R, T>,
                                   gfx::handle::DepthStencilView<R, S>), gfx::CombinedError>
{
    let kind = gfx::texture::Kind::D2(width, height, gfx::texture::AaMode::Single);
    let levels = 1;
    let rtv_cty = <T::Channel as gfx::format::ChannelTyped>::get_channel_type();
    let tex_rtv = try!(factory.create_texture(kind, levels, gfx::memory::RENDER_TARGET | gfx::memory::TRANSFER_SRC, gfx::memory::Usage::Data, Some(rtv_cty)));
    let rtv = try!(factory.view_texture_as_render_target(&tex_rtv, 0, None));

    let dsv_cty = <S::Channel as gfx::format::ChannelTyped>::get_channel_type();
    let tex_dsv = try!(factory.create_texture(kind, 1, gfx::memory::DEPTH_STENCIL, gfx::memory::Usage::Data, Some(dsv_cty)));
    let dsv = try!(factory.view_texture_as_depth_stencil_trivial(&tex_dsv));
    Ok((rtv, dsv))
}

pub fn init_existing<Cf, Df>(window: &glutin::Window) ->
                            (device_gl::Device,device_gl::Factory,
                             gfx::handle::RenderTargetView<R, Cf>, gfx::handle::DepthStencilView<R, Df>)
where Cf: gfx::format::RenderFormat, Df: gfx::format::DepthFormat,
{
    unsafe { window.make_current().unwrap() };
    let (device, factory) = device_gl::create(|s|
        window.get_proc_address(s) as *const std::os::raw::c_void);

    let (width, height) = window.get_inner_size().unwrap();
    let aa = window.get_pixel_format().multisampling.unwrap_or(0) as gfx::texture::NumSamples;
    let dim = ((width as f32 * window.hidpi_factor()) as gfx::texture::Size,
               (height as f32 * window.hidpi_factor()) as gfx::texture::Size,
               1,
               aa.into());

    let (color_view, ds_view) = device_gl::create_main_targets_raw(dim, Cf::get_format().0, Df::get_format().0);
    (device, factory, Typed::new(color_view), Typed::new(ds_view))
}
