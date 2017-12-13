/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::ColorF;
use debug_render::{DebugFontVertex, DebugColorVertex};
use device::{Device, DEVICE_PIXEL_RATIO, MAX_INSTANCE_COUNT, TextureId};
use euclid::{Matrix4D, Transform3D};
use gfx;
use gfx::state::{Blend, BlendChannel, BlendValue, Comparison, Depth, Equation, Factor};
use gfx::memory::Typed;
use gfx::Factory;
use gfx::traits::FactoryExt;
use gfx::format::DepthStencil as DepthFormat;
use backend::Resources as R;
use gfx::format::Format;
use gpu_types::{BlurInstance, ClipMaskInstance, PrimitiveInstance};
use renderer::{BlendMode, RendererError, TextureSampler};

const ALPHA: Blend = Blend {
    color: BlendChannel {
        equation: Equation::Add,
        source: Factor::ZeroPlus(BlendValue::SourceAlpha),
        destination: Factor::OneMinus(BlendValue::SourceAlpha),
    },
    alpha: BlendChannel {
        equation: Equation::Add,
        source: Factor::One,
        destination: Factor::OneMinus(BlendValue::SourceAlpha),
    },
};

const PREM_ALPHA: Blend = Blend {
    color: BlendChannel {
        equation: Equation::Add,
        source: Factor::One,
        destination: Factor::OneMinus(BlendValue::SourceAlpha),
    },
    alpha: BlendChannel {
        equation: Equation::Add,
        source: Factor::One,
        destination: Factor::OneMinus(BlendValue::SourceAlpha),
    },
};

const SUBPIXEL_PASS0: Blend = Blend {
    color: BlendChannel {
        equation: Equation::Add,
        source: Factor::Zero,
        destination: Factor::OneMinus(BlendValue::SourceColor),
    },
    alpha: BlendChannel {
        equation: Equation::Add,
        source: Factor::Zero,
        destination: Factor::OneMinus(BlendValue::SourceColor),
    },
};

const SUBPIXEL_PASS1: Blend = Blend {
    color: BlendChannel {
        equation: Equation::Add,
        source: Factor::One,
        destination: Factor::One,
    },
    alpha: BlendChannel {
        equation: Equation::Add,
        source: Factor::One,
        destination: Factor::One,
    },
};

const MULTIPLY: Blend = Blend {
    color: BlendChannel {
        equation: Equation::Add,
        source: Factor::Zero,
        destination: Factor::ZeroPlus(BlendValue::SourceColor),
    },
    alpha: BlendChannel {
        equation: Equation::Add,
        source: Factor::Zero,
        destination: Factor::ZeroPlus(BlendValue::SourceAlpha),
    },
};

const MAX: Blend = Blend {
    color: BlendChannel {
        equation: Equation::Max,
        source: Factor::One,
        destination: Factor::One,
    },
    alpha: BlendChannel {
        equation: Equation::Add,
        source: Factor::One,
        destination: Factor::One,
    },
};

gfx_defines! {
    vertex Position {
        pos: [f32; 3] = "aPosition",
    }

    vertex PrimitiveInstances {
            data0: [i32; 4] = "aDataA",
            data1: [i32; 4] = "aDataB",
    }

    vertex BlurInstances {
        task_address: i32 = "aBlurRenderTaskAddress",
        src_task_address: i32 = "aBlurSourceTaskAddress",
        blur_direction: i32 = "aBlurDirection",
        region: [f32; 4] = "aBlurRegion",
    }

    vertex ClipMaskInstances {
        render_task_address: i32 = "aClipRenderTaskAddress",
        layer_address: i32 = "aClipLayerAddress",
        segment: i32 = "aClipSegment",
        data_resource_address: [i32; 4] = "aClipDataResourceAddress",
    }

    constant Locals {
        transform: [[f32; 4]; 4] = "uTransform",
        device_pixel_ratio: f32 = "uDevicePixelRatio",
        mode: i32 = "uMode",
    }

    pipeline primitive {
        locals: gfx::ConstantBuffer<Locals> = "Locals",
        mode: gfx::Global<i32> = "uMode",
        transform: gfx::Global<[[f32; 4]; 4]> = "uTransform",
        device_pixel_ratio: gfx::Global<f32> = "uDevicePixelRatio",
        vbuf: gfx::VertexBuffer<Position> = (),
        ibuf: gfx::InstanceBuffer<PrimitiveInstances> = (),

        color0: gfx::TextureSampler<[f32; 4]> = "sColor0",
        color1: gfx::TextureSampler<[f32; 4]> = "sColor1",
        color2: gfx::TextureSampler<[f32; 4]> = "sColor2",
        cache_a8: gfx::TextureSampler<[f32; 4]> = "sCacheA8",
        cache_rgba8: gfx::TextureSampler<[f32; 4]> = "sCacheRGBA8",
        shared_cache_a8: gfx::TextureSampler<[f32; 4]> = "sSharedCacheA8",

        resource_cache: gfx::TextureSampler<[f32; 4]> = "sResourceCache",
        layers: gfx::TextureSampler<[f32; 4]> = "sLayers",
        render_tasks: gfx::TextureSampler<[f32; 4]> = "sRenderTasks",
        dither: gfx::TextureSampler<f32> = "sDither",

        out_color: gfx::RawRenderTarget = ("Target0",
                                           Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                                           gfx::state::MASK_ALL,
                                           None),
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
        blend_value: gfx::BlendRef = (),
    }

    pipeline brush {
        locals: gfx::ConstantBuffer<Locals> = "Locals",
        mode: gfx::Global<i32> = "uMode",
        transform: gfx::Global<[[f32; 4]; 4]> = "uTransform",
        device_pixel_ratio: gfx::Global<f32> = "uDevicePixelRatio",
        vbuf: gfx::VertexBuffer<Position> = (),
        ibuf: gfx::InstanceBuffer<PrimitiveInstances> = (),

        color0: gfx::TextureSampler<[f32; 4]> = "sColor0",
        color1: gfx::TextureSampler<[f32; 4]> = "sColor1",
        color2: gfx::TextureSampler<[f32; 4]> = "sColor2",
        cache_a8: gfx::TextureSampler<[f32; 4]> = "sCacheA8",
        cache_rgba8: gfx::TextureSampler<[f32; 4]> = "sCacheRGBA8",
        shared_cache_a8: gfx::TextureSampler<[f32; 4]> = "sSharedCacheA8",

        resource_cache: gfx::TextureSampler<[f32; 4]> = "sResourceCache",
        layers: gfx::TextureSampler<[f32; 4]> = "sLayers",
        render_tasks: gfx::TextureSampler<[f32; 4]> = "sRenderTasks",

        out_color: gfx::RawRenderTarget = ("Target0",
                                           Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                                           gfx::state::MASK_ALL,
                                           None),
    }

    pipeline blur {
        locals: gfx::ConstantBuffer<Locals> = "Locals",
        mode: gfx::Global<i32> = "uMode",
        transform: gfx::Global<[[f32; 4]; 4]> = "uTransform",
        device_pixel_ratio: gfx::Global<f32> = "uDevicePixelRatio",
        vbuf: gfx::VertexBuffer<Position> = (),
        ibuf: gfx::InstanceBuffer<BlurInstances> = (),

        cache_a8: gfx::TextureSampler<[f32; 4]> = "sCacheA8",
        cache_rgba8: gfx::TextureSampler<[f32; 4]> = "sCacheRGBA8",

        resource_cache: gfx::TextureSampler<[f32; 4]> = "sResourceCache",
        layers: gfx::TextureSampler<[f32; 4]> = "sLayers",
        render_tasks: gfx::TextureSampler<[f32; 4]> = "sRenderTasks",

        out_color: gfx::RawRenderTarget = ("Target0",
                                           Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                                           gfx::state::MASK_ALL,
                                           None),
    }

    pipeline clip {
        locals: gfx::ConstantBuffer<Locals> = "Locals",
        mode: gfx::Global<i32> = "uMode",
        transform: gfx::Global<[[f32; 4]; 4]> = "uTransform",
        device_pixel_ratio: gfx::Global<f32> = "uDevicePixelRatio",
        vbuf: gfx::VertexBuffer<Position> = (),
        ibuf: gfx::InstanceBuffer<ClipMaskInstances> = (),

        color0: gfx::TextureSampler<[f32; 4]> = "sColor0",
        color1: gfx::TextureSampler<[f32; 4]> = "sColor1",
        color2: gfx::TextureSampler<[f32; 4]> = "sColor2",
        cache_a8: gfx::TextureSampler<[f32; 4]> = "sCacheA8",
        cache_rgba8: gfx::TextureSampler<[f32; 4]> = "sCacheRGBA8",
        shared_cache_a8: gfx::TextureSampler<[f32; 4]> = "sSharedCacheA8",

        resource_cache: gfx::TextureSampler<[f32; 4]> = "sResourceCache",
        layers: gfx::TextureSampler<[f32; 4]> = "sLayers",
        render_tasks: gfx::TextureSampler<[f32; 4]> = "sRenderTasks",
        dither: gfx::TextureSampler<f32> = "sDither",

        out_color: gfx::RawRenderTarget = ("Target0",
                                           Format(gfx::format::SurfaceType::R8, gfx::format::ChannelType::Unorm),
                                           gfx::state::MASK_ALL,
                                           None),
    }
    
    vertex DebugColorVertices {
        pos: [f32; 2] = "aPosition",
        color: [f32; 4] = "aColor",
    }

    pipeline debug_color {
        locals: gfx::ConstantBuffer<Locals> = "Locals",
        mode: gfx::Global<i32> = "uMode",
        transform: gfx::Global<[[f32; 4]; 4]> = "uTransform",
        device_pixel_ratio: gfx::Global<f32> = "uDevicePixelRatio",
        vbuf: gfx::VertexBuffer<DebugColorVertices> = (),
        out_color: gfx::RawRenderTarget = ("Target0",
                                           Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                                           gfx::state::MASK_ALL,
                                           None),
    }

    vertex DebugFontVertices {
        pos: [f32; 2] = "aPosition",
        color: [f32; 4] = "aColor",
        tex_coord: [f32; 2] = "aColorTexCoord",
    }

    pipeline debug_font {
        locals: gfx::ConstantBuffer<Locals> = "Locals",
        mode: gfx::Global<i32> = "uMode",
        transform: gfx::Global<[[f32; 4]; 4]> = "uTransform",
        device_pixel_ratio: gfx::Global<f32> = "uDevicePixelRatio",
        vbuf: gfx::VertexBuffer<DebugFontVertices> = (),
        color0: gfx::TextureSampler<[f32; 4]> = "sColor0",
        out_color: gfx::RawRenderTarget = ("Target0",
                                           Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                                           gfx::state::MASK_ALL,
                                           Some(ALPHA)),
    }
}

type PrimPSO = gfx::PipelineState<R, primitive::Meta>;
type BrushPSO = gfx::PipelineState<R, brush::Meta>;
type ClipPSO = gfx::PipelineState<R, clip::Meta>;
type BlurPSO = gfx::PipelineState<R, blur::Meta>;
type DebugColorPSO = gfx::PipelineState<R, debug_color::Meta>;
type DebugFontPSO = gfx::PipelineState<R, debug_font::Meta>;

impl Position {
    pub fn new(p: [f32; 2]) -> Position {
        Position {
            pos: [p[0], p[1], 0.0],
        }
    }
}

impl PrimitiveInstances {
    pub fn new() -> PrimitiveInstances {
        PrimitiveInstances {
            data0: [0; 4],
            data1: [0; 4],
        }
    }

    pub fn update(&mut self, instance: &PrimitiveInstance) {
        self.data0 = [instance.data[0], instance.data[1], instance.data[2], instance.data[3]];
        self.data1 = [instance.data[4], instance.data[5], instance.data[6], instance.data[7]];
    }
}


impl DebugColorVertices {
    pub fn new(pos: [f32; 2], color: [f32; 4]) -> DebugColorVertices {
        DebugColorVertices {
            pos: pos,
            color: color,
        }
    }
}

impl DebugFontVertices {
    pub fn new(pos: [f32; 2], color: [f32; 4], tex_coord: [f32; 2]) -> DebugFontVertices {
        DebugFontVertices {
            pos: pos,
            color: color,
            tex_coord: tex_coord,
        }
    }
}

impl BlurInstances {
    pub fn new() -> BlurInstances {
        BlurInstances {
            task_address: 0,
            src_task_address: 0,
            blur_direction: 0,
            region: [0.0; 4],
        }
    }

    pub fn update(&mut self, blur_instance: &BlurInstance) {
        self.task_address = blur_instance.task_address.0 as i32;
        self.src_task_address = blur_instance.src_task_address.0 as i32;
        self.blur_direction = blur_instance.blur_direction as i32;
        // TODO check if this is the correct way to pass arguments from rect to the shader
        self.region = [blur_instance.region.origin.x,
                       blur_instance.region.origin.y,
                       blur_instance.region.size.width,
                       blur_instance.region.size.height];
    }
}

impl ClipMaskInstances {
    pub fn new() -> ClipMaskInstances {
        ClipMaskInstances {
            render_task_address: 0,
            layer_address: 0,
            segment: 0,
            data_resource_address: [0; 4],
        }
    }

    pub fn update(&mut self, instance: &ClipMaskInstance) {
        self.render_task_address = instance.render_task_address.0 as i32;
        self.layer_address = instance.layer_address.0 as i32;
        self.segment = instance.segment;
        self.data_resource_address[0] = instance.clip_data_address.u as i32;
        self.data_resource_address[1] = instance.clip_data_address.v as i32;
        self.data_resource_address[2] = instance.resource_address.u as i32;
        self.data_resource_address[3] = instance.resource_address.v as i32;
    }
}

/*fn update_texture_srv_and_sampler(program_texture_id: &mut TextureId,
                                  device_texture_id: TextureId,
                                  device: &mut Device,
                                  tex_sampler: &mut (ShaderResourceView<R, [f32; 4]>, Sampler<R>)) {
    if *program_texture_id != device_texture_id {
        *program_texture_id = device_texture_id;
        if device_texture_id.is_skipable() {
            tex_sampler.0 = device.dummy_tex.srv.clone();
        } else {
            let tex = device.textures.get(&device_texture_id).unwrap();
            let sampler = match tex.filter {
                TextureFilter::Nearest => device.sampler.clone().0,
                TextureFilter::Linear => device.sampler.clone().1,
            };
            *tex_sampler = (tex.srv.clone(), sampler);
        }
    }
}*/

#[derive(Debug)]
pub struct Program {
    pub data: primitive::Data<R>,
    pub pso: (PrimPSO, PrimPSO),
    pub pso_alpha: (PrimPSO, PrimPSO),
    pub pso_prem_alpha: (PrimPSO, PrimPSO),
    pub slice: gfx::Slice<R>,
    pub upload: (gfx::handle::Buffer<R, PrimitiveInstances>, usize),
}

impl Program {
    pub fn new(data: primitive::Data<R>,
           psos: (PrimPSO, PrimPSO, PrimPSO, PrimPSO, PrimPSO, PrimPSO),
           slice: gfx::Slice<R>,
           upload: gfx::handle::Buffer<R, PrimitiveInstances>)
           -> Program {
        Program {
            data: data,
            pso: (psos.0, psos.1),
            pso_alpha: (psos.2, psos.3),
            pso_prem_alpha: (psos.4, psos.5),
            slice: slice,
            upload: (upload, 0),
        }
    }

    pub fn get_pso(&self, blend: &BlendMode, depth_write: bool) -> &PrimPSO {
        match *blend {
            BlendMode::Alpha => if depth_write { &self.pso_alpha.0 } else { &self.pso_alpha.1 },
            BlendMode::PremultipliedAlpha => if depth_write { &self.pso_prem_alpha.0 } else { &self.pso_prem_alpha.1 },
            _ => if depth_write { &self.pso.0 } else { &self.pso.1 },
        }
    }

    pub fn reset_upload_offset(&mut self) {
        self.upload.1 = 0;
    }

    pub fn bind(
        &mut self,
        device: &mut Device,
        projection: &Transform3D<f32>,
        instances: &[PrimitiveInstance],
        render_target: Option<(&TextureId, i32)>,
        renderer_errors: &mut Vec<RendererError>,
        mode: i32,
    ) {
        self.data.transform = projection.to_row_arrays();
        self.data.mode = mode;
        let locals = Locals {
            transform: self.data.transform,
            device_pixel_ratio: self.data.device_pixel_ratio,
            mode: self.data.mode,
        };
        device.encoder.update_buffer(&self.data.locals, &[locals], 0).unwrap();

        {
            let mut writer = device.factory.write_mapping(&self.upload.0).unwrap();
            for (i, inst) in instances.iter().enumerate() {
                writer[i + self.upload.1].update(inst);
            }
        }

        {
            self.slice.instances = Some((instances.len() as u32, 0));
        }
        device.encoder.copy_buffer(&self.upload.0, &self.data.ibuf, self.upload.1, 0, instances.len()).unwrap();
        self.upload.1 += instances.len();

        self.data.color0 = device.get_texture_srv_and_sampler(TextureSampler::Color0);
        self.data.color1 = device.get_texture_srv_and_sampler(TextureSampler::Color1);
        self.data.color2 = device.get_texture_srv_and_sampler(TextureSampler::Color2);
        self.data.cache_a8.0 = device.get_texture_srv_and_sampler(TextureSampler::CacheA8).0;
        self.data.cache_rgba8.0 = device.get_texture_srv_and_sampler(TextureSampler::CacheRGBA8).0;
        self.data.shared_cache_a8.0 = device.get_texture_srv_and_sampler(TextureSampler::SharedCacheA8).0;

        if render_target.is_some() {
            let tex = device.cache_rgba8_textures.get(&render_target.unwrap().0).unwrap();
            self.data.out_color = tex.rtv.raw().clone();
            self.data.out_depth = tex.dsv.clone();
        } else {
            self.data.out_color = device.main_color.raw().clone();
            self.data.out_depth = device.main_depth.clone();
        }
    }

    pub fn draw(&mut self, device: &mut Device, blendmode: &BlendMode, enable_depth_write: bool)
    {
        device.encoder.draw(&self.slice, &self.get_pso(blendmode, enable_depth_write), &self.data);
    }
}

#[derive(Debug)]
pub struct BrushProgram {
    pub data: brush::Data<R>,
    pub pso: BrushPSO,
    pub pso_alpha: BrushPSO,
    pub pso_prem_alpha: BrushPSO,
    pub slice: gfx::Slice<R>,
    pub upload: (gfx::handle::Buffer<R, PrimitiveInstances>, usize),
}

impl BrushProgram {
    pub fn new(
        data: brush::Data<R>,
        psos: (BrushPSO, BrushPSO, BrushPSO),
        slice: gfx::Slice<R>,
        upload: gfx::handle::Buffer<R, PrimitiveInstances>,
    ) -> BrushProgram {
        BrushProgram {
            data: data,
            pso: psos.0,
            pso_alpha: psos.1,
            pso_prem_alpha: psos.2,
            slice: slice,
            upload: (upload, 0),
        }
    }

    pub fn get_pso(&self, blend: &BlendMode) -> &BrushPSO {
        match *blend {
            BlendMode::Alpha => &self.pso_alpha,
            BlendMode::PremultipliedAlpha => &self.pso_prem_alpha,
            _ => &self.pso,
        }
    }

    pub fn reset_upload_offset(&mut self) {
        self.upload.1 = 0;
    }

    pub fn bind(
        &mut self,
        device: &mut Device,
        projection: &Transform3D<f32>,
        instances: &[PrimitiveInstance],
        render_target: Option<(&TextureId, i32)>,
        renderer_errors: &mut Vec<RendererError>,
        mode: i32,
    ) {
        self.data.transform = projection.to_row_arrays();
        self.data.mode = mode;
        let locals = Locals {
            transform: self.data.transform,
            device_pixel_ratio: self.data.device_pixel_ratio,
            mode: self.data.mode,
        };
        device.encoder.update_buffer(&self.data.locals, &[locals], 0).unwrap();

        {
            let mut writer = device.factory.write_mapping(&self.upload.0).unwrap();
            for (i, inst) in instances.iter().enumerate() {
                writer[i + self.upload.1].update(inst);
            }
        }

        {
            self.slice.instances = Some((instances.len() as u32, 0));
        }
        device.encoder.copy_buffer(&self.upload.0, &self.data.ibuf, self.upload.1, 0, instances.len()).unwrap();
        self.upload.1 += instances.len();

        self.data.color0 = device.get_texture_srv_and_sampler(TextureSampler::Color0);
        self.data.color1 = device.get_texture_srv_and_sampler(TextureSampler::Color1);
        self.data.color2 = device.get_texture_srv_and_sampler(TextureSampler::Color2);
        self.data.cache_a8.0 = device.get_texture_srv_and_sampler(TextureSampler::CacheA8).0;
        self.data.cache_rgba8.0 = device.get_texture_srv_and_sampler(TextureSampler::CacheRGBA8).0;
        self.data.shared_cache_a8.0 = device.get_texture_srv_and_sampler(TextureSampler::SharedCacheA8).0;

        if render_target.is_some() {
            let tex = device.cache_rgba8_textures
                    .get(&render_target.unwrap().0)
                    .unwrap_or(device.cache_a8_textures.get(&render_target.unwrap().0)
                    .unwrap_or(device.dummy_cache_a8()));
            self.data.out_color = tex.rtv.raw().clone();
        } else {
            self.data.out_color = device.main_color.raw().clone();
        }
    }

    pub fn draw(&mut self, device: &mut Device, blendmode: &BlendMode)
    {
        device.encoder.draw(&self.slice, &self.get_pso(blendmode), &self.data);
    }
}

#[derive(Debug)]
pub struct TextProgram {
    pub data: primitive::Data<R>,
    // Depth write is always disabled for the text drawing pass,
    // so we don't need duplicate the PSO-s here
    pub pso_prem_alpha: PrimPSO,
    pub pso_subpixel_pass0: PrimPSO,
    pub pso_subpixel_pass1: PrimPSO,
    pub slice: gfx::Slice<R>,
    pub upload: (gfx::handle::Buffer<R, PrimitiveInstances>, usize),
}

impl TextProgram {
    pub fn new(data: primitive::Data<R>,
           psos: (PrimPSO, PrimPSO, PrimPSO),
           slice: gfx::Slice<R>,
           upload: gfx::handle::Buffer<R, PrimitiveInstances>)
           -> TextProgram {
        TextProgram {
            data: data,
            pso_prem_alpha: psos.0,
            pso_subpixel_pass0: psos.1,
            pso_subpixel_pass1: psos.2,
            slice: slice,
            upload: (upload, 0),
        }
    }

    pub fn get_pso(&self, blend: &BlendMode, pass_number: Option<i32>) -> &PrimPSO {
        match *blend {
            BlendMode::PremultipliedAlpha => &self.pso_prem_alpha,
            BlendMode::Subpixel => match pass_number {
                Some(0) => &self.pso_subpixel_pass0,
                Some(1) => &self.pso_subpixel_pass1,
                _ => unreachable!(),
            }
            _ => unreachable!(),
        }
    }

    pub fn reset_upload_offset(&mut self) {
        self.upload.1 = 0;
    }

    pub fn bind(
        &mut self,
        device: &mut Device,
        projection: &Transform3D<f32>,
        instances: &[PrimitiveInstance],
        render_target: Option<(&TextureId, i32)>,
        renderer_errors: &mut Vec<RendererError>,
        mode: i32,
    ) {
        self.data.transform = projection.to_row_arrays();
        self.data.mode = mode;
        let locals = Locals {
            transform: self.data.transform,
            device_pixel_ratio: self.data.device_pixel_ratio,
            mode: self.data.mode,
        };
        device.encoder.update_buffer(&self.data.locals, &[locals], 0).unwrap();

        {
            let mut writer = device.factory.write_mapping(&self.upload.0).unwrap();
            for (i, inst) in instances.iter().enumerate() {
                writer[i + self.upload.1].update(inst);
            }
        }

        {
            self.slice.instances = Some((instances.len() as u32, 0));
        }
        device.encoder.copy_buffer(&self.upload.0, &self.data.ibuf, self.upload.1, 0, instances.len()).unwrap();
        self.upload.1 += instances.len();

        self.data.color0 = device.get_texture_srv_and_sampler(TextureSampler::Color0);
        self.data.color1 = device.get_texture_srv_and_sampler(TextureSampler::Color1);
        self.data.color2 = device.get_texture_srv_and_sampler(TextureSampler::Color2);
        self.data.cache_a8.0 = device.get_texture_srv_and_sampler(TextureSampler::CacheA8).0;
        self.data.cache_rgba8.0 = device.get_texture_srv_and_sampler(TextureSampler::CacheRGBA8).0;
        self.data.shared_cache_a8.0 = device.get_texture_srv_and_sampler(TextureSampler::SharedCacheA8).0;

        if render_target.is_some() {
            let tex = device.cache_rgba8_textures.get(&render_target.unwrap().0).unwrap();
            self.data.out_color = tex.rtv.raw().clone();
            self.data.out_depth = tex.dsv.clone();
        } else {
            self.data.out_color = device.main_color.raw().clone();
            self.data.out_depth = device.main_depth.clone();
        }
    }

    pub fn draw(&mut self, device: &mut Device, blendmode: &BlendMode, pass_number: Option<i32>)
    {
        device.encoder.draw(&self.slice, &self.get_pso(blendmode, pass_number), &self.data);
    }
}

pub struct BlurProgram {
    pub data: blur::Data<R>,
    pub pso: BlurPSO,
    pub slice: gfx::Slice<R>,
    pub upload: (gfx::handle::Buffer<R, BlurInstances>, usize),
}

impl BlurProgram {
    pub fn new(data: blur::Data<R>,
           pso: BlurPSO,
           slice: gfx::Slice<R>,
           upload: gfx::handle::Buffer<R, BlurInstances>)
           -> BlurProgram {
        BlurProgram {
            data: data,
            pso: pso,
            slice: slice,
            upload: (upload, 0),
        }
    }

    pub fn reset_upload_offset(&mut self) {
        self.upload.1 = 0;
    }

    pub fn bind(
        &mut self,
        device: &mut Device,
        projection: &Transform3D<f32>,
        instances: &[BlurInstance],
        render_target: Option<(&TextureId, i32)>,
        renderer_errors: &mut Vec<RendererError>,
        mode: i32,
    ) {
        self.data.transform = projection.to_row_arrays();
        self.data.mode = mode;
        let locals = Locals {
            transform: self.data.transform,
            device_pixel_ratio: self.data.device_pixel_ratio,
            mode: self.data.mode,
        };
        device.encoder.update_buffer(&self.data.locals, &[locals], 0).unwrap();

        {
            let mut writer = device.factory.write_mapping(&self.upload.0).unwrap();
            for (i, inst) in instances.iter().enumerate() {
                writer[i + self.upload.1].update(inst);
            }
        }

        {
            self.slice.instances = Some((instances.len() as u32, 0));
        }
        device.encoder.copy_buffer(&self.upload.0, &self.data.ibuf, self.upload.1, 0, instances.len()).unwrap();
        self.upload.1 += instances.len();

        self.data.cache_rgba8.0 = device.get_texture_srv_and_sampler(TextureSampler::CacheRGBA8).0;
        self.data.cache_a8.0 = device.get_texture_srv_and_sampler(TextureSampler::CacheA8).0;

        if render_target.is_some() {
            let tex = device.cache_rgba8_textures
                    .get(&render_target.unwrap().0)
                    .unwrap_or(device.cache_a8_textures.get(&render_target.unwrap().0)
                    .unwrap_or(device.dummy_cache_a8()));
            self.data.out_color = tex.rtv.raw().clone();
            //self.data.out_depth = tex.dsv.clone();
        } else {
            self.data.out_color = device.main_color.raw().clone();
            //self.data.out_depth = device.main_depth.clone();
        }
    }

    pub fn draw(&mut self, device: &mut Device)
    {
        device.encoder.draw(&self.slice, &self.pso, &self.data);
    }
}

pub struct ClipProgram {
    pub data: clip::Data<R>,
    pub pso: ClipPSO,
    pub pso_multiply: ClipPSO,
    pub pso_max: ClipPSO,
    pub slice: gfx::Slice<R>,
    pub upload: (gfx::handle::Buffer<R, ClipMaskInstances>, usize),
}

impl ClipProgram {
    pub fn new(data: clip::Data<R>,
           psos: (ClipPSO, ClipPSO, ClipPSO),
           slice: gfx::Slice<R>,
           upload: gfx::handle::Buffer<R, ClipMaskInstances>)
           -> ClipProgram {
        ClipProgram {
            data: data,
            pso: psos.0,
            pso_multiply: psos.1,
            pso_max: psos.2,
            slice: slice,
            upload: (upload, 0),
        }
    }

    pub fn get_pso(&self, blend: &BlendMode) -> &ClipPSO {
        match *blend {
            BlendMode::Multiply => &self.pso_multiply,
            BlendMode::Max => &self.pso_max,
            _ => &self.pso,
        }
    }

    pub fn reset_upload_offset(&mut self) {
        self.upload.1 = 0;
    }

    pub fn bind(
        &mut self,
        device: &mut Device,
        projection: &Transform3D<f32>,
        instances: &[ClipMaskInstance],
        render_target: &TextureId,
        renderer_errors: &mut Vec<RendererError>,
        mode: i32,
    ) {
        self.data.transform = projection.to_row_arrays();
        self.data.mode = mode;
        let locals = Locals {
            transform: self.data.transform,
            device_pixel_ratio: self.data.device_pixel_ratio,
            mode: self.data.mode,
        };
        device.encoder.update_buffer(&self.data.locals, &[locals], 0).unwrap();

        {
            let mut writer = device.factory.write_mapping(&self.upload.0).unwrap();
            for (i, inst) in instances.iter().enumerate() {
                writer[i + self.upload.1].update(inst);
            }
        }

        {
            self.slice.instances = Some((instances.len() as u32, 0));
        }
        device.encoder.copy_buffer(&self.upload.0, &self.data.ibuf, self.upload.1, 0, instances.len()).unwrap();
        self.upload.1 += instances.len();
        self.data.out_color = device.cache_a8_textures.get(&render_target).unwrap().rtv.raw().clone();
        self.data.color0 = device.get_texture_srv_and_sampler(TextureSampler::Color0);
        self.data.color1 = device.get_texture_srv_and_sampler(TextureSampler::Color1);
        self.data.color2 = device.get_texture_srv_and_sampler(TextureSampler::Color2);
        self.data.cache_a8.0 = device.get_texture_srv_and_sampler(TextureSampler::CacheA8).0;
        self.data.cache_rgba8.0 = device.get_texture_srv_and_sampler(TextureSampler::CacheRGBA8).0;
        self.data.shared_cache_a8.0 = device.get_texture_srv_and_sampler(TextureSampler::SharedCacheA8).0;
    }

    pub fn draw(&mut self, device: &mut Device, blendmode: &BlendMode)
    {
        device.encoder.draw(&self.slice, &self.get_pso(blendmode), &self.data);
    }
}

pub struct DebugColorProgram {
    pub data: debug_color::Data<R>,
    pub pso: DebugColorPSO,
    pub slice: gfx::Slice<R>,
}

impl DebugColorProgram {
    pub fn new(data: debug_color::Data<R>, pso: DebugColorPSO, slice: gfx::Slice<R>) -> DebugColorProgram {
        DebugColorProgram {
            data,
            pso,
            slice
        }
    }

    pub fn bind(
        &mut self,
        device: &mut Device,
        projection: &Transform3D<f32>,
        indices: &[u32],
        vertices: &[DebugColorVertex],
        render_target: Option<(&TextureId, i32)>,
    ) {
        self.data.transform = projection.to_row_arrays();
        let quad_vertices: Vec<DebugColorVertices> = vertices.iter().map(|v| DebugColorVertices::new([v.x, v.y], ColorF::from(v.color).to_array())).collect();
        let (vbuf, slice) = device.factory.create_vertex_buffer_with_slice(&quad_vertices, indices);

        {
            self.data.vbuf = vbuf;
            self.slice = slice;
        }

        let locals = Locals {
            transform: self.data.transform,
            device_pixel_ratio: self.data.device_pixel_ratio,
            mode: self.data.mode,
        };
        device.encoder.update_buffer(&self.data.locals, &[locals], 0).unwrap();
        if render_target.is_some() {
            let tex = device.cache_rgba8_textures
                    .get(&render_target.unwrap().0)
                    .unwrap_or(device.cache_a8_textures.get(&render_target.unwrap().0)
                    .unwrap_or(device.dummy_cache_a8()));
            self.data.out_color = tex.rtv.raw().clone();
        } else {
            self.data.out_color = device.main_color.raw().clone();
        }
    }

    pub fn draw(&mut self, device: &mut Device) {
        device.encoder.draw(&self.slice, &self.pso, &self.data);
    }
}

pub struct DebugFontProgram {
    pub data: debug_font::Data<R>,
    pub pso: DebugFontPSO,
    pub slice: gfx::Slice<R>,
}

impl DebugFontProgram {
    pub fn new(data: debug_font::Data<R>, pso: DebugFontPSO, slice: gfx::Slice<R>) -> DebugFontProgram {
        DebugFontProgram {
            data,
            pso,
            slice
        }
    }

    pub fn bind(
        &mut self,
        device: &mut Device,
        projection: &Transform3D<f32>,
        indices: &[u32],
        vertices: &[DebugFontVertex],
    ) {
        self.data.transform = projection.to_row_arrays();
        let quad_vertices: Vec<DebugFontVertices> = vertices.iter().map(|v| DebugFontVertices::new([v.x, v.y], ColorF::from(v.color).to_array(), [v.u, v.v])).collect();
        let (vbuf, slice) = device.factory.create_vertex_buffer_with_slice(&quad_vertices, indices);

        {
            self.data.vbuf = vbuf;
            self.slice = slice;
        }

        self.data.color0 = device.get_texture_srv_and_sampler(TextureSampler::Color0);

        let locals = Locals {
            transform: self.data.transform,
            device_pixel_ratio: self.data.device_pixel_ratio,
            mode: self.data.mode,
        };
        device.encoder.update_buffer(&self.data.locals, &[locals], 0).unwrap();
    }

    pub fn draw(&mut self, device: &mut Device) {
        device.encoder.draw(&self.slice, &self.pso, &self.data);
    }
}

impl Device {
    pub fn create_prim_psos(&mut self, vert_src: &[u8],frag_src: &[u8]) -> (PrimPSO, PrimPSO, PrimPSO, PrimPSO, PrimPSO, PrimPSO) {
        let pso_depth_write = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::new()
        ).unwrap();

        let pso = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_depth: gfx::preset::depth::LESS_EQUAL_TEST,
                .. primitive::new()
            }
        ).unwrap();

        let pso_alpha_depth_write = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("Target0",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                            gfx::state::MASK_ALL,
                            Some(ALPHA)),
                .. primitive::new()
            }
        ).unwrap();

        let pso_alpha = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("Target0",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                            gfx::state::MASK_ALL,
                            Some(ALPHA)),
                out_depth: gfx::preset::depth::LESS_EQUAL_TEST,
                .. primitive::new()
            }
        ).unwrap();

        let pso_prem_alpha_depth_write = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("Target0",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                            gfx::state::MASK_ALL,
                            Some(PREM_ALPHA)),
                .. primitive::new()
            }
        ).unwrap();

        let pso_prem_alpha = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("Target0",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                            gfx::state::MASK_ALL,
                            Some(PREM_ALPHA)),
            out_depth: gfx::preset::depth::LESS_EQUAL_TEST,
                .. primitive::new()
            }
        ).unwrap();


        (pso_depth_write, pso, pso_alpha_depth_write, pso_alpha, pso_prem_alpha_depth_write, pso_prem_alpha)
    }

    pub fn create_brush_psos(&mut self, vert_src: &[u8],frag_src: &[u8]) -> (BrushPSO, BrushPSO, BrushPSO) {
        let pso = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            brush::new()
        ).unwrap();

        let pso_alpha = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            brush::Init {
                out_color: ("Target0",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                            gfx::state::MASK_ALL,
                            Some(ALPHA)),
                .. brush::new()
            }
        ).unwrap();

        let pso_prem_alpha = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            brush::Init {
                out_color: ("Target0",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                            gfx::state::MASK_ALL,
                            Some(PREM_ALPHA)),
                .. brush::new()
            }
        ).unwrap();


        (pso, pso_alpha, pso_prem_alpha)
    }

    pub fn create_text_psos(&mut self, vert_src: &[u8],frag_src: &[u8]) -> (PrimPSO, PrimPSO, PrimPSO) {
        let pso_prem_alpha = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("Target0",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                            gfx::state::MASK_ALL,
                            Some(PREM_ALPHA)),
            out_depth: gfx::preset::depth::LESS_EQUAL_TEST,
                .. primitive::new()
            }
        ).unwrap();

        let pso_subpixel_pass0 = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("Target0",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                            gfx::state::MASK_ALL,
                            Some(SUBPIXEL_PASS0)),
                out_depth: gfx::preset::depth::LESS_EQUAL_TEST,
                .. primitive::new()
            }
        ).unwrap();

        let pso_subpixel_pass1 = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("Target0",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                            gfx::state::MASK_ALL,
                            Some(SUBPIXEL_PASS1)),
                out_depth: gfx::preset::depth::LESS_EQUAL_TEST,
                .. primitive::new()
            }
        ).unwrap();

        (pso_prem_alpha, pso_subpixel_pass0, pso_subpixel_pass1)
    }

    pub fn create_clip_psos(&mut self, vert_src: &[u8],frag_src: &[u8]) -> (ClipPSO, ClipPSO, ClipPSO) {
        let pso = self.factory.create_pipeline_simple(vert_src, frag_src, clip::new()).unwrap();

        let pso_multiply = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            clip::Init {
                out_color: ("Target0",
                            Format(gfx::format::SurfaceType::R8, gfx::format::ChannelType::Unorm),
                            gfx::state::MASK_ALL,
                            Some(MULTIPLY)),
                .. clip::new()
            }
        ).unwrap();

        let pso_max = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            clip::Init {
                out_color: ("Target0",
                            Format(gfx::format::SurfaceType::R8, gfx::format::ChannelType::Unorm),
                            gfx::state::MASK_ALL,
                            Some(MAX)),
                .. clip::new()
            }
        ).unwrap();
        (pso, pso_multiply, pso_max)
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
            locals: self.factory.create_constant_buffer(1),
            transform: [[0f32; 4]; 4],
            device_pixel_ratio: DEVICE_PIXEL_RATIO,
            mode: 0,
            vbuf: self.vertex_buffer.clone(),
            ibuf: instances,
            color0: (self.dummy_image().srv.clone(), self.sampler.0.clone()),
            color1: (self.dummy_image().srv.clone(), self.sampler.0.clone()),
            color2: (self.dummy_image().srv.clone(), self.sampler.0.clone()),
            cache_a8: (self.dummy_cache_a8().srv.clone(), self.sampler.0.clone()),
            cache_rgba8: (self.dummy_cache_rgba8().srv.clone(), self.sampler.1.clone()),
            shared_cache_a8: (self.dummy_cache_a8().srv.clone(), self.sampler.0.clone()),
            resource_cache: (self.resource_cache.srv.clone(), self.sampler.0.clone()),
            layers: (self.layers.srv.clone(), self.sampler.0.clone()),
            render_tasks: (self.render_tasks.srv.clone(), self.sampler.0.clone()),
            dither: (self.dither().srv.clone(), self.sampler.0.clone()),
            out_color: self.main_color.raw().clone(),
            out_depth: self.main_depth.clone(),
            blend_value: [0.0, 0.0, 0.0, 0.0]
        };
        let psos = self.create_prim_psos(vert_src, frag_src);
        Program::new(data, psos, self.slice.clone(), upload)
    }

    pub fn create_brush_program(&mut self, vert_src: &[u8], frag_src: &[u8]) -> BrushProgram {
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

        let data = brush::Data {
            locals: self.factory.create_constant_buffer(1),
            transform: [[0f32; 4]; 4],
            device_pixel_ratio: DEVICE_PIXEL_RATIO,
            mode: 0,
            vbuf: self.vertex_buffer.clone(),
            ibuf: instances,
            color0: (self.dummy_image().srv.clone(), self.sampler.0.clone()),
            color1: (self.dummy_image().srv.clone(), self.sampler.0.clone()),
            color2: (self.dummy_image().srv.clone(), self.sampler.0.clone()),
            cache_a8: (self.dummy_cache_a8().srv.clone(), self.sampler.0.clone()),
            cache_rgba8: (self.dummy_cache_rgba8().srv.clone(), self.sampler.1.clone()),
            shared_cache_a8: (self.dummy_cache_a8().srv.clone(), self.sampler.0.clone()),
            resource_cache: (self.resource_cache.srv.clone(), self.sampler.0.clone()),
            layers: (self.layers.srv.clone(), self.sampler.0.clone()),
            render_tasks: (self.render_tasks.srv.clone(), self.sampler.0.clone()),
            out_color: self.main_color.raw().clone(),
        };
        let psos = self.create_brush_psos(vert_src, frag_src);
        BrushProgram::new(data, psos, self.slice.clone(), upload)
    }

    pub fn create_text_program(&mut self, vert_src: &[u8], frag_src: &[u8]) -> TextProgram {
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
            locals: self.factory.create_constant_buffer(1),
            transform: [[0f32; 4]; 4],
            device_pixel_ratio: DEVICE_PIXEL_RATIO,
            mode: 0,
            vbuf: self.vertex_buffer.clone(),
            ibuf: instances,
            color0: (self.dummy_image().srv.clone(), self.sampler.0.clone()),
            color1: (self.dummy_image().srv.clone(), self.sampler.0.clone()),
            color2: (self.dummy_image().srv.clone(), self.sampler.0.clone()),
            cache_a8: (self.dummy_cache_a8().srv.clone(), self.sampler.0.clone()),
            cache_rgba8: (self.dummy_cache_rgba8().srv.clone(), self.sampler.1.clone()),
            shared_cache_a8: (self.dummy_cache_a8().srv.clone(), self.sampler.0.clone()),
            resource_cache: (self.resource_cache.srv.clone(), self.sampler.0.clone()),
            layers: (self.layers.srv.clone(), self.sampler.0.clone()),
            render_tasks: (self.render_tasks.srv.clone(), self.sampler.0.clone()),
            dither: (self.dither().srv.clone(), self.sampler.0.clone()),
            out_color: self.main_color.raw().clone(),
            out_depth: self.main_depth.clone(),
            blend_value: [0.0, 0.0, 0.0, 0.0]
        };
        let psos = self.create_text_psos(vert_src, frag_src);
        TextProgram::new(data, psos, self.slice.clone(), upload)
    }

    pub fn create_blur_program(&mut self, vert_src: &[u8], frag_src: &[u8]) -> BlurProgram {
        let upload = self.factory.create_upload_buffer(MAX_INSTANCE_COUNT).unwrap();
        {
            let mut writer = self.factory.write_mapping(&upload).unwrap();
            for i in 0..MAX_INSTANCE_COUNT {
                writer[i] = BlurInstances::new();
            }
        }

        let blur_instances = self.factory.create_buffer(MAX_INSTANCE_COUNT,
                                                        gfx::buffer::Role::Vertex,
                                                        gfx::memory::Usage::Data,
                                                        gfx::TRANSFER_DST).unwrap();

        let data = blur::Data {
            locals: self.factory.create_constant_buffer(1),
            transform: [[0f32; 4]; 4],
            device_pixel_ratio: DEVICE_PIXEL_RATIO,
            mode: 0,
            vbuf: self.vertex_buffer.clone(),
            ibuf: blur_instances,
            cache_a8: (self.dummy_cache_a8().srv.clone(), self.sampler.1.clone()),
            cache_rgba8: (self.dummy_cache_rgba8().srv.clone(), self.sampler.1.clone()),
            resource_cache: (self.resource_cache.srv.clone(), self.sampler.0.clone()),
            layers: (self.layers.srv.clone(), self.sampler.0.clone()),
            render_tasks: (self.render_tasks.srv.clone(), self.sampler.0.clone()),
            out_color: self.main_color.raw().clone(),
        };
        let pso = self.factory.create_pipeline_simple(vert_src, frag_src, blur::new()).unwrap();
        BlurProgram {data: data, pso: pso, slice: self.slice.clone(), upload:(upload,0)}
    }

    pub fn create_clip_program(&mut self, vert_src: &[u8], frag_src: &[u8]) -> ClipProgram {
        let upload = self.factory.create_upload_buffer(MAX_INSTANCE_COUNT).unwrap();
        {
            let mut writer = self.factory.write_mapping(&upload).unwrap();
            for i in 0..MAX_INSTANCE_COUNT {
                writer[i] = ClipMaskInstances::new();
            }
        }

        let cache_instances = self.factory.create_buffer(MAX_INSTANCE_COUNT,
                                                         gfx::buffer::Role::Vertex,
                                                         gfx::memory::Usage::Data,
                                                         gfx::TRANSFER_DST).unwrap();

        let data = clip::Data {
            locals: self.factory.create_constant_buffer(1),
            transform: [[0f32; 4]; 4],
            device_pixel_ratio: DEVICE_PIXEL_RATIO,
            mode: 0,
            vbuf: self.vertex_buffer.clone(),
            ibuf: cache_instances,
            color0: (self.dummy_image().srv.clone(), self.sampler.0.clone()),
            color1: (self.dummy_image().srv.clone(), self.sampler.0.clone()),
            color2: (self.dummy_image().srv.clone(), self.sampler.0.clone()),
            dither: (self.dither().srv.clone(), self.sampler.0.clone()),
            cache_a8: (self.dummy_cache_a8().srv.clone(), self.sampler.0.clone()),
            cache_rgba8: (self.dummy_cache_rgba8().srv.clone(), self.sampler.1.clone()),
            shared_cache_a8: (self.dummy_cache_a8().srv.clone(), self.sampler.0.clone()),
            resource_cache: (self.resource_cache.srv.clone(), self.sampler.0.clone()),
            layers: (self.layers.srv.clone(), self.sampler.0.clone()),
            render_tasks: (self.render_tasks.srv.clone(), self.sampler.0.clone()),
            out_color: self.dummy_cache_a8().rtv.raw().clone(),
        };
        let psos = self.create_clip_psos(vert_src, frag_src);
        ClipProgram::new(data, psos, self.slice.clone(), upload)
    }

    pub fn create_debug_color_program(&mut self, vert_src: &[u8], frag_src: &[u8]) -> DebugColorProgram {
        // Creating a dummy vertexbuffer here. This is replaced in the draw_debug_color call.
        let quad_indices: &[u16] = &[0];
        let quad_vertices = [DebugColorVertices::new([0.0, 0.0], [0.0, 0.0, 0.0, 0.0])];
        let (vertex_buffer, mut slice) = self.factory.create_vertex_buffer_with_slice(&quad_vertices, quad_indices);

        let data = debug_color::Data {
            locals: self.factory.create_constant_buffer(1),
            transform: [[0f32; 4]; 4],
            device_pixel_ratio: DEVICE_PIXEL_RATIO,
            mode: 0,
            vbuf: vertex_buffer,
            out_color: self.main_color.raw().clone(),
        };
        let pso = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            debug_color::Init {
                out_color: ("Target0",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Srgb),
                            gfx::state::MASK_ALL,
                            Some(ALPHA)),
                .. debug_color::new()
            },
        ).unwrap();
        DebugColorProgram::new(data, pso, self.slice.clone())
    }

    pub fn create_clear_program(&mut self, vert_src: &[u8], frag_src: &[u8]) -> DebugColorProgram {
        // Creating a dummy vertexbuffer here. This is replaced in the draw_debug_color call.
        let quad_indices: &[u16] = &[0];
        let quad_vertices = [DebugColorVertices::new([0.0, 0.0], [0.0, 0.0, 0.0, 0.0])];
        let (vertex_buffer, mut slice) = self.factory.create_vertex_buffer_with_slice(&quad_vertices, quad_indices);

        let data = debug_color::Data {
            locals: self.factory.create_constant_buffer(1),
            transform: [[0f32; 4]; 4],
            device_pixel_ratio: DEVICE_PIXEL_RATIO,
            mode: 0,
            vbuf: vertex_buffer,
            out_color: self.main_color.raw().clone(),
        };
        let pso = self.factory.create_pipeline_simple(vert_src, frag_src, debug_color::new()).unwrap();
        DebugColorProgram::new(data, pso, self.slice.clone())
    }

    pub fn create_debug_font_program(&mut self, vert_src: &[u8], frag_src: &[u8]) -> DebugFontProgram {
        // Creating a dummy vertexbuffer here. This is replaced in the draw_debug_font call.
        let quad_indices: &[u16] = &[ 0,];
        let quad_vertices = [DebugFontVertices::new([0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0])];
        let (vertex_buffer, mut slice) = self.factory.create_vertex_buffer_with_slice(&quad_vertices, quad_indices);

        let data = debug_font::Data {
            locals: self.factory.create_constant_buffer(1),
            transform: [[0f32; 4]; 4],
            device_pixel_ratio: DEVICE_PIXEL_RATIO,
            mode: 0,
            vbuf: vertex_buffer,
            color0: (self.dummy_image().srv.clone(), self.sampler.0.clone()),
            out_color: self.main_color.raw().clone(),
        };
        let pso = self.factory.create_pipeline_simple(vert_src, frag_src, debug_font::new()).unwrap();
        DebugFontProgram::new(data, pso, slice)
    }
}
