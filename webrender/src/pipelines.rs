/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


use device::{Device, DEVICE_PIXEL_RATIO, MAX_INSTANCE_COUNT};
use euclid::Matrix4D;
use gfx;
use gfx::state::{Blend, BlendChannel, BlendValue, Comparison, Depth, Equation, Factor};
use gfx::memory::Typed;
use gfx::Factory;
use gfx::traits::FactoryExt;
use gfx::format::DepthStencil as DepthFormat;
use gfx_device_gl::Resources as R;
use gfx::format::Format;
use tiling::{BlurCommand, CacheClipInstance, PrimitiveInstance};
use renderer::BlendMode;

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

const SUBPIXEL: Blend = Blend {
    color: BlendChannel {
        equation: Equation::Add,
        source: Factor::ZeroPlus(BlendValue::ConstColor),
        destination: Factor::OneMinus(BlendValue::SourceColor),
    },
    alpha: BlendChannel {
        equation: Equation::Add,
        source: Factor::ZeroPlus(BlendValue::ConstColor),
        destination: Factor::OneMinus(BlendValue::SourceColor),
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
            data0: [i32; 4] = "aData0",
            data1: [i32; 4] = "aData1",
    }

    vertex BlurInstances {
        render_task_index: i32 = "aBlurRenderTaskIndex",
        source_task_index: i32 = "aBlurSourceTaskIndex",
        direction: i32 = "aBlurDirection",
    }

    vertex ClipInstances {
        render_task_index: i32 = "aClipRenderTaskIndex",
        layer_index: i32 = "aClipLayerIndex",
        data_index: i32 = "aClipDataIndex",
        segment_index: i32 = "aClipSegmentIndex",
        resource_address: i32 = "aClipResourceAddress",
    }

    pipeline primitive {
        transform: gfx::Global<[[f32; 4]; 4]> = "uTransform",
        device_pixel_ratio: gfx::Global<f32> = "uDevicePixelRatio",
        vbuf: gfx::VertexBuffer<Position> = (),
        ibuf: gfx::InstanceBuffer<PrimitiveInstances> = (),

        color0: gfx::TextureSampler<[f32; 4]> = "sColor0",
        color1: gfx::TextureSampler<[f32; 4]> = "sColor1",
        color2: gfx::TextureSampler<[f32; 4]> = "sColor2",
        dither: gfx::TextureSampler<f32> = "sDither",
        cache_a8: gfx::TextureSampler<[f32; 4]> = "sCacheA8",
        cache_rgba8: gfx::TextureSampler<[f32; 4]> = "sCacheRGBA8",

        layers: gfx::TextureSampler<[f32; 4]> = "sLayers",
        render_tasks: gfx::TextureSampler<[f32; 4]> = "sRenderTasks",
        resource_cache: gfx::TextureSampler<[f32; 4]> = "sResourceCache",

        out_color: gfx::RawRenderTarget = ("oFragColor",
                                           Format(gfx::format::SurfaceType::R8_G8_B8_A8,
                                                  gfx::format::ChannelType::Unorm),
                                           gfx::state::MASK_ALL,
                                           None),
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
        blend_value: gfx::BlendRef = (),
    }

    pipeline cache {
        transform: gfx::Global<[[f32; 4]; 4]> = "uTransform",
        device_pixel_ratio: gfx::Global<f32> = "uDevicePixelRatio",
        vbuf: gfx::VertexBuffer<Position> = (),
        ibuf: gfx::InstanceBuffer<PrimitiveInstances> = (),

        color0: gfx::TextureSampler<[f32; 4]> = "sColor0",
        cache_a8: gfx::TextureSampler<[f32; 4]> = "sCacheA8",
        cache_rgba8: gfx::TextureSampler<[f32; 4]> = "sCacheRGBA8",

        layers: gfx::TextureSampler<[f32; 4]> = "sLayers",
        render_tasks: gfx::TextureSampler<[f32; 4]> = "sRenderTasks",
        resource_cache: gfx::TextureSampler<[f32; 4]> = "sResourceCache",

        out_color: gfx::RawRenderTarget = ("oFragColor",
                                           Format(gfx::format::SurfaceType::R8_G8_B8_A8,
                                                  gfx::format::ChannelType::Unorm),
                                           gfx::state::MASK_ALL,
                                           None),
        out_depth: gfx::DepthTarget<DepthFormat> = Depth{fun: Comparison::Never , write: false},
    }

    pipeline blur {
        transform: gfx::Global<[[f32; 4]; 4]> = "uTransform",
        device_pixel_ratio: gfx::Global<f32> = "uDevicePixelRatio",
        vbuf: gfx::VertexBuffer<Position> = (),
        ibuf: gfx::InstanceBuffer<BlurInstances> = (),

        color0: gfx::TextureSampler<[f32; 4]> = "sColor0",
        cache_a8: gfx::TextureSampler<[f32; 4]> = "sCacheA8",
        cache_rgba8: gfx::TextureSampler<[f32; 4]> = "sCacheRGBA8",

        layers: gfx::TextureSampler<[f32; 4]> = "sLayers",
        render_tasks: gfx::TextureSampler<[f32; 4]> = "sRenderTasks",
        resource_cache: gfx::TextureSampler<[f32; 4]> = "sResourceCache",

        out_color: gfx::RawRenderTarget = ("oFragColor",
                                           Format(gfx::format::SurfaceType::R8_G8_B8_A8,
                                                  gfx::format::ChannelType::Unorm),
                                           gfx::state::MASK_ALL,
                                           None),
        out_depth: gfx::DepthTarget<DepthFormat> = Depth{fun: Comparison::Never , write: false},
    }

    pipeline clip {
        transform: gfx::Global<[[f32; 4]; 4]> = "uTransform",
        device_pixel_ratio: gfx::Global<f32> = "uDevicePixelRatio",
        vbuf: gfx::VertexBuffer<Position> = (),
        ibuf: gfx::InstanceBuffer<ClipInstances> = (),

        color0: gfx::TextureSampler<[f32; 4]> = "sColor0",
        color1: gfx::TextureSampler<[f32; 4]> = "sColor1",
        color2: gfx::TextureSampler<[f32; 4]> = "sColor2",
        dither: gfx::TextureSampler<f32> = "sDither",
        cache_a8: gfx::TextureSampler<[f32; 4]> = "sCacheA8",
        cache_rgba8: gfx::TextureSampler<[f32; 4]> = "sCacheRGBA8",

        layers: gfx::TextureSampler<[f32; 4]> = "sLayers",
        render_tasks: gfx::TextureSampler<[f32; 4]> = "sRenderTasks",
        resource_cache: gfx::TextureSampler<[f32; 4]> = "sResourceCache",

        out_color: gfx::RawRenderTarget = ("oFragColor",
                                           Format(gfx::format::SurfaceType::R8_G8_B8_A8,
                                                  gfx::format::ChannelType::Unorm),
                                           gfx::state::MASK_ALL,
                                           None),
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
    }
}

type PrimPSO = gfx::PipelineState<R, primitive::Meta>;
type CachePSO = gfx::PipelineState<R, cache::Meta>;
type ClipPSO = gfx::PipelineState<R, clip::Meta>;
type BlurPSO = gfx::PipelineState<R, blur::Meta>;

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

impl BlurInstances {
    pub fn new() -> BlurInstances {
        BlurInstances {
            render_task_index: 0,
            source_task_index: 0,
            direction: 0,
        }
    }

    pub fn update(&mut self, blur_command: &BlurCommand) {
        self.render_task_index = blur_command.task_id;
        self.source_task_index = blur_command.src_task_id;
        self.direction = blur_command.blur_direction;
    }
}

impl ClipInstances {
    pub fn new() -> ClipInstances {
        ClipInstances {
            render_task_index: 0,
            layer_index: 0,
            data_index: 0,
            segment_index: 0,
            resource_address: 0,
        }
    }

    pub fn update(&mut self, instance: &CacheClipInstance) {
        self.render_task_index = instance.task_id;
        self.layer_index = instance.layer_index;
        self.data_index = instance.address;
        self.segment_index = instance.segment;
        self.resource_address = instance.resource_address;
    }
}

pub struct Program {
    pub data: primitive::Data<R>,
    pub pso: (PrimPSO, PrimPSO),
    pub pso_alpha: (PrimPSO, PrimPSO),
    pub pso_prem_alpha: (PrimPSO, PrimPSO),
    pub pso_subpixel: (PrimPSO, PrimPSO),
    pub slice: gfx::Slice<R>,
    pub upload: gfx::handle::Buffer<R, PrimitiveInstances>,
}

impl Program {
    pub fn new(data: primitive::Data<R>,
           psos: (PrimPSO, PrimPSO, PrimPSO, PrimPSO, PrimPSO, PrimPSO, PrimPSO, PrimPSO),
           slice: gfx::Slice<R>,
           upload: gfx::handle::Buffer<R, PrimitiveInstances>)
           -> Program {
        Program {
            data: data,
            pso: (psos.0, psos.1),
            pso_alpha: (psos.2, psos.3),
            pso_prem_alpha: (psos.4, psos.5),
            pso_subpixel: (psos.6, psos.7),
            slice: slice,
            upload: upload,
        }
    }

    pub fn get_pso(&self, blend: &BlendMode, depth_write: bool) -> &PrimPSO {
        match *blend {
            BlendMode::Alpha => if depth_write { &self.pso_alpha.0 } else { &self.pso_alpha.1 },
            BlendMode::PremultipliedAlpha => if depth_write { &self.pso_prem_alpha.0 } else { &self.pso_prem_alpha.1 },
            BlendMode::Subpixel(..) => if depth_write { &self.pso_subpixel.0 } else { &self.pso_subpixel.1 },
            _ => if depth_write { &self.pso.0 } else { &self.pso.1 },
        }
    }
}

#[allow(dead_code)]
pub struct CacheProgram {
    pub data: cache::Data<R>,
    pub pso: CachePSO,
    pub pso_alpha: CachePSO,
    pub slice: gfx::Slice<R>,
    pub upload: gfx::handle::Buffer<R, PrimitiveInstances>,
}

#[allow(dead_code)]
impl CacheProgram {
    pub fn new(data: cache::Data<R>,
           psos: (CachePSO, CachePSO),
           slice: gfx::Slice<R>,
           upload: gfx::handle::Buffer<R, PrimitiveInstances>)
           -> CacheProgram {
        CacheProgram {
            data: data,
            pso: psos.0,
            pso_alpha: psos.1,
            slice: slice,
            upload: upload,
        }
    }

    pub fn get_pso(&self, blend: &BlendMode) -> &CachePSO {
        match *blend {
            BlendMode::Alpha => &self.pso_alpha,
            _ => &self.pso,
        }
    }
}

#[allow(dead_code)]
pub struct BlurProgram {
    pub data: blur::Data<R>,
    pub pso: BlurPSO,
    pub slice: gfx::Slice<R>,
    pub upload: gfx::handle::Buffer<R, BlurInstances>,
}

#[allow(dead_code)]
pub struct ClipProgram {
    pub data: clip::Data<R>,
    pub pso: ClipPSO,
    pub pso_multiply: ClipPSO,
    pub pso_max: ClipPSO,
    pub slice: gfx::Slice<R>,
    pub upload: gfx::handle::Buffer<R, ClipInstances>,
}

#[allow(dead_code)]
impl ClipProgram {
    pub fn new(data: clip::Data<R>,
           psos: (ClipPSO, ClipPSO, ClipPSO),
           slice: gfx::Slice<R>,
           upload: gfx::handle::Buffer<R, ClipInstances>)
           -> ClipProgram {
        ClipProgram {
            data: data,
            pso: psos.0,
            pso_multiply: psos.1,
            pso_max: psos.2,
            slice: slice,
            upload: upload,
        }
    }

    pub fn get_pso(&self, blend: &BlendMode) -> &ClipPSO {
        match *blend {
            BlendMode::Multiply => &self.pso_multiply,
            BlendMode::Max => &self.pso_max,
            _ => &self.pso,
        }
    }
}

impl Device {
    pub fn create_prim_psos(&mut self, vert_src: &[u8],frag_src: &[u8]) -> (PrimPSO, PrimPSO, PrimPSO, PrimPSO, PrimPSO, PrimPSO, PrimPSO, PrimPSO) {
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
                out_color: ("oFragColor",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Unorm),
                            gfx::state::MASK_ALL,
                            Some(ALPHA)),
                .. primitive::new()
            }
        ).unwrap();

        let pso_alpha = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("oFragColor",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Unorm),
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
                out_color: ("oFragColor",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Unorm),
                            gfx::state::MASK_ALL,
                            Some(PREM_ALPHA)),
                .. primitive::new()
            }
        ).unwrap();

        let pso_prem_alpha = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("oFragColor",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Unorm),
                            gfx::state::MASK_ALL,
                            Some(PREM_ALPHA)),
            out_depth: gfx::preset::depth::LESS_EQUAL_TEST,
                .. primitive::new()
            }
        ).unwrap();

        let pso_subpixel_depth_write = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("oFragColor",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Unorm),
                            gfx::state::MASK_ALL,
                            Some(SUBPIXEL)),
                .. primitive::new()
            }
        ).unwrap();

        let pso_subpixel = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("oFragColor",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Unorm),
                            gfx::state::MASK_ALL,
                            Some(SUBPIXEL)),
                out_depth: gfx::preset::depth::LESS_EQUAL_TEST,
                .. primitive::new()
            }
        ).unwrap();

        (pso_depth_write, pso, pso_alpha_depth_write, pso_alpha, pso_prem_alpha_depth_write,
         pso_prem_alpha, pso_subpixel_depth_write, pso_subpixel)
    }

    pub fn create_cache_psos(&mut self, vert_src: &[u8],frag_src: &[u8]) -> (CachePSO, CachePSO) {
        let pso = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            cache::new()
        ).unwrap();


        let pso_alpha = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            cache::Init {
                out_color: ("oFragColor",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Unorm),
                            gfx::state::MASK_ALL,
                            Some(ALPHA)),
                .. cache::new()
            }
        ).unwrap();

        (pso, pso_alpha)
    }

    pub fn create_clip_psos(&mut self, vert_src: &[u8],frag_src: &[u8]) -> (ClipPSO, ClipPSO, ClipPSO) {
        let pso = self.factory.create_pipeline_simple(vert_src, frag_src, clip::new()).unwrap();

        let pso_multiply = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            clip::Init {
                out_color: ("oFragColor",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Unorm),
                            gfx::state::MASK_ALL,
                            Some(MULTIPLY)),
                .. clip::new()
            }
        ).unwrap();

        let pso_max = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            clip::Init {
                out_color: ("oFragColor",
                            Format(gfx::format::SurfaceType::R8_G8_B8_A8, gfx::format::ChannelType::Unorm),
                            gfx::state::MASK_ALL,
                            Some(MAX)),
                .. clip::new()
            }
        ).unwrap();
        (pso, pso_multiply, pso_max)
    }

    pub fn create_cache_program(&mut self, vert_src: &[u8], frag_src: &[u8]) -> CacheProgram {
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

        let data = cache::Data {
            transform: [[0f32; 4]; 4],
            device_pixel_ratio: DEVICE_PIXEL_RATIO,
            vbuf: self.vertex_buffer.clone(),
            ibuf: instances,
            color0: (self.color0.clone().view, self.color0.clone().sampler),
            cache_a8: (self.cache_a8.clone().view, self.cache_a8.clone().sampler),
            cache_rgba8: (self.cache_rgba8.clone().view, self.cache_rgba8.clone().sampler),
            layers: (self.layers.clone().view, self.layers.clone().sampler),
            render_tasks: (self.render_tasks.clone().view, self.render_tasks.clone().sampler),
            resource_cache: (self.resource_cache.clone().view, self.resource_cache.clone().sampler),
            out_color: self.main_color.raw().clone(),
            out_depth: self.main_depth.clone(),
        };
        let psos = self.create_cache_psos(vert_src, frag_src);
        CacheProgram::new(data, psos, self.slice.clone(), upload)
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
            transform: [[0f32; 4]; 4],
            device_pixel_ratio: DEVICE_PIXEL_RATIO,
            vbuf: self.vertex_buffer.clone(),
            ibuf: blur_instances,
            color0: (self.color0.clone().view, self.color0.clone().sampler),
            cache_a8: (self.cache_a8.clone().view, self.cache_a8.clone().sampler),
            cache_rgba8: (self.cache_rgba8.clone().view, self.cache_rgba8.clone().sampler),
            layers: (self.layers.clone().view, self.layers.clone().sampler),
            render_tasks: (self.render_tasks.clone().view, self.render_tasks.clone().sampler),
            resource_cache: (self.resource_cache.clone().view, self.resource_cache.clone().sampler),
            out_color: self.main_color.raw().clone(),
            out_depth: self.main_depth.clone(),
        };
        let pso = self.factory.create_pipeline_simple(vert_src, frag_src, blur::new()).unwrap();
        BlurProgram {data: data, pso: pso, slice: self.slice.clone(), upload:upload}
    }

    pub fn create_clip_program(&mut self, vert_src: &[u8], frag_src: &[u8]) -> ClipProgram {
        let upload = self.factory.create_upload_buffer(MAX_INSTANCE_COUNT).unwrap();
        {
            let mut writer = self.factory.write_mapping(&upload).unwrap();
            for i in 0..MAX_INSTANCE_COUNT {
                writer[i] = ClipInstances::new();
            }
        }

        let cache_instances = self.factory.create_buffer(MAX_INSTANCE_COUNT,
                                                         gfx::buffer::Role::Vertex,
                                                         gfx::memory::Usage::Data,
                                                         gfx::TRANSFER_DST).unwrap();

        let data = clip::Data {
            transform: [[0f32; 4]; 4],
            device_pixel_ratio: DEVICE_PIXEL_RATIO,
            vbuf: self.vertex_buffer.clone(),
            ibuf: cache_instances,
            color0: (self.color0.clone().view, self.color0.clone().sampler),
            color1: (self.color1.clone().view, self.color1.clone().sampler),
            color2: (self.color2.clone().view, self.color2.clone().sampler),
            dither: (self.dither.clone().view, self.dither.clone().sampler),
            cache_a8: (self.cache_a8.clone().view, self.cache_a8.clone().sampler),
            cache_rgba8: (self.cache_rgba8.clone().view, self.cache_rgba8.clone().sampler),
            layers: (self.layers.clone().view, self.layers.clone().sampler),
            render_tasks: (self.render_tasks.clone().view, self.render_tasks.clone().sampler),
            resource_cache: (self.resource_cache.clone().view, self.resource_cache.clone().sampler),
            out_color: self.main_color.raw().clone(),
            out_depth: self.main_depth.clone(),
        };
        let psos = self.create_clip_psos(vert_src, frag_src);
        ClipProgram::new(data, psos, self.slice.clone(), upload)
    }

    pub fn draw_cache(&mut self, program: &mut CacheProgram, proj: &Matrix4D<f32>, instances: &[PrimitiveInstance], blendmode: &BlendMode) {
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

        self.encoder.copy_buffer(&program.upload, &program.data.ibuf, 0, 0, program.upload.len()).unwrap();
        self.encoder.draw(&program.slice, &program.get_pso(blendmode), &program.data);
    }

    pub fn draw_blur(&mut self, program: &mut BlurProgram, proj: &Matrix4D<f32>, blur_commands: &[BlurCommand]) {
       program.data.transform = proj.to_row_arrays();

        {
            let mut writer = self.factory.write_mapping(&program.upload).unwrap();
            for (i, blur_command) in blur_commands.iter().enumerate() {
                writer[i].update(blur_command);
            }
        }

        {
            program.slice.instances = Some((blur_commands.len() as u32, 0));
        }

        self.encoder.copy_buffer(&program.upload, &program.data.ibuf, 0, 0, program.upload.len()).unwrap();
        self.encoder.draw(&program.slice, &program.pso, &program.data);
    }
}
