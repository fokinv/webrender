/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! The webrender API.
//!
//! The `webrender::renderer` module provides the interface to webrender, which
//! is accessible through [`Renderer`][renderer]
//!
//! [renderer]: struct.Renderer.html

use api::{channel, BlobImageRenderer, FontRenderMode};
use api::{ColorF, Epoch, PipelineId, RenderApiSender, RenderNotifier};
use api::{DeviceIntPoint, DeviceIntRect, DeviceIntSize, DeviceUintRect, DeviceUintSize};
use api::{ExternalImageId, ExternalImageType, ImageFormat};
use api::{YUV_COLOR_SPACES, YUV_FORMATS};
use api::{YuvColorSpace, YuvFormat};
#[cfg(not(feature = "debugger"))]
use api::ApiMsg;
use api::{ColorU, DebugCommand};
#[cfg(not(feature = "debugger"))]
use api::channel::MsgSender;
use debug_colors;
use debug_render::{DebugRenderer, DebugColorVertex};
#[cfg(feature = "debugger")]
use debug_server::{self, DebugServer};
use device::{BackendDevice, Device, DeviceInitParams, FrameId, VertexDescriptor, GpuMarker, GpuProfiler};
use device::{GpuTimer, TextureFilter, VertexUsageHint, TextureTarget, ShaderError,FileWatcherHandler};
use device::{TextureSlot, TextureStorage, VertexAttribute, VertexAttributeKind};
use device::{TextureId, DUMMY_ID};
use euclid::{rect, Transform3D};
use frame_builder::FrameBuilderConfig;
use glyph_rasterizer::GlyphFormat;
use gpu_cache::{GpuBlockData, GpuCacheUpdate, GpuCacheUpdateList};
use gpu_types::PrimitiveInstance;
use internal_types::{BatchTextures, SourceTexture, ORTHO_FAR_PLANE, ORTHO_NEAR_PLANE};
use internal_types::{CacheTextureId, FastHashMap, RendererFrame, ResultMsg, TextureUpdateOp};
use internal_types::{DebugOutput, RenderTargetMode, TextureUpdateList, TextureUpdateSource};
use pipelines::{BlurProgram, BrushProgram, ClipProgram, DebugColorProgram, DebugFontProgram, Program, TextProgram};
use profiler::{BackendProfileCounters, Profiler};
use profiler::{GpuProfileTag, RendererProfileCounters, RendererProfileTimers};
use rayon::Configuration as ThreadPoolConfig;
use rayon::ThreadPool;
use record::ApiRecordingReceiver;
use render_backend::RenderBackend;
use render_task::RenderTaskTree;
#[cfg(feature = "debugger")]
use serde_json;
use std;
use std::cmp;
use std::collections::VecDeque;
use std::collections::hash_map::Entry;
use std::f32;
use std::fs::File;
use std::mem;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use texture_cache::TextureCache;
use thread_profiler::{register_thread_with_profiler, write_profile};
use tiling::{AlphaRenderTarget, ColorRenderTarget, RenderTargetKind};
use tiling::{BatchKey, BatchKind, BrushBatchKind, Frame, RenderTarget, TransformBatchKind};
use time::precise_time_ns;
use util::TransformedRectKind;

use backend::{self, Resources as R};
use gfx;
use gfx::format::{DepthStencil as DepthFormat, Rgba8 as ColorFormat};
use window::ExistingWindow;

pub const MAX_VERTEX_TEXTURE_WIDTH: usize = 1024;

const GPU_TAG_BRUSH_MASK: GpuProfileTag = GpuProfileTag {
    label: "B_Mask",
    color: debug_colors::BLACK,
};
const GPU_TAG_BRUSH_IMAGE: GpuProfileTag = GpuProfileTag {
    label: "B_Image",
    color: debug_colors::SILVER,
};
const GPU_TAG_CACHE_CLIP: GpuProfileTag = GpuProfileTag {
    label: "C_Clip",
    color: debug_colors::PURPLE,
};
const GPU_TAG_CACHE_TEXT_RUN: GpuProfileTag = GpuProfileTag {
    label: "C_TextRun",
    color: debug_colors::MISTYROSE,
};
const GPU_TAG_CACHE_LINE: GpuProfileTag = GpuProfileTag {
    label: "C_Line",
    color: debug_colors::BROWN,
};
const GPU_TAG_SETUP_TARGET: GpuProfileTag = GpuProfileTag {
    label: "target",
    color: debug_colors::SLATEGREY,
};
const GPU_TAG_SETUP_DATA: GpuProfileTag = GpuProfileTag {
    label: "data init",
    color: debug_colors::LIGHTGREY,
};
const GPU_TAG_PRIM_RECT: GpuProfileTag = GpuProfileTag {
    label: "Rect",
    color: debug_colors::RED,
};
const GPU_TAG_PRIM_LINE: GpuProfileTag = GpuProfileTag {
    label: "Line",
    color: debug_colors::DARKRED,
};
const GPU_TAG_PRIM_IMAGE: GpuProfileTag = GpuProfileTag {
    label: "Image",
    color: debug_colors::GREEN,
};
const GPU_TAG_PRIM_YUV_IMAGE: GpuProfileTag = GpuProfileTag {
    label: "YuvImage",
    color: debug_colors::DARKGREEN,
};
const GPU_TAG_PRIM_BLEND: GpuProfileTag = GpuProfileTag {
    label: "Blend",
    color: debug_colors::LIGHTBLUE,
};
const GPU_TAG_PRIM_HW_COMPOSITE: GpuProfileTag = GpuProfileTag {
    label: "HwComposite",
    color: debug_colors::DODGERBLUE,
};
const GPU_TAG_PRIM_SPLIT_COMPOSITE: GpuProfileTag = GpuProfileTag {
    label: "SplitComposite",
    color: debug_colors::DARKBLUE,
};
const GPU_TAG_PRIM_COMPOSITE: GpuProfileTag = GpuProfileTag {
    label: "Composite",
    color: debug_colors::MAGENTA,
};
const GPU_TAG_PRIM_TEXT_RUN: GpuProfileTag = GpuProfileTag {
    label: "TextRun",
    color: debug_colors::BLUE,
};
const GPU_TAG_PRIM_GRADIENT: GpuProfileTag = GpuProfileTag {
    label: "Gradient",
    color: debug_colors::YELLOW,
};
const GPU_TAG_PRIM_ANGLE_GRADIENT: GpuProfileTag = GpuProfileTag {
    label: "AngleGradient",
    color: debug_colors::POWDERBLUE,
};
const GPU_TAG_PRIM_RADIAL_GRADIENT: GpuProfileTag = GpuProfileTag {
    label: "RadialGradient",
    color: debug_colors::LIGHTPINK,
};
const GPU_TAG_PRIM_BORDER_CORNER: GpuProfileTag = GpuProfileTag {
    label: "BorderCorner",
    color: debug_colors::DARKSLATEGREY,
};
const GPU_TAG_PRIM_BORDER_EDGE: GpuProfileTag = GpuProfileTag {
    label: "BorderEdge",
    color: debug_colors::LAVENDER,
};
const GPU_TAG_BLUR: GpuProfileTag = GpuProfileTag {
    label: "Blur",
    color: debug_colors::VIOLET,
};

const GPU_SAMPLER_TAG_ALPHA: GpuProfileTag = GpuProfileTag {
    label: "Alpha Targets",
    color: debug_colors::BLACK,
};
const GPU_SAMPLER_TAG_OPAQUE: GpuProfileTag = GpuProfileTag {
    label: "Opaque Pass",
    color: debug_colors::BLACK,
};
const GPU_SAMPLER_TAG_TRANSPARENT: GpuProfileTag = GpuProfileTag {
    label: "Transparent Pass",
    color: debug_colors::BLACK,
};

#[cfg(feature = "debugger")]
impl BatchKind {
    fn debug_name(&self) -> &'static str {
        match *self {
            BatchKind::Composite { .. } => "Composite",
            BatchKind::HardwareComposite => "HardwareComposite",
            BatchKind::SplitComposite => "SplitComposite",
            BatchKind::Blend => "Blend",
            BatchKind::Brush(kind) => {
                match kind {
                    BrushBatchKind::Image(..) => "Brush (Image)",
                }
            }
            BatchKind::Transformable(_, kind) => match kind {
                TransformBatchKind::Rectangle(..) => "Rectangle",
                TransformBatchKind::TextRun(..) => "TextRun",
                TransformBatchKind::Image(image_buffer_kind, ..) => match image_buffer_kind {
                    ImageBufferKind::Texture2D => "Image (2D)",
                    ImageBufferKind::TextureRect => "Image (Rect)",
                    ImageBufferKind::TextureExternal => "Image (External)",
                    ImageBufferKind::Texture2DArray => "Image (Array)",
                },
                TransformBatchKind::YuvImage(..) => "YuvImage",
                TransformBatchKind::AlignedGradient => "AlignedGradient",
                TransformBatchKind::AngleGradient => "AngleGradient",
                TransformBatchKind::RadialGradient => "RadialGradient",
                TransformBatchKind::BorderCorner => "BorderCorner",
                TransformBatchKind::BorderEdge => "BorderEdge",
                TransformBatchKind::Line => "Line",
            },
        }
    }
}

bitflags! {
    #[derive(Default)]
    pub struct DebugFlags: u32 {
        const PROFILER_DBG      = 1 << 0;
        const RENDER_TARGET_DBG = 1 << 1;
        const TEXTURE_CACHE_DBG = 1 << 2;
        const ALPHA_PRIM_DBG    = 1 << 3;
    }
}

// A generic mode that can be passed to shaders to change
// behaviour per draw-call.
type ShaderMode = i32;

#[repr(C)]
enum TextShaderMode {
    Alpha = 0,
    SubpixelPass0 = 1,
    SubpixelPass1 = 2,
    ColorBitmap = 3,
}

impl Into<ShaderMode> for TextShaderMode {
    fn into(self) -> i32 {
        self as i32
    }
}

impl From<GlyphFormat> for TextShaderMode {
    fn from(format: GlyphFormat) -> TextShaderMode {
        match format {
            GlyphFormat::Mono | GlyphFormat::Alpha => TextShaderMode::Alpha,
            GlyphFormat::Subpixel => {
                panic!("Subpixel glyph format must be handled separately.");
            }
            GlyphFormat::ColorBitmap => TextShaderMode::ColorBitmap,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TextureSampler {
    Color0,
    Color1,
    Color2,
    CacheA8,
    CacheRGBA8,
    ResourceCache,
    Layers,
    RenderTasks,
    Dither,
    // A special sampler that is bound to the A8 output of
    // the *first* pass. Items rendered in this target are
    // available as inputs to tasks in any subsequent pass.
    SharedCacheA8,
}

impl TextureSampler {
    fn color(n: usize) -> TextureSampler {
        match n {
            0 => TextureSampler::Color0,
            1 => TextureSampler::Color1,
            2 => TextureSampler::Color2,
            _ => {
                panic!("There are only 3 color samplers.");
            }
        }
    }
}

impl Into<TextureSlot> for TextureSampler {
    fn into(self) -> TextureSlot {
        match self {
            TextureSampler::Color0 => TextureSlot(0),
            TextureSampler::Color1 => TextureSlot(1),
            TextureSampler::Color2 => TextureSlot(2),
            TextureSampler::CacheA8 => TextureSlot(3),
            TextureSampler::CacheRGBA8 => TextureSlot(4),
            TextureSampler::ResourceCache => TextureSlot(5),
            TextureSampler::Layers => TextureSlot(6),
            TextureSampler::RenderTasks => TextureSlot(7),
            TextureSampler::Dither => TextureSlot(8),
            TextureSampler::SharedCacheA8 => TextureSlot(9),
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PackedVertex {
    pub pos: [f32; 2],
}

/*const DESC_PRIM_INSTANCES: VertexDescriptor = VertexDescriptor {
    vertex_attributes: &[
        VertexAttribute {
            name: "aPosition",
            count: 2,
            kind: VertexAttributeKind::F32,
        },
    ],
    instance_attributes: &[
        VertexAttribute {
            name: "aData0",
            count: 4,
            kind: VertexAttributeKind::I32,
        },
        VertexAttribute {
            name: "aData1",
            count: 4,
            kind: VertexAttributeKind::I32,
        },
    ],
};

const DESC_BLUR: VertexDescriptor = VertexDescriptor {
    vertex_attributes: &[
        VertexAttribute {
            name: "aPosition",
            count: 2,
            kind: VertexAttributeKind::F32,
        },
    ],
    instance_attributes: &[
        VertexAttribute {
            name: "aBlurRenderTaskAddress",
            count: 1,
            kind: VertexAttributeKind::I32,
        },
        VertexAttribute {
            name: "aBlurSourceTaskAddress",
            count: 1,
            kind: VertexAttributeKind::I32,
        },
        VertexAttribute {
            name: "aBlurDirection",
            count: 1,
            kind: VertexAttributeKind::I32,
        },
        VertexAttribute {
            name: "aBlurRegion",
            count: 4,
            kind: VertexAttributeKind::F32
        },
    ],
};

const DESC_CLIP: VertexDescriptor = VertexDescriptor {
    vertex_attributes: &[
        VertexAttribute {
            name: "aPosition",
            count: 2,
            kind: VertexAttributeKind::F32,
        },
    ],
    instance_attributes: &[
        VertexAttribute {
            name: "aClipRenderTaskAddress",
            count: 1,
            kind: VertexAttributeKind::I32,
        },
        VertexAttribute {
            name: "aClipLayerAddress",
            count: 1,
            kind: VertexAttributeKind::I32,
        },
        VertexAttribute {
            name: "aClipSegment",
            count: 1,
            kind: VertexAttributeKind::I32,
        },
        VertexAttribute {
            name: "aClipDataResourceAddress",
            count: 4,
            kind: VertexAttributeKind::U16,
        },
    ],
};*/

#[derive(Debug, Copy, Clone)]
enum VertexArrayKind {
    Primitive,
    Blur,
    Clip,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphicsApi {
    OpenGL,
    D3D11,
}

#[derive(Clone, Debug)]
pub struct GraphicsApiInfo {
    pub kind: GraphicsApi,
    pub renderer: String,
    pub version: String,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum ImageBufferKind {
    Texture2D = 0,
    TextureRect = 1,
    TextureExternal = 2,
    Texture2DArray = 3,
}

pub const IMAGE_BUFFER_KINDS: [ImageBufferKind; 4] = [
    ImageBufferKind::Texture2D,
    ImageBufferKind::TextureRect,
    ImageBufferKind::TextureExternal,
    ImageBufferKind::Texture2DArray,
];

impl ImageBufferKind {
    pub fn get_feature_string(&self) -> &'static str {
        match *self {
            ImageBufferKind::Texture2D => "TEXTURE_2D",
            ImageBufferKind::Texture2DArray => "",
            ImageBufferKind::TextureRect => "TEXTURE_RECT",
            ImageBufferKind::TextureExternal => "TEXTURE_EXTERNAL",
        }
    }

    pub fn has_platform_support(&self) -> bool {
        match *self {
            ImageBufferKind::Texture2D => true,
            ImageBufferKind::Texture2DArray => true,
            ImageBufferKind::TextureRect => true,
            ImageBufferKind::TextureExternal => false,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum RendererKind {
    Native,
    OSMesa,
}

#[derive(Debug)]
pub struct GpuProfile {
    pub frame_id: FrameId,
    pub paint_time_ns: u64,
}

impl GpuProfile {
    fn new<T>(frame_id: FrameId, timers: &[GpuTimer<T>]) -> GpuProfile {
        let mut paint_time_ns = 0;
        for timer in timers {
            paint_time_ns += timer.time_ns;
        }
        GpuProfile {
            frame_id,
            paint_time_ns,
        }
    }
}

#[derive(Debug)]
pub struct CpuProfile {
    pub frame_id: FrameId,
    pub backend_time_ns: u64,
    pub composite_time_ns: u64,
    pub draw_calls: usize,
}

impl CpuProfile {
    fn new(
        frame_id: FrameId,
        backend_time_ns: u64,
        composite_time_ns: u64,
        draw_calls: usize,
    ) -> CpuProfile {
        CpuProfile {
            frame_id,
            backend_time_ns,
            composite_time_ns,
            draw_calls,
        }
    }
}

struct SourceTextureResolver {
    /// A vector for fast resolves of texture cache IDs to
    /// native texture IDs. This maps to a free-list managed
    /// by the backend thread / texture cache. We free the
    /// texture memory associated with a TextureId when its
    /// texture cache ID is freed by the texture cache, but
    /// reuse the TextureId when the texture caches's free
    /// list reuses the texture cache ID. This saves having to
    /// use a hashmap, and allows a flat vector for performance.
    cache_texture_map: Vec<TextureId>,

    /// Map of external image IDs to native textures.
    //external_images: FastHashMap<(ExternalImageId, u8), ExternalTexture>,

    /// A special 1x1 dummy cache texture used for shaders that expect to work
    /// with the cache but are actually running in the first pass
    /// when no target is yet provided as a cache texture input.
    dummy_cache_rgba8_texture: TextureId,
    dummy_cache_a8_texture: TextureId,

    /// The current cache textures.
    cache_rgba8_texture: Option<TextureId>,
    cache_a8_texture: Option<TextureId>,
}

impl SourceTextureResolver {
    fn new(/*device: &mut Device*/) -> SourceTextureResolver {
        SourceTextureResolver {
            cache_texture_map: Vec::new(),
            //external_images: FastHashMap::default(),
            dummy_cache_a8_texture: DUMMY_ID,
            dummy_cache_rgba8_texture: DUMMY_ID,
            cache_a8_texture: None,
            cache_rgba8_texture: None,
        }
    }

    fn deinit(self, device: &mut Device) {
        //device.delete_texture(self.dummy_cache_texture);

        /*for texture in self.cache_texture_map {
            device.delete_texture(texture);
        }*/
    }

    fn end_pass(
        &mut self,
        pass_index: usize,
        pass_count: usize,
        mut a8_texture: Option<TextureId>,
        mut rgba8_texture: Option<TextureId>,
        a8_pool: &mut Vec<TextureId>,
        rgba8_pool: &mut Vec<TextureId>,
    ) {
        // If we have cache textures from previous pass, return them to the pool.
        rgba8_pool.extend(self.cache_rgba8_texture.take());
        a8_pool.extend(self.cache_a8_texture.take());

        if pass_index == pass_count - 1 {
            // On the last pass, return the textures from this pass to the pool.
            if let Some(texture) = rgba8_texture.take() {
                rgba8_pool.push(texture);
            }
            if let Some(texture) = a8_texture.take() {
                a8_pool.push(texture);
            }
        } else {
            // We have another pass to process, make these textures available
            // as inputs to the next pass.
            self.cache_rgba8_texture = rgba8_texture.take();
            self.cache_a8_texture = a8_texture.take();
        }
    }

    // Bind a source texture to the device.
    fn bind(&self, texture_id: &SourceTexture, sampler: TextureSampler, device: &mut Device) {
        match *texture_id {
            SourceTexture::Invalid => {}
            SourceTexture::CacheA8 => {
                let texture = self.cache_a8_texture
                    .unwrap_or(self.dummy_cache_a8_texture);
                device.bind_texture(sampler, texture, TextureStorage::CacheA8);
            }
            SourceTexture::CacheRGBA8 => {
                let texture = self.cache_rgba8_texture
                    .unwrap_or(self.dummy_cache_rgba8_texture);
                device.bind_texture(sampler, texture, TextureStorage::CacheRGBA8);
            }
            SourceTexture::External(external_image) => {
                /*let texture = self.external_images
                    .get(&(external_image.id, external_image.channel_index))
                    .expect("BUG: External image should be resolved by now!");
                device.bind_external_texture(sampler, texture);*/
            }
            SourceTexture::TextureCache(index) => {
                let texture = self.cache_texture_map[index.0];
                device.bind_texture(sampler, texture, TextureStorage::Image);
            }
        }
    }

    // Get the real (OpenGL) texture ID for a given source texture.
    // For a texture cache texture, the IDs are stored in a vector
    // map for fast access.
    fn resolve(&self, texture_id: &SourceTexture) -> Option<TextureId> {
        match *texture_id {
            SourceTexture::Invalid => None,
            SourceTexture::CacheA8 => Some(
                self.cache_a8_texture
                    .unwrap_or(self.dummy_cache_a8_texture),
            ),
            SourceTexture::CacheRGBA8 => Some(
                self.cache_rgba8_texture
                    .unwrap_or(self.dummy_cache_rgba8_texture),
            ),
            SourceTexture::External(..) => {
                panic!("BUG: External textures cannot be resolved, they can only be bound.");
            }
            SourceTexture::TextureCache(index) => Some(self.cache_texture_map[index.0]),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BlendMode {
    None,
    Alpha,
    PremultipliedAlpha,
    Subpixel,
    Multiply,
    Max,
}

// Tracks the state of each row in the GPU cache texture.
struct CacheRow {
    is_dirty: bool,
}

impl CacheRow {
    fn new() -> CacheRow {
        CacheRow { is_dirty: false }
    }
}

/// The device-specific representation of the cache texture in gpu_cache.rs
struct CacheTexture {
    rows: Vec<CacheRow>,
    cpu_blocks: Vec<GpuBlockData>,
}

impl CacheTexture {
    fn new(device: &mut Device) -> CacheTexture {
        CacheTexture {
            rows: Vec::new(),
            cpu_blocks: Vec::new(),
        }
    }

    fn apply_patch(&mut self, update: &GpuCacheUpdate, blocks: &[GpuBlockData]) {
        match update {
            &GpuCacheUpdate::Copy {
                block_index,
                block_count,
                address,
            } => {
                let row = address.v as usize;

                // Ensure that the CPU-side shadow copy of the GPU cache data has enough
                // rows to apply this patch.
                while self.rows.len() <= row {
                    // Add a new row.
                    self.rows.push(CacheRow::new());
                    // Add enough GPU blocks for this row.
                    self.cpu_blocks
                        .extend_from_slice(&[GpuBlockData::empty(); MAX_VERTEX_TEXTURE_WIDTH]);
                }

                // This row is dirty (needs to be updated in GPU texture).
                self.rows[row].is_dirty = true;

                // Copy the blocks from the patch array in the shadow CPU copy.
                let block_offset = row * MAX_VERTEX_TEXTURE_WIDTH + address.u as usize;
                let data = &mut self.cpu_blocks[block_offset .. (block_offset + block_count)];
                for i in 0 .. block_count {
                    data[i] = blocks[block_index + i];
                }
            }
        }
    }

    fn update(&mut self, device: &mut Device, updates: &GpuCacheUpdateList) {
        // See if we need to create or resize the texture.
        //let current_dimensions = self.texture.get_dimensions();
        /*if updates.height > current_dimensions.height {
            panic!("add resize")
            // Create a f32 texture that can be used for the vertex shader
            // to fetch data from.
            device.init_texture(
                &mut self.texture,
                MAX_VERTEX_TEXTURE_WIDTH as u32,
                updates.height as u32,
                ImageFormat::RGBAF32,
                TextureFilter::Nearest,
                RenderTargetMode::None,
                1,
                None,
            );

            // Copy the current texture into the newly resized texture.
            if current_dimensions.height > 0 {
                // If we had to resize the texture, just mark all rows
                // as dirty so they will be uploaded to the texture
                // during the next flush.
                for row in &mut self.rows {
                    row.is_dirty = true;
                }
            }
        }*/

        for update in &updates.updates {
            self.apply_patch(update, &updates.blocks);
        }
    }

    #[cfg(all(target_os = "windows", feature="dx11"))]
    fn flush(&mut self, device: &mut Device) {
        let is_dirty = self.rows.iter().any(|r| r.is_dirty);
        if is_dirty {
            let cpu_blocks = &self.cpu_blocks[..];

            device.update_data_texture(TextureSampler::ResourceCache, [0, 0], [MAX_VERTEX_TEXTURE_WIDTH as u16, self.rows.len() as u16], cpu_blocks);

            for row in self.rows.iter_mut() {
                row.is_dirty = false;
            }
        }
    }

    #[cfg(not(feature = "dx11"))]
    fn flush(&mut self, device: &mut Device) {
        for (row_index, row) in self.rows.iter_mut().enumerate() {
            if row.is_dirty {
                let block_index = row_index * MAX_VERTEX_TEXTURE_WIDTH;
                let cpu_blocks =
                    &self.cpu_blocks[block_index .. (block_index + MAX_VERTEX_TEXTURE_WIDTH)];
                /*device.update_pbo_data(cpu_blocks);

                // Insert a command to copy the PBO data to the right place in
                // the GPU-side cache texture.
                device.update_texture_from_pbo(
                    &self.texture,
                    0,
                    row_index as u32,
                    MAX_VERTEX_TEXTURE_WIDTH as u32,
                    1,
                    0,
                    None,
                    0,
                );

                // Orphan the PBO. This is the recommended way to hint to the
                // driver to detach the underlying storage from this PBO id.
                // Keeping the size the same gives the driver a hint for future
                // use of this PBO.
                device.orphan_pbo(mem::size_of::<GpuBlockData>() * MAX_VERTEX_TEXTURE_WIDTH);*/
                
                device.update_data_texture(
                    TextureSampler::ResourceCache,
                    [0, row_index as u16],
                    [MAX_VERTEX_TEXTURE_WIDTH as u16, 1],
                    cpu_blocks,
                );

                row.is_dirty = false;
            }
        }
    }
}

struct VertexDataTexture {
    sampler: TextureSampler,
}

impl VertexDataTexture {
    fn new(sampler: TextureSampler) -> VertexDataTexture {
        VertexDataTexture {
            sampler: sampler,
        }
    }

    fn update<T>(&mut self, device: &mut Device, data: &mut Vec<T>)
        where T: gfx::traits::Pod
    {
        if data.is_empty() {
            return;
        }

        debug_assert!(mem::size_of::<T>() % 16 == 0);
        let texels_per_item = mem::size_of::<T>() / 16;
        let items_per_row = MAX_VERTEX_TEXTURE_WIDTH / texels_per_item;

        // Extend the data array to be a multiple of the row size.
        // This ensures memory safety when the array is passed to
        // OpenGL to upload to the GPU.
        if items_per_row != 0 {
            while data.len() % items_per_row != 0 {
                data.push(unsafe { mem::uninitialized() });
            }
        }

        let width =
            (MAX_VERTEX_TEXTURE_WIDTH - (MAX_VERTEX_TEXTURE_WIDTH % texels_per_item)) as u32;
        let needed_height = (data.len() / items_per_row) as u32;

        // Determine if the texture needs to be resized.
        /*let texture_size = self.texture.get_dimensions();

        if needed_height > texture_size.height {
            let new_height = (needed_height + 127) & !127;

            device.init_texture(
                &mut self.texture,
                width,
                new_height,
                ImageFormat::RGBAF32,
                TextureFilter::Nearest,
                RenderTargetMode::None,
                1,
                None,
            );
        }*/

        device.update_data_texture(self.sampler, [0, 0], [width as u16, needed_height as u16], data);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReadPixelsFormat {
    Rgba8,
    Bgra8,
}

struct FileWatcher {
    notifier: Box<RenderNotifier>,
    result_tx: Sender<ResultMsg>,
}

impl FileWatcherHandler for FileWatcher {
    fn file_changed(&self, path: PathBuf) {
        self.result_tx.send(ResultMsg::RefreshShader(path)).ok();
        self.notifier.new_frame_ready();
    }
}

#[derive(Debug)]
struct ProgramPair((Box<Program>, Box<Program>));

impl ProgramPair {
    fn get(&mut self, transform_kind: TransformedRectKind) -> &mut Box<Program> {
        match transform_kind {
            TransformedRectKind::AxisAligned => &mut (self.0).0,
            TransformedRectKind::Complex => &mut (self.0).1,
        }
    }

    pub fn reset_upload_offset(&mut self) {
        (self.0).0.reset_upload_offset();
        (self.0).1.reset_upload_offset();
    }

    pub fn bind(
        &mut self,
        device: &mut Device,
        transform_kind: TransformedRectKind,
        projection: &Transform3D<f32>,
        instances: &[PrimitiveInstance],
        render_target: Option<(&TextureId, i32)>,
        renderer_errors: &mut Vec<RendererError>,
        mode: i32,
    ) {
        self.get(transform_kind).bind(device, projection, instances, render_target, renderer_errors, mode);
    }
}

#[derive(Debug)]
struct BrushProgramPair((Box<BrushProgram>, Box<BrushProgram>));

impl BrushProgramPair {
    fn get(&mut self, blend_mode: BlendMode) -> &mut Box<BrushProgram> {
        match blend_mode {
            BlendMode::None => &mut (self.0).0,
            BlendMode::Alpha |
            BlendMode::PremultipliedAlpha |
            BlendMode::Subpixel |
            BlendMode::Max |
            BlendMode::Multiply => {
                &mut (self.0).1
            }
        }
    }

    pub fn reset_upload_offset(&mut self) {
        (self.0).0.reset_upload_offset();
        (self.0).1.reset_upload_offset();
    }

    pub fn bind(
        &mut self,
        device: &mut Device,
        blend_mode: BlendMode,
        projection: &Transform3D<f32>,
        instances: &[PrimitiveInstance],
        render_target: Option<(&TextureId, i32)>,
        renderer_errors: &mut Vec<RendererError>,
        mode: i32,
    ) {
        self.get(blend_mode).bind(device, projection, instances, render_target, renderer_errors, mode);
    }
}

#[derive(Debug)]
struct TextProgramPair((Box<TextProgram>, Box<TextProgram>));

impl TextProgramPair {
    fn get(&mut self, transform_kind: TransformedRectKind) -> &mut Box<TextProgram> {
        match transform_kind {
            TransformedRectKind::AxisAligned => &mut (self.0).0,
            TransformedRectKind::Complex => &mut (self.0).1,
        }
    }

    pub fn reset_upload_offset(&mut self) {
        (self.0).0.reset_upload_offset();
        (self.0).1.reset_upload_offset();
    }

    pub fn bind(
        &mut self,
        device: &mut Device,
        transform_kind: TransformedRectKind,
        projection: &Transform3D<f32>,
        instances: &[PrimitiveInstance],
        render_target: Option<(&TextureId, i32)>,
        renderer_errors: &mut Vec<RendererError>,
        mode: i32,
    ) {
        self.get(transform_kind).bind(device, projection, instances, render_target, renderer_errors, mode);
    }
}

trait BindDraw {
    fn bind_resources(
        &mut self,
        device: &mut Device,
        projection: &Transform3D<f32>,
        instances: &[PrimitiveInstance],
        render_target: Option<(&TextureId, i32)>,
        renderer_errors: &mut Vec<RendererError>,
        mode: i32
    );
    fn call_draw(&mut self, device: &mut Device, blend_mode: &BlendMode, enable_depth_write: bool);
}

impl BindDraw for Box<Program> {
    fn bind_resources(
        &mut self,
        device: &mut Device,
        projection: &Transform3D<f32>,
        instances: &[PrimitiveInstance],
        render_target: Option<(&TextureId, i32)>,
        renderer_errors: &mut Vec<RendererError>,
        mode: i32
    ) {
        self.bind(device, projection, instances, render_target, renderer_errors, mode)
    }

    fn call_draw(&mut self, device: &mut Device, blend_mode: &BlendMode, enable_depth_write: bool) {
        self.draw(device, blend_mode, enable_depth_write)
    }
}

impl BindDraw for Box<BrushProgram> {
    fn bind_resources(
        &mut self,
        device: &mut Device,
        projection: &Transform3D<f32>,
        instances: &[PrimitiveInstance],
        render_target: Option<(&TextureId, i32)>,
        renderer_errors: &mut Vec<RendererError>,
        mode: i32
    ) {
        self.bind(device, projection, instances, render_target, renderer_errors, mode)
    }

    fn call_draw(&mut self, device: &mut Device, blend_mode: &BlendMode, enable_depth_write: bool) {
        self.draw(device, blend_mode)
    }
}


fn create_prim_programs(device: &mut Device, filename: &str) -> ProgramPair {
    let program = create_program(device, filename);
    ProgramPair((program, create_program(device, &format!("{}_transform", filename))))
}

fn create_brush_programs(device: &mut Device, filename: &str) -> BrushProgramPair {
    let program = create_brush_program(device, filename);
    BrushProgramPair((program, create_brush_program(device, &format!("{}_alpha_pass", filename))))
}

fn create_text_programs(device: &mut Device, filename: &str) -> TextProgramPair {
    let program = create_text_program(device, filename);
    TextProgramPair((program, create_text_program(device, &format!("{}_transform", filename))))
}

#[cfg(not(feature = "dx11"))]
fn create_program(device: &mut Device, filename: &str) -> Box<Program> {
    let vs = get_shader_source(filename, ".vert");
    let ps = get_shader_source(filename, ".frag");
    Box::new(device.create_program(vs.as_slice(), ps.as_slice()))
}

#[cfg(all(target_os = "windows", feature="dx11"))]
fn create_program(device: &mut Device, filename: &str) -> Box<Program> {
    let vs = get_shader_source(filename, ".vert.fx");
    let ps = get_shader_source(filename, ".frag.fx");
    Box::new(device.create_program(vs.as_slice(), ps.as_slice()))
}

#[cfg(not(feature = "dx11"))]
fn create_brush_program(device: &mut Device, filename: &str) -> Box<BrushProgram> {
    let vs = get_shader_source(filename, ".vert");
    let ps = get_shader_source(filename, ".frag");
    Box::new(device.create_brush_program(vs.as_slice(), ps.as_slice()))
}

#[cfg(all(target_os = "windows", feature="dx11"))]
fn create_brush_program(device: &mut Device, filename: &str) -> Box<BrushProgram> {
    let vs = get_shader_source(filename, ".vert.fx");
    let ps = get_shader_source(filename, ".frag.fx");
    Box::new(device.create_brush_program(vs.as_slice(), ps.as_slice()))
}


#[cfg(not(feature = "dx11"))]
fn create_text_program(device: &mut Device, filename: &str) -> Box<TextProgram> {
    let vs = get_shader_source(filename, ".vert");
    let ps = get_shader_source(filename, ".frag");
    Box::new(device.create_text_program(vs.as_slice(), ps.as_slice()))
}

#[cfg(all(target_os = "windows", feature="dx11"))]
fn create_text_program(device: &mut Device, filename: &str) -> Box<TextProgram> {
    let vs = get_shader_source(filename, ".vert.fx");
    let ps = get_shader_source(filename, ".frag.fx");
    Box::new(device.create_text_program(vs.as_slice(), ps.as_slice()))
}

#[cfg(not(feature = "dx11"))]
fn create_clip_program(device: &mut Device, filename: &str) -> Box<ClipProgram> {
    let vs = get_shader_source(filename, ".vert");
    let ps = get_shader_source(filename, ".frag");
    Box::new(device.create_clip_program(vs.as_slice(), ps.as_slice()))
}

#[cfg(all(target_os = "windows", feature="dx11"))]
fn create_clip_program(device: &mut Device, filename: &str) -> Box<ClipProgram> {
    let vs = get_shader_source(filename, ".vert.fx");
    let ps = get_shader_source(filename, ".frag.fx");
    Box::new(device.create_clip_program(vs.as_slice(), ps.as_slice()))
}

#[cfg(not(feature = "dx11"))]
fn create_blur_program(device: &mut Device, filename: &str) -> Box<BlurProgram> {
    let vs = get_shader_source(filename, ".vert");
    let ps = get_shader_source(filename, ".frag");
    Box::new(device.create_blur_program(vs.as_slice(), ps.as_slice()))
}

#[cfg(all(target_os = "windows", feature="dx11"))]
fn create_blur_program(device: &mut Device, filename: &str) -> Box<BlurProgram> {
    let vs = get_shader_source(filename, ".vert.fx");
    let ps = get_shader_source(filename, ".frag.fx");
    Box::new(device.create_blur_program(vs.as_slice(), ps.as_slice()))
}

#[cfg(not(feature = "dx11"))]
pub fn create_debug_color_program(device: &mut Device, filename: &str) -> DebugColorProgram {
    let vs = get_shader_source(filename, ".vert");
    let ps = get_shader_source(filename, ".frag");
    device.create_debug_color_program(vs.as_slice(), ps.as_slice())
}

#[cfg(all(target_os = "windows", feature="dx11"))]
pub fn create_debug_color_program(device: &mut Device, filename: &str) -> DebugColorProgram {
    let vs = get_shader_source(filename, ".vert.fx");
    let ps = get_shader_source(filename, ".frag.fx");
    device.create_debug_color_program(vs.as_slice(), ps.as_slice())
}

#[cfg(not(feature = "dx11"))]
pub fn create_clear_program(device: &mut Device, filename: &str) -> Box<DebugColorProgram> {
    let vs = get_shader_source(filename, ".vert");
    let ps = get_shader_source(filename, ".frag");
    Box::new(device.create_clear_program(vs.as_slice(), ps.as_slice()))
}

#[cfg(all(target_os = "windows", feature="dx11"))]
pub fn create_clear_program(device: &mut Device, filename: &str) -> Box<DebugColorProgram> {
    let vs = get_shader_source(filename, ".vert.fx");
    let ps = get_shader_source(filename, ".frag.fx");
    Box::new(device.create_clear_program(vs.as_slice(), ps.as_slice()))
}

#[cfg(not(feature = "dx11"))]
pub fn create_debug_font_program(device: &mut Device, filename: &str) -> DebugFontProgram {
    let vs = get_shader_source(filename, ".vert");
    let ps = get_shader_source(filename, ".frag");
    device.create_debug_font_program(vs.as_slice(), ps.as_slice())
}

 #[cfg(all(target_os = "windows", feature="dx11"))]
pub fn create_debug_font_program(device: &mut Device, filename: &str) -> DebugFontProgram {
    let vs = get_shader_source(filename, ".vert.fx");
    let ps = get_shader_source(filename, ".frag.fx");
    device.create_debug_font_program(vs.as_slice(), ps.as_slice())
 }

fn get_shader_source(filename: &str, extension: &str) -> Vec<u8> {
    use std::io::Read;
    let path_str = format!("{}/{}{}", env!("OUT_DIR"), filename, extension);
    let mut file = File::open(path_str).unwrap();
    let mut shader = Vec::new();
    file.read_to_end(&mut shader);
    shader
}

#[cfg(all(target_os = "windows", feature="dx11"))]
pub fn transform_projection(projection: Transform3D<f32>) -> Transform3D<f32> {
    let transform = Transform3D::row_major(1.0, 0.0, 0.0, 0.0,
                                           0.0, 1.0, 0.0, 0.0,
                                           0.0, 0.0, 0.5, 0.5,
                                           0.0, 0.0, 0.0, 1.0);
    transform.post_mul(&Transform3D::from_array(projection.to_column_major_array()))
}

#[cfg(not(feature = "dx11"))]
pub fn transform_projection(projection: Transform3D<f32>) -> Transform3D<f32> {
    projection
}

#[cfg(all(target_os = "windows", feature="dx11"))]
pub fn alpha_transform_projection(projection: Transform3D<f32>) -> Transform3D<f32> {
    let transform = Transform3D::row_major(1.0, 0.0, 0.0, 0.0,
                                           0.0, -1.0, 0.0, 0.0,
                                           0.0, 0.0, 1.0, 0.0,
                                           0.0, 0.0, 0.0, 1.0);
    transform.post_mul(&Transform3D::from_array(projection.to_column_major_array()))
}

#[cfg(not(feature = "dx11"))]
pub fn alpha_transform_projection(projection: Transform3D<f32>) -> Transform3D<f32> {
    projection
}

struct FrameOutput {
    last_access: FrameId,
}

/// The renderer is responsible for submitting to the GPU the work prepared by the
/// RenderBackend.
pub struct Renderer {
    result_rx: Receiver<ResultMsg>,
    debug_server: DebugServer,
    device: Device,
    pending_texture_updates: Vec<TextureUpdateList>,
    pending_gpu_cache_updates: Vec<GpuCacheUpdateList>,
    pending_shader_updates: Vec<PathBuf>,
    current_frame: Option<RendererFrame>,

    // These are "cache shaders". These shaders are used to
    // draw intermediate results to cache targets. The results
    // of these shaders are then used by the primitive shaders.
    cs_text_run: Option<Box<Program>>,
    cs_line: Option<Box<Program>>,
    cs_blur_a8: Option<Box<BlurProgram>>,
    cs_blur_rgba8: Option<Box<BlurProgram>>,

    brush_mask: Option<BrushProgramPair>,
    brush_image_rgba8: Option<BrushProgramPair>,
    brush_image_a8: Option<BrushProgramPair>,

    /// These are "cache clip shaders". These shaders are used to
    /// draw clip instances into the cached clip mask. The results
    /// of these shaders are also used by the primitive shaders.
    cs_clip_rectangle: Option<Box<ClipProgram>>,
    cs_clip_image: Option<Box<ClipProgram>>,
    cs_clip_border: Option<Box<ClipProgram>>,

    // The are "primitive shaders". These shaders draw and blend
    // final results on screen. They are aware of tile boundaries.
    // Most draw directly to the framebuffer, but some use inputs
    // from the cache shaders to draw. Specifically, the box
    // shadow primitive shader stretches the box shadow cache
    // output, and the cache_image shader blits the results of
    // a cache shader (e.g. blur) to the screen.
    ps_rectangle: Option<ProgramPair>,
    ps_rectangle_clip: Option<ProgramPair>,
    ps_text_run: Option<TextProgramPair>,
    ps_image: Option<ProgramPair>,
    ps_yuv_image: Option<Vec<ProgramPair>>,
    ps_border_corner: Option<ProgramPair>,
    ps_border_edge: Option<ProgramPair>,
    ps_gradient: Option<ProgramPair>,
    ps_angle_gradient: Option<ProgramPair>,
    ps_radial_gradient: Option<ProgramPair>,
    ps_line: Option<ProgramPair>,

    ps_blend: Option<Box<Program>>,
    ps_hw_composite: Option<Box<Program>>,
    ps_split_composite: Option<Box<Program>>,
    ps_composite: Option<Box<Program>>,

    ps_clear: Option<Box<DebugColorProgram>>,

    max_texture_size: u32,

    max_recorded_profiles: usize,
    clear_framebuffer: bool,
    clear_color: ColorF,
    enable_clear_scissor: bool,
    debug: DebugRenderer,
    debug_flags: DebugFlags,
    enable_batcher: bool,
    backend_profile_counters: BackendProfileCounters,
    profile_counters: RendererProfileCounters,
    profiler: Profiler,
    last_time: u64,

    color_render_targets: Vec<TextureId>,
    alpha_render_targets: Vec<TextureId>,

    gpu_profile: GpuProfiler<GpuProfileTag>,

    layer_texture: VertexDataTexture,
    render_task_texture: VertexDataTexture,
    gpu_cache_texture: CacheTexture,

    pipeline_epoch_map: FastHashMap<PipelineId, Epoch>,

    // Manages and resolves source textures IDs to real texture IDs.
    texture_resolver: SourceTextureResolver,

    /// Optional trait object that allows the client
    /// application to provide external buffers for image data.
    external_image_handler: Option<Box<ExternalImageHandler>>,

    /// Optional trait object that allows the client
    /// application to provide a texture handle to
    /// copy the WR output to.
    output_image_handler: Option<Box<OutputImageHandler>>,

    // Currently allocated FBOs for output frames.
    output_targets: FastHashMap<u32, FrameOutput>,

    renderer_errors: Vec<RendererError>,

    /// List of profile results from previous frames. Can be retrieved
    /// via get_frame_profiles().
    cpu_profiles: VecDeque<CpuProfile>,
    gpu_profiles: VecDeque<GpuProfile>,
    enable_dithering: bool,
}

#[derive(Debug)]
pub enum RendererError {
    Shader(ShaderError),
    Thread(std::io::Error),
    MaxTextureSize,
}

impl From<ShaderError> for RendererError {
    fn from(err: ShaderError) -> Self {
        RendererError::Shader(err)
    }
}

impl From<std::io::Error> for RendererError {
    fn from(err: std::io::Error) -> Self {
        RendererError::Thread(err)
    }
}

impl Renderer {
    /// Initializes webrender and creates a `Renderer` and `RenderApiSender`.
    ///
    /// # Examples
    /// Initializes a `Renderer` with some reasonable values. For more information see
    /// [`RendererOptions`][rendereroptions].
    ///
    /// ```rust,ignore
    /// # use webrender::renderer::Renderer;
    /// # use std::path::PathBuf;
    /// let opts = webrender::RendererOptions {
    ///    device_pixel_ratio: 1.0,
    ///    resource_override_path: None,
    ///    enable_aa: false,
    /// };
    /// let (renderer, sender) = Renderer::new(opts);
    /// ```
    /// [rendereroptions]: struct.RendererOptions.html
    pub fn new(
        notifier: Box<RenderNotifier>,
        mut options: RendererOptions,
        mut params: DeviceInitParams,
    ) -> Result<(Renderer, RenderApiSender), RendererError> {
        let (api_tx, api_rx) = try!{ channel::msg_channel() };
        let (payload_tx, payload_rx) = try!{ channel::payload_channel() };
        let (result_tx, result_rx) = channel();

        let debug_server = DebugServer::new(api_tx.clone());

        let file_watch_handler = FileWatcher {
            result_tx: result_tx.clone(),
            notifier: notifier.clone(),
        };

        let mut device = Device::new(
            options.resource_override_path.clone(),
            params,
            Box::new(file_watch_handler),
        );

        let device_max_size = device.max_texture_size();
        // 512 is the minimum that the texture cache can work with.
        // Broken GL contexts can return a max texture size of zero (See #1260). Better to
        // gracefully fail now than panic as soon as a texture is allocated.
        let min_texture_size = 512;
        if device_max_size < min_texture_size {
            return Err(RendererError::MaxTextureSize);
        }
        let max_device_size = cmp::max(
            cmp::min(
                device_max_size,
                options.max_texture_size.unwrap_or(device_max_size),
            ),
            min_texture_size,
        );

        register_thread_with_profiler("Compositor".to_owned());

        // device-pixel ratio doesn't matter here - we are just creating resources.
        device.begin_frame(1.0);

        let texture_cache = TextureCache::new(max_device_size);
        let max_texture_size = texture_cache.max_texture_size();
        let backend_profile_counters = BackendProfileCounters::new();
        let debug_renderer = DebugRenderer::new(&mut device);
        let texture_resolver = SourceTextureResolver::new();
        let layer_texture = VertexDataTexture::new(TextureSampler::Layers);
        let render_task_texture = VertexDataTexture::new(TextureSampler::RenderTasks);

        device.end_frame();

        let backend_notifier = notifier.clone();

        let default_font_render_mode = match (options.enable_aa, options.enable_subpixel_aa) {
            (true, true) => FontRenderMode::Subpixel,
            (true, false) => FontRenderMode::Alpha,
            (false, _) => FontRenderMode::Mono,
        };

        let config = FrameBuilderConfig {
            enable_scrollbars: options.enable_scrollbars,
            default_font_render_mode,
            debug: options.debug,
        };

        let device_pixel_ratio = options.device_pixel_ratio;
        let debug_flags = options.debug_flags;
        let payload_tx_for_backend = payload_tx.clone();
        let recorder = options.recorder;
        let worker_config = ThreadPoolConfig::new()
            .thread_name(|idx| format!("WebRender:Worker#{}", idx))
            .start_handler(|idx| {
                register_thread_with_profiler(format!("WebRender:Worker#{}", idx));
            });
        let workers = options
            .workers
            .take()
            .unwrap_or_else(|| Arc::new(ThreadPool::new(worker_config).unwrap()));
        let enable_render_on_scroll = options.enable_render_on_scroll;

        let blob_image_renderer = options.blob_image_renderer.take();
        try!{ thread::Builder::new().name("RenderBackend".to_string()).spawn(move || {
            let mut backend = RenderBackend::new(api_rx,
                                                 payload_rx,
                                                 payload_tx_for_backend,
                                                 result_tx,
                                                 device_pixel_ratio,
                                                 texture_cache,
                                                 workers,
                                                 backend_notifier,
                                                 config,
                                                 recorder,
                                                 blob_image_renderer,
                                                 enable_render_on_scroll);
            backend.run(backend_profile_counters);
        })};

        let gpu_cache_texture = CacheTexture::new(&mut device);

        let gpu_profile = GpuProfiler::new();

        let renderer = Renderer {
            result_rx,
            debug_server,
            device,
            current_frame: None,
            pending_texture_updates: Vec::new(),
            pending_gpu_cache_updates: Vec::new(),
            pending_shader_updates: Vec::new(),
            cs_text_run: None,
            cs_line: None,
            cs_blur_a8: None,
            cs_blur_rgba8: None,
            brush_mask: None,
            brush_image_rgba8: None,
            brush_image_a8: None,
            cs_clip_rectangle: None,
            cs_clip_border: None,
            cs_clip_image: None,
            ps_rectangle: None,
            ps_rectangle_clip: None,
            ps_text_run: None,
            ps_image: None,
            ps_yuv_image: None,
            ps_border_corner: None,
            ps_border_edge: None,
            ps_gradient: None,
            ps_angle_gradient: None,
            ps_radial_gradient: None,
            ps_line: None,
            ps_blend: None,
            ps_hw_composite: None,
            ps_split_composite: None,
            ps_composite: None,
            ps_clear: None,
            debug: debug_renderer,
            debug_flags,
            enable_batcher: options.enable_batcher,
            backend_profile_counters: BackendProfileCounters::new(),
            profile_counters: RendererProfileCounters::new(),
            profiler: Profiler::new(),
            max_texture_size: max_texture_size,
            max_recorded_profiles: options.max_recorded_profiles,
            clear_framebuffer: options.clear_framebuffer,
            clear_color: options.clear_color,
            enable_clear_scissor: options.enable_clear_scissor,
            last_time: 0,
            color_render_targets: Vec::new(),
            alpha_render_targets: Vec::new(),
            gpu_profile,
            layer_texture,
            render_task_texture,
            pipeline_epoch_map: FastHashMap::default(),
            external_image_handler: None,
            output_image_handler: None,
            output_targets: FastHashMap::default(),
            cpu_profiles: VecDeque::new(),
            gpu_profiles: VecDeque::new(),
            gpu_cache_texture,
            texture_resolver,
            renderer_errors: Vec::new(),
            enable_dithering: options.enable_dithering,
        };

        let sender = RenderApiSender::new(api_tx, payload_tx);
        Ok((renderer, sender))
    }

    pub fn get_max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    pub fn get_graphics_api_info(&self) -> GraphicsApiInfo {
        GraphicsApiInfo {
            kind: GraphicsApi::OpenGL,
            version: "TODO graphics api version".to_owned(),
            renderer: "TODO graphics api renderer".to_owned(),
        }
    }

    fn get_yuv_shader_index(
        buffer_kind: ImageBufferKind,
        format: YuvFormat,
        color_space: YuvColorSpace,
        modulo: usize,
    ) -> usize {
        (((buffer_kind as usize) * YUV_FORMATS.len() + (format as usize)) * YUV_COLOR_SPACES.len() +
            (color_space as usize)) % modulo
    }

    /// Returns the Epoch of the current frame in a pipeline.
    pub fn current_epoch(&self, pipeline_id: PipelineId) -> Option<Epoch> {
        self.pipeline_epoch_map.get(&pipeline_id).cloned()
    }

    /// Returns a HashMap containing the pipeline ids that have been received by the renderer and
    /// their respective epochs since the last time the method was called.
    pub fn flush_rendered_epochs(&mut self) -> FastHashMap<PipelineId, Epoch> {
        mem::replace(&mut self.pipeline_epoch_map, FastHashMap::default())
    }

    /// Processes the result queue.
    ///
    /// Should be called before `render()`, as texture cache updates are done here.
    pub fn update(&mut self) {
        profile_scope!("update");

        // Pull any pending results and return the most recent.
        while let Ok(msg) = self.result_rx.try_recv() {
            match msg {
                ResultMsg::NewFrame(
                    _document_id,
                    mut frame,
                    texture_update_list,
                    profile_counters,
                ) => {
                    //TODO: associate `document_id` with target window
                    self.pending_texture_updates.push(texture_update_list);
                    if let Some(ref mut frame) = frame.frame {
                        // TODO(gw): This whole message / Frame / RendererFrame stuff
                        //           is really messy and needs to be refactored!!
                        if let Some(update_list) = frame.gpu_cache_updates.take() {
                            self.pending_gpu_cache_updates.push(update_list);
                        }
                    }
                    self.backend_profile_counters = profile_counters;

                    // Update the list of available epochs for use during reftests.
                    // This is a workaround for https://github.com/servo/servo/issues/13149.
                    for (pipeline_id, epoch) in &frame.pipeline_epoch_map {
                        self.pipeline_epoch_map.insert(*pipeline_id, *epoch);
                    }

                    self.current_frame = Some(frame);
                }
                ResultMsg::UpdateResources {
                    updates,
                    cancel_rendering,
                } => {
                    self.pending_texture_updates.push(updates);
                    self.update_texture_cache();
                    // If we receive a NewFrame message followed by this one within
                    // the same update we need ot cancel the frame because we might
                    // have deleted the resources in use in the frame dut to a memory
                    // pressure event.
                    if cancel_rendering {
                        self.current_frame = None;
                    }
                }
                ResultMsg::RefreshShader(path) => {
                    self.pending_shader_updates.push(path);
                }
                ResultMsg::DebugOutput(output) => match output {
                    DebugOutput::FetchDocuments(string) |
                    DebugOutput::FetchClipScrollTree(string) => {
                        self.debug_server.send(string);
                    }
                },
                ResultMsg::DebugCommand(command) => {
                    self.handle_debug_command(command);
                }
            }
        }
    }

    #[cfg(not(feature = "debugger"))]
    fn get_passes_for_debugger(&self) -> String {
        // Avoid unused param warning.
        let _ = &self.debug_server;
        String::new()
    }

    #[cfg(feature = "debugger")]
    fn get_passes_for_debugger(&self) -> String {
        let mut debug_passes = debug_server::PassList::new();

        if let Some(frame) = self.current_frame
            .as_ref()
            .and_then(|frame| frame.frame.as_ref())
        {
            for pass in &frame.passes {
                let mut debug_pass = debug_server::Pass::new();

                for target in &pass.alpha_targets.targets {
                    let mut debug_target = debug_server::Target::new("A8");

                    debug_target.add(
                        debug_server::BatchKind::Clip,
                        "Clear",
                        target.clip_batcher.border_clears.len(),
                    );
                    debug_target.add(
                        debug_server::BatchKind::Clip,
                        "Borders",
                        target.clip_batcher.borders.len(),
                    );
                    debug_target.add(
                        debug_server::BatchKind::Cache,
                        "Vertical Blur",
                        target.vertical_blurs.len(),
                    );
                    debug_target.add(
                        debug_server::BatchKind::Cache,
                        "Horizontal Blur",
                        target.horizontal_blurs.len(),
                    );
                    debug_target.add(
                        debug_server::BatchKind::Clip,
                        "Rectangles",
                        target.clip_batcher.rectangles.len(),
                    );
                    debug_target.add(
                        debug_server::BatchKind::Cache,
                        "Rectangle Brush",
                        target.rect_cache_prims.len(),
                    );
                    for (_, items) in target.clip_batcher.images.iter() {
                        debug_target.add(debug_server::BatchKind::Clip, "Image mask", items.len());
                    }

                    debug_pass.add(debug_target);
                }

                for target in &pass.color_targets.targets {
                    let mut debug_target = debug_server::Target::new("RGBA8");

                    debug_target.add(
                        debug_server::BatchKind::Cache,
                        "Vertical Blur",
                        target.vertical_blurs.len(),
                    );
                    debug_target.add(
                        debug_server::BatchKind::Cache,
                        "Horizontal Blur",
                        target.horizontal_blurs.len(),
                    );
                    for (_, batch) in &target.text_run_cache_prims {
                        debug_target.add(
                            debug_server::BatchKind::Cache,
                            "Text Shadow",
                            batch.len(),
                        );
                    }
                    debug_target.add(
                        debug_server::BatchKind::Cache,
                        "Lines",
                        target.line_cache_prims.len(),
                    );

                    for batch in target
                        .alpha_batcher
                        .batch_list
                        .opaque_batch_list
                        .batches
                        .iter()
                        .rev()
                    {
                        debug_target.add(
                            debug_server::BatchKind::Opaque,
                            batch.key.kind.debug_name(),
                            batch.instances.len(),
                        );
                    }

                    for batch in &target.alpha_batcher.batch_list.alpha_batch_list.batches {
                        debug_target.add(
                            debug_server::BatchKind::Alpha,
                            batch.key.kind.debug_name(),
                            batch.instances.len(),
                        );
                    }

                    debug_pass.add(debug_target);
                }

                debug_passes.add(debug_pass);
            }
        }

        serde_json::to_string(&debug_passes).unwrap()
    }

    fn handle_debug_command(&mut self, command: DebugCommand) {
        match command {
            DebugCommand::EnableProfiler(enable) => if enable {
                self.debug_flags.insert(DebugFlags::PROFILER_DBG);
            } else {
                self.debug_flags.remove(DebugFlags::PROFILER_DBG);
            },
            DebugCommand::EnableTextureCacheDebug(enable) => if enable {
                self.debug_flags.insert(DebugFlags::TEXTURE_CACHE_DBG);
            } else {
                self.debug_flags.remove(DebugFlags::TEXTURE_CACHE_DBG);
            },
            DebugCommand::EnableRenderTargetDebug(enable) => if enable {
                self.debug_flags.insert(DebugFlags::RENDER_TARGET_DBG);
            } else {
                self.debug_flags.remove(DebugFlags::RENDER_TARGET_DBG);
            },
            DebugCommand::EnableAlphaRectsDebug(enable) => if enable {
                self.debug_flags.insert(DebugFlags::ALPHA_PRIM_DBG);
            } else {
                self.debug_flags.remove(DebugFlags::ALPHA_PRIM_DBG);
            },
            DebugCommand::FetchDocuments => {}
            DebugCommand::FetchClipScrollTree => {}
            DebugCommand::FetchPasses => {
                let json = self.get_passes_for_debugger();
                self.debug_server.send(json);
            }
        }
    }

    /// Set a callback for handling external images.
    pub fn set_external_image_handler(&mut self, handler: Box<ExternalImageHandler>) {
        self.external_image_handler = Some(handler);
    }

    /// Set a callback for handling external outputs.
    pub fn set_output_image_handler(&mut self, handler: Box<OutputImageHandler>) {
        self.output_image_handler = Some(handler);
    }

    /// Retrieve (and clear) the current list of recorded frame profiles.
    pub fn get_frame_profiles(&mut self) -> (Vec<CpuProfile>, Vec<GpuProfile>) {
        let cpu_profiles = self.cpu_profiles.drain(..).collect();
        let gpu_profiles = self.gpu_profiles.drain(..).collect();
        (cpu_profiles, gpu_profiles)
    }

    /// Renders the current frame.
    ///
    /// A Frame is supplied by calling [`generate_frame()`][genframe].
    /// [genframe]: ../../webrender_api/struct.DocumentApi.html#method.generate_frame
    pub fn render(&mut self, framebuffer_size: DeviceUintSize) -> Result<(), Vec<RendererError>> {
        profile_scope!("render");

        if let Some(mut frame) = self.current_frame.take() {
            if let Some(ref mut frame) = frame.frame {
                let mut profile_timers = RendererProfileTimers::new();
                let mut profile_samplers = Vec::new();

                {
                    //Note: avoiding `self.gpu_profile.add_marker` - it would block here
                    let _gm = GpuMarker::new("build samples");
                    // Block CPU waiting for last frame's GPU profiles to arrive.
                    // In general this shouldn't block unless heavily GPU limited.
                    if let Some((gpu_frame_id, timers, samplers)) = self.gpu_profile.build_samples()
                    {
                        if self.max_recorded_profiles > 0 {
                            while self.gpu_profiles.len() >= self.max_recorded_profiles {
                                self.gpu_profiles.pop_front();
                            }
                            self.gpu_profiles
                                .push_back(GpuProfile::new(gpu_frame_id, &timers));
                        }
                        profile_timers.gpu_samples = timers;
                        profile_samplers = samplers;
                    }
                }

                let cpu_frame_id = profile_timers.cpu_time.profile(|| {
                    let cpu_frame_id = {
                        let _gm = GpuMarker::new("begin frame");
                        let frame_id = self.device.begin_frame(frame.device_pixel_ratio);
                        self.gpu_profile.begin_frame(frame_id);

                        self.update_texture_cache();

                        self.update_gpu_cache(frame);

                        frame_id
                    };

                    self.draw_tile_frame(frame, framebuffer_size);
                    self.flush();

                    self.gpu_profile.end_frame();
                    cpu_frame_id
                });

                let current_time = precise_time_ns();
                let ns = current_time - self.last_time;
                self.profile_counters.frame_time.set(ns);

                if self.max_recorded_profiles > 0 {
                    while self.cpu_profiles.len() >= self.max_recorded_profiles {
                        self.cpu_profiles.pop_front();
                    }
                    let cpu_profile = CpuProfile::new(
                        cpu_frame_id,
                        self.backend_profile_counters.total_time.get(),
                        profile_timers.cpu_time.get(),
                        self.profile_counters.draw_calls.get(),
                    );
                    self.cpu_profiles.push_back(cpu_profile);
                }

                if self.debug_flags.contains(DebugFlags::PROFILER_DBG) {
                    let screen_fraction = 1.0 / //TODO: take device/pixel ratio into equation?
                        (framebuffer_size.width as f32 * framebuffer_size.height as f32);
                    self.profiler.draw_profile(
                        &mut self.device,
                        &frame.profile_counters,
                        &self.backend_profile_counters,
                        &self.profile_counters,
                        &mut profile_timers,
                        &profile_samplers,
                        screen_fraction,
                        &mut self.debug,
                    );
                }

                self.profile_counters.reset();
                self.profile_counters.frame_counter.inc();

                let debug_size = DeviceUintSize::new(
                    framebuffer_size.width as u32,
                    framebuffer_size.height as u32,
                );
                self.debug.render(&mut self.device, &debug_size);
                self.flush();
                {
                    let _gm = GpuMarker::new("end frame");
                    self.device.end_frame();
                }
                self.last_time = current_time;
            }

            // Restore frame - avoid borrow checker!
            self.current_frame = Some(frame);
        }
        if !self.renderer_errors.is_empty() {
            let errors = mem::replace(&mut self.renderer_errors, Vec::new());
            return Err(errors);
        }
        Ok(())
    }

    fn flush(&mut self) {
        self.device.flush();
        self.cs_text_run.as_mut().map_or((), |p| p.reset_upload_offset());;
        self.cs_line.as_mut().map_or((), |p| p.reset_upload_offset());;
        self.cs_blur_a8.as_mut().map_or((), |p| p.reset_upload_offset());;
        self.cs_blur_rgba8.as_mut().map_or((), |p| p.reset_upload_offset());;
        self.brush_mask.as_mut().map_or((), |p| p.reset_upload_offset());;
        self.brush_image_rgba8.as_mut().map_or((), |p| p.reset_upload_offset());;
        self.brush_image_a8.as_mut().map_or((), |p| p.reset_upload_offset());;
        self.cs_clip_rectangle.as_mut().map_or((), |p| p.reset_upload_offset());;
        self.cs_clip_border.as_mut().map_or((), |p| p.reset_upload_offset());;
        self.cs_clip_image.as_mut().map_or((), |p| p.reset_upload_offset());;
        self.ps_rectangle.as_mut().map_or((), |p| p.reset_upload_offset());
        self.ps_rectangle_clip.as_mut().map_or((), |p| p.reset_upload_offset());
        self.ps_text_run.as_mut().map_or((), |p| p.reset_upload_offset());
        self.ps_image.as_mut().map_or((), |p| p.reset_upload_offset());
        self.ps_border_corner.as_mut().map_or((), |p| p.reset_upload_offset());
        self.ps_border_edge.as_mut().map_or((), |p| p.reset_upload_offset());
        self.ps_gradient.as_mut().map_or((), |p| p.reset_upload_offset());();
        self.ps_angle_gradient.as_mut().map_or((), |p| p.reset_upload_offset());
        self.ps_radial_gradient.as_mut().map_or((), |p| p.reset_upload_offset());
        self.ps_blend.as_mut().map_or((), |p| p.reset_upload_offset());
        self.ps_hw_composite.as_mut().map_or((), |p| p.reset_upload_offset());
        self.ps_split_composite.as_mut().map_or((), |p| p.reset_upload_offset());
        self.ps_composite.as_mut().map_or((), |p| p.reset_upload_offset());
        self.ps_line.as_mut().map_or((), |p| p.reset_upload_offset());
        if let Some(ref mut ps_yuv_image) = self.ps_yuv_image {
            for mut program in ps_yuv_image {
                program.reset_upload_offset();
            }
        }
    }

    pub fn layers_are_bouncing_back(&self) -> bool {
        match self.current_frame {
            None => false,
            Some(ref current_frame) => !current_frame.layers_bouncing_back.is_empty(),
        }
    }

    fn update_gpu_cache(&mut self, frame: &mut Frame) {
        let _gm = GpuMarker::new("gpu cache update");
        for update_list in self.pending_gpu_cache_updates.drain(..) {
            self.gpu_cache_texture
                .update(&mut self.device, &update_list);
        }
        self.update_deferred_resolves(frame);
        self.gpu_cache_texture.flush(&mut self.device);
    }

    fn update_texture_cache(&mut self) {
        let _gm = GpuMarker::new("texture cache update");
        let mut pending_texture_updates = mem::replace(&mut self.pending_texture_updates, vec![]);

        for update_list in pending_texture_updates.drain(..) {
            for update in update_list.updates {
                match update.op {
                    TextureUpdateOp::Create {
                        width,
                        height,
                        layer_count,
                        format,
                        filter,
                        mode,
                    } => {
                        let CacheTextureId(cache_texture_index) = update.id;
                        if self.texture_resolver.cache_texture_map.len() == cache_texture_index {
                            // Create a new native texture, as requested by the texture cache.
                            let texture = self.device.create_image_texture(width, height, layer_count, filter, format);
                            self.texture_resolver.cache_texture_map.push(texture);
                        }
                    }
                    TextureUpdateOp::Update {
                        rect,
                        source,
                        stride,
                        layer_index,
                        offset,
                    } => {
                        let texture = &self.texture_resolver.cache_texture_map[update.id.0];

                        match source {
                            TextureUpdateSource::Bytes { data }  => {
                                self.device.update_image_data(
                                    &data[offset as usize..],
                                    texture,
                                    rect.origin.x,
                                    rect.origin.y,
                                    rect.size.width,
                                    rect.size.height,
                                    layer_index,
                                    stride,
                                    0,
                                );
                            }
                            TextureUpdateSource::External { id, channel_index } => {
                                /*let handler = self.external_image_handler
                                                  .as_mut()
                                                  .expect("Found external image, but no handler set!");
                                match handler.lock(id, channel_index).source {
                                    ExternalImageSource::RawData(data) => {
                                        self.device.update_pbo_data(&data[offset as usize ..]);
                                    }
                                    ExternalImageSource::Invalid => {
                                        // Create a local buffer to fill the pbo.
                                        let bpp = texture.get_bpp();
                                        let width = stride.unwrap_or(rect.size.width * bpp);
                                        let total_size = width * rect.size.height;
                                        // WR haven't support RGBAF32 format in texture_cache, so
                                        // we use u8 type here.
                                        let dummy_data: Vec<u8> = vec![255; total_size as usize];
                                        self.device.update_pbo_data(&dummy_data);
                                    }
                                    _ => panic!("No external buffer found"),
                                };
                                handler.unlock(id, channel_index);*/
                            }
                        }
                    }
                    TextureUpdateOp::Free => {
                        let texture = &mut self.texture_resolver.cache_texture_map[update.id.0];
                        self.device.free_texture_storage(texture);
                    }
                }
            }
        }
    }

    fn submit_batch(
        &mut self,
        key: &BatchKey,
        instances: &[PrimitiveInstance],
        projection: &Transform3D<f32>,
        render_tasks: &RenderTaskTree,
        render_target: Option<(&TextureId, i32)>,
        target_dimensions: DeviceUintSize,
        enable_depth_write: bool,
    ) {
        let (program, marker) = match key.kind {
            BatchKind::Composite { .. } => {
                (self.ps_composite.get_or_insert(create_program(&mut self.device, "ps_composite")) as &mut BindDraw, GPU_TAG_PRIM_COMPOSITE)
            }
            BatchKind::HardwareComposite => {
                (self.ps_hw_composite.get_or_insert(create_program(&mut self.device, "ps_hardware_composite")) as &mut BindDraw, GPU_TAG_PRIM_COMPOSITE)
            }
            BatchKind::SplitComposite => {
                (self.ps_split_composite.get_or_insert(create_program(&mut self.device, "ps_split_composite")) as &mut BindDraw, GPU_TAG_PRIM_SPLIT_COMPOSITE)
            }
            BatchKind::Blend => {
                (self.ps_blend.get_or_insert(create_program(&mut self.device, "ps_blend")) as &mut BindDraw, GPU_TAG_PRIM_BLEND)
            }
            BatchKind::Brush(brush_kind) => {
                match brush_kind {
                    BrushBatchKind::Image(target_kind) => {
                        let shader = match target_kind {
                            RenderTargetKind::Alpha => {
                                let brush_image_a8 = self.brush_image_a8.get_or_insert(create_brush_programs(&mut self.device, "brush_image_alpha_target"));
                                brush_image_a8.get(key.blend_mode)
                            },
                            RenderTargetKind::Color => {
                                let brush_image_rgba8 = self.brush_image_rgba8.get_or_insert(create_brush_programs(&mut self.device, "brush_image_color_target"));
                                brush_image_rgba8.get(key.blend_mode)
                            },
                        };
                        (shader  as &mut BindDraw, GPU_TAG_BRUSH_IMAGE)
                    }
                }
            }
            BatchKind::Transformable(transform_kind, batch_kind) => match batch_kind {
                TransformBatchKind::Rectangle(needs_clipping) => {
                    debug_assert!(
                        !needs_clipping || match key.blend_mode {
                            BlendMode::Alpha |
                            BlendMode::PremultipliedAlpha |
                            BlendMode::Subpixel |
                            BlendMode::Max |
                            BlendMode::Multiply => true,
                            BlendMode::None => false,
                        }
                    );
                    if needs_clipping {
                        let ps_rectangle_clip = self.ps_rectangle.get_or_insert(create_prim_programs(&mut self.device, "ps_rectangle_clip"));
                        (ps_rectangle_clip.get(transform_kind)  as &mut BindDraw, GPU_TAG_PRIM_RECT)
                    } else {
                        let ps_rectangle = self.ps_rectangle.get_or_insert(create_prim_programs(&mut self.device, "ps_rectangle"));
                        (ps_rectangle.get(transform_kind)  as &mut BindDraw, GPU_TAG_PRIM_RECT)
                    }
                }
                TransformBatchKind::Line => {
                    let ps_line = self.ps_line.get_or_insert(create_prim_programs(&mut self.device, "ps_line"));
                    (ps_line.get(transform_kind)  as &mut BindDraw, GPU_TAG_PRIM_LINE)
                }
                TransformBatchKind::TextRun(..) => {
                    unreachable!("bug: text batches are special cased");
                }
                TransformBatchKind::Image(image_buffer_kind) => {
                    let ps_image = self.ps_image.get_or_insert(create_prim_programs(&mut self.device, "ps_image"));
                    (ps_image.get(transform_kind)  as &mut BindDraw, GPU_TAG_PRIM_IMAGE)
                }
                TransformBatchKind::YuvImage(image_buffer_kind, format, color_space) => {
                    let ps_yuv_image = self.ps_yuv_image.get_or_insert(
                        vec![create_prim_programs(&mut self.device, "ps_yuv_image_nv12"),
                             create_prim_programs(&mut self.device, "ps_yuv_image_nv12_yuv_rec709"),
                             create_prim_programs(&mut self.device, "ps_yuv_image"),
                             create_prim_programs(&mut self.device, "ps_yuv_image_yuv_rec709"),
                             create_prim_programs(&mut self.device, "ps_yuv_image_interleaved_y_cb_cr"),
                             create_prim_programs(&mut self.device, "ps_yuv_image_interleaved_y_cb_cr_yuv_rec709")]
                    );
                    let shader_index = Renderer::get_yuv_shader_index(image_buffer_kind,
                                                                      format,
                                                                      color_space,
                                                                      ps_yuv_image.len());
                    (ps_yuv_image[shader_index].get(transform_kind)  as &mut BindDraw, GPU_TAG_PRIM_YUV_IMAGE)
                }
                TransformBatchKind::BorderCorner => {
                    let ps_border_corner = self.ps_border_corner.get_or_insert(create_prim_programs(&mut self.device, "ps_border_corner"));
                    (ps_border_corner.get(transform_kind)  as &mut BindDraw, GPU_TAG_PRIM_BORDER_CORNER)
                }
                TransformBatchKind::BorderEdge => {
                    let ps_border_edge = self.ps_border_edge.get_or_insert(create_prim_programs(&mut self.device, "ps_border_edge"));
                    (ps_border_edge.get(transform_kind)  as &mut BindDraw, GPU_TAG_PRIM_BORDER_EDGE)
                }
                TransformBatchKind::AlignedGradient => {
                    let ps_gradient = {
                        if self.enable_dithering {
                            self.ps_gradient.get_or_insert(create_prim_programs(&mut self.device, "ps_gradient_dithering"))
                        } else {
                            self.ps_gradient.get_or_insert(create_prim_programs(&mut self.device, "ps_gradient"))
                        }
                    };
                    (ps_gradient.get(transform_kind)  as &mut BindDraw, GPU_TAG_PRIM_GRADIENT)
                }
                TransformBatchKind::AngleGradient => {
                    let ps_angle_gradient = {
                        if self.enable_dithering {
                            self.ps_angle_gradient.get_or_insert(create_prim_programs(&mut self.device, "ps_angle_gradient_dithering"))
                        } else {
                            self.ps_angle_gradient.get_or_insert(create_prim_programs(&mut self.device, "ps_angle_gradient"))
                        }
                    };
                    (ps_angle_gradient.get(transform_kind)  as &mut BindDraw, GPU_TAG_PRIM_ANGLE_GRADIENT)
                }
                TransformBatchKind::RadialGradient => {
                    let ps_radial_gradient = {
                        if self.enable_dithering {
                            self.ps_radial_gradient.get_or_insert(create_prim_programs(&mut self.device, "ps_radial_gradient_dithering"))
                        } else {
                            self.ps_radial_gradient.get_or_insert(create_prim_programs(&mut self.device, "ps_radial_gradient"))
                        }
                    };
                    (ps_radial_gradient.get(transform_kind)  as &mut BindDraw, GPU_TAG_PRIM_RADIAL_GRADIENT)
                }
            },
        };

        // Handle special case readback for composites.
        match key.kind {
            BatchKind::Composite {
                task_id,
                source_id,
                backdrop_id,
            } => {
                // composites can't be grouped together because
                // they may overlap and affect each other.
                debug_assert!(instances.len() == 1);
                let cache_texture = self.texture_resolver
                    .resolve(&SourceTexture::CacheRGBA8)
                    .unwrap();

                // Before submitting the composite batch, do the
                // framebuffer readbacks that are needed for each
                // composite operation in this batch.
                //let cache_texture_dimensions = cache_texture.get_dimensions();

                let source = render_tasks.get(source_id);
                let backdrop = render_tasks.get(task_id);
                let readback = render_tasks.get(backdrop_id);

                let (readback_rect, readback_layer) = readback.get_target_rect();
                let (backdrop_rect, _) = backdrop.get_target_rect();
                let backdrop_screen_origin = backdrop.as_alpha_batch().screen_origin;
                let source_screen_origin = source.as_alpha_batch().screen_origin;

                // Bind the FBO to blit the backdrop to.
                // Called per-instance in case the layer (and therefore FBO)
                // changes. The device will skip the GL call if the requested
                // target is already bound.
                let cache_draw_target = (cache_texture, readback_layer.0 as i32);
                //self.device.bind_draw_target(Some(cache_draw_target), Some(cache_texture_dimensions));

                let src_x =
                    backdrop_rect.origin.x - backdrop_screen_origin.x + source_screen_origin.x;
                let src_y =
                    backdrop_rect.origin.y - backdrop_screen_origin.y + source_screen_origin.y;

                let dest_x = readback_rect.origin.x;
                let dest_y = readback_rect.origin.y;

                let width = readback_rect.size.width;
                let height = readback_rect.size.height;

                let mut src = DeviceIntRect::new(
                    DeviceIntPoint::new(src_x as i32, src_y as i32),
                    DeviceIntSize::new(width as i32, height as i32),
                );
                let mut dest = DeviceIntRect::new(
                    DeviceIntPoint::new(dest_x as i32, dest_y as i32),
                    DeviceIntSize::new(width as i32, height as i32),
                );

                // Need to invert the y coordinates and flip the image vertically when
                // reading back from the framebuffer.
                if render_target.is_none() && cfg!(not(feature = "dx11")) {
                    src.origin.y = target_dimensions.height as i32 - src.size.height - src.origin.y;
                    //dest.origin.y += dest.size.height;
                    //dest.size.height = -dest.size.height;
                } else if render_target.is_some() && cfg!(feature = "dx11") {
                    src.origin.y = target_dimensions.height as i32 - src.size.height - src.origin.y;
                }

                // self.device.bind_read_target(render_target);
                // self.device.blit_render_target(src, dest);

                // Restore draw target to current pass render target + layer.
                //self.device.bind_draw_target(render_target, Some(target_dimensions));
                self.device.copy_texture(render_target, &cache_texture, Some(src), dest);
            }
            _ => {}
        }

        let _gm = self.gpu_profile.add_marker(marker);
        for i in 0..key.textures.colors.len() {
            self.texture_resolver.bind(&key.textures.colors[i], TextureSampler::color(i), &mut self.device);
        }
        let mode = 0;
        program.bind_resources(&mut self.device, projection, instances, render_target, &mut self.renderer_errors, mode);
        self.profile_counters.vertices.add(6 * instances.len());
        program.call_draw(&mut self.device, &key.blend_mode, enable_depth_write);
}

    fn draw_color_target(
        &mut self,
        render_target: Option<(&TextureId, i32)>,
        target: &ColorRenderTarget,
        target_size: DeviceUintSize,
        clear_color: Option<[f32; 4]>,
        render_tasks: &RenderTaskTree,
        projection: &Transform3D<f32>,
    ) {
        {
            let _gm = self.gpu_profile.add_marker(GPU_TAG_SETUP_TARGET);
            match render_target {
                /*Some(..) if self.enable_clear_scissor => {
                    // TODO(gw): Applying a scissor rect and minimal clear here
                    // is a very large performance win on the Intel and nVidia
                    // GPUs that I have tested with. It's possible it may be a
                    // performance penalty on other GPU types - we should test this
                    // and consider different code paths.
                    self.device
                        .clear_target_rect(clear_color, Some(1.0), target.used_rect());
                }
                let clear_color = [1.0, 1.0, 1.0, 1.0];*/
                Some((tex_id, _)) => {
                    self.device.clear_render_target_color(&tex_id, clear_color, 1.0);
                }
                _ => {
                    self.device.clear_target(clear_color, Some(1.0));
                }
            }
        }

        // Draw any blurs for this target.
        // Blurs are rendered as a standard 2-pass
        // separable implementation.
        // TODO(gw): In the future, consider having
        //           fast path blur shaders for common
        //           blur radii with fixed weights.
        if !target.vertical_blurs.is_empty() || !target.horizontal_blurs.is_empty() {
            let _gm = self.gpu_profile.add_marker(GPU_TAG_BLUR);
            let cs_blur_rgba8 = self.cs_blur_rgba8.get_or_insert(create_blur_program(&mut self.device, "cs_blur_rgba8"));
            if !target.vertical_blurs.is_empty() {
                cs_blur_rgba8.bind(&mut self.device, projection, &target.vertical_blurs, render_target, &mut self.renderer_errors, 0);
                cs_blur_rgba8.draw(&mut self.device);
            }

            if !target.horizontal_blurs.is_empty() {
                cs_blur_rgba8.bind(&mut self.device, projection, &target.horizontal_blurs, render_target, &mut self.renderer_errors, 0);
                cs_blur_rgba8.draw(&mut self.device);
            }
        }

        // Draw any textrun caches for this target. For now, this
        // is only used to cache text runs that are to be blurred
        // for shadow support. In the future it may be worth
        // considering using this for (some) other text runs, since
        // it removes the overhead of submitting many small glyphs
        // to multiple tiles in the normal text run case.
        if !target.text_run_cache_prims.is_empty() {
            let _gm = self.gpu_profile.add_marker(GPU_TAG_CACHE_TEXT_RUN);

            for (texture_id, instances) in &target.text_run_cache_prims {
                self.texture_resolver.bind(&texture_id, TextureSampler::Color0, &mut self.device);
                let cs_text_run = self.cs_text_run.get_or_insert(create_program(&mut self.device, "cs_text_run"));
                cs_text_run.bind(&mut self.device, projection, &instances, render_target, &mut self.renderer_errors, 0);
                cs_text_run.draw(&mut self.device, &BlendMode::Alpha, false);
            }
        }
        if !target.line_cache_prims.is_empty() {
            let cs_line = self.cs_line.get_or_insert(create_program(&mut self.device, "cs_line"));
            cs_line.bind(&mut self.device, projection, &target.line_cache_prims, render_target, &mut self.renderer_errors, 0);
            cs_line.draw(&mut self.device, &BlendMode::Alpha, false);
        }

        //TODO: record the pixel count for cached primitives

        if !target.alpha_batcher.is_empty() {
            let _gm2 = GpuMarker::new("alpha batches");

            self.gpu_profile.add_sampler(GPU_SAMPLER_TAG_OPAQUE);

            let mut enable_depth_write = true;
            for batch in target.alpha_batcher
                               .batch_list
                               .opaque_batch_list
                               .batches
                               .iter()
                               .rev() {
                self.submit_batch(&batch.key,
                                  &batch.instances,
                                  &projection,
                                  render_tasks,
                                  render_target,
                                  target_size,
                                  enable_depth_write);
            }

            self.gpu_profile.add_sampler(GPU_SAMPLER_TAG_TRANSPARENT);
            enable_depth_write = false;
            for batch in &target.alpha_batcher.batch_list.alpha_batch_list.batches {
                if self.debug_flags.contains(DebugFlags::ALPHA_PRIM_DBG) {
                    let color = match batch.key.blend_mode {
                        BlendMode::None => ColorF::new(0.3, 0.3, 0.3, 1.0),
                        BlendMode::Alpha => ColorF::new(0.0, 0.9, 0.1, 1.0),
                        BlendMode::PremultipliedAlpha => ColorF::new(0.0, 0.3, 0.7, 1.0),
                        BlendMode::Subpixel => ColorF::new(0.5, 0.0, 0.4, 1.0),
                        _ => ColorF::new(0.0, 0.0, 0.0, 0.0),
                    }.into();
                    for item_rect in &batch.item_rects {
                        self.debug.add_rect(item_rect, color);
                    }
                }

                match batch.key.kind {
                    BatchKind::Transformable(transform_kind, TransformBatchKind::TextRun(glyph_format)) => {
                        // Text run batches are handled by this special case branch.
                        // In the case of subpixel text, we draw it as a two pass
                        // effect, to ensure we can apply clip masks correctly.
                        // In the future, there are several optimizations available:
                        // 1) Use dual source blending where available (almost all recent hardware).
                        // 2) Use frame buffer fetch where available (most modern hardware).
                        // 3) Consider the old constant color blend method where no clip is applied.
                        let _gm = self.gpu_profile.add_marker(GPU_TAG_PRIM_TEXT_RUN);

                        let ps_text_run = self.ps_text_run.get_or_insert(create_text_programs(&mut self.device, "ps_text_run"));
                        let mut ps_text_run = ps_text_run.get(transform_kind);
                        for i in 0..batch.key.textures.colors.len() {
                            self.texture_resolver.bind(&batch.key.textures.colors[i], TextureSampler::color(i), &mut self.device);
                        }
                        match batch.key.blend_mode {
                            BlendMode::PremultipliedAlpha => {
                                ps_text_run.bind(
                                    &mut self.device,
                                    projection,
                                    &batch.instances,
                                    render_target,
                                    &mut self.renderer_errors,
                                    TextShaderMode::from(glyph_format).into(),
                                );

                                ps_text_run.draw(
                                    &mut self.device,
                                    &BlendMode::PremultipliedAlpha,
                                    // No subpixel pass
                                    None,
                                );
                            }
                            BlendMode::Subpixel => {
                                // Using the two pass component alpha rendering technique:
                                //
                                // http://anholt.livejournal.com/32058.html

                                ps_text_run.bind(
                                    &mut self.device,
                                    projection,
                                    &batch.instances,
                                    render_target,
                                    &mut self.renderer_errors,
                                    TextShaderMode::SubpixelPass0.into(),
                                );

                                ps_text_run.draw(
                                    &mut self.device,
                                    &BlendMode::Subpixel,
                                    // First subpixel pass
                                    Some(0),
                                );

                                ps_text_run.bind(
                                    &mut self.device,
                                    projection,
                                    &batch.instances,
                                    render_target,
                                    &mut self.renderer_errors,
                                    TextShaderMode::SubpixelPass1.into(),
                                );

                                // When drawing the 2nd pass, we know that the VAO, textures etc
                                // are all set up from the previous draw_instanced_batch call,
                                // so just issue a draw call here to avoid re-uploading the
                                // instances and re-binding textures etc.

                                ps_text_run.draw(
                                    &mut self.device,
                                    &BlendMode::Subpixel,
                                    // Second subpixel pass
                                    Some(1),
                                );
                            }
                            //BlendMode::Alpha | BlendMode::None => {
                            _ => {
                                unreachable!("bug: bad blend mode for text");
                            }
                        }
                    }
                    _ => {
                        /*if batch.key.blend_mode != prev_blend_mode {
                            match batch.key.blend_mode {
                                BlendMode::None => {
                                    self.device.set_blend(false);
                                }
                                BlendMode::Alpha => {
                                    self.device.set_blend(true);
                                    self.device.set_blend_mode_alpha();
                                }
                                BlendMode::PremultipliedAlpha => {
                                    self.device.set_blend(true);
                                    self.device.set_blend_mode_premultiplied_alpha();
                                }
                                BlendMode::Subpixel => {
                                    unreachable!("bug: subpx text handled earlier");
                                }
                            }
                            prev_blend_mode = batch.key.blend_mode;
                        }*/

                        self.submit_batch(
                            &batch.key,
                            &batch.instances,
                            &projection,
                            render_tasks,
                            render_target,
                            target_size,
                            enable_depth_write,
                        );
                    }
                }
/*
                self.submit_batch(&batch.key,
                                  &batch.instances,
                                  &projection,
                                  render_tasks,
                                  render_target,
                                  target_size,
                                  enable_depth_write);*/
            }

            self.gpu_profile.done_sampler();
        }

        // TODO(zgy): we don't support this new feature yet
        // For any registered image outputs on this render target,
        // get the texture from caller and blit it.
        /*for output in &target.outputs {
            let handler = self.output_image_handler
                .as_mut()
                .expect("Found output image, but no handler set!");
            if let Some((texture_id, output_size)) = handler.lock(output.pipeline_id) {
                let device = &mut self.device;
                let fbo_id = match self.output_targets.entry(texture_id) {
                    Entry::Vacant(entry) => {
                        let fbo_id = device.create_fbo_for_external_texture(texture_id);
                        entry.insert(FrameOutput {
                            fbo_id,
                            last_access: frame_id,
                        });
                        fbo_id
                    }
                    Entry::Occupied(mut entry) => {
                        let target = entry.get_mut();
                        target.last_access = frame_id;
                        target.fbo_id
                    }
                };
                let task = render_tasks.get(output.task_id);
                let (src_rect, _) = task.get_target_rect();
                let dest_rect = DeviceIntRect::new(DeviceIntPoint::zero(), output_size);
                device.bind_read_target(render_target);
                device.bind_external_draw_target(fbo_id);
                device.blit_render_target(src_rect, dest_rect);
                handler.unlock(output.pipeline_id);
            }
        }*/
    }

    fn draw_alpha_target(
        &mut self,
        render_target: (&TextureId, i32),
        target: &AlphaRenderTarget,
        target_size: DeviceUintSize,
        projection: &Transform3D<f32>,
        render_tasks: &RenderTaskTree,
    ) {
        self.gpu_profile.add_sampler(GPU_SAMPLER_TAG_ALPHA);

        {
            let _gm = self.gpu_profile.add_marker(GPU_TAG_SETUP_TARGET);

            // TODO(gw): Applying a scissor rect and minimal clear here
            // is a very large performance win on the Intel and nVidia
            // GPUs that I have tested with. It's possible it may be a
            // performance penalty on other GPU types - we should test this
            // and consider different code paths.
            let clear_color = [1.0, 1.0, 1.0, 0.0];
            self.device.clear_render_target_alpha(render_target.0, clear_color);

            let (mut tri_vertices, mut tri_indices) = (Vec::new(), Vec::new());
            let zero_color = ColorU::new(0, 0, 0, 0);
            for task_id in &target.zero_clears {
                let task = render_tasks.get(*task_id);
                let (rect, _) = task.get_target_rect();
                let vertex_count = tri_vertices.len() as u32;

                tri_indices.push(vertex_count + 0);
                tri_indices.push(vertex_count + 1);
                tri_indices.push(vertex_count + 2);
                tri_indices.push(vertex_count + 2);
                tri_indices.push(vertex_count + 1);
                tri_indices.push(vertex_count + 3);

                let (x0, y0, x1, y1) = (
                    rect.origin.x as f32,
                    rect.origin.y as f32,
                    (rect.origin.x + rect.size.width) as f32,
                    (rect.origin.y + rect.size.height) as f32,
                );
                tri_vertices.push(DebugColorVertex::new(x0, y0, zero_color));
                tri_vertices.push(DebugColorVertex::new(x1, y0, zero_color));
                tri_vertices.push(DebugColorVertex::new(x0, y1, zero_color));
                tri_vertices.push(DebugColorVertex::new(x1, y1, zero_color));

            }
            if !tri_vertices.is_empty() {
                let ps_clear = self.ps_clear.get_or_insert(create_clear_program(&mut self.device, "debug_color"));
                ps_clear.bind(
                    &mut self.device,
                    projection,
                    &tri_indices,
                    &tri_vertices,
                    Some(render_target),
                );
                ps_clear.draw(&mut self.device);
            }
        }

        // Draw any blurs for this target.
        // Blurs are rendered as a standard 2-pass
        // separable implementation.
        // TODO(gw): In the future, consider having
        //           fast path blur shaders for common
        //           blur radii with fixed weights.
        if !target.vertical_blurs.is_empty() || !target.horizontal_blurs.is_empty() {
            let _gm = self.gpu_profile.add_marker(GPU_TAG_BLUR);

            let cs_blur_a8 = self.cs_blur_a8.get_or_insert(create_blur_program(&mut self.device, "cs_blur_a8"));
            if !target.vertical_blurs.is_empty() {
                cs_blur_a8.bind(&mut self.device, projection, &target.vertical_blurs, Some(render_target), &mut self.renderer_errors, 0);
                cs_blur_a8.draw(&mut self.device);
            }

            if !target.horizontal_blurs.is_empty() {
                cs_blur_a8.bind(&mut self.device, projection, &target.horizontal_blurs, Some(render_target), &mut self.renderer_errors, 0);
                cs_blur_a8.draw(&mut self.device);
            }
        }

        if !target.rect_cache_prims.is_empty() {
            let _gm = self.gpu_profile.add_marker(GPU_TAG_BRUSH_MASK);
            let brush_mask = self.brush_mask.get_or_insert(create_brush_programs(&mut self.device, "brush_mask"));
            let shader = brush_mask.get(BlendMode::None);
            shader.bind(&mut self.device, projection, &target.rect_cache_prims, Some(render_target), &mut self.renderer_errors, 0);
            shader.draw(&mut self.device, &BlendMode::None);
        }


        // Draw the clip items into the tiled alpha mask.
        {
            let _gm = self.gpu_profile.add_marker(GPU_TAG_CACHE_CLIP);

            // If we have border corner clips, the first step is to clear out the
            // area in the clip mask. This allows drawing multiple invididual clip
            // in regions below.
            if !target.clip_batcher.border_clears.is_empty() {
                let _gm2 = GpuMarker::new("clip borders [clear]");
                let cs_clip_border = self.cs_clip_border.get_or_insert(create_clip_program(&mut self.device, "cs_clip_border_transform"));
                cs_clip_border.bind(&mut self.device, projection, &target.clip_batcher.border_clears, render_target.0, &mut self.renderer_errors, 0);
                self.profile_counters.vertices.add(6 * &target.clip_batcher.border_clears.len());
                cs_clip_border.draw(&mut self.device, &BlendMode::None);
            }

            // Draw any dots or dashes for border corners.
            if !target.clip_batcher.borders.is_empty() {
                let _gm2 = GpuMarker::new("clip borders");
                // We are masking in parts of the corner (dots or dashes) here.
                // Blend mode is set to max to allow drawing multiple dots.
                // The individual dots and dashes in a border never overlap, so using
                // a max blend mode here is fine.
                let cs_clip_border = self.cs_clip_border.get_or_insert(create_clip_program(&mut self.device, "cs_clip_border_transform"));
                cs_clip_border.bind(&mut self.device, projection, &target.clip_batcher.borders, render_target.0, &mut self.renderer_errors, 0);
                self.profile_counters.vertices.add(6 * &target.clip_batcher.borders.len());
                cs_clip_border.draw(&mut self.device, &BlendMode::Max);
            }

            // switch to multiplicative blending
            let blend_mode = BlendMode::Multiply;

            // draw rounded cornered rectangles
            if !target.clip_batcher.rectangles.is_empty() {
                let _gm2 = GpuMarker::new("clip rectangles");
                let cs_clip_rectangle = self.cs_clip_rectangle.get_or_insert(create_clip_program(&mut self.device, "cs_clip_rectangle_transform"));
                cs_clip_rectangle.bind(&mut self.device, projection, &target.clip_batcher.rectangles, render_target.0, &mut self.renderer_errors, 0);
                self.profile_counters.vertices.add(6 * &target.clip_batcher.rectangles.len());
                cs_clip_rectangle.draw(&mut self.device, &blend_mode);
            }

            // draw image masks
            for (mask_texture_id, items) in target.clip_batcher.images.iter() {
                let _gm2 = GpuMarker::new("clip images");
                self.texture_resolver.bind(&mask_texture_id, TextureSampler::Color0, &mut self.device);
                let cs_clip_image = self.cs_clip_image.get_or_insert(create_clip_program(&mut self.device, "cs_clip_image_transform"));
                cs_clip_image.bind(&mut self.device, projection, &items, render_target.0, &mut self.renderer_errors, 0);
                self.profile_counters.vertices.add(6 * &items.len());
                cs_clip_image.draw(&mut self.device, &blend_mode);
            }
        }

        self.gpu_profile.done_sampler();
    }

    fn update_deferred_resolves(&mut self, frame: &mut Frame) {
        // The first thing we do is run through any pending deferred
        // resolves, and use a callback to get the UV rect for this
        // custom item. Then we patch the resource_rects structure
        // here before it's uploaded to the GPU.
        if !frame.deferred_resolves.is_empty() {
            let handler = self.external_image_handler
                .as_mut()
                .expect("Found external image, but no handler set!");

            for deferred_resolve in &frame.deferred_resolves {
                GpuMarker::fire("deferred resolve");
                let props = &deferred_resolve.image_properties;
                let ext_image = props
                    .external_image
                    .expect("BUG: Deferred resolves must be external images!");
                let image = handler.lock(ext_image.id, ext_image.channel_index);
                let texture_target = match ext_image.image_type {
                    ExternalImageType::Texture2DHandle => TextureTarget::Default,
                    ExternalImageType::Texture2DArrayHandle => TextureTarget::Array,
                    ExternalImageType::TextureRectHandle => TextureTarget::Rect,
                    ExternalImageType::TextureExternalHandle => TextureTarget::External,
                    ExternalImageType::ExternalBuffer => {
                        panic!(
                            "{:?} is not a suitable image type in update_deferred_resolves().",
                            ext_image.image_type
                        );
                    }
                };

                // In order to produce the handle, the external image handler may call into
                // the GL context and change some states.
                self.device.reset_state();

                /*let texture = match image.source {
                    ExternalImageSource::NativeTexture(texture_id) => {
                        ExternalTexture::new(texture_id, texture_target)
                    }
                    ExternalImageSource::Invalid => {
                        warn!(
                            "Invalid ext-image for ext_id:{:?}, channel:{}.",
                            ext_image.id,
                            ext_image.channel_index
                        );
                        // Just use 0 as the gl handle for this failed case.
                        ExternalTexture::new(0, texture_target)
                    }
                    _ => panic!("No native texture found."),
                };

                self.texture_resolver
                    .external_images
                    .insert((ext_image.id, ext_image.channel_index), texture);*/

                let update = GpuCacheUpdate::Copy {
                    block_index: 0,
                    block_count: 1,
                    address: deferred_resolve.address,
                };

                let blocks = [
                    [image.u0, image.v0, image.u1, image.v1].into(),
                    [0.0; 4].into(),
                ];
                self.gpu_cache_texture.apply_patch(&update, &blocks);
            }
        }
    }

    /*fn unlock_external_images(&mut self) {
        if !self.texture_resolver.external_images.is_empty() {
            let handler = self.external_image_handler
                .as_mut()
                .expect("Found external image, but no handler set!");

            for (ext_data, _) in self.texture_resolver.external_images.drain() {
                handler.unlock(ext_data.0, ext_data.1);
            }
        }
    }*/

    fn start_frame(&mut self, frame: &mut Frame) {
        let _gm = self.gpu_profile.add_marker(GPU_TAG_SETUP_DATA);

        // Assign render targets to the passes.
        for pass in &mut frame.passes {
            debug_assert!(pass.color_texture.is_none());
            debug_assert!(pass.alpha_texture.is_none());

            if pass.needs_render_target_kind(RenderTargetKind::Color) {
                pass.color_texture = Some(self.color_render_targets
                                              .pop()
                                              .unwrap_or_else(|| {
                                                  self.device
                                                      .create_cache_texture(
                                                          pass.max_color_target_size.width,
                                                          pass.max_color_target_size.height,
                                                          RenderTargetKind::Color)
                                               }));
            }

            if pass.needs_render_target_kind(RenderTargetKind::Alpha) {
                pass.alpha_texture = Some(self.alpha_render_targets
                                              .pop()
                                              .unwrap_or_else(|| {
                                                  self.device
                                                      .create_cache_texture(
                                                          pass.max_alpha_target_size.width,
                                                          pass.max_alpha_target_size.height,
                                                          RenderTargetKind::Alpha)
                                               }));
            }
        }

        self.layer_texture.update(&mut self.device, &mut frame.layer_texture_data);
        self.render_task_texture.update(&mut self.device, &mut frame.render_tasks.task_data);

        debug_assert!(self.texture_resolver.cache_a8_texture.is_none());
        debug_assert!(self.texture_resolver.cache_rgba8_texture.is_none());
    }

    fn draw_tile_frame(
        &mut self,
        frame: &mut Frame,
        framebuffer_size: DeviceUintSize,
        //frame_id: FrameId,
    ) {
        let _gm = GpuMarker::new("tile frame draw");

        // Some tests use a restricted viewport smaller than the main screen size.
        // Ensure we clear the framebuffer in these tests.
        // TODO(gw): Find a better solution for this?
        let needs_clear = frame.window_size.width < framebuffer_size.width ||
            frame.window_size.height < framebuffer_size.height;

        if frame.passes.is_empty() {
            self.device
                .clear_target(Some(self.clear_color.to_array()), Some(1.0));
        } else {
            self.start_frame(frame);
            let pass_count = frame.passes.len();

            for (pass_index, pass) in frame.passes.iter_mut().enumerate() {
                self.texture_resolver.bind(
                    &SourceTexture::CacheA8,
                    TextureSampler::CacheA8,
                    &mut self.device,
                );
                self.texture_resolver.bind(
                    &SourceTexture::CacheRGBA8,
                    TextureSampler::CacheRGBA8,
                    &mut self.device,
                );

                for (target_index, target) in pass.alpha_targets.targets.iter().enumerate() {
                    let projection = alpha_transform_projection(Transform3D::ortho(
                        0.0,
                        pass.max_alpha_target_size.width as f32,
                        0.0,
                        pass.max_alpha_target_size.height as f32,
                        ORTHO_NEAR_PLANE,
                        ORTHO_FAR_PLANE,
                    ));

                    self.draw_alpha_target(
                        (pass.alpha_texture.as_ref().unwrap(), target_index as i32),
                        target,
                        pass.max_alpha_target_size,
                        &projection,
                        &frame.render_tasks,
                    );
                }

                for (target_index, target) in pass.color_targets.targets.iter().enumerate() {
                    let size;
                    let clear_color;
                    let projection;

                    if pass.is_framebuffer {
                        clear_color = if self.clear_framebuffer || needs_clear {
                            Some(
                                frame
                                    .background_color
                                    .map_or(self.clear_color.to_array(), |color| color.to_array()),
                            )
                        } else {
                            None
                        };
                        size = framebuffer_size;
                        projection = Transform3D::ortho(
                            0.0,
                            size.width as f32,
                            size.height as f32,
                            0.0,
                            ORTHO_NEAR_PLANE,
                            ORTHO_FAR_PLANE,
                        )
                    } else {
                        size = pass.max_color_target_size;
                        clear_color = Some([0.0, 0.0, 0.0, 0.0]);
                        projection = Transform3D::ortho(
                            0.0,
                            size.width as f32,
                            0.0,
                            size.height as f32,
                            ORTHO_NEAR_PLANE,
                            ORTHO_FAR_PLANE,
                        );
                    }

                    let render_target = pass.color_texture
                        .as_ref()
                        .map(|texture| (texture, target_index as i32));
                    self.draw_color_target(
                        render_target,
                        target,
                        size,
                        clear_color,
                        &frame.render_tasks,
                        &transform_projection(projection),
                    );
                }

                self.texture_resolver.end_pass(
                    pass_index,
                    pass_count,
                    pass.alpha_texture.take(),
                    pass.color_texture.take(),
                    &mut self.alpha_render_targets,
                    &mut self.color_render_targets,
                );

                // After completing the first pass, make the A8 target available as an
                // input to any subsequent passes.
                if pass_index == 0 {
                    if let Some(shared_alpha_texture) =
                        self.texture_resolver.resolve(&SourceTexture::CacheA8)
                    {
                        self.device
                            .bind_texture(TextureSampler::SharedCacheA8, shared_alpha_texture, TextureStorage::CacheA8);
                    }
                }
            }

            self.color_render_targets.reverse();
            self.alpha_render_targets.reverse();
            self.draw_render_target_debug(framebuffer_size);
            self.draw_texture_cache_debug(framebuffer_size);

            // Garbage collect any frame outputs that weren't used this frame.
            /*let device = &mut self.device;
            self.output_targets
                .retain(|_, target| if target.last_access != frame_id {
                    device.delete_fbo(target.fbo_id);
                    false
                } else {
                    true
                });*/
        }

        //self.unlock_external_images();
}

    pub fn debug_renderer<'a>(&'a mut self) -> &'a mut DebugRenderer {
        &mut self.debug
    }

    pub fn get_debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }

    pub fn set_debug_flags(&mut self, flags: DebugFlags) {
        self.debug_flags = flags;
    }

    pub fn save_cpu_profile(&self, filename: &str) {
        write_profile(filename);
    }

    fn draw_render_target_debug(&mut self, framebuffer_size: DeviceUintSize) {
        /*if !self.debug_flags.contains(DebugFlags::RENDER_TARGET_DBG) {
            return;
        }

        let mut spacing = 16;
        let mut size = 512;
        let fb_width = framebuffer_size.width as i32;
        let num_layers: i32 = self.color_render_targets
            .iter()
            .chain(self.alpha_render_targets.iter())
            .map(|texture| texture.get_render_target_layer_count() as i32)
            .sum();

        if num_layers * (size + spacing) > fb_width {
            let factor = fb_width as f32 / (num_layers * (size + spacing)) as f32;
            size = (size as f32 * factor) as i32;
            spacing = (spacing as f32 * factor) as i32;
        }

        let mut target_index = 0;
        for texture in self.color_render_targets
            .iter()
            .chain(self.alpha_render_targets.iter())
        {
            let dimensions = texture.get_dimensions();
            let src_rect = DeviceIntRect::new(DeviceIntPoint::zero(), dimensions.to_i32());

            let layer_count = texture.get_render_target_layer_count();
            for layer_index in 0 .. layer_count {
                self.device
                    .bind_read_target(Some((texture, layer_index as i32)));
                let x = fb_width - (spacing + size) * (target_index + 1);
                let y = spacing;

                let dest_rect = rect(x, y, size, size);
                self.device.blit_render_target(src_rect, dest_rect);
                target_index += 1;
            }
        }*/
    }

    fn draw_texture_cache_debug(&mut self, framebuffer_size: DeviceUintSize) {
        /*if !self.debug_flags.contains(DebugFlags::TEXTURE_CACHE_DBG) {
            return;
        }

        let mut spacing = 16;
        let mut size = 512;
        let fb_width = framebuffer_size.width as i32;
        let num_layers: i32 = self.texture_resolver
            .cache_texture_map
            .iter()
            .map(|texture| {
                1//texture.get_layer_count() TODO(zgy): check what happens, if we use the layercount
            })
            .sum();

        if num_layers * (size + spacing) > fb_width {
            let factor = fb_width as f32 / (num_layers * (size + spacing)) as f32;
            size = (size as f32 * factor) as i32;
            spacing = (spacing as f32 * factor) as i32;
        }

        let mut i = 0;
        for texture in &self.texture_resolver.cache_texture_map {
            let y = spacing + if self.debug_flags.contains(DebugFlags::RENDER_TARGET_DBG) {
                528
            } else {
                0
            };
            let dimensions = texture.get_dimensions();
            let src_rect = DeviceIntRect::new(
                DeviceIntPoint::zero(),
                DeviceIntSize::new(dimensions.width as i32, dimensions.height as i32),
            );

            let layer_count = 1; //texture.get_layer_count();
            for layer_index in 0 .. layer_count {
                // self.device.bind_read_target(Some((texture, layer_index)));
                let x = fb_width - (spacing + size) * (i as i32 + 1);

                // If we have more targets than fit on one row in screen, just early exit.
                if x > fb_width {
                    return;
                }

                //let dest_rect = rect(x, y, size, size);
                //self.device.blit_render_target(src_rect, dest_rect);
                i += 1;
            }
        }*/
    }

    pub fn read_pixels_rgba8(&mut self, rect: DeviceUintRect) -> Vec<u8> {
        let mut pixels = vec![0u8; (4 * rect.size.width * rect.size.height) as usize];
        self.read_pixels_into(rect, ReadPixelsFormat::Rgba8, &mut pixels);
        pixels
    }

    pub fn read_pixels_into(
        &mut self,
        rect: DeviceUintRect,
        _format: ReadPixelsFormat,
        output: &mut [u8],
    ) {
        /*let stride = match _format {
            ReadPixelsFormat::Rgba8 => RGBA_STRIDE,
            ReadPixelsFormat::Bgra8 => RGBA_STRIDE,
        };*/
        assert_eq!(output.len(), 4 * (rect.size.width * rect.size.height) as usize);
        self.device.flush();
        self.device.read_pixels(rect, output);
    }

    // De-initialize the Renderer safely, assuming the GL is still alive and active.
    pub fn deinit(mut self) {
        //Note: this is a fake frame, only needed because texture deletion is require to happen inside a frame
        self.device.begin_frame(1.0);
        self.texture_resolver.deinit(&mut self.device);
        self.debug.deinit(&mut self.device);
        self.device.end_frame();
    }
}

pub enum ExternalImageSource<'a> {
    RawData(&'a [u8]),  // raw buffers.
    NativeTexture(u32), // It's a gl::GLuint texture handle
    Invalid,
}

/// The data that an external client should provide about
/// an external image. The timestamp is used to test if
/// the renderer should upload new texture data this
/// frame. For instance, if providing video frames, the
/// application could call wr.render() whenever a new
/// video frame is ready. If the callback increments
/// the returned timestamp for a given image, the renderer
/// will know to re-upload the image data to the GPU.
/// Note that the UV coords are supplied in texel-space!
pub struct ExternalImage<'a> {
    pub u0: f32,
    pub v0: f32,
    pub u1: f32,
    pub v1: f32,
    pub source: ExternalImageSource<'a>,
}

/// The interfaces that an application can implement to support providing
/// external image buffers.
/// When the the application passes an external image to WR, it should kepp that
/// external image life time. People could check the epoch id in RenderNotifier
/// at the client side to make sure that the external image is not used by WR.
/// Then, do the clean up for that external image.
pub trait ExternalImageHandler {
    /// Lock the external image. Then, WR could start to read the image content.
    /// The WR client should not change the image content until the unlock()
    /// call.
    fn lock(&mut self, key: ExternalImageId, channel_index: u8) -> ExternalImage;
    /// Unlock the external image. The WR should not read the image content
    /// after this call.
    fn unlock(&mut self, key: ExternalImageId, channel_index: u8);
}

/// Allows callers to receive a texture with the contents of a specific
/// pipeline copied to it. Lock should return the native texture handle
/// and the size of the texture. Unlock will only be called if the lock()
/// call succeeds, when WR has issued the GL commands to copy the output
/// to the texture handle.
pub trait OutputImageHandler {
    fn lock(&mut self, pipeline_id: PipelineId) -> Option<(u32, DeviceIntSize)>;
    fn unlock(&mut self, pipeline_id: PipelineId);
}

pub struct RendererOptions {
    pub device_pixel_ratio: f32,
    pub resource_override_path: Option<PathBuf>,
    pub enable_aa: bool,
    pub enable_dithering: bool,
    pub max_recorded_profiles: usize,
    pub debug: bool,
    pub enable_scrollbars: bool,
    pub precache_shaders: bool,
    pub renderer_kind: RendererKind,
    pub enable_subpixel_aa: bool,
    pub clear_framebuffer: bool,
    pub clear_color: ColorF,
    pub enable_clear_scissor: bool,
    pub enable_batcher: bool,
    pub max_texture_size: Option<u32>,
    pub workers: Option<Arc<ThreadPool>>,
    pub blob_image_renderer: Option<Box<BlobImageRenderer>>,
    pub recorder: Option<Box<ApiRecordingReceiver>>,
    pub enable_render_on_scroll: bool,
    pub debug_flags: DebugFlags,
}

impl Default for RendererOptions {
    fn default() -> RendererOptions {
        RendererOptions {
            device_pixel_ratio: 1.0,
            resource_override_path: None,
            enable_aa: true,
            enable_dithering: true,
            debug_flags: DebugFlags::empty(),
            max_recorded_profiles: 0,
            debug: false,
            enable_scrollbars: false,
            precache_shaders: false,
            renderer_kind: RendererKind::Native,
            enable_subpixel_aa: false,
            clear_framebuffer: true,
            clear_color: ColorF::new(1.0, 1.0, 1.0, 1.0),
            enable_clear_scissor: true,
            enable_batcher: true,
            max_texture_size: None,
            workers: None,
            blob_image_renderer: None,
            recorder: None,
            enable_render_on_scroll: true,
        }
    }
}

#[cfg(not(feature = "debugger"))]
pub struct DebugServer;

#[cfg(not(feature = "debugger"))]
impl DebugServer {
    pub fn new(_: MsgSender<ApiMsg>) -> DebugServer {
        DebugServer
    }

    pub fn send(&mut self, _: String) {}
}
