/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! The webrender API.
//!
//! The `webrender::renderer` module provides the interface to webrender, which
//! is accessible through [`Renderer`][renderer]
//!
//! [renderer]: struct.Renderer.html

use device::{Device, TextureId, TextureFilter, TextureTarget, Program, ShaderError};
use device::{VECS_PER_DATA_16, VECS_PER_DATA_32, VECS_PER_DATA_64, VECS_PER_DATA_128};
use device::{VECS_PER_LAYER, VECS_PER_PRIM_GEOM, VECS_PER_RENDER_TASK};
use device::{VECS_PER_RESOURCE_RECTS, VECS_PER_GRADIENT_DATA, VECS_PER_SPLIT_GEOM};
use device::RGBA_STRIDE;
use euclid::Matrix4D;
use fnv::FnvHasher;
use frame_builder::FrameBuilderConfig;
use gleam::gl;
use gpu_store::{GpuStore, GpuStoreLayout};
use internal_types::{CacheTextureId, RendererFrame, ResultMsg, TextureUpdateOp};
use internal_types::{TextureUpdateList, RenderTargetMode};
use internal_types::{ORTHO_NEAR_PLANE, ORTHO_FAR_PLANE, SourceTexture};
use internal_types::TextureSampler;
use prim_store::{GradientData, SplitGeometry};
use record::ApiRecordingReceiver;
use render_backend::RenderBackend;
use render_task::RenderTaskData;
use std;
use std::cmp;
use std::collections::HashMap;
use std::f32;
use std::hash::BuildHasherDefault;
use std::marker::PhantomData;
use std::mem;
use std::path::PathBuf;
use std::slice;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Receiver};
use std::thread;
use texture_cache::TextureCache;
use rayon::ThreadPool;
use rayon::Configuration as ThreadPoolConfig;
use tiling::{AlphaBatchKind, Frame, PrimitiveBatch};
use tiling::{AlphaRenderTarget, ColorRenderTarget, RenderTargetKind};
use thread_profiler::{register_thread_with_profiler};
use util::TransformedRectKind;
use webgl_types::GLContextHandleWrapper;
use webrender_traits::{ColorF, Epoch, PipelineId, RenderNotifier, RenderDispatcher};
use webrender_traits::{ExternalImageId, ExternalImageType, ImageData, ImageFormat, RenderApiSender};
use webrender_traits::{DevicePoint, DeviceUintSize};
use webrender_traits::BlobImageRenderer;
use webrender_traits::{channel, FontRenderMode};
use webrender_traits::VRCompositorHandler;
use webrender_traits::{YuvColorSpace, YuvFormat};
use webrender_traits::{YUV_COLOR_SPACES, YUV_FORMATS};

use glutin;

pub const MAX_VERTEX_TEXTURE_WIDTH: usize = 1024;
pub const DUMMY_RGBA8_ID: u32 = 2;
pub const DUMMY_A8_ID: u32 = 3;
pub const DITHER_ID: u32 = 4;

macro_rules! create_program (
    ($device: ident, $shader: expr) => {
        $device.create_program(include_bytes!(concat!(env!("OUT_DIR"), "/", $shader, ".vert")),
                               include_bytes!(concat!(env!("OUT_DIR"), "/", $shader, ".frag")))
    };
);

macro_rules! create_programs (
    ($device: ident, $shader: expr) => {
        (create_program!($device, $shader), create_program!($device, concat!($shader, "_transform")))
    };
);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum ImageBufferKind {
    Texture2D = 0,
    TextureRect = 1,
    TextureExternal = 2,
}
pub const IMAGE_BUFFER_KINDS: [ImageBufferKind; 3] = [
    ImageBufferKind::Texture2D,
    ImageBufferKind::TextureRect,
    ImageBufferKind::TextureExternal
];

impl ImageBufferKind {
    pub fn get_feature_string(&self) -> &'static str {
        match *self {
            ImageBufferKind::Texture2D => "",
            ImageBufferKind::TextureRect => "TEXTURE_RECT",
            ImageBufferKind::TextureExternal => "TEXTURE_EXTERNAL",
        }
    }

    pub fn has_platform_support(&self, gl_type: &gl::GlType) -> bool {
        match *gl_type {
            gl::GlType::Gles => {
                match *self {
                    ImageBufferKind::Texture2D => true,
                    ImageBufferKind::TextureRect => true,
                    ImageBufferKind::TextureExternal => true,
                }
            }
            gl::GlType::Gl => {
                match *self {
                    ImageBufferKind::Texture2D => true,
                    ImageBufferKind::TextureRect => true,
                    ImageBufferKind::TextureExternal => false,
                }
            }
        }
    }
}
#[derive(Debug, Copy, Clone)]
pub enum RendererKind {
    Native,
    OSMesa,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BlendMode {
    None,
    Alpha,
    PremultipliedAlpha,

    // Use the color of the text itself as a constant color blend factor.
    Subpixel(ColorF),
}

struct GpuDataTexture<L> {
    layout: PhantomData<L>,
}

impl<L: GpuStoreLayout> GpuDataTexture<L> {
    fn new() -> GpuDataTexture<L> {
        GpuDataTexture {
            layout: PhantomData,
        }
    }

    fn init<T: Default>(&mut self,
                        device: &mut Device,
                        sampler: TextureSampler,
                        data: &mut Vec<T>,
                        size: usize) {
        if data.is_empty() {
            return;
        }

        let items_per_row = L::items_per_row::<T>();

        // Extend the data array to be a multiple of the row size.
        // This ensures memory safety when the array is passed to
        // OpenGL to upload to the GPU.
        if items_per_row != 0 {
            while data.len() % items_per_row != 0 {
                data.push(T::default());
            }
        }

        match L::image_format() {
            ImageFormat::RGBAF32 => {
                device.update_sampler_f32(sampler,
                                          unsafe { slice::from_raw_parts(data.as_ptr() as *const f32, data.len() * size as usize) });
            },
            ImageFormat::RGBA8 => {
                device.update_sampler_u8(sampler,
                                         unsafe { slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * size as usize) });
            },
            _ => unimplemented!(), // Invalid, A8, RGB8, RG8
        }
    }
}

pub struct VertexDataTextureLayout {}

impl GpuStoreLayout for VertexDataTextureLayout {
    fn image_format() -> ImageFormat {
        ImageFormat::RGBAF32
    }

    fn texture_width<T>() -> usize {
        MAX_VERTEX_TEXTURE_WIDTH - (MAX_VERTEX_TEXTURE_WIDTH % Self::texels_per_item::<T>())
    }

    fn texture_filter() -> TextureFilter {
        TextureFilter::Nearest
    }
}

type VertexDataTexture = GpuDataTexture<VertexDataTextureLayout>;
pub type VertexDataStore<T> = GpuStore<T, VertexDataTextureLayout>;

pub struct GradientDataTextureLayout;

impl GpuStoreLayout for GradientDataTextureLayout {
    fn image_format() -> ImageFormat {
        ImageFormat::RGBA8
    }

    fn texture_width<T>() -> usize {
        mem::size_of::<GradientData>() / Self::texel_size() / 2
    }

    fn texture_filter() -> TextureFilter {
        TextureFilter::Linear
    }
}

type GradientDataTexture = GpuDataTexture<GradientDataTextureLayout>;
pub type GradientDataStore = GpuStore<GradientData, GradientDataTextureLayout>;

pub struct SplitGeometryTextureLayout;

impl GpuStoreLayout for SplitGeometryTextureLayout {
    fn image_format() -> ImageFormat {
        //TODO: use normalized integers
        ImageFormat::RGBAF32
    }

    fn texture_width<T>() -> usize {
        MAX_VERTEX_TEXTURE_WIDTH - (MAX_VERTEX_TEXTURE_WIDTH % Self::texels_per_item::<T>())
    }

    fn texture_filter() -> TextureFilter {
        TextureFilter::Nearest
    }
}

type SplitGeometryTexture = GpuDataTexture<SplitGeometryTextureLayout>;
pub type SplitGeometryStore = GpuStore<SplitGeometry, SplitGeometryTextureLayout>;

struct GpuDataTextures {
    layer_texture: VertexDataTexture,
    render_task_texture: VertexDataTexture,
    prim_geom_texture: VertexDataTexture,
    data16_texture: VertexDataTexture,
    data32_texture: VertexDataTexture,
    data64_texture: VertexDataTexture,
    data128_texture: VertexDataTexture,
    resource_rects_texture: VertexDataTexture,
    gradient_data_texture: GradientDataTexture,
    split_geometry_texture: SplitGeometryTexture,
}

impl GpuDataTextures {
    fn new() -> GpuDataTextures {
        GpuDataTextures {
            layer_texture: VertexDataTexture::new(),
            render_task_texture: VertexDataTexture::new(),
            prim_geom_texture: VertexDataTexture::new(),
            data16_texture: VertexDataTexture::new(),
            data32_texture: VertexDataTexture::new(),
            data64_texture: VertexDataTexture::new(),
            data128_texture: VertexDataTexture::new(),
            resource_rects_texture: VertexDataTexture::new(),
            gradient_data_texture: GradientDataTexture::new(),
            split_geometry_texture: SplitGeometryTexture::new(),
        }
    }

    fn init_frame(&mut self, device: &mut Device, frame: &mut Frame) {
        self.data16_texture.init(device, TextureSampler::Data16, &mut frame.gpu_data16, VECS_PER_DATA_16 * RGBA_STRIDE);
        self.data32_texture.init(device, TextureSampler::Data32, &mut frame.gpu_data32, VECS_PER_DATA_32 * RGBA_STRIDE);
        self.data64_texture.init(device, TextureSampler::Data64, &mut frame.gpu_data64, VECS_PER_DATA_64 * RGBA_STRIDE);
        self.data128_texture.init(device, TextureSampler::Data128, &mut frame.gpu_data128, VECS_PER_DATA_128 * RGBA_STRIDE);
        self.prim_geom_texture.init(device, TextureSampler::Geometry, &mut frame.gpu_geometry, VECS_PER_PRIM_GEOM * RGBA_STRIDE);
        self.resource_rects_texture.init(device, TextureSampler::ResourceRects, &mut frame.gpu_resource_rects, VECS_PER_RESOURCE_RECTS * RGBA_STRIDE);
        self.layer_texture.init(device, TextureSampler::Layers, &mut frame.layer_texture_data, VECS_PER_LAYER * RGBA_STRIDE);
        self.render_task_texture.init(device, TextureSampler::RenderTasks, &mut frame.render_task_data, VECS_PER_RENDER_TASK * RGBA_STRIDE);
        self.split_geometry_texture.init(device, TextureSampler::SplitGeometry, &mut frame.gpu_split_geometry, VECS_PER_SPLIT_GEOM * RGBA_STRIDE);
        self.gradient_data_texture.init(device, TextureSampler::Gradients, &mut frame.gpu_gradient_data, VECS_PER_GRADIENT_DATA * RGBA_STRIDE);
    }
}

/// The renderer is responsible for submitting to the GPU the work prepared by the
/// RenderBackend.
pub struct Renderer {
    result_rx: Receiver<ResultMsg>,
    device: Device,
    pending_texture_updates: Vec<TextureUpdateList>,
    pending_shader_updates: Vec<PathBuf>,
    current_frame: Option<RendererFrame>,

    // These are "cache shaders". These shaders are used to
    // draw intermediate results to cache targets. The results
    // of these shaders are then used by the primitive shaders.
    //cs_box_shadow: Program,
    //cs_text_run: Program,
    //cs_blur: Program,
    /// These are "cache clip shaders". These shaders are used to
    /// draw clip instances into the cached clip mask. The results
    /// of these shaders are also used by the primitive shaders.
    //cs_clip_rectangle: Program,
    //cs_clip_image: Program,
    //cs_clip_border: Program,

    // The are "primitive shaders". These shaders draw and blend
    // final results on screen. They are aware of tile boundaries.
    // Most draw directly to the framebuffer, but some use inputs
    // from the cache shaders to draw. Specifically, the box
    // shadow primitive shader stretches the box shadow cache
    // output, and the cache_image shader blits the results of
    // a cache shader (e.g. blur) to the screen.
    ps_rectangle: ProgramPair,
    ps_rectangle_clip: ProgramPair,
    ps_text_run: ProgramPair,
    ps_text_run_subpixel: ProgramPair,
    ps_image: ProgramPair,
    ps_yuv_image: Vec<ProgramPair>,
    ps_border_corner: ProgramPair,
    ps_border_edge: ProgramPair,
    ps_gradient: ProgramPair,
    ps_angle_gradient: ProgramPair,
    ps_radial_gradient: ProgramPair,
    ps_box_shadow: ProgramPair,
    ps_cache_image: ProgramPair,

    ps_blend: Program,
    ps_hw_composite: Program,
    ps_split_composite: Program,
    ps_composite: Program,

    notifier: Arc<Mutex<Option<Box<RenderNotifier>>>>,

    clear_framebuffer: bool,
    clear_color: ColorF,

    color_render_targets: Vec<TextureId>,
    alpha_render_targets: Vec<TextureId>,

    gpu_data_textures: GpuDataTextures,

    pipeline_epoch_map: HashMap<PipelineId, Epoch, BuildHasherDefault<FnvHasher>>,
    /// Used to dispatch functions to the main thread's event loop.
    /// Required to allow GLContext sharing in some implementations like WGL.
    main_thread_dispatcher: Arc<Mutex<Option<Box<RenderDispatcher>>>>,

    /// A vector for fast resolves of texture cache IDs to
    /// native texture IDs. This maps to a free-list managed
    /// by the backend thread / texture cache. We free the
    /// texture memory associated with a TextureId when its
    /// texture cache ID is freed by the texture cache, but
    /// reuse the TextureId when the texture caches's free
    /// list reuses the texture cache ID. This saves having to
    /// use a hashmap, and allows a flat vector for performance.
    cache_texture_id_map: Vec<TextureId>,

    /// A special 1x1 dummy cache texture used for shaders that expect to work
    /// with the cache but are actually running in the first pass
    /// when no target is yet provided as a cache texture input.
    dummy_cache_texture_id: TextureId,
    dummy_cache_texture_a8_id: TextureId,

    dither_matrix_texture_id: Option<TextureId>,

    /// Optional trait object that allows the client
    /// application to provide external buffers for image data.
    external_image_handler: Option<Box<ExternalImageHandler>>,

    /// Map of external image IDs to native textures.
    external_images: HashMap<(ExternalImageId, u8), TextureId, BuildHasherDefault<FnvHasher>>,

    // Optional trait object that handles WebVR commands.
    // Some WebVR commands such as SubmitFrame must be synced with the WebGL render thread.
    vr_compositor_handler: Arc<Mutex<Option<Box<VRCompositorHandler>>>>,
}

#[derive(Debug)]
pub enum InitError {
    Shader(ShaderError),
    Thread(std::io::Error),
}

impl From<ShaderError> for InitError {
    fn from(err: ShaderError) -> Self { InitError::Shader(err) }
}

impl From<std::io::Error> for InitError {
    fn from(err: std::io::Error) -> Self { InitError::Thread(err) }
}

struct ProgramPair((Program, Program));

impl ProgramPair {
    fn get(&mut self, transform_kind: TransformedRectKind) -> &mut Program {
        match transform_kind {
            TransformedRectKind::AxisAligned => &mut (self.0).0,
            TransformedRectKind::Complex => &mut (self.0).1,
        }
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
    ///    enable_profiler: false,
    /// };
    /// let (renderer, sender) = Renderer::new(opts);
    /// ```
    /// [rendereroptions]: struct.RendererOptions.html
    pub fn new(window: &glutin::Window,
               mut options: RendererOptions,
               initial_window_size: DeviceUintSize) -> Result<(Renderer, RenderApiSender), InitError> {
        let (api_tx, api_rx) = try!{ channel::msg_channel() };
        let (payload_tx, payload_rx) = try!{ channel::payload_channel() };
        let (result_tx, result_rx) = channel();

        register_thread_with_profiler("Compositor".to_owned());

        let notifier = Arc::new(Mutex::new(None));

        let mut device = Device::new(window);

        //let cs_box_shadow = create_program!(device, "cs_box_shadow");
        //let cs_text_run = create_program!(device, "cs_text_run");
        //let cs_blur = create_program!(device, "cs_blur");
        //let cs_clip_rectangle = create_program!(device, "cs_clip_rectangle");
        //let cs_clip_image = create_program!(device, "cs_clip_image");
        //let cs_clip_border = create_program!(device, "cs_clip_border");

        let ps_rectangle = create_programs!(device, "ps_rectangle");
        let ps_rectangle_clip = create_programs!(device, "ps_rectangle_clip");
        let ps_text_run = create_programs!(device, "ps_text_run");
        let ps_text_run_subpixel = create_programs!(device, "ps_text_run_subpixel");
        let ps_image = create_programs!(device, "ps_image");
        let ps_yuv_image =
            vec![ProgramPair(create_programs!(device, "ps_yuv_image_nv12_601")),
                 ProgramPair(create_programs!(device, "ps_yuv_image_planar_601")),
                 ProgramPair(create_programs!(device, "ps_yuv_image_interleaved_601")),
                 ProgramPair(create_programs!(device, "ps_yuv_image_nv12_709")),
                 ProgramPair(create_programs!(device, "ps_yuv_image_planar_709")),
                 ProgramPair(create_programs!(device, "ps_yuv_image_interleaved_709"))];

        let ps_border_corner = create_programs!(device, "ps_border_corner");
        let ps_border_edge = create_programs!(device, "ps_border_edge");

        let (ps_gradient, ps_angle_gradient, ps_radial_gradient) =
            if options.enable_dithering {
                (create_programs!(device, "ps_gradient_dither"),
                 create_programs!(device, "ps_angle_gradient_dither"),
                 create_programs!(device, "ps_radial_gradient_dither"))
            } else {
                (create_programs!(device, "ps_gradient"),
                 create_programs!(device, "ps_angle_gradient"),
                 create_programs!(device, "ps_radial_gradient"))
            };

        let ps_box_shadow = create_programs!(device, "ps_box_shadow");
        let ps_cache_image = create_programs!(device, "ps_cache_image");

        let ps_blend = create_program!(device, "ps_blend");
        let ps_hw_composite = create_program!(device, "ps_hardware_composite");
        let ps_split_composite = create_program!(device, "ps_split_composite");
        let ps_composite = create_program!(device, "ps_composite");

        let device_max_size = device.max_texture_size();
        let max_texture_size = cmp::min(device_max_size, options.max_texture_size.unwrap_or(device_max_size));

        let texture_cache = TextureCache::new(max_texture_size);
        let dummy_cache_texture_id = TextureId::new(DUMMY_RGBA8_ID, TextureTarget::Default);
        let dummy_cache_texture_a8_id = TextureId::new(DUMMY_A8_ID, TextureTarget::Default);
        let dither_matrix_texture_id = if options.enable_dithering {
                                           Some(TextureId::new(DITHER_ID, TextureTarget::Default))
                                       } else {
                                           None
                                       };

        let gpu_data_textures = GpuDataTextures::new();

        let main_thread_dispatcher = Arc::new(Mutex::new(None));
        let backend_notifier = Arc::clone(&notifier);
        let backend_main_thread_dispatcher = Arc::clone(&main_thread_dispatcher);

        let vr_compositor = Arc::new(Mutex::new(None));
        let backend_vr_compositor = Arc::clone(&vr_compositor);

        // We need a reference to the webrender context from the render backend in order to share
        // texture ids
        let context_handle = match options.renderer_kind {
            RendererKind::Native => GLContextHandleWrapper::current_native_handle(),
            RendererKind::OSMesa => GLContextHandleWrapper::current_osmesa_handle(),
        };

        let default_font_render_mode = match (options.enable_aa, options.enable_subpixel_aa) {
            (true, true) => FontRenderMode::Subpixel,
            (true, false) => FontRenderMode::Alpha,
            (false, _) => FontRenderMode::Mono,
        };

        let config = FrameBuilderConfig::new(options.enable_scrollbars,
                                             default_font_render_mode,
                                             options.debug);

        let device_pixel_ratio = options.device_pixel_ratio;
        let payload_tx_for_backend = payload_tx.clone();
        let recorder = options.recorder;
        let worker_config = ThreadPoolConfig::new().thread_name(|idx|{ format!("WebRender:Worker#{}", idx) });
        let workers = options.workers.take().unwrap_or_else(||{
            Arc::new(ThreadPool::new(worker_config).unwrap())
        });

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
                                                 context_handle,
                                                 config,
                                                 recorder,
                                                 backend_main_thread_dispatcher,
                                                 blob_image_renderer,
                                                 backend_vr_compositor,
                                                 initial_window_size);
            backend.run();
        })};

        let renderer = Renderer {
            result_rx: result_rx,
            device: device,
            current_frame: None,
            pending_texture_updates: Vec::new(),
            pending_shader_updates: Vec::new(),
            //cs_box_shadow: cs_box_shadow,
            //cs_text_run: cs_text_run,
            //cs_blur: cs_blur,
            //cs_clip_rectangle: cs_clip_rectangle,
            //cs_clip_border: cs_clip_border,
            //cs_clip_image: cs_clip_image,
            ps_rectangle: ProgramPair(ps_rectangle),
            ps_rectangle_clip: ProgramPair(ps_rectangle_clip),
            ps_text_run: ProgramPair(ps_text_run),
            ps_text_run_subpixel: ProgramPair(ps_text_run_subpixel),
            ps_image: ProgramPair(ps_image),
            ps_yuv_image: ps_yuv_image,
            ps_border_corner: ProgramPair(ps_border_corner),
            ps_border_edge: ProgramPair(ps_border_edge),
            ps_gradient: ProgramPair(ps_gradient),
            ps_angle_gradient: ProgramPair(ps_angle_gradient),
            ps_radial_gradient: ProgramPair(ps_radial_gradient),
            ps_box_shadow: ProgramPair(ps_box_shadow),
            ps_cache_image: ProgramPair(ps_cache_image),
            ps_blend: ps_blend,
            ps_hw_composite: ps_hw_composite,
            ps_split_composite: ps_split_composite,
            ps_composite: ps_composite,
            notifier: notifier,
            clear_framebuffer: options.clear_framebuffer,
            clear_color: options.clear_color,
            color_render_targets: Vec::new(),
            alpha_render_targets: Vec::new(),
            gpu_data_textures: gpu_data_textures,
            pipeline_epoch_map: HashMap::with_hasher(Default::default()),
            main_thread_dispatcher: main_thread_dispatcher,
            cache_texture_id_map: Vec::new(),
            dummy_cache_texture_id: dummy_cache_texture_id,
            dummy_cache_texture_a8_id: dummy_cache_texture_a8_id,
            dither_matrix_texture_id: dither_matrix_texture_id,
            external_image_handler: None,
            external_images: HashMap::with_hasher(Default::default()),
            vr_compositor_handler: vr_compositor,
        };

        let sender = RenderApiSender::new(api_tx, payload_tx);
        Ok((renderer, sender))
    }

    fn get_yuv_shader_index(buffer_kind: ImageBufferKind, format: YuvFormat, color_space: YuvColorSpace) -> usize {
        ((buffer_kind as usize) * YUV_FORMATS.len() + (format as usize)) * YUV_COLOR_SPACES.len() + (color_space as usize)
    }

    /// Sets the new RenderNotifier.
    ///
    /// The RenderNotifier will be called when processing e.g. of a (scrolling) frame is done,
    /// and therefore the screen should be updated.
    pub fn set_render_notifier(&self, notifier: Box<RenderNotifier>) {
        let mut notifier_arc = self.notifier.lock().unwrap();
        *notifier_arc = Some(notifier);
    }

    /// Sets the new main thread dispatcher.
    ///
    /// Allows to dispatch functions to the main thread's event loop.
    pub fn set_main_thread_dispatcher(&self, dispatcher: Box<RenderDispatcher>) {
        let mut dispatcher_arc = self.main_thread_dispatcher.lock().unwrap();
        *dispatcher_arc = Some(dispatcher);
    }

    /// Sets the VRCompositorHandler.
    ///
    /// It's used to handle WebVR render commands.
    /// Some WebVR commands such as Vsync and SubmitFrame must be called in the WebGL render thread.
    pub fn set_vr_compositor_handler(&self, creator: Box<VRCompositorHandler>) {
        let mut handler_arc = self.vr_compositor_handler.lock().unwrap();
        *handler_arc = Some(creator);
    }

    /// Returns the Epoch of the current frame in a pipeline.
    pub fn current_epoch(&self, pipeline_id: PipelineId) -> Option<Epoch> {
        self.pipeline_epoch_map.get(&pipeline_id).cloned()
    }

    /// Returns a HashMap containing the pipeline ids that have been received by the renderer and
    /// their respective epochs since the last time the method was called.
    pub fn flush_rendered_epochs(&mut self) -> HashMap<PipelineId, Epoch, BuildHasherDefault<FnvHasher>> {
        mem::replace(&mut self.pipeline_epoch_map, HashMap::default())
    }

    /// Processes the result queue.
    ///
    /// Should be called before `render()`, as texture cache updates are done here.
    pub fn update(&mut self) {
        profile_scope!("update");

        // Pull any pending results and return the most recent.
        while let Ok(msg) = self.result_rx.try_recv() {
            match msg {
                ResultMsg::NewFrame(frame, texture_update_list) => {
                    self.pending_texture_updates.push(texture_update_list);

                    // Update the list of available epochs for use during reftests.
                    // This is a workaround for https://github.com/servo/servo/issues/13149.
                    for (pipeline_id, epoch) in &frame.pipeline_epoch_map {
                        self.pipeline_epoch_map.insert(*pipeline_id, *epoch);
                    }

                    self.current_frame = Some(frame);
                }
                ResultMsg::RefreshShader(path) => {
                    self.pending_shader_updates.push(path);
                }
            }
        }
    }

    // Get the real (OpenGL) texture ID for a given source texture.
    // For a texture cache texture, the IDs are stored in a vector
    // map for fast access. For WebGL textures, the native texture ID
    // is stored inline. When we add support for external textures,
    // we will add a callback here that is able to ask the caller
    // for the image data.
    fn resolve_source_texture(&mut self, texture_id: &SourceTexture) -> TextureId {
        match *texture_id {
            SourceTexture::Invalid => TextureId::invalid(),
            SourceTexture::WebGL(id) => TextureId::new(id, TextureTarget::Default),
            SourceTexture::External(external_image) => {
                *self.external_images
                     .get(&(external_image.id, external_image.channel_index))
                     .expect("BUG: External image should be resolved by now!")
            }
            SourceTexture::TextureCache(index) => {
                self.cache_texture_id_map[index.0]
            }
        }
    }

    /// Set a callback for handling external images.
    pub fn set_external_image_handler(&mut self, handler: Box<ExternalImageHandler>) {
        self.external_image_handler = Some(handler);
    }

    /// Retrieve (and clear) the current list of recorded frame profiles.
    /*pub fn get_frame_profiles(&mut self) -> (Vec<CpuProfile>, Vec<GpuProfile>) {
        let cpu_profiles = self.cpu_profiles.drain(..).collect();
        let gpu_profiles = self.gpu_profiles.drain(..).collect();
        (cpu_profiles, gpu_profiles)
    }*/

    /// Renders the current frame.
    ///
    /// A Frame is supplied by calling [`set_display_list()`][newframe].
    /// [newframe]: ../../webrender_traits/struct.RenderApi.html#method.set_display_list
    pub fn render(&mut self, framebuffer_size: DeviceUintSize) {
        profile_scope!("render");

        if let Some(mut frame) = self.current_frame.take() {
            if let Some(ref mut frame) = frame.frame {
                // self.device.begin_frame(frame.device_pixel_ratio);
                // self.device.disable_scissor();
                // self.device.disable_depth();
                // self.device.set_blend(false);

                // self.update_shaders();
                self.update_texture_cache();
                self.draw_tile_frame(frame, &framebuffer_size);
                // self.device.end_frame();
                self.device.flush();
            }

            // Restore frame - avoid borrow checker!
            self.current_frame = Some(frame);
        }
    }

    pub fn layers_are_bouncing_back(&self) -> bool {
        match self.current_frame {
            None => false,
            Some(ref current_frame) => !current_frame.layers_bouncing_back.is_empty(),
        }
    }

/*
    fn update_shaders(&mut self) {
        let update_uniforms = !self.pending_shader_updates.is_empty();

        for path in self.pending_shader_updates.drain(..) {
            panic!("todo");
            //self.device.refresh_shader(path);
        }

        if update_uniforms {
            self.update_uniform_locations();
        }
    }
*/

    fn update_texture_cache(&mut self) {
        //let _gm = GpuMarker::new(self.device.rc_gl(), "texture cache update");
        let mut pending_texture_updates = mem::replace(&mut self.pending_texture_updates, vec![]);
        for update_list in pending_texture_updates.drain(..) {
            for update in update_list.updates {
                match update.op {
                    TextureUpdateOp::Create { width, height, format, filter, mode, data } => {
                        let CacheTextureId(cache_texture_index) = update.id;
                        if self.cache_texture_id_map.len() == cache_texture_index {
                            // Create a new native texture, as requested by the texture cache.
                            /*let texture_id = self.device
                                                 .create_texture_ids(1, TextureTarget::Default, format)[0];*/
                            let texture_id = self.device.create_texture_id(TextureTarget::Default, format);
                            self.cache_texture_id_map.push(texture_id);
                        }
                        let texture_id = self.cache_texture_id_map[cache_texture_index];

                        if let Some(image) = data {
                            match image {
                                ImageData::Raw(raw) => {
                                    self.device.init_texture(texture_id,
                                                             width,
                                                             height,
                                                             format,
                                                             filter,
                                                             mode,
                                                             Some(raw.as_slice()));
                                }
                                ImageData::External(ext_image) => {
                                    match ext_image.image_type {
                                        ExternalImageType::ExternalBuffer => {
                                            let handler = self.external_image_handler
                                                              .as_mut()
                                                              .expect("Found external image, but no handler set!");

                                            match handler.lock(ext_image.id, ext_image.channel_index).source {
                                                ExternalImageSource::RawData(raw) => {
                                                    self.device.init_texture(texture_id,
                                                                             width,
                                                                             height,
                                                                             format,
                                                                             filter,
                                                                             mode,
                                                                             Some(raw));
                                                }
                                                _ => panic!("No external buffer found"),
                                            };
                                            handler.unlock(ext_image.id, ext_image.channel_index);
                                        }
                                        ExternalImageType::Texture2DHandle |
                                        ExternalImageType::TextureRectHandle |
                                        ExternalImageType::TextureExternalHandle => {
                                            panic!("External texture handle should not use TextureUpdateOp::Create.");
                                        }
                                    }
                                }
                                _ => {
                                    panic!("No suitable image buffer for TextureUpdateOp::Create.");
                                }
                            }
                        } else {
                            self.device.init_texture(texture_id,
                                                     width,
                                                     height,
                                                     format,
                                                     filter,
                                                     mode,
                                                     None);
                        }
                    }
                    TextureUpdateOp::Grow { width, height, format, filter, mode } => {
                        let texture_id = self.cache_texture_id_map[update.id.0];
                        self.device.resize_texture(texture_id,
                                                   width,
                                                   height,
                                                   format,
                                                   filter,
                                                   mode);
                    }
                    TextureUpdateOp::Update { page_pos_x, page_pos_y, width, height, data, stride, offset } => {
                        let texture_id = self.cache_texture_id_map[update.id.0];
                        self.device.update_texture(texture_id,
                                                   page_pos_x,
                                                   page_pos_y,
                                                   width, height, stride,
                                                   &data[offset as usize..]);
                    }
                    TextureUpdateOp::UpdateForExternalBuffer { rect, id, channel_index, stride, offset } => {
                        let handler = self.external_image_handler
                                          .as_mut()
                                          .expect("Found external image, but no handler set!");
                        let device = &mut self.device;
                        let cached_id = self.cache_texture_id_map[update.id.0];

                        match handler.lock(id, channel_index).source {
                            ExternalImageSource::RawData(data) => {
                                device.update_texture(cached_id,
                                                      rect.origin.x,
                                                      rect.origin.y,
                                                      rect.size.width,
                                                      rect.size.height,
                                                      stride,
                                                      &data[offset as usize..]);
                            }
                            _ => panic!("No external buffer found"),
                        };
                        handler.unlock(id, channel_index);
                    }
                    TextureUpdateOp::Free => {
                        let texture_id = self.cache_texture_id_map[update.id.0];
                        self.device.deinit_texture(texture_id);
                    }
                }
            }
        }
    }

    /*fn draw_instanced_batch<T>(&mut self,
                               data: &[T],
                               vao: VAOId,
                               shader: ProgramId,
                               textures: &BatchTextures,
                               projection: &Matrix4D<f32>) {
        self.device.bind_vao(vao);
        self.device.bind_program(shader, projection);

        for i in 0..textures.colors.len() {
            let texture_id = self.resolve_source_texture(&textures.colors[i]);
            self.device.bind_texture(TextureSampler::color(i), texture_id);
        }

        // TODO: this probably isn't the best place for this.
        if let Some(id) = self.dither_matrix_texture_id {
            self.device.bind_texture(TextureSampler::Dither, id);
        }

        if self.enable_batcher {
            self.device.update_vao_instances(vao, data, VertexUsageHint::Stream);
            self.device.draw_indexed_triangles_instanced_u16(6, data.len() as i32);
            self.profile_counters.draw_calls.inc();
        } else {
            for i in 0 .. data.len() {
                self.device.update_vao_instances(vao, &data[i..i+1], VertexUsageHint::Stream);
                self.device.draw_triangles_u16(0, 6);
                self.profile_counters.draw_calls.inc();
            }
        }

        self.profile_counters.vertices.add(6 * data.len());
        self.profile_counters.draw_calls.inc();
    }*/

    fn submit_batch(&mut self,
                    batch: &PrimitiveBatch,
                    projection: &Matrix4D<f32>,
                    _render_task_data: &[RenderTaskData],
                    _render_target: Option<(TextureId, i32)>,
                    _target_dimensions: DeviceUintSize) {
        let transform_kind = batch.key.flags.transform_kind();
        let needs_clipping = batch.key.flags.needs_clipping();
        debug_assert!(!needs_clipping ||
                      match batch.key.blend_mode {
                          BlendMode::Alpha |
                          BlendMode::PremultipliedAlpha |
                          BlendMode::Subpixel(..) => true,
                          BlendMode::None => false,
                      });

        match batch.key.kind {
            AlphaBatchKind::YuvImage(..) => {
                self.device.flush();
                for i in 0..batch.key.textures.colors.len() {
                    let texture_id = self.resolve_source_texture(&batch.key.textures.colors[i]);
                    self.device.bind_yuv_texture(TextureSampler::color(i), texture_id);
                }
            },
            _ => {
                for i in 0..batch.key.textures.colors.len() {
                    let texture_id = self.resolve_source_texture(&batch.key.textures.colors[i]);
                    self.device.bind_texture(TextureSampler::color(i), texture_id);
                }
            },
        }

        {
            let mut program = match batch.key.kind {
                AlphaBatchKind::Rectangle => {
                    if needs_clipping {
                        self.ps_rectangle_clip.get(transform_kind)
                    } else {
                        self.ps_rectangle.get(transform_kind)
                    }
                },
                AlphaBatchKind::Composite => &mut self.ps_composite,
                AlphaBatchKind::SplitComposite => &mut self.ps_split_composite,
                AlphaBatchKind::HardwareComposite => &mut self.ps_hw_composite,
                AlphaBatchKind::Blend => &mut self.ps_blend,
                AlphaBatchKind::TextRun => {
                    match batch.key.blend_mode {
                        BlendMode::Subpixel(..) => self.ps_text_run_subpixel.get(transform_kind),
                        _ => self.ps_text_run.get(transform_kind),
                    }
                },
                AlphaBatchKind::Image(..) => self.ps_image.get(transform_kind),
                AlphaBatchKind::YuvImage(_, format, color_space) => {
                    let shader_index = Renderer::get_yuv_shader_index(ImageBufferKind::Texture2D,
                                                                      format,
                                                                      color_space);
                    self.ps_yuv_image[shader_index].get(transform_kind)
                },
                AlphaBatchKind::BorderCorner => self.ps_border_corner.get(transform_kind),
                AlphaBatchKind::BorderEdge => self.ps_border_edge.get(transform_kind),
                AlphaBatchKind::AlignedGradient => self.ps_gradient.get(transform_kind),
                AlphaBatchKind::AngleGradient => self.ps_angle_gradient.get(transform_kind),
                AlphaBatchKind::RadialGradient => self.ps_radial_gradient.get(transform_kind),
                AlphaBatchKind::BoxShadow => self.ps_box_shadow.get(transform_kind),
                AlphaBatchKind::CacheImage => self.ps_cache_image.get(transform_kind),
            };

            if let Some(id) = self.dither_matrix_texture_id {
                self.device.bind_texture(TextureSampler::Dither, id);
            }
            self.device.draw(&mut program, projection, &batch.instances, &batch.key.blend_mode);
        }

        // Handle special case readback for composites.
        /*if batch.key.kind == AlphaBatchKind::Composite {
            // composites can't be grouped together because
            // they may overlap and affect each other.
            debug_assert!(batch.instances.len() == 1);
            let instance = &batch.instances[0];

            // TODO(gw): This code branch is all a bit hacky. We rely
            // on pulling specific values from the render target data
            // and also cloning the single primitive instance to be
            // able to pass to draw_instanced_batch(). We should
            // think about a cleaner way to achieve this!

            // Before submitting the composite batch, do the
            // framebuffer readbacks that are needed for each
            // composite operation in this batch.
            let cache_texture_dimensions = self.device.get_texture_dimensions(cache_texture);

            let backdrop = &render_task_data[instance.task_index as usize];
            let readback = &render_task_data[instance.user_data[0] as usize];
            let source = &render_task_data[instance.user_data[1] as usize];

            // Bind the FBO to blit the backdrop to.
            // Called per-instance in case the layer (and therefore FBO)
            // changes. The device will skip the GL call if the requested
            // target is already bound.
            let cache_draw_target = (cache_texture, readback.data[4] as i32);
            self.device.bind_draw_target(Some(cache_draw_target), Some(cache_texture_dimensions));

            let src_x = backdrop.data[0] - backdrop.data[4] + source.data[4];
            let src_y = backdrop.data[1] - backdrop.data[5] + source.data[5];

            let dest_x = readback.data[0];
            let dest_y = readback.data[1];

            let width = readback.data[2];
            let height = readback.data[3];

            let mut src = DeviceIntRect::new(DeviceIntPoint::new(src_x as i32, src_y as i32),
                                             DeviceIntSize::new(width as i32, height as i32));
            let mut dest = DeviceIntRect::new(DeviceIntPoint::new(dest_x as i32, dest_y as i32),
                                              DeviceIntSize::new(width as i32, height as i32));

            // Need to invert the y coordinates and flip the image vertically when
            // reading back from the framebuffer.
            if render_target.is_none() {
                src.origin.y = target_dimensions.height as i32 - src.size.height - src.origin.y;
                dest.origin.y += dest.size.height;
                dest.size.height = -dest.size.height;
            }

            self.device.blit_render_target(render_target,
                                           Some(src),
                                           dest);

            // Restore draw target to current pass render target + layer.
            self.device.bind_draw_target(render_target, Some(target_dimensions));
        }

        self.draw_instanced_batch(&batch.instances,
                                  vao,
                                  shader,
                                  &batch.key.textures,
                                  projection);*/
    }

    fn draw_color_target(&mut self,
                         render_target: Option<(TextureId, i32)>,
                         target: &ColorRenderTarget,
                         target_size: DeviceUintSize,
                         _color_cache_texture: TextureId,
                         clear_color: Option<[f32; 4]>,
                         render_task_data: &[RenderTaskData],
                         projection: &Matrix4D<f32>) {
        {
            match render_target {
                Some(..) => {
                    // TODO(gw): Applying a scissor rect and minimal clear here
                    // is a very large performance win on the Intel and nVidia
                    // GPUs that I have tested with. It's possible it may be a
                    // performance penalty on other GPU types - we should test this
                    // and consider different code paths.
                    // self.device.clear_target_rect(clear_color,
                    //                               Some(1.0),
                    //                               target.used_rect());
                }
                None => {
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
        /*if !target.vertical_blurs.is_empty() || !target.horizontal_blurs.is_empty() {
            let _gm = self.gpu_profile.add_marker(GPU_TAG_BLUR);
            let vao = self.blur_vao_id;

            self.device.set_blend(false);
            let shader = self.cs_blur.get(&mut self.device).unwrap();

            self.draw_instanced_batch(&target.vertical_blurs,
                                      vao,
                                      shader,
                                      &BatchTextures::no_texture(),
                                      &projection);
            self.draw_instanced_batch(&target.horizontal_blurs,
                                      vao,
                                      shader,
                                      &BatchTextures::no_texture(),
                                      &projection);
        }

        // Draw any box-shadow caches for this target.
        if !target.box_shadow_cache_prims.is_empty() {
            self.device.set_blend(false);
            let _gm = self.gpu_profile.add_marker(GPU_TAG_CACHE_BOX_SHADOW);
            let vao = self.prim_vao_id;
            let shader = self.cs_box_shadow.get(&mut self.device).unwrap();
            self.draw_instanced_batch(&target.box_shadow_cache_prims,
                                      vao,
                                      shader,
                                      &BatchTextures::no_texture(),
                                      &projection);
        }

        // Draw any textrun caches for this target. For now, this
        // is only used to cache text runs that are to be blurred
        // for text-shadow support. In the future it may be worth
        // considering using this for (some) other text runs, since
        // it removes the overhead of submitting many small glyphs
        // to multiple tiles in the normal text run case.
        if !target.text_run_cache_prims.is_empty() {
            self.device.set_blend(true);
            self.device.set_blend_mode_alpha();

            let _gm = self.gpu_profile.add_marker(GPU_TAG_CACHE_TEXT_RUN);
            let vao = self.prim_vao_id;
            let shader = self.cs_text_run.get(&mut self.device).unwrap();

            self.draw_instanced_batch(&target.text_run_cache_prims,
                                      vao,
                                      shader,
                                      &target.text_run_textures,
                                      &projection);
        }*/

        for batch in &target.alpha_batcher.batch_list.opaque_batches {
            self.submit_batch(batch,
                              &projection,
                              render_task_data,
                              render_target,
                              target_size);
        }

        for batch in &target.alpha_batcher.batch_list.alpha_batches {
            self.submit_batch(batch,
                              &projection,
                              render_task_data,
                              //color_cache_texture,
                              render_target,
                              target_size);

        }
    }

    fn draw_alpha_target(&mut self,
                         _render_target: (TextureId, i32),
                         _target: &AlphaRenderTarget,
                         _target_size: DeviceUintSize,
                         _projection: &Matrix4D<f32>) {
        {
            // let _gm = self.gpu_profile.add_marker(GPU_TAG_SETUP_TARGET);
            // self.device.bind_draw_target(Some(render_target), Some(target_size));
            // self.device.disable_depth();
            // self.device.disable_depth_write();

            // TODO(gw): Applying a scissor rect and minimal clear here
            // is a very large performance win on the Intel and nVidia
            // GPUs that I have tested with. It's possible it may be a
            // performance penalty on other GPU types - we should test this
            // and consider different code paths.
            //let clear_color = [1.0, 1.0, 1.0, 0.0];
            // self.device.clear_target_rect(Some(clear_color),
            //                               None,
            //                               target.used_rect());
        }

        // Draw the clip items into the tiled alpha mask.
        /*{
            let _gm = self.gpu_profile.add_marker(GPU_TAG_CACHE_CLIP);
            let vao = self.clip_vao_id;

            // If we have border corner clips, the first step is to clear out the
            // area in the clip mask. This allows drawing multiple invididual clip
            // in regions below.
            if !target.clip_batcher.border_clears.is_empty() {
                let _gm2 = GpuMarker::new(self.device.rc_gl(), "clip borders [clear]");
                self.device.set_blend(false);
                let shader = self.cs_clip_border.get(&mut self.device).unwrap();
                self.draw_instanced_batch(&target.clip_batcher.border_clears,
                                          vao,
                                          shader,
                                          &BatchTextures::no_texture(),
                                          &projection);
            }

            // Draw any dots or dashes for border corners.
            if !target.clip_batcher.borders.is_empty() {
                let _gm2 = GpuMarker::new(self.device.rc_gl(), "clip borders");
                // We are masking in parts of the corner (dots or dashes) here.
                // Blend mode is set to max to allow drawing multiple dots.
                // The individual dots and dashes in a border never overlap, so using
                // a max blend mode here is fine.
                self.device.set_blend(true);
                self.device.set_blend_mode_max();
                let shader = self.cs_clip_border.get(&mut self.device).unwrap();
                self.draw_instanced_batch(&target.clip_batcher.borders,
                                          vao,
                                          shader,
                                          &BatchTextures::no_texture(),
                                          &projection);
            }

            // switch to multiplicative blending
            self.device.set_blend(true);
            self.device.set_blend_mode_multiply();

            // draw rounded cornered rectangles
            if !target.clip_batcher.rectangles.is_empty() {
                let _gm2 = GpuMarker::new(self.device.rc_gl(), "clip rectangles");
                let shader = self.cs_clip_rectangle.get(&mut self.device).unwrap();
                self.draw_instanced_batch(&target.clip_batcher.rectangles,
                                          vao,
                                          shader,
                                          &BatchTextures::no_texture(),
                                          &projection);
            }
            // draw image masks
            for (mask_texture_id, items) in target.clip_batcher.images.iter() {
                let _gm2 = GpuMarker::new(self.device.rc_gl(), "clip images");
                let textures = BatchTextures {
                    colors: [
                        mask_texture_id.clone(),
                        SourceTexture::Invalid,
                        SourceTexture::Invalid,
                    ]
                };
                let shader = self.cs_clip_image.get(&mut self.device).unwrap();
                self.draw_instanced_batch(items,
                                          vao,
                                          shader,
                                          &textures,
                                          &projection);
            }
        }*/
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
                //GpuMarker::fire(self.device.gl(), "deferred resolve");
                let props = &deferred_resolve.image_properties;
                let ext_image = props.external_image
                                     .expect("BUG: Deferred resolves must be external images!");
                let image = handler.lock(ext_image.id, ext_image.channel_index);
                let texture_target = match ext_image.image_type {
                    ExternalImageType::Texture2DHandle => TextureTarget::Default,
                    ExternalImageType::TextureRectHandle => TextureTarget::Rect,
                    ExternalImageType::TextureExternalHandle => TextureTarget::External,
                    ExternalImageType::ExternalBuffer => {
                        panic!("{:?} is not a suitable image type in update_deferred_resolves().",
                            ext_image.image_type);
                    }
                };

                let texture_id = match image.source {
                    ExternalImageSource::NativeTexture(texture_id) => TextureId::new(texture_id, texture_target),
                    _ => panic!("No native texture found."),
                };

                self.external_images.insert((ext_image.id, ext_image.channel_index), texture_id);
                let resource_rect_index = deferred_resolve.resource_address.0 as usize;
                let resource_rect = &mut frame.gpu_resource_rects[resource_rect_index];
                resource_rect.uv0 = DevicePoint::new(image.u0, image.v0);
                resource_rect.uv1 = DevicePoint::new(image.u1, image.v1);
            }
        }
    }

    fn unlock_external_images(&mut self) {
        if !self.external_images.is_empty() {
            let handler = self.external_image_handler
                              .as_mut()
                              .expect("Found external image, but no handler set!");

            for (ext_data, _) in self.external_images.drain() {
                handler.unlock(ext_data.0, ext_data.1);
            }
        }
    }

    fn draw_tile_frame(&mut self,
                       frame: &mut Frame,
                       framebuffer_size: &DeviceUintSize) {
        //let _gm = GpuMarker::new(self.device.rc_gl(), "tile frame draw");
        self.update_deferred_resolves(frame);

        // Some tests use a restricted viewport smaller than the main screen size.
        // Ensure we clear the framebuffer in these tests.
        // TODO(gw): Find a better solution for this?
        let needs_clear = frame.window_size.width < framebuffer_size.width ||
                          frame.window_size.height < framebuffer_size.height;

        if frame.passes.is_empty() {
            //self.device.clear_target(Some(self.clear_color.to_array()), Some(1.0));
        } else {
            // Assign render targets to the passes.
            for pass in &mut frame.passes {
                debug_assert!(pass.color_texture_id.is_none());
                debug_assert!(pass.alpha_texture_id.is_none());

                if pass.needs_render_target_kind(RenderTargetKind::Color) {
                    pass.color_texture_id = Some(self.color_render_targets
                                                     .pop()
                                                     .unwrap_or_else(|| {
                                                         self.device.create_texture_id(TextureTarget::Default, ImageFormat::RGBA8)
                                                      }));
                }

                if pass.needs_render_target_kind(RenderTargetKind::Alpha) {
                    pass.alpha_texture_id = Some(self.alpha_render_targets
                                                     .pop()
                                                     .unwrap_or_else(|| {
                                                         self.device.create_texture_id(TextureTarget::Default, ImageFormat::A8)
                                                      }));
                }
            }

            // Init textures and render targets to match this scene.
            for pass in &frame.passes {
                if let Some(texture_id) = pass.color_texture_id {
                    let target_count = pass.required_target_count(RenderTargetKind::Color);
                    self.device.init_texture(texture_id,
                                             frame.cache_size.width as u32,
                                             frame.cache_size.height as u32,
                                             ImageFormat::RGBA8,
                                             TextureFilter::Linear,
                                             RenderTargetMode::LayerRenderTarget(target_count as i32),
                                             None);
                }
                if let Some(texture_id) = pass.alpha_texture_id {
                    let target_count = pass.required_target_count(RenderTargetKind::Alpha);
                    self.device.init_texture(texture_id,
                                             frame.cache_size.width as u32,
                                             frame.cache_size.height as u32,
                                             ImageFormat::A8,
                                             TextureFilter::Nearest,
                                             RenderTargetMode::LayerRenderTarget(target_count as i32),
                                             None);
                }
            }

            self.gpu_data_textures.init_frame(&mut self.device, frame);

            let mut src_color_id = self.dummy_cache_texture_id;
            let mut src_alpha_id = self.dummy_cache_texture_a8_id;

            for pass in &mut frame.passes {
                let size;
                let clear_color;
                let projection;

                if pass.is_framebuffer {
                    clear_color = if self.clear_framebuffer || needs_clear {
                        Some(frame.background_color.map_or(self.clear_color.to_array(), |color| {
                            color.to_array()
                        }))
                    } else {
                        None
                    };
                    size = framebuffer_size;
                    projection = Matrix4D::ortho(0.0,
                                                 size.width as f32,
                                                 size.height as f32,
                                                 0.0,
                                                 ORTHO_NEAR_PLANE,
                                                 ORTHO_FAR_PLANE)
                } else {
                    size = &frame.cache_size;
                    clear_color = Some([0.0, 0.0, 0.0, 0.0]);
                    projection = Matrix4D::ortho(0.0,
                                                 size.width as f32,
                                                 0.0,
                                                 size.height as f32,
                                                 ORTHO_NEAR_PLANE,
                                                 ORTHO_FAR_PLANE);
                }

                 self.device.bind_texture(TextureSampler::CacheA8, src_alpha_id);
                 self.device.bind_texture(TextureSampler::CacheRGBA8, src_color_id);

                for (target_index, target) in pass.alpha_targets.targets.iter().enumerate() {
                    self.draw_alpha_target((pass.alpha_texture_id.unwrap(), target_index as i32),
                                           target,
                                           *size,
                                           &projection);
                }

                for (target_index, target) in pass.color_targets.targets.iter().enumerate() {
                    let render_target = pass.color_texture_id.map(|texture_id| {
                        (texture_id, target_index as i32)
                    });
                    self.draw_color_target(render_target,
                                           target,
                                           *size,
                                           src_color_id,
                                           clear_color,
                                           &frame.render_task_data,
                                           &projection);

                }

                 src_color_id = pass.color_texture_id.unwrap_or(self.dummy_cache_texture_id);
                 src_alpha_id = pass.alpha_texture_id.unwrap_or(self.dummy_cache_texture_a8_id);

                // Return the texture IDs to the pool for next frame.
                if let Some(texture_id) = pass.color_texture_id.take() {
                    self.color_render_targets.push(texture_id);
                }
                if let Some(texture_id) = pass.alpha_texture_id.take() {
                    self.alpha_render_targets.push(texture_id);
                }
            }

            self.color_render_targets.reverse();
            self.alpha_render_targets.reverse();
            // self.draw_render_target_debug(framebuffer_size);
        }

        self.unlock_external_images();
    }

    /*pub fn debug_renderer<'a>(&'a mut self) -> &'a mut DebugRenderer {
        &mut self.debug
    }

    pub fn get_profiler_enabled(&mut self) -> bool {
        self.enable_profiler
    }

    pub fn set_profiler_enabled(&mut self, enabled: bool) {
        self.enable_profiler = enabled;
    }

    pub fn save_cpu_profile(&self, filename: &str) {
        write_profile(filename);
    }

    fn draw_render_target_debug(&mut self,
                                framebuffer_size: &DeviceUintSize) {
        if self.render_target_debug {
            // TODO(gw): Make the layout of the render targets a bit more sophisticated.
            // Right now, it just draws them in one row at the bottom of the screen,
            // with a fixed size.
            let rt_debug_x0 = 16;
            let rt_debug_y0 = 16;
            let rt_debug_spacing = 16;
            let rt_debug_size = 512;
            let mut current_target = 0;

            for texture_id in self.color_render_targets.iter().chain(self.alpha_render_targets.iter()) {
                let layer_count = self.device.get_render_target_layer_count(*texture_id);
                for layer_index in 0..layer_count {
                    let x0 = rt_debug_x0 + (rt_debug_spacing + rt_debug_size) * current_target;
                    let y0 = rt_debug_y0;

                    // If we have more targets than fit on one row in screen, just early exit.
                    if x0 > framebuffer_size.width as i32 {
                        return;
                    }

                    let dest_rect = DeviceIntRect::new(DeviceIntPoint::new(x0, y0),
                                                       DeviceIntSize::new(rt_debug_size, rt_debug_size));
                    self.device.blit_render_target(Some((*texture_id, layer_index as i32)),
                                                   None,
                                                   dest_rect);

                    current_target += 1;
                }
            }
        }
    }*/

    // De-initialize the Renderer safely, assuming the GL is still alive and active.
    /*pub fn deinit(mut self) {
        //Note: this is a fake frame, only needed because texture deletion is require to happen inside a frame
        // self.device.begin_frame(1.0);
        // self.device.deinit_texture(self.dummy_cache_texture_id);
        // self.device.end_frame();
    }*/
}

pub enum ExternalImageSource<'a> {
    RawData(&'a [u8]),      // raw buffers.
    NativeTexture(u32),     // Is a gl::GLuint texture handle
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

pub struct RendererOptions {
    pub device_pixel_ratio: f32,
    pub resource_override_path: Option<PathBuf>,
    pub enable_aa: bool,
    pub enable_dithering: bool,
    pub enable_profiler: bool,
    pub max_recorded_profiles: usize,
    pub debug: bool,
    pub enable_scrollbars: bool,
    pub precache_shaders: bool,
    pub renderer_kind: RendererKind,
    pub enable_subpixel_aa: bool,
    pub clear_framebuffer: bool,
    pub clear_color: ColorF,
    pub enable_batcher: bool,
    pub render_target_debug: bool,
    pub max_texture_size: Option<u32>,
    pub workers: Option<Arc<ThreadPool>>,
    pub blob_image_renderer: Option<Box<BlobImageRenderer>>,
    pub recorder: Option<Box<ApiRecordingReceiver>>,
}

impl Default for RendererOptions {
    fn default() -> RendererOptions {
        RendererOptions {
            device_pixel_ratio: 1.0,
            resource_override_path: None,
            enable_aa: true,
            enable_dithering: true,
            enable_profiler: false,
            max_recorded_profiles: 0,
            debug: false,
            enable_scrollbars: false,
            precache_shaders: false,
            renderer_kind: RendererKind::Native,
            enable_subpixel_aa: false,
            clear_framebuffer: true,
            clear_color: ColorF::new(1.0, 1.0, 1.0, 1.0),
            enable_batcher: true,
            render_target_debug: false,
            max_texture_size: None,
            workers: None,
            blob_image_renderer: None,
            recorder: None,
        }
    }
}
