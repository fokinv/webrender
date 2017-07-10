/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate euclid;
extern crate winit;
extern crate webrender;
extern crate webrender_traits;

use app_units::Au;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use webrender_traits::{ClipRegionToken, ColorF, DisplayListBuilder, Epoch, GlyphInstance};
use webrender_traits::{DeviceIntPoint, DeviceUintSize, LayoutPoint, LayoutRect, LayoutSize};
use webrender_traits::{ImageData, ImageDescriptor, ImageFormat};
use webrender_traits::{PipelineId, RenderApi, TransformStyle, BoxShadowClipMode};
use euclid::vec2;

fn push_sub_clip(api: &RenderApi, builder: &mut DisplayListBuilder, bounds: &LayoutRect)
                 -> ClipRegionToken {
    let mask_image = api.generate_image_key();
    api.add_image(mask_image,
                  ImageDescriptor::new(2, 2, ImageFormat::A8, true),
                  ImageData::new(vec![0, 80, 180, 255]),
                  None);
    let mask = webrender_traits::ImageMask {
        image: mask_image,
        rect: LayoutRect::new(LayoutPoint::new(75.0, 75.0), LayoutSize::new(100.0, 100.0)),
        repeat: false,
    };
    let complex = webrender_traits::ComplexClipRegion::new(
        LayoutRect::new(LayoutPoint::new(50.0, 50.0), LayoutSize::new(100.0, 100.0)),
        webrender_traits::BorderRadius::uniform(20.0));

    builder.push_clip_region(bounds, vec![/*complex*/], None/*Some(mask)*/)
}


fn main() {
    let args: Vec<String> = env::args().collect();
    let res_path = if args.len() > 1 {
        Some(PathBuf::from(&args[1]))
    } else {
        None
    };

    let events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new()
                .with_title("WebRender Sample")
                .build(&events_loop)
                .unwrap();

    let (width, height) = window.get_inner_size_pixels().unwrap();

    let opts = webrender::RendererOptions {
        resource_override_path: res_path,
        debug: true,
        precache_shaders: true,
        device_pixel_ratio: window.hidpi_factor(),
        .. Default::default()
    };

    let size = DeviceUintSize::new(width, height);
    let (mut window, mut renderer, sender) = webrender::renderer::Renderer::new(window, opts, size).unwrap();
    let api = sender.create_api();

    let epoch = Epoch(0);
    let root_background_color = ColorF::new(0.3, 0.0, 0.0, 1.0);

    let pipeline_id = PipelineId(0, 0);
    let layout_size = LayoutSize::new(width as f32, height as f32);
    let mut builder = webrender_traits::DisplayListBuilder::new(pipeline_id, layout_size);

    let bounds = LayoutRect::new(LayoutPoint::zero(), layout_size);
    builder.push_stacking_context(webrender_traits::ScrollPolicy::Scrollable,
                                  bounds,
                                  None,
                                  TransformStyle::Flat,
                                  None,
                                  webrender_traits::MixBlendMode::Normal,
                                  Vec::new());

    let clip = push_sub_clip(&api, &mut builder, &bounds);
    builder.push_rect(LayoutRect::new(LayoutPoint::new(100.0, 100.0), LayoutSize::new(100.0, 100.0)),
                      clip,
                      ColorF::new(0.0, 1.0, 0.0, 1.0));

    /*let clip = push_sub_clip(&api, &mut builder, &bounds);
    builder.push_rect(LayoutRect::new(LayoutPoint::new(250.0, 100.0), LayoutSize::new(100.0, 100.0)),
                      clip,
                      ColorF::new(0.0, 1.0, 0.0, 1.0));
    let border_side = webrender_traits::BorderSide {
        color: ColorF::new(0.0, 0.0, 1.0, 1.0),
        style: webrender_traits::BorderStyle::Groove,
    };
    let border_widths = webrender_traits::BorderWidths {
        top: 10.0,
        left: 10.0,
        bottom: 10.0,
        right: 10.0,
    };
    let border_details = webrender_traits::BorderDetails::Normal(webrender_traits::NormalBorder {
        top: border_side,
        right: border_side,
        bottom: border_side,
        left: border_side,
        radius: webrender_traits::BorderRadius::uniform(20.0),
    });

    let clip = push_sub_clip(&api, &mut builder, &bounds);
    builder.push_border(LayoutRect::new(LayoutPoint::new(100.0, 100.0), LayoutSize::new(100.0, 100.0)),
                        clip,
                        border_widths,
                        border_details);*/

    builder.pop_stacking_context();

    api.set_display_list(
        Some(root_background_color),
        epoch,
        LayoutSize::new(width as f32, height as f32),
        builder.finalize(),
        true);
    api.set_root_pipeline(pipeline_id);
    api.generate_frame(None);

    events_loop.run_forever(|event| {
        renderer.update();
        renderer.render(DeviceUintSize::new(width, height));
        window.swap_buffers(1);
    });
}
