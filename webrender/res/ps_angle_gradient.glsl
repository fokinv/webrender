/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

LAYOUT(6, flat varying int vGradientAddress);
LAYOUT(7, flat varying float vGradientRepeat);

LAYOUT(8, flat varying vec2 vScaledDir);
LAYOUT(9, flat varying vec2 vStartPoint);

LAYOUT(10, flat varying vec2 vTileSize);
LAYOUT(11, flat varying vec2 vTileRepeat);

LAYOUT(12, varying vec2 vPos);

#ifdef WR_VERTEX_SHADER
void main(void) {
    Primitive prim = load_primitive();
    Gradient gradient = fetch_gradient(prim.specific_prim_address);

    VertexInfo vi = write_vertex(prim.local_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.layer,
                                 prim.task,
                                 prim.local_rect);

    vPos = vi.local_pos - prim.local_rect.p0;

    vec2 start_point = gradient.start_end_point.xy;
    vec2 end_point = gradient.start_end_point.zw;
    vec2 dir = end_point - start_point;

    vStartPoint = start_point;
    vScaledDir = dir / dot(dir, dir);

    vTileSize = gradient.tile_size_repeat.xy;
    vTileRepeat = gradient.tile_size_repeat.zw;

    vGradientAddress = prim.specific_prim_address + VECS_PER_GRADIENT;

    // Whether to repeat the gradient instead of clamping.
    vGradientRepeat = float(int(gradient.extend_mode.x) != EXTEND_MODE_CLAMP);

    write_clip(vi.screen_pos, prim.clip_area);
}
#endif

#ifdef WR_FRAGMENT_SHADER
void main(void) {
    vec2 pos = mod(vPos, vTileRepeat);

    if (pos.x >= vTileSize.x ||
        pos.y >= vTileSize.y) {
        discard;
    }

    float offset = dot(pos - vStartPoint, vScaledDir);

    vec4 color = sample_gradient(vGradientAddress,
                                 offset,
                                 vGradientRepeat);

    Target0 = color * do_clip();
}
#endif
