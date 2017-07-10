//#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef WR_DX11
void main(void) {
#else
void main(in a2v IN, out v2p OUT) {
    vec3 aPosition = IN.pos;
    ivec4 aData0 = IN.data0;
    ivec4 aData1 = IN.data1;
#endif
    Primitive prim = load_primitive(aData0, aData1);
    Rectangle rect = fetch_rectangle(prim.specific_prim_address);
#ifdef WR_DX11
    OUT.vColor = rect.color;
#else
    vColor = rect.color;
#endif
#ifdef WR_FEATURE_TRANSFORM
    TransformVertexInfo vi = write_transform_vertex(//TODO vertex_id,
                                                    prim.local_rect,
                                                    prim.local_clip_rect,
                                                    prim.z,
                                                    prim.layer,
                                                    prim.task,
                                                    prim.local_rect);
#ifdef WR_DX11
    OUT.vLocalPos = vi.local_pos;
#else
    vLocalPos = vi.local_pos;
#endif
#else
    VertexInfo vi = write_vertex(aPosition,
                                 prim.local_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.layer,
                                 prim.task,
                                 prim.local_rect);
#endif

#ifdef WR_DX11
    OUT.Position = vi.out_pos;
#endif

#ifdef WR_FEATURE_CLIP
    write_clip(vi.screen_pos, prim.clip_area);
#endif
}
