/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_DX11
    #define vec2 float2
    #define vec3 float3
    #define vec4 float4
    #define ivec2 int2
    #define ivec3 int3
    #define ivec4 int4
    #define bvec2 bool2
    #define bvec3 bool3
    #define bvec4 bool4
    #define mat2 float2x2
    #define mat3 float3x3
    #define mat4 float4x4
    #define mix lerp
    #define fract frac
    #define uniform
    #define sampler2D Texture2D
    #define sampler2DArray Texture2DArray
    #define gl_Position SV_Position
    #define gl_FragCoord SV_Position
    #define point p
    #define flat
    #define texelFetchOffset(sampler, loc, index, offset) sampler.Load(vec3(loc, 0.0), offset)
    #define texelFetch(sampler, loc, index) sampler.Load(vec3(loc, 0.0))
#endif

#ifdef WR_FEATURE_TEXTURE_EXTERNAL
// Please check https://www.khronos.org/registry/OpenGL/extensions/OES/OES_EGL_image_external_essl3.txt
// for this extension.
#extension GL_OES_EGL_image_external_essl3 : require
#endif

// The textureLod() doesn't support samplerExternalOES for WR_FEATURE_TEXTURE_EXTERNAL.
// https://www.khronos.org/registry/OpenGL/extensions/OES/OES_EGL_image_external_essl3.txt
//
// The textureLod() doesn't support sampler2DRect for WR_FEATURE_TEXTURE_RECT, too.
//
// Use texture() instead.
#if defined(WR_FEATURE_TEXTURE_EXTERNAL) || defined(WR_FEATURE_TEXTURE_RECT)
#define TEX_SAMPLE(sampler, tex_coord) texture(sampler, tex_coord)
#else
// In normal case, we use textureLod(). We haven't used the lod yet. So, we always pass 0.0 now.
#define TEX_SAMPLE(sampler, tex_coord) textureLod(sampler, tex_coord, 0.0)
#endif

//======================================================================================
// Vertex shader attributes and uniforms
//======================================================================================
#ifdef WR_VERTEX_SHADER
    //#define varying out

    // Uniform inputs
    uniform mat4 uTransform;       // Orthographic projection
    uniform float uDevicePixelRatio;

    // Attribute inputs
    //in vec3 aPosition;
#endif

//======================================================================================
// Fragment shader attributes and uniforms
//======================================================================================
#ifdef WR_FRAGMENT_SHADER
    precision highp float;

    //#define varying in

    // Uniform inputs

    // Fragment shader outputs
    //out vec4 oFragColor;
#endif

//======================================================================================
// Shared shader uniforms
//======================================================================================
#ifdef WR_FEATURE_TEXTURE_RECT
uniform sampler2DRect sColor0;
uniform sampler2DRect sColor1;
uniform sampler2DRect sColor2;
#elif defined WR_FEATURE_TEXTURE_EXTERNAL
uniform samplerExternalOES sColor0;
uniform samplerExternalOES sColor1;
uniform samplerExternalOES sColor2;
#else
uniform sampler2D sColor0;
uniform sampler2D sColor1;
uniform sampler2D sColor2;
#endif

#ifdef WR_FEATURE_DITHERING
uniform sampler2D sDither;
#endif

//======================================================================================
// Interpolator definitions
//======================================================================================

//======================================================================================
// VS only types and UBOs
//======================================================================================

//======================================================================================
// VS only functions
//======================================================================================
