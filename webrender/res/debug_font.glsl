/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,shared_other

LAYOUT(0, varying vec2 vColorTexCoord);
LAYOUT(1, varying vec4 vColor);

#ifdef WR_VERTEX_SHADER
LAYOUT(0, in vec4 aColor);
LAYOUT(1, in vec4 aColorTexCoord);

void main(void) {
    vColor = aColor;
    vColorTexCoord = aColorTexCoord.xy;
    vec4 pos = vec4(aPosition, 1.0);
    pos.xy = floor(pos.xy * uDevicePixelRatio + 0.5) / uDevicePixelRatio;
    gl_Position = uTransform * pos;
}
#endif

#ifdef WR_FRAGMENT_SHADER
void main(void) {
    float alpha = texture(sColor0, vec3(vColorTexCoord.xy, 0.0)).r;
    Target0 = vec4(vColor.xyz, vColor.w * alpha);
}
#endif
