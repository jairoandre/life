#version 330

in vec2 in_position;
in float in_color;

uniform float zoom;
uniform vec2 offset;

out vec3 frag_color;

vec3 hsv2rgb(float h) {
    vec3 c = vec3(h * 0.95, 1., 1.);
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {

    vec2 pos = 2 * (in_position + offset - vec2(0.5)); // because the x, y varying from zero to one.

    vec2 repeated = mod(pos + 1, 2.0) - 1.0;

    repeated *= zoom;

    gl_Position = vec4(repeated, 0.0, 1.0);
    gl_PointSize = 2.0 * zoom;
    frag_color = hsv2rgb(in_color);
}
