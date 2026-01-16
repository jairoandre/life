#version 330

in vec3 frag_color;
out vec4 out_color;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    if(length(coord) > 1.0)
       discard;
    out_color = vec4(frag_color, 0.5);
}
