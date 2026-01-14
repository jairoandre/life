#version 330

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D scene;
uniform sampler2D bloom_blur;
uniform float bloom_strength;

void main() {
    vec3 hdr_color = texture(scene, v_uv).rgb;
    vec3 bloom_color = texture(bloom_blur, v_uv).rgb;
    
    // Additive blending with strength multiplier
    vec3 result = hdr_color + bloom_color * bloom_strength; 
    
    // Tone mapping (Reinhard)
    // vec3 result = hdr_color + bloom_color; // simple addition for now often looks cleaner for "glow" without complex HDR logic, but let's see. 
    // Usually Glow is just added on top.
    
    out_color = vec4(result, 1.0);
}
