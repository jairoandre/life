#version 430
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 5) buffer GridHead { int grid_head[]; };

uniform int num_cells;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= num_cells) {
        return;
    }
    grid_head[gid] = -1;
}
