#version 430
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 3) buffer InputPositions { vec2 input_positions[]; };
layout(std430, binding = 5) buffer GridHead { int grid_head[]; };
layout(std430, binding = 6) buffer ParticleNext { int particle_next[]; };

uniform int num_particles;
uniform int grid_width;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= num_particles) {
        return;
    }

    vec2 pos = input_positions[gid];
    // Map 0.0-1.0 to grid cell
    int cell_x = int(clamp(pos.x * grid_width, 0.0, grid_width - 1.0));
    int cell_y = int(clamp(pos.y * grid_width, 0.0, grid_width - 1.0));
    int cell_index = cell_y * grid_width + cell_x;

    // Atomic exchange to insert into linked list
    int old_head = atomicExchange(grid_head[cell_index], int(gid));
    particle_next[gid] = old_head;
}
