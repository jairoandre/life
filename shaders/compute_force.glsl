#version 430
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(rgba32f, binding = 0) uniform image2D force_matrix_tex;
layout(std430, binding =  1) buffer ParticleType { int particles_types[]; };
layout(std430, binding =  2) buffer InputVelocities { vec2 velocities[]; };
layout(std430, binding =  3) buffer InputPositions { vec2 input_positions[]; };
layout(std430, binding =  4) buffer OutputPositions { vec2 output_positions[]; };
// New bindings for grid
layout(std430, binding = 5) buffer GridHead { int grid_head[]; };
layout(std430, binding = 6) buffer ParticleNext { int particle_next[]; };

uniform int num_particles;
uniform float beta;
uniform float r_max;
uniform float friction_rate;
uniform float dt;
uniform int grid_width;

float wrap_pos(float o, float t){
    float d = abs(t - o);
    if(d > .5){
        return t < o ? t + 1. : t - 1.;
    }
    return t;
}

float clamp_pos(float v){
    return v > 1. ? (v - 1.) : v < 0. ? (1. + v) : v;
}

float force(float r, uint col_1, uint col_2){
    if (r < beta) {
        return r / beta - 1.;
    } else if (r > beta && r < 1.) {
        float a = imageLoad(force_matrix_tex, ivec2(col_1,col_2)).r;
        return a * (1. - abs(2. * r - 1. - beta) / (1. - beta));
    }
    return 0.;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= num_particles) {
        return;
    }
    vec2 current_pos = input_positions[gid];
    vec2 current_vel = velocities[gid];
    uint particle_type = uint(particles_types[gid]);
    vec2 total_force = vec2(0);

    // Identify my cell
    int my_cell_x = int(clamp(current_pos.x * grid_width, 0.0, grid_width - 1.0));
    int my_cell_y = int(clamp(current_pos.y * grid_width, 0.0, grid_width - 1.0));

    // Iterate over 3x3 neighbors
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            // Check bounds (wrapping)
            int check_x = my_cell_x + x;
            int check_y = my_cell_y + y;

            // Handle wrapping for grid lookup?
            // Since particle positions wrap 0..1, the grid should also wrap.
            // e.g. if I am at x=0.01 (cell 0), neigbor x=-1 is cell N-1.
            
            if (check_x < 0) check_x += grid_width;
            if (check_x >= grid_width) check_x -= grid_width;
            if (check_y < 0) check_y += grid_width;
            if (check_y >= grid_width) check_y -= grid_width;
            
            int cell_idx = check_y * grid_width + check_x;
            
            // Iterate through linked last in this cell
            int other_id = grid_head[cell_idx];
            
            // Limit iterations to prevent TDR (Timeout Detection Recovery) if too many particles cluster?
            // For now, simple loop.
            int iter_count = 0;
            while (other_id != -1 && iter_count < 1000) {
                if (other_id != gid) {
                    vec2 other_pos = input_positions[other_id];
                    float wraped_x = wrap_pos(current_pos.x, other_pos.x);
                    float wraped_y = wrap_pos(current_pos.y, other_pos.y);
                    vec2 wrapped_other_pos = vec2(wraped_x, wraped_y);
                    
                    vec2 dist_vec = wrapped_other_pos - current_pos;
                    float dist = length(dist_vec);
                    
                    if (dist < r_max && dist > 0.0) {
                        uint other_type = uint(particles_types[other_id]);
                        float f = force(dist/r_max, particle_type, other_type);
                        total_force += (f * (dist_vec/dist));
                    }
                }
                other_id = particle_next[other_id];
                iter_count += 1;
            }
        }
    }
    vec2 new_vel = current_vel * friction_rate;
    new_vel += total_force * dt;
    vec2 new_pos = current_pos + new_vel * dt;
    new_pos.x = clamp_pos(new_pos.x);
    new_pos.y = clamp_pos(new_pos.y);
    output_positions[gid] = new_pos;
    velocities[gid] = new_vel;
}
