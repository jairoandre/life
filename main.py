import os
import sys
import numpy as np
import math

import moderngl
import pygame
import OpenGL.GL as gl
import array

from utils.ui import UIManager, Slider

os.environ["SDL_WINDOWS_DPI_AWARENESS"] = "permonitorv2"

WIDTH = 1920
HEIGHT = 1080

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# SIMULATION PARAMS

NUM_PARTICLES = 10000 
NUM_TYPE_OF_PARTICLES = 4
BETA = 0.15 
R_MAX = 0.02 
FRICTION_RATE = 0.040
DT = 0.015

# Derived Grid Params
GRID_CELL_SIZE = R_MAX
GRID_WIDTH = int(math.ceil(1.0 / GRID_CELL_SIZE))
NUM_CELLS = GRID_WIDTH * GRID_WIDTH

# SHADER PARAMS
GROUP_SIZE = 64

def load_shader(filename):
    with open(os.path.join("shaders", filename), "r") as f:
        return f.read()

class Scene:
    def __init__(self):
        pygame.init()
        display_size = (WIDTH, HEIGHT)
        self.display = pygame.display.set_mode(
            display_size,
            flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
            vsync=True,
        )
        self.ctx = moderngl.get_context()
        self.use_first_buffer_set = True
        self.zoom = 1.0
        self.pause = False
        self.dragging = False
        self.offset = np.array([0.0, 0.0])
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        
        # UI Setup
        self.ui_manager = UIManager(WIDTH, HEIGHT)
        self.setup_ui()
        
        # Load Shaders
        self.program = self.ctx.program(
            vertex_shader=load_shader("vertex.glsl"),
            fragment_shader=load_shader("fragment.glsl"),
        )
        
        # UI Texture Shader
        self.ui_program = self.ctx.program(
            vertex_shader=load_shader("ui_vertex.glsl"),
            fragment_shader=load_shader("ui_fragment.glsl"),
        )
        # Fullscreen quad for UI
        self.quad_fs = self.ctx.buffer(np.array([
            # x, y, u, v
            -1.0, 1.0, 0.0, 0.0,  # TL
            -1.0, -1.0, 0.0, 1.0, # BL
            1.0, 1.0, 1.0, 0.0,   # TR
            1.0, -1.0, 1.0, 1.0,  # BR
        ], dtype='f4').tobytes())
        self.quad_vao = self.ctx.vertex_array(self.ui_program, [(self.quad_fs, '2f 2f', 'in_vert', 'in_uv')])


        self.program["zoom"] = self.zoom
        self.program["offset"] = self.offset

        # Compute Shaders
        self.compute_shader = self.ctx.compute_shader(load_shader("compute_force.glsl"))
        self.clear_grid_shader = self.ctx.compute_shader(load_shader("clear_grid.glsl"))
        self.build_grid_shader = self.ctx.compute_shader(load_shader("build_grid.glsl"))

        self.beta = BETA
        self.r_max = R_MAX
        self.friction = np.power(0.5, DT / FRICTION_RATE, dtype="f4")
        self.dt = DT

        # Set Uniforms
        self.compute_shader["num_particles"] = NUM_PARTICLES
        self.compute_shader["beta"] = self.beta
        self.compute_shader["r_max"] = self.r_max
        self.compute_shader["friction_rate"] = self.friction
        self.compute_shader["dt"] = self.dt
        self.compute_shader["grid_width"] = GRID_WIDTH

        self.build_grid_shader["num_particles"] = NUM_PARTICLES
        self.build_grid_shader["grid_width"] = GRID_WIDTH
        
        self.clear_grid_shader["num_cells"] = NUM_CELLS

        # Initialize positions, velocities and particle types
        positions = np.random.uniform(0.0, 1.0, size=(NUM_PARTICLES, 2)).astype("f4")
        velocities = np.zeros((NUM_PARTICLES, 2)).astype("f4")
        self.particle_types = np.random.choice(
            range(NUM_TYPE_OF_PARTICLES), size=(NUM_PARTICLES, 1)
        ).astype(int)

        # Create buffers
        self.gen_force_matrix()

        self.pos_buffer1 = self.ctx.buffer(positions.tobytes())
        self.pos_buffer2 = self.ctx.buffer(positions.tobytes())
        self.vel_buffer = self.ctx.buffer(velocities.tobytes())
        self.types_buffer = self.ctx.buffer(self.particle_types.tobytes())
        
        # Grid Buffers
        # Initialize grid_head with -1
        grid_head_init = np.full(NUM_CELLS, -1, dtype="i4")
        self.grid_head_buffer = self.ctx.buffer(grid_head_init.tobytes())
        
        # Particle Next buffer (uninitialized is fine, but lets zero it)
        self.particle_next_buffer = self.ctx.buffer(reserve=NUM_PARTICLES * 4) # 4 bytes per int

        self.gen_colors()

        # Bindings
        # 0: Force Matrix (Image) - Bound in gen_force_matrix
        # 1: Particle Types
        self.types_buffer.bind_to_storage_buffer(1)
        # 2: Velocities
        self.vel_buffer.bind_to_storage_buffer(2)
        # 3, 4: Positions (Swapped per frame)
        # 5: Grid Head
        self.grid_head_buffer.bind_to_storage_buffer(5)
        # 6: Particle Next
        self.particle_next_buffer.bind_to_storage_buffer(6)

        # Compute Group Sizes
        self.x_wg = NUM_PARTICLES // GROUP_SIZE
        if self.x_wg == 0 or NUM_PARTICLES % GROUP_SIZE > 0:
            self.x_wg += 1
            
        self.grid_wg = NUM_CELLS // GROUP_SIZE
        if self.grid_wg == 0 or NUM_CELLS % GROUP_SIZE > 0:
            self.grid_wg += 1

        self.clock = pygame.time.Clock()
        self.delta_time = 0.0
        self.running = True
        
        self.ui_texture = self.ctx.texture((WIDTH, HEIGHT), 4)

    def setup_ui(self):
        # x, y, w, h, min, max, initial, label
        # BETA
        self.slider_beta = Slider(20, 20, 200, 20, 0.0, 1.0, BETA, "Beta")
        self.ui_manager.add_slider(self.slider_beta)
        
        # FRICTION RATE
        self.slider_friction = Slider(20, 70, 200, 20, 0.001, 0.1, FRICTION_RATE, "Friction Rate")
        self.ui_manager.add_slider(self.slider_friction)
        
        # DT
        self.slider_dt = Slider(20, 120, 200, 20, 0.001, 0.05, DT, "Delta Time")
        self.ui_manager.add_slider(self.slider_dt)

    def update_compute_shader_uniforms(self):
        # Read from UI
        self.beta = self.slider_beta.value
        f_rate = self.slider_friction.value
        self.dt = self.slider_dt.value
        
        self.friction = np.power(0.5, self.dt / f_rate, dtype="f4")

        self.compute_shader["beta"] = float(self.beta)
        # self.compute_shader["r_max"] = self.r_max # R_max not changable yet
        self.compute_shader["friction_rate"] = self.friction
        self.compute_shader["dt"] = self.dt

    def gen_force_matrix(self):
        matrix_size = (NUM_TYPE_OF_PARTICLES, NUM_TYPE_OF_PARTICLES)
        force_matrix = np.random.uniform(-1.0, 1.0, matrix_size).astype("f4")
        force_texture = self.ctx.texture(matrix_size, 1, dtype="f4")
        force_texture.write(force_matrix.tobytes())
        force_texture.bind_to_image(0)

    def gen_colors(self):
        random_colors = np.random.rand(NUM_TYPE_OF_PARTICLES)
        colors = np.array(
            [[random_colors[c] for c in self.particle_types.T[0]]]
        ).T.astype("f4")
        self.colors_buffer = self.ctx.buffer(colors.tobytes())

    def render(self):
        self.display.fill((0, 0, 0))
        # self.delta_time = self.clock.tick(60) / 1000

        if not self.pause:
            
            # Check UI updates
            self.update_compute_shader_uniforms()
            
            # 1. Clear Grid
            # grid_head_buffer is bound to 5
            self.clear_grid_shader.run(self.grid_wg)
            
            # Select Buffers
            if self.use_first_buffer_set:
                pos_in = self.pos_buffer1
                pos_out = self.pos_buffer2
            else:
                pos_in = self.pos_buffer2
                pos_out = self.pos_buffer1
            
            pos_in.bind_to_storage_buffer(3)
            pos_out.bind_to_storage_buffer(4)

            # 2. Build Grid
            # Inputs: pos_in (3), grid_head (5), particle_next (6)
            self.build_grid_shader.run(self.x_wg)

            # 3. Simulate
            # Inputs: pos_in (3), velocities (2), types (1), grid_head (5), particle_next (6)
            # Outputs: pos_out (4), velocities (2)
            self.compute_shader.run(self.x_wg)

            self.use_first_buffer_set = not self.use_first_buffer_set

        self.ctx.clear()
        
        # 1. Render Particles
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE) # Additive blending for cool look? Or standard. Let's do standard for now or disable for performance.
        # Actually, standard depth test is off, so order matters if alpha. But particles are opaque points usually.
        # Let's disable blend for particles for speed.
        gl.glDisable(gl.GL_BLEND)

        pos_buffer = self.pos_buffer2 if self.use_first_buffer_set else self.pos_buffer1
        
        vao = self.ctx.vertex_array(
            self.program,
            [(pos_buffer, "2f", "in_position"), (self.colors_buffer, "1f", "in_color")],
        )
        vao.render(moderngl.POINTS)
        
        # 2. Render UI
        ui_surf = self.ui_manager.draw()
        texture_data = ui_surf.get_view('1')
        self.ui_texture.write(texture_data)
        self.ui_texture.use(location=0)
        
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)
        
        # Draw FPS
        pygame.display.set_caption(f"Particle Life - FPS: {self.clock.get_fps():.2f} - Particles: {NUM_PARTICLES}")
        self.clock.tick()

        pygame.display.flip()

    def resize(self, w, h):
        size = (w, h)
        self.display = pygame.display.set_mode(
            size, flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync=True
        )
        # Update UI manager size
        # Recreate texture?
        self.ui_manager.width = w
        self.ui_manager.height = h
        self.ui_manager.surface = pygame.Surface((w, h), pygame.SRCALPHA)
        self.ui_texture = self.ctx.texture((w, h), 4)


    def run(self):
        while self.running:
            for event in pygame.event.get():
                # Pass to UI
                if self.ui_manager.handle_event(event):
                    pass # Event handled by UI

                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.resize(event.w, event.h)
                elif event.type == pygame.MOUSEWHEEL:
                    self.zoom += 0.1 * event.y
                    if self.zoom < 0.1:
                        self.zoom = 0.1
                    self.program["zoom"].value = self.zoom
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.ui_manager.sliders[0].dragging: # Hacky check if UI is using it
                         # We should probably check if event was consumed properly
                         pass
                    
                    # Logic to drag screen
                    if event.button == 1:
                        # Check collision with UI?
                        # For now, if UI didn't take it (handled inside UI check above? No, UI returns True if updated)
                        # Let's improve UI handling logic later.
                        self.dragging = True
                        x, y = event.pos
                        self.previous_pos = np.array([x, y])
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                         # Simplified: Disable screen drag if interacting with UI sliders
                         # But dragging is False in slider.
                         # Need to ensure we don't drag screen when dragging slider.
                         # Since UI `handle_event` returns True, we can use that.
                         pass
                    
                    if self.dragging:
                        x, y = event.pos
                        current_pos = np.array([x, y])
                        delta = current_pos - self.previous_pos
                        self.offset += delta
                        self.previous_pos = current_pos
                        normalized = np.array(
                            [self.offset[0] / WIDTH, -self.offset[1] / HEIGHT]
                        )
                        self.program["offset"].value = normalized
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.gen_force_matrix()
                    elif event.key == pygame.K_c:
                        self.gen_colors()
                    elif event.key == pygame.K_p:
                        self.pause = not self.pause
            self.render()
        print("Closing...")
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    scene = Scene()
    scene.run()
