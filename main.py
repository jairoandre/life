import os
import sys
import numpy as np
import math
import json
import tkinter as tk
from tkinter import filedialog

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

INITIAL_NUM_PARTICLES = 1_000 
MAX_NUM_PARTICLES = 200_000
NUM_TYPE_OF_PARTICLES = 4
BETA = 0.15 
FRICTION_RATE = 0.040
DT = 0.015
ZOOM_FACTOR = 0.3

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
        
        # Post Processing Programs
        self.blur_program = self.ctx.program(
            vertex_shader=load_shader("blur_vertex.glsl"),
            fragment_shader=load_shader("blur_fragment.glsl"),
        )
        self.composite_program = self.ctx.program(
            vertex_shader=load_shader("composite_vertex.glsl"),
            fragment_shader=load_shader("composite_fragment.glsl"),
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
        self.friction = np.power(0.5, DT / FRICTION_RATE, dtype="f4")
        self.dt = DT
        self.num_types = NUM_TYPE_OF_PARTICLES
        self.num_particles = INITIAL_NUM_PARTICLES
        self.adding_particles = False
        self.removing_particles = False
        self.increase_force = False
        self.decrease_force = False

        # Initialize Buffers (MAX SIZE)
        self.init_buffers()
        
        # Initialize Uniforms and Grid infrastructure
        self.update_simulation_params()
        
        # Fill Initial Particles
        self.fill_initial_particles()

        self.clock = pygame.time.Clock()
        self.delta_time = 0.0
        self.running = True
        
        self.ui_texture = self.ctx.texture((WIDTH, HEIGHT), 4)

        # FBO Setup
        self.init_fbo(WIDTH, HEIGHT)

    def init_fbo(self, w, h):
        # Scene FBO
        self.scene_texture = self.ctx.texture((w, h), 4)
        self.scene_fbo = self.ctx.framebuffer(color_attachments=[self.scene_texture])
        
        # Ping Pong FBOs for Blur (Downscale by 2 for performance and larger glow radius effect)
        blur_w, blur_h = w // 2, h // 2
        
        self.pingpong_textures = [
            self.ctx.texture((blur_w, blur_h), 4),
            self.ctx.texture((blur_w, blur_h), 4)
        ]
        self.pingpong_fbos = [
            self.ctx.framebuffer(color_attachments=[self.pingpong_textures[0]]),
            self.ctx.framebuffer(color_attachments=[self.pingpong_textures[1]])
        ]

    def init_buffers(self):
        # We assume positions and velocities are 2 floats (8 bytes)
        # Type is 1 int (4 bytes)
        # Color is 1 float (hue) or 3 for rgb? Fragment shader usually takes hue or RGB. 
        # previous code used '1f' for in_color, so it is Hue.
        
        # Create full sized buffers, but we will mostly write to them dynamically or initially
        self.pos_buffer1 = self.ctx.buffer(reserve=MAX_NUM_PARTICLES * 8) # 2 floats
        self.pos_buffer2 = self.ctx.buffer(reserve=MAX_NUM_PARTICLES * 8)
        self.vel_buffer = self.ctx.buffer(reserve=MAX_NUM_PARTICLES * 8)
        self.types_buffer = self.ctx.buffer(reserve=MAX_NUM_PARTICLES * 4) # 1 int
        self.colors_buffer = self.ctx.buffer(reserve=MAX_NUM_PARTICLES * 4) # 1 float (4 bytes)
        
        self.particle_next_buffer = self.ctx.buffer(reserve=MAX_NUM_PARTICLES * 4) # 1 int

        self.gen_force_matrix()
        
        # Bindings
        self.types_buffer.bind_to_storage_buffer(1)
        self.vel_buffer.bind_to_storage_buffer(2)
        self.particle_next_buffer.bind_to_storage_buffer(6)
        
        # We need to generate base colors for types to pick from later
        self.base_type_colors = np.random.rand(self.num_types).astype("f4")


    def update_simulation_params(self):
        # Recalculate R_MAX based on user formula
        self.r_max = min(0.1, 1000.0 / self.num_particles) if self.num_particles > 0 else 0.1
        
        # Recalculate Grid
        self.grid_cell_size = self.r_max
        self.grid_width = int(math.ceil(1.0 / self.grid_cell_size))
        self.num_cells = self.grid_width * self.grid_width
        
        # Update Uniforms
        self.compute_shader["num_particles"] = self.num_particles
        self.compute_shader["beta"] = self.beta
        self.compute_shader["r_max"] = self.r_max
        self.compute_shader["friction_rate"] = self.friction
        self.compute_shader["dt"] = self.dt
        self.compute_shader["grid_width"] = self.grid_width

        self.build_grid_shader["num_particles"] = self.num_particles
        self.build_grid_shader["grid_width"] = self.grid_width
        
        self.clear_grid_shader["num_cells"] = self.num_cells
        
        # Reallocate Grid Head Buffer
        # This is small enough to reallocate safely
        grid_head_init = np.full(self.num_cells, -1, dtype="i4")
        self.grid_head_buffer = self.ctx.buffer(grid_head_init.tobytes())
        self.grid_head_buffer.bind_to_storage_buffer(5)
        
        # Compute Group Sizes
        self.x_wg = self.num_particles // GROUP_SIZE
        if self.x_wg == 0 or self.num_particles % GROUP_SIZE > 0:
            self.x_wg += 1
            
        self.grid_wg = self.num_cells // GROUP_SIZE
        if self.grid_wg == 0 or self.num_cells % GROUP_SIZE > 0:
            self.grid_wg += 1

    def fill_initial_particles(self):
        # Fill only the initial count
        # Generate data
        self.positions = np.random.uniform(0.0, 1.0, size=(MAX_NUM_PARTICLES, 2)).astype("f4")
        self.velocities = np.zeros((MAX_NUM_PARTICLES, 2)).astype("f4")
        self.types = np.random.choice(range(self.num_types), size=(MAX_NUM_PARTICLES)).astype("i4")
        
        # Map types to colors
        self.colors = np.array([self.base_type_colors[t] for t in self.types]).astype("f4")
        
        # Write to buffers
        # We write only the size needed
        self.pos_buffer1.write(self.positions.tobytes())
        self.pos_buffer2.write(self.positions.tobytes())
        self.vel_buffer.write(self.velocities.tobytes())
        self.types_buffer.write(self.types.tobytes())
        self.colors_buffer.write(self.colors.tobytes())

    def add_particle(self):
        if self.num_particles >= MAX_NUM_PARTICLES:
            return
        self.num_particles += 100
        self.update_simulation_params()
        
    def remove_particle(self):
        if self.num_particles <= 0:
            return
        self.num_particles -= 100
        if self.num_particles < 0:
            self.num_particles = 0
        self.update_simulation_params()
    
    def do_increase_force(self):
        self.apply_force_matrix(0.001)
    
    def do_decrease_force(self):
        self.apply_force_matrix(-0.001)

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
        # self.compute_shader["r_max"] = self.r_max # R_max controlled by particle logic
        self.compute_shader["friction_rate"] = self.friction
        self.compute_shader["dt"] = self.dt

    def gen_force_matrix(self):
        matrix_size = (self.num_types, self.num_types)
        self.force_matrix = np.random.uniform(-1.0, 1.0, matrix_size).astype("f4")
        self.apply_force_matrix()
    
    def apply_force_matrix(self, force = 0.0):
        matrix_size = (self.num_types, self.num_types)
        self.force_matrix += force
        self.force_texture = self.ctx.texture(matrix_size, 1, dtype="f4")
        self.force_texture.write(self.force_matrix.tobytes())
        self.force_texture.bind_to_image(0)

    def gen_colors(self):
        self.base_type_colors = np.random.rand(self.num_types).astype("f4")
        colors = np.array([self.base_type_colors[t] for t in self.types]).astype("f4")
        self.colors_buffer.write(colors.tobytes())
        
    def save_config(self):
        root = tk.Tk()
        root.withdraw()
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        root.destroy()
        
        if not filename:
            return

        data = {
            "parameters": {
                "beta": float(self.beta),
                # "r_max": float(self.r_max), # Derived
                "friction_values": { 
                    "rate": float(self.slider_friction.value), 
                    "dt": float(self.dt)
                },
                "num_types": int(self.num_types)
            },
            "force_matrix": self.force_matrix.tolist(),
            "base_type_colors": self.base_type_colors.tolist()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved config to {filename}")

    def load_config(self):
        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        root.destroy()
        
        if not filename:
            return

        with open(filename, 'r') as f:
            data = json.load(f)
            
        params = data["parameters"]
        
        # Load Parameters
        self.beta = params["beta"]
        
        # Update sliders
        self.slider_beta.value = self.beta
        self.slider_friction.value = params["friction_values"]["rate"]
        self.slider_dt.value = params["friction_values"]["dt"]
        
        # Force Matrix
        self.force_matrix = np.array(data["force_matrix"], dtype="f4")
        self.num_types = len(self.force_matrix)
        
        self.force_texture.write(self.force_matrix.tobytes())
        
        # Colors
        self.base_type_colors = np.array(data["base_type_colors"], dtype="f4")
        
        # Re-generate particle colors for *current* types
        # Reading types buffer
        types_data = self.types_buffer.read(size=self.num_particles * 4)
        types = np.frombuffer(types_data, dtype="i4")
        
        # Map to new colors
        colors = np.array([self.base_type_colors[t] for t in types]).astype("f4")
        self.colors_buffer.write(colors.tobytes())
        
        print(f"Loaded config from {filename}")


    def render(self):
        self.display.fill((0, 0, 0))

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



        # Render Pass 1: Render Scene to FBO
        self.scene_fbo.use()
        self.scene_fbo.clear(0, 0, 0, 0) # Clear with transparent black

        
        # 1. Render Particles
        pos_buffer = self.pos_buffer2 if self.use_first_buffer_set else self.pos_buffer1
        
        # Render only active particles
        vao = self.ctx.vertex_array(
            self.program,
            [(pos_buffer, "2f", "in_position"), (self.colors_buffer, "1f", "in_color")],
        )

        vao.render(moderngl.POINTS, vertices=self.num_particles)
        
        # Render Pass 2: Blur
        # Horizontal
        amount = 10 # Blur iterations
        horizontal = True
        first_iteration = True
        
        self.quad_fs_blur = self.ctx.vertex_array(self.blur_program, [(self.quad_fs, '2f 2f', 'in_vert', 'in_uv')])
        
        for i in range(amount):
            self.pingpong_fbos[int(horizontal)].use()
            self.blur_program['horizontal'] = horizontal
            
            if first_iteration:
                self.scene_texture.use(location=0)
            else:
                self.pingpong_textures[int(not horizontal)].use(location=0)
                
            self.quad_fs_blur.render(moderngl.TRIANGLE_STRIP)
            horizontal = not horizontal
            if first_iteration:
                first_iteration = False
                
        # Render Pass 3: Composite to Screen
        self.ctx.screen.use()
        self.ctx.clear()
        
        self.scene_texture.use(location=0) # Original scene
        self.pingpong_textures[int(not horizontal)].use(location=1) # Blurred bloom
        
        self.composite_program['scene'] = 0
        self.composite_program['bloom_blur'] = 1
        self.composite_program['bloom_strength'] = 3.0
        
        self.quad_fs_composite = self.ctx.vertex_array(self.composite_program, [(self.quad_fs, '2f 2f', 'in_vert', 'in_uv')])
        self.quad_fs_composite.render(moderngl.TRIANGLE_STRIP)
        
        # 2. Render UI
        ui_surf = self.ui_manager.draw()
        texture_data = ui_surf.get_view('1')
        self.ui_texture.write(texture_data)
        self.ui_texture.use(location=0)
        
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)
        
        # Draw FPS
        pygame.display.set_caption(f"Particle Life - FPS: {self.clock.get_fps():.2f} - Particles: {self.num_particles}")
        self.clock.tick()

        pygame.display.flip()

    def resize(self, w, h):
        # Update Viewport
        self.ctx.viewport = (0, 0, w, h)
        
        # NOTE: calling pygame.display.set_mode on resize with OPENGL usually isn't necessary 
        # for moderngl to work, but it updates pygame's internal surface.
        # However, it implies context destruction on some OS. 
        # For now, we rely on the event.w/h which are accurate.
        
        self.ui_manager.width = w
        self.ui_manager.height = h
        
        # Recreate UI Surface and Texture
        self.ui_manager.surface = pygame.Surface((w, h), pygame.SRCALPHA)
        self.ui_texture = self.ctx.texture((w, h), 4)
        
        # Recreate FBOs to match new size
        self.init_fbo(w, h)



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
                    self.zoom += ZOOM_FACTOR * event.y
                    if self.zoom < ZOOM_FACTOR * 0.1:
                        self.zoom = ZOOM_FACTOR * 0.1
                    self.program["zoom"].value = self.zoom
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    slider_dragging = False
                    for slider in self.ui_manager.sliders:
                        if slider.dragging:
                            slider_dragging = True
                            break
                    if event.button == 1 and not slider_dragging:
                        self.dragging = True
                        x, y = event.pos
                        self.previous_pos = np.array([x, y])
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False
                elif event.type == pygame.MOUSEMOTION:
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
                    elif event.key == pygame.K_p:
                        self.pause = not self.pause
                    elif event.key == pygame.K_r:
                        self.zoom = 1.0
                        self.program["zoom"].value = self.zoom
                    elif event.key == pygame.K_c:
                        self.gen_colors()
                    elif event.key == pygame.K_s:
                        self.save_config()
                    elif event.key == pygame.K_l:
                        self.load_config()
                    elif event.key == pygame.K_x:
                        self.adding_particles = True
                    elif event.key == pygame.K_z:
                        self.removing_particles = True
                    elif event.key == pygame.K_UP:
                        self.increase_force = True
                    elif event.key == pygame.K_DOWN:
                        self.decrease_force = True
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_x:
                        self.adding_particles = False
                    elif event.key == pygame.K_z:
                        self.removing_particles = False
                    elif event.key == pygame.K_UP:
                        self.increase_force = False
                    elif event.key == pygame.K_DOWN:
                        self.decrease_force = False
            if self.adding_particles:
                self.add_particle()
            if self.removing_particles:
                self.remove_particle()  
            if self.increase_force:
                self.do_increase_force()
            if self.decrease_force:
                self.do_decrease_force()
            self.render()
        print("Closing...")
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    scene = Scene()
    scene.run()
