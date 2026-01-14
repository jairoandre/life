# Particle Life Simulation

A GPU-accelerated particle simulation where particles interact based on attraction/repulsion rules defined by their types. This project demonstrates high-performance simulation using Compute Shaders and efficient spatial partitioning.

## Application Description

This application simulates "artificial life" using simple rules. Particles have types (colors), and each type applies a force (attraction or repulsion) to other types. By tuning these forces and other parameters like friction and beta (momentum), complex emergent behaviors and structures can be observed.

Key features:
-   **GPU Acceleration**: All physics calculations are performed on the GPU for maximum performance.
-   **Real-time Interaction**: Adjust parameters like Friction, Beta, and Delta Time in real-time.
-   **Zoom and Pan**: Navigate the simulation space.
-   **Save/Load Configs**: Save interesting stable states and load them later.

## Libraries & Dependencies

The following Python libraries are used:
-   **[ModernGL](https://github.com/moderngl/moderngl)**: For high-performance rendering and Compute Shader access.
-   **[Pygame](https://www.pygame.org/)**: For window creation, context management, and handling user input.
-   **[NumPy](https://numpy.org/)**: For initial data generation and array manipulations.
-   **Tkinter**: For file dialogs (Save/Load).

## Algorithms

### Spatial Partitioning (Grid-based Linked List)
To optimize the simulation complexity from a naive $O(N^2)$ to a much faster linear-approaching $O(N)$, the simulation uses a Grid-based Spatial Partitioning technique.

1.  **Grid Construction**: The world is divided into a grid of cells.
2.  **Atomic Linked List**: A linked list is built for each cell on the GPU using atomic operations.
    -   `clear_grid`: Resets the "head" pointer of every cell.
    -   `build_grid`: Particles calculate which cell they belong to and atomically insert themselves into the linked list for that cell.
3.  **Neighbor Search**: When calculating forces for a particle, instead of checking every other particle, the shader only iterates through particles in the 9 neighboring grid cells (using the linked list traversal).

This allows the simulation to support hundreds of thousands of particles at real-time framerates.

### Physics simulation
-   **Euler Integration**: Simple semi-implicit Euler integration is used for updating positions based on velocity and acceleration.
-   **Force Matrix**: A random or user-defined matrix defining the interaction strength between every pair of particle types.

## Usage

### Installation

1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Run the main script using Python:
```bash
python main.py
```

### Controls

| Key / Action | Function |
| :--- | :--- |
| **Mouse Drag** | Pan the view |
| **Mouse Wheel** | Zoom in / out |
| **P** | Pause / Resume simulation |
| **A** | Randomize attraction/repulsion forces |
| **C** | Randomize particle colors |
| **R** | Reset zoom and camera position |
| **S** | Save current configuration (params, forces, colors) to JSON |
| **L** | Load configuration from JSON |
| **Hold X** | spawns particles |
| **Hold Z** | remove particles |
| **Hold UP** | Increase global force strength |
| **Hold DOWN** | Decrease global force strength |

