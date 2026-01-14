import pygame

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
        self.handle_w = 10
        self.font = pygame.font.SysFont("Arial", 16)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.update_value(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.update_value(event.pos[0])
                return True
        return False

    def update_value(self, mouse_x):
        ratio = (mouse_x - self.rect.x) / self.rect.w
        ratio = max(0, min(1, ratio))
        self.value = self.min_val + ratio * (self.max_val - self.min_val)

    def draw(self, surface):
        # Draw Label
        val_str = f"{self.label}: {self.value:.4f}"
        text_surf = self.font.render(val_str, True, (255, 255, 255))
        surface.blit(text_surf, (self.rect.x, self.rect.y - 20))

        # Draw Background Line
        pygame.draw.rect(surface, (100, 100, 100), self.rect)

        # Draw Handle
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + (ratio * self.rect.w) - (self.handle_w // 2)
        handle_rect = pygame.Rect(handle_x, self.rect.y - 2, self.handle_w, self.rect.h + 4)
        pygame.draw.rect(surface, (200, 200, 200), handle_rect)


class Switch:
    def __init__(self, x, y, w, h, initial_state, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.state = initial_state
        self.label = label
        self.font = pygame.font.SysFont("Arial", 16)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.rect.collidepoint(event.pos):
                    self.state = not self.state
                    return True
        return False
        
    def draw(self, surface):
        # Draw Label
        text_surf = self.font.render(self.label, True, (255, 255, 255))
        surface.blit(text_surf, (self.rect.x, self.rect.y - 20))
        
        # Draw Box
        pygame.draw.rect(surface, (100, 100, 100), self.rect)
        
        # Draw State
        if self.state:
            # Draw ON (Green)
            inner_rect = self.rect.inflate(-4, -4)
            inner_rect.x += self.rect.w // 2
            inner_rect.w = self.rect.w // 2 - 2
            pygame.draw.rect(surface, (0, 255, 0), inner_rect)
        else:
            # Draw OFF (Red)
            inner_rect = self.rect.inflate(-4, -4)
            inner_rect.w = self.rect.w // 2 - 2
            pygame.draw.rect(surface, (200, 50, 50), inner_rect)
            
        # Draw border just for clarity
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2)


class UIManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.sliders = []
        self.switches = []

    def add_slider(self, slider):
        self.sliders.append(slider)
        
    def add_switch(self, switch):
        self.switches.append(switch)

    def handle_event(self, event):
        updated = False
        for s in self.sliders:
            if s.handle_event(event):
                updated = True
        for s in self.switches:
            if s.handle_event(event):
                updated = True
        return updated

    def draw(self):
        self.surface.fill((0, 0, 0, 0)) # Clear with transparency
        for s in self.sliders:
            s.draw(self.surface)
        for s in self.switches:
            s.draw(self.surface)
        return self.surface
