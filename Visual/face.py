import pygame
import os

# Config
WIDTH, HEIGHT = 1000, 900
FPS = 30
ASSETS_DIR = "assets"
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 40
PADDING = 10

class JarvisFace:
    def __init__(self, screen):
        self.screen = screen
        self.bg_color = (30, 30, 30)
        self.expression = "neutral"
        self.hat = None
        self.glasses = None
        self.load_assets()

    def load_assets(self):
        self.expressions = self.load_images("expressions")
        self.hats = self.load_images("hats")
        self.glasses_options = self.load_images("glasses")

    def load_images(self, subfolder):
        assets = {}
        path = os.path.join(ASSETS_DIR, subfolder)
        for fname in os.listdir(path):
            name = fname.split('.')[0]
            assets[name] = pygame.image.load(os.path.join(path, fname)).convert_alpha()
        return assets

    def set_expression(self, name):
        if name in self.expressions:
            self.expression = name
        else:
            print(f"[WARN] Unknown expression: {name}")

    def set_hat(self, name):
        if name in self.hats:
            self.hat = name
        else:
            print(f"[WARN] Unknown hat: {name}")

    def set_glasses(self, name):
        if name in self.glasses_options:
            self.glasses = name
        else:
            print(f"[WARN] Unknown glasses: {name}")

    def draw(self):
        self.screen.fill(self.bg_color)

        expr_img = self.expressions.get(self.expression)
        if expr_img:
            rect = expr_img.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            self.screen.blit(expr_img, rect)

        if self.hat:
            hat_img = self.hats.get(self.hat)
            if hat_img:
                rect = hat_img.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 400))
                self.screen.blit(hat_img, rect)

        if self.glasses:
            glasses_img = self.glasses_options.get(self.glasses)
            if glasses_img:
                rect = glasses_img.get_rect(center=(WIDTH // 2 - 10, HEIGHT // 2 - 120))
                self.screen.blit(glasses_img, rect)

def draw_button(screen, x, y, text, color=(70, 70, 70)):
    rect = pygame.Rect(x, y, BUTTON_WIDTH, BUTTON_HEIGHT)
    pygame.draw.rect(screen, color, rect)
    font = pygame.font.SysFont(None, 24)
    text_surf = font.render(text, True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)
    return rect

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("JARVIS Face")
    clock = pygame.time.Clock()
    face = JarvisFace(screen)

    expression_names = list(face.expressions.keys())
    hat_names = list(face.hats.keys())
    glasses_names = list(face.glasses_options.keys())
    expr_index = hat_index = glasses_index = 0

    running = True
    while running:
        screen.fill((30, 30, 30))
        face.draw()

        # Draw buttons
        expr_btn = draw_button(screen, 10, HEIGHT - 130, f"Expression: {expression_names[expr_index]}")
        hat_btn = draw_button(screen, 10, HEIGHT - 80, f"Hat: {hat_names[hat_index]}")
        glasses_btn = draw_button(screen, 10, HEIGHT - 30, f"Glasses: {glasses_names[glasses_index]}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if expr_btn.collidepoint(event.pos):
                    expr_index = (expr_index + 1) % len(expression_names)
                    face.set_expression(expression_names[expr_index])
                elif hat_btn.collidepoint(event.pos):
                    hat_index = (hat_index + 1) % len(hat_names)
                    face.set_hat(hat_names[hat_index])
                elif glasses_btn.collidepoint(event.pos):
                    glasses_index = (glasses_index + 1) % len(glasses_names)
                    face.set_glasses(glasses_names[glasses_index])

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
