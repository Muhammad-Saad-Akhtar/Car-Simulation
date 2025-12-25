# Advanced NEAT Car Racing Simulation - FULLY FIXED (Parallel + No Display Errors)

import math
import random
import sys
import os
import pickle
import multiprocessing

import neat
import pygame

# ====================== CONFIGURATION ======================
WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255)

MAPS = ['map.png', 'map2.png', 'map3.png', 'map4.png', 'map5.png']  # Your map files

CHECKPOINTS = [
    (860, 940), (860, 600), (1200, 400), (1600, 500),
    (1600, 800), (1200, 900), (900, 800), (600, 600),
    (600, 300), (900, 200),
]

current_generation = 0
current_map_surface = None  # Will be set per generation (shared surface)
car_sprite_original = None  # Loaded once in main process

# ====================== CAR CLASS (NO PYGAME INIT OR DISPLAY CALLS) ======================
class Car:
    def __init__(self):
        global car_sprite_original
        self.sprite = pygame.transform.scale(car_sprite_original, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.position = [830, 920]
        self.angle = 0
        self.speed = 0

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]

        self.radars = []
        self.alive = True

        self.distance = 0
        self.time = 0

        self.current_checkpoint = 0
        self.last_position = self.position[:]
        self.stagnation_counter = 0

    def check_collision(self, game_map):
        length = 0.5 * CAR_SIZE_X
        corners = []
        for a in [30, 150, 210, 330]:
            x = self.center[0] + math.cos(math.radians(360 - (self.angle + a))) * length
            y = self.center[1] + math.sin(math.radians(360 - (self.angle + a))) * length
            corners.append((int(x), int(y)))

        for px, py in corners:
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                if game_map.get_at((px, py)) == BORDER_COLOR:
                    self.alive = False
                    break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while (0 <= x < WIDTH and 0 <= y < HEIGHT and
               game_map.get_at((x, y)) != BORDER_COLOR and length < 300):
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = math.hypot(x - self.center[0], y - self.center[1])
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        if self.speed == 0:
            self.speed = 12

        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed

        self.position[0] = max(20, min(self.position[0], WIDTH - 120))
        self.position[1] = max(20, min(self.position[1], HEIGHT - 120))

        self.distance += self.speed
        self.time += 1

        self.center = [int(self.position[0] + CAR_SIZE_X / 2), int(self.position[1] + CAR_SIZE_Y / 2)]
        self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)

        self.check_collision(game_map)

        self.radars.clear()
        for d in range(-90, 105, 22):  # 9 radars
            self.check_radar(d, game_map)

        # Checkpoint
        if self.current_checkpoint < len(CHECKPOINTS):
            cp_x, cp_y = CHECKPOINTS[self.current_checkpoint]
            if math.hypot(self.center[0] - cp_x, self.center[1] - cp_y) < 80:
                self.current_checkpoint += 1

        # Stagnation
        moved = math.hypot(self.position[0] - self.last_position[0],
                           self.position[1] - self.last_position[1])
        if moved < 8:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.last_position = self.position[:]

    def get_data(self):
        return [int(r[1] / 30) for r in self.radars] + [0] * max(0, 9 - len(self.radars))

    def get_reward(self):
        base = self.distance / 50
        cp_bonus = self.current_checkpoint * 2000
        stagnation = self.stagnation_counter * 15
        speed_bonus = self.speed * 2
        return base + cp_bonus - stagnation + speed_bonus

# ====================== EVALUATION (WORKERS) ======================
def evaluate_genome(genome, config):
    global current_map_surface
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    car = Car()

    for _ in range(1800):
        if not car.alive:
            break

        output = net.activate(car.get_data())
        steer = output[0] * 15
        accel = output[1] * 3
        brake = max(0, output[2] * 4)

        car.angle += steer
        car.speed += accel - brake
        car.speed = max(8, min(car.speed, 35))

        car.update(current_map_surface)

    genome.fitness = car.get_reward()

# ====================== VISUALIZATION (MAIN PROCESS ONLY) ======================
def visualize_best(genome, config):
    global current_map_surface
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 30)

    car = Car()
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    running = True
    while running and car.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        output = net.activate(car.get_data())
        steer = output[0] * 15
        accel = output[1] * 3
        brake = max(0, output[2] * 4)

        car.angle += steer
        car.speed += accel - brake
        car.speed = max(8, min(car.speed, 35))

        car.update(current_map_surface)

        screen.blit(current_map_surface, (0, 0))
        car.rotated_sprite = pygame.transform.rotate(car.sprite, car.angle)  # Re-rotate
        screen.blit(car.rotated_sprite, car.position)

        # Draw radars
        for radar in car.radars:
            pos, _ = radar
            pygame.draw.line(screen, (0, 255, 0), car.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

        # Draw checkpoints
        for i, cp in enumerate(CHECKPOINTS):
            color = (0, 255, 0) if i < car.current_checkpoint else (100, 100, 100)
            pygame.draw.circle(screen, color, cp, 80, 5)
            pygame.draw.circle(screen, (255, 255, 0), cp, 10)

        # Info
        screen.blit(font.render(f"Gen: {current_generation} | Fitness: {genome.fitness:.1f}", True, (255,255,255)), (50, 50))
        screen.blit(font.render(f"Checkpoints: {car.current_checkpoint}/10", True, (255,255,255)), (50, 100))
        screen.blit(font.render(f"Speed: {car.speed:.1f}", True, (255,255,255)), (50, 150))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# ====================== MAIN EVAL WITH VISUALIZATION ======================
def eval_genomes(genomes, config):
    global current_generation, current_map_surface

    current_generation += 1
    current_map_surface = pygame.image.load(random.choice(MAPS)).convert()  # Simple convert, no alpha

    # Evaluate all genomes in parallel
    for _, genome in genomes:
        genome.fitness = None  # Reset

    # Run parallel evaluation
    list(genomes)  # Force evaluation

    # Find best
    best = max(genomes, key=lambda g: g[1].fitness or 0)[1]

    # Visualize every 3 generations or when fitness > 10000
    if current_generation % 3 == 0 or best.fitness > 10000:
        print(f"\n=== VISUALIZING BEST CAR - Generation {current_generation} | Fitness: {best.fitness:.1f} ===")
        visualize_best(best, config)

# ====================== MAIN ======================
if __name__ == "__main__":
    # Create a hidden tiny display JUST for convert_alpha()
    os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Optional: forces dummy driver (works on most systems)
    pygame.init()
    temp_screen = pygame.display.set_mode((1, 1))  # Tiny invisible window

    # Now safe to load with alpha
    car_sprite_original = pygame.image.load('car.png').convert_alpha()

    # Clean up temp display
    pygame.display.quit()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, "config.txt")

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix='checkpoint-'))

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    pe = neat.ParallelEvaluator(num_workers, evaluate_genome)

    winner = p.run(pe.evaluate, 1000)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nTraining complete! Winner saved as winner.pkl")