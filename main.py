import pygame
import math
import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev


# Initialize Pygame
pygame.init()

# Set up the window dimensions
width, height = 1200, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("JAI HIND")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WALL_COLOR = (255, 0, 0)  # Color for the track (e.g., red)
ROAD_COLOR = (100, 100, 100)  # Color for the wider road (gray)

# Load the resized coordinates from the txt file
ins = "resized_track_coordinates.txt"
inside_wall = []
with open(ins, "r") as file:
    for line in file:
        x, y = map(int, line.strip().split(", "))
        inside_wall.append((x, y))
outs = "track_coordinates.txt"
outside_wall = []
with open(outs, "r") as file:
    for line in file:
        x, y = map(int, line.strip().split(", "))
        outside_wall.append((x, y))

class Wall:
    def __init__(self, x1, y1, x2, y2, color=(255, 0, 0), thickness=5):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.color = color
        self.thickness = thickness

    def draw(self, screen):
        # Draw the wall as a line between (x1, y1) and (x2, y2)
        pygame.draw.line(screen, self.color, (self.x1, self.y1), (self.x2, self.y2), self.thickness)

# Function to read the contour points from a file
def load_contour(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            x1, y1 = map(int, lines[i].strip().split(','))
            if i+1 >= len(lines):
                second = 0
            else:
                second = i+1
            x2, y2 = map(int, lines[second].strip().split(','))
            coordinates.append((x1, y1, x2, y2))
    return coordinates


# Create wall objects from contour points
walls_ins = [Wall(x1, y1, x2, y2) for x1, y1, x2, y2 in load_contour(ins)]
walls_out = [Wall(x1, y1, x2, y2) for x1, y1, x2, y2 in load_contour(outs)]

# Function to calculate the perpendicular offset
def get_offset_point(x1, y1, x2, y2, distance):
    # Calculate the angle of the line
    angle = math.atan2(y2 - y1, x2 - x1)

    # Calculate the offset direction (perpendicular to the line)
    offset_x = -distance * math.sin(angle)
    offset_y = distance * math.cos(angle)

    # Apply the offset to the points
    return (x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y)


# Define margin
MARGIN = 10


CAR_COLOR = (0, 255, 0)
CAR_WIDTH,CAR_HEIGHT = 3,6
class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0  # Facing right initially
        self.speed = 0
        self.max_speed = 10
        self.min_speed = 0
        self.acceleration = 0.2
        self.brake_deceleration = 0.5
        self.turn_speed = 5
        self.image = pygame.Surface((CAR_WIDTH, CAR_HEIGHT))
        self.image.fill(CAR_COLOR)
        self.rect = self.image.get_rect(center=(self.x, self.y))

    def update(self, action):
        # Parse the action
        steer, accel = action

        # Steering
        if steer == 1:  # Left
            self.angle += self.turn_speed
        elif steer == 2:  # Right
            self.angle -= self.turn_speed

        # Acceleration
        if accel == 0:  # Brake
            self.speed -= self.brake_deceleration
        elif accel == 2:  # Accelerate
            self.speed += self.acceleration

        # Clamp speed
        self.speed = max(self.min_speed, min(self.speed, self.max_speed))

        # Update position
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y -= math.sin(rad) * self.speed

        # Update car rect
        self.rect = self.image.get_rect(center=(self.x, self.y))

    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        new_rect = rotated_image.get_rect(center=self.rect.center)
        screen.blit(rotated_image, new_rect.topleft)


# Define discrete action space
def get_action(action_index):
    # Convert action index to (steer, accel)
    steer = action_index // 3  # 0: straight, 1: left, 2: right
    accel = action_index % 3  # 0: brake, 1: maintain, 2: accelerate
    return steer, accel

# Function to check for collision
def check_collision(carx, cary, track_coords, threshold):
    for (x, y) in track_coords:
        distance = math.hypot(carx - x, cary - y)
        if distance < threshold:
            return True
    return False

START = (680,325)
collision_threshold = 4

track_points = outside_wall + list(reversed(inside_wall))

# Main loop
running = True
car = Car(START[0],START[1])
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)
    action_index = np.random.randint(0,8)  # Random action (1: straight, 2: accelerate)
    action = get_action(action_index)
    car.update(action)
    carx = car.x
    cary = car.y
    pygame.draw.polygon(screen, ROAD_COLOR, track_points)
    pygame.draw.polygon(screen, ROAD_COLOR, [(603,135),(608,135),(608,150)])
    for wall in walls_ins:
        wall.draw(screen)
    for wall1 in walls_out:
        wall1.draw(screen)
    mouse_x, mouse_y = pygame.mouse.get_pos()
    font = pygame.font.Font(None, 36)
    coord_text = font.render(f"Mouse: ({mouse_x}, {mouse_y})", True, (0,0,0))
    # Display the text on the screen
    screen.blit(coord_text, (20, 20))
    #state = state_gen(statex,statey)
    car.draw(screen)
    pygame.display.flip()

# Quit Pygame
pygame.quit()
