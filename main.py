import pygame
import math
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def interpolate_points(points, num_points):
    # Convert points to numpy array
    points = np.array(points)

    # Calculate cumulative distances along the contour
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Normalize cumulative distances
    total_distance = cumulative_distances[-1]
    normalized_distances = cumulative_distances / total_distance

    # Interpolate x and y coordinates based on normalized distances
    interp_func_x = interp1d(normalized_distances, points[:, 0], kind='linear')
    interp_func_y = interp1d(normalized_distances, points[:, 1], kind='linear')

    # Generate evenly spaced points
    even_spaced = np.linspace(0, 1, num_points)
    new_points = [(int(interp_func_x(t)), int(interp_func_y(t))) for t in even_spaced]

    return new_points


def find_closest_point(point, points):
    min_distance = float('inf')
    closest_point = None
    for p in points:
        distance = math.dist(point, p)
        if distance < min_distance:
            min_distance = distance
            closest_point = p
    return closest_point


def draw_reward_lines(screen, outer_points, inner_points, num_lines, line_color):
    # Interpolate points to get evenly spaced coordinates
    segments = []
    outer_interp = interpolate_points(outer_points, num_lines)
    inner_interp = interpolate_points(inner_points, num_lines)

    # Draw lines between the interpolated outer and inner points
    for outer_point in outer_interp:
        closest_inner_point = find_closest_point(outer_point, inner_interp)
        if closest_inner_point:
            pygame.draw.line(screen, line_color, outer_point, closest_inner_point, 2)
            segments.append([outer_point,closest_inner_point])
    return(segments)


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

def distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def scale_outer_wall(walls_ins, walls_out, scale_factor=2):
    scaled_walls_out = []

    # Convert inner wall points to a list of tuples for easy distance calculation
    ins_coords = [(wall.x1, wall.y1) for wall in walls_ins] + [(wall.x2, wall.y2) for wall in walls_ins]

    for wall in walls_out:
        x1_out, y1_out, x2_out, y2_out = wall.x1, wall.y1, wall.x2, wall.y2

        def scale_point(x_out, y_out):
            # Step 2: Find the closest point on the inner wall
            closest_point = min(ins_coords, key=lambda p: distance(x_out, y_out, p[0], p[1]))
            x_in, y_in = closest_point

            # Step 3: Calculate the direction vector from inner to outer point
            dx = x_out - x_in
            dy = y_out - y_in

            # Step 4: Calculate the new scaled outer point
            x_scaled = x_in + dx * scale_factor
            y_scaled = y_in + dy * scale_factor

            return int(x_scaled), int(y_scaled)

        # Scale both endpoints of the wall segment
        new_x1_out, new_y1_out = scale_point(x1_out, y1_out)
        new_x2_out, new_y2_out = scale_point(x2_out, y2_out)

        # Step 5: Create a new Wall object with the scaled points
        scaled_walls_out.append(Wall(new_x1_out, new_y1_out, new_x2_out, new_y2_out))

    return scaled_walls_out

# Example usage:
walls_ins = scale_outer_wall(walls_out, walls_ins, scale_factor=2)

# Function to calculate the perpendicular offset
# Define margin
MARGIN = 10

WI = []
for j in range(len(walls_ins)):
    WI.append((walls_ins[j].x1,walls_ins[j].y1))
    WI.append((walls_ins[j].x2, walls_ins[j].y2))

WO = []
for j in range(len(walls_out)):
    WO.append([walls_out[j].x1,walls_out[j].y1])
    WO.append([walls_out[j].x2, walls_out[j].y2])

# Initialize Pygame
pygame.init()

# Set up the window dimensions
width, height = 1600, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("JAI HIND")
# Main loop
running = True
screen.fill(WHITE)
for wall in walls_ins:
    wall.draw(screen)
pygame.draw.polygon(screen, ROAD_COLOR, WI)
for wall1 in walls_out:
    wall1.draw(screen)
pygame.draw.polygon(screen, WHITE, WO)
draw_reward_lines(screen, WO, WI, 200, (0, 0, 255))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    pygame.display.flip()

# Quit Pygame
pygame.quit()
