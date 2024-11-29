import pygame
import math
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


def distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def scale_point(x_out, y_out, ins_coords, scale_factor):
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


def scale_wall(walls_ins, walls_out, scale_factor=4):
    scaled_walls_out = []

    # Convert inner wall points to a list of tuples for easy distance calculation
    ins_coords = [(wall.x1, wall.y1) for wall in walls_ins] + [(wall.x2, wall.y2) for wall in walls_ins]

    for wall in walls_out:
        x1_out, y1_out, x2_out, y2_out = wall.x1, wall.y1, wall.x2, wall.y2

        # Scale both endpoints of the wall segment
        new_x1_out, new_y1_out = scale_point(x1_out, y1_out, ins_coords, scale_factor)
        new_x2_out, new_y2_out = scale_point(x2_out, y2_out, ins_coords, scale_factor)

        # Step 5: Create a new Wall object with the scaled points
        scaled_walls_out.append(Wall(new_x1_out, new_y1_out, new_x2_out, new_y2_out))

    return scaled_walls_out

def get_walls():
    ins = "track_coordinates.txt"
    inside_wall = []
    with open(ins, "r") as file:
        for line in file:
            x, y = map(int, line.strip().split(", "))
            inside_wall.append((x, y))
    outs = "resized_track_coordinates.txt"
    outside_wall = []
    with open(outs, "r") as file:
        for line in file:
            x, y = map(int, line.strip().split(", "))
            outside_wall.append((x, y))

    # Create wall objects from contour points
    walls_in = [Wall(x1, y1, x2, y2) for x1, y1, x2, y2 in load_contour(ins)]
    walls_out = [Wall(x1, y1, x2, y2) for x1, y1, x2, y2 in load_contour(outs)]
    return walls_in + walls_out
