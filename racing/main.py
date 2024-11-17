import pygame
import math

# Initialize Pygame
pygame.init()

# Set up the window dimensions
width, height = 1200, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("JAI HIND")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
TRACK_COLOR = (255, 0, 0)  # Color for the track (e.g., red)
ROAD_COLOR = (100, 100, 100)  # Color for the wider road (gray)

# Load the resized coordinates from the txt file
coordinates_file = "resized_track_coordinates.txt"  # The file with the resized coordinates

# Define the widening distance for the road
widening_distance = 15  # Change this value to make the road wider or thinner


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

# Function to filter coordinates near the margin
def is_within_margin(x, y):
    return MARGIN <= x <= width - MARGIN and MARGIN <= y <= height - MARGIN

# Read the coordinates and filter based on margin
coordinates = []
with open(coordinates_file, "r") as file:
    for line in file:
        x, y = map(int, line.strip().split(", "))
        if is_within_margin(x, y):
            coordinates.append((x, y))

CAR_COLOR = (0, 255, 0)
car_width,car_height = 2,4
def draw_car(x, y):
    pygame.draw.rect(screen, CAR_COLOR, (x, y, car_width, car_height))

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with black
    screen.fill(WHITE)

    mouse_x, mouse_y = pygame.mouse.get_pos()
    font = pygame.font.Font(None, 36)
    text = font.render(f"Mouse Coordinates: ({mouse_x}, {mouse_y})", True, (0, 0, 0))
    screen.blit(text, (10, 10))
    if len(coordinates) > 2:
        pygame.draw.polygon(screen, ROAD_COLOR, coordinates)
        pygame.draw.polygon(screen, ROAD_COLOR, [(604, 150), (603, 133), (608, 133), (607, 151)])
    carx, cary = mouse_x, mouse_y
    draw_car(carx, cary)
    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
