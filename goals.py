import pygame

class Goal:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.isactiv = False
    
    def draw(self, win):
        pygame.draw.line(win, (0, 255, 0), (self.x1, self.y1), (self.x2, self.y2), 2)
        if self.isactiv:
            pygame.draw.line(win, (255, 0, 0), (self.x1, self.y1), (self.x2, self.y2), 2)

# Function to parse reward barriers into Goal objects
def get_goals():
    goals = []
    
    # Open and read the file
    with open("reward_barriers.txt", "r") as file:
        lines = file.readlines()
    
    for line in lines:
        # Remove whitespace and split the coordinates
        line = line.strip()
        if line:
            try:
                # Parse coordinate pairs
                coord1, coord2 = line.split("),(")
                x1, y1 = map(int, coord1.strip("()").split(","))
                x2, y2 = map(int, coord2.strip("()").split(","))
                # Create a Goal object and append to the list
                goals.append(Goal(x1, y1, x2, y2))
            except ValueError:
                print(f"Error parsing line: {line}")
    
    if goals:
        goals[-1].isactiv = True

    return goals