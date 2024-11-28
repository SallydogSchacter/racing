import pygame
import math
from walls import Wall, getWalls
from goals import Goal, getGoals
from utils import Point, Line, Ray, distance, rotate, rotate_rect

GOALREWARD = 10
LIFE_REWARD = -5
PENALTY = -10

class Car:
    def __init__(self, x, y):
        self.position = Point(x, y)
        self.width = 14
        self.height = 30
        self.points = 0

        # Load car image
        self.original_image = pygame.image.load("assets/car.png").convert()
        self.original_image.set_colorkey((0, 0, 0))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))

        # Motion parameters
        self.angle = math.radians(180)
        self.target_angle = self.angle
        self.velocity = 0
        self.max_velocity = 15
        self.acceleration = 1

        # Define corners of the car
        self.update_corners()

    def update_corners(self):
        half_width, half_height = self.width / 2, self.height / 2
        self.pt1 = Point(self.position.x - half_width, self.position.y - half_height)
        self.pt2 = Point(self.position.x + half_width, self.position.y - half_height)
        self.pt3 = Point(self.position.x + half_width, self.position.y + half_height)
        self.pt4 = Point(self.position.x - half_width, self.position.y + half_height)

        # Rotated points
        self.p1, self.p2, self.p3, self.p4 = rotate_rect(self.pt1, self.pt2, self.pt3, self.pt4, self.target_angle)

    def action(self, choice):
        """
        Define car actions based on the input choice:
        0: Do nothing
        1: Move forward
        2: Turn left
        3: Turn right
        4: Move backward
        5: Move backward, turn right
        6: Move backward, turn left
        7: Move forward, turn left
        8: Move forward, turn right
        """
        if choice == 0:
            pass
        elif choice == 1:
            self.accelerate(self.acceleration)
        elif choice == 2:
            self.turn(-1)
        elif choice == 3:
            self.turn(1)
        elif choice == 4:
            self.accelerate(-self.acceleration)
        elif choice == 5:
            self.accelerate(-self.acceleration)
            self.turn(1)
        elif choice == 6:
            self.accelerate(-self.acceleration)
            self.turn(-1)
        elif choice == 7:
            self.accelerate(self.acceleration)
            self.turn(-1)
        elif choice == 8:
            self.accelerate(self.acceleration)
            self.turn(1)


    def accelerate(self, delta_velocity):
        self.velocity = max(-self.max_velocity, min(self.max_velocity, self.velocity + delta_velocity))

    def turn(self, direction):
        self.target_angle += direction * math.radians(15)

    def update(self):
        self.angle = self.target_angle

        # Calculate velocity vector and update position
        velocity_vector = rotate(Point(0, 0), Point(0, self.velocity), self.angle)
        self.position.x += velocity_vector.x
        self.position.y += velocity_vector.y

        self.rect.center = (self.position.x, self.position.y)

        self.update_corners()
        self.image = pygame.transform.rotate(self.original_image, 90 - math.degrees(self.angle))
        self.rect = self.image.get_rect(center=self.rect.center)


    def cast(self, walls):
        """
        Cast rays from the car's position to detect distances to walls.

        Args:
            walls: A list of wall objects to check for intersections.

        Returns:
            observations: A list of normalized distances from the car to the walls.
        """
        # Define angles for the rays (relative to the car's target angle)
        angles = [
            0, -30, 30, -45, 45, -90, 90, 180,
            10, -10, 135, -135, 20, -20, 90, -90, 0, 0
        ]

        # Define starting positions for the rays
        positions = [self.position] * 14 + [self.p1, self.p2, self.p1, self.p2]

        # Ensure angles and positions match in length
        assert len(angles) == len(positions), "Mismatch between angles and positions."

        # Generate rays
        self.rays = [
            Ray(positions[i], self.target_angle + math.radians(angles[i]))
            for i in range(len(angles))
        ]

        # Initialize observations and closest rays
        observations = []
        self.closestRays = []

        # Cast each ray and record the closest intersection
        for ray in self.rays:
            closest = None
            record = math.inf
            for wall in walls:
                pt = ray.cast(wall)
                if pt:
                    dist = distance(self.position, pt)
                    if dist < record:
                        record = dist
                        closest = pt

            if closest:
                self.closestRays.append(closest)
                observations.append(record)
            else:
                observations.append(1000)  # Default value if no intersection

        # Normalize observations to range [0, 1]
        normalized_observations = [(1000 - obs) / 1000 for obs in observations]

        # Append velocity as a normalized value
        normalized_observations.append(self.velocity / self.max_velocity)

        return normalized_observations


    def collision(self, wall):
        car_lines = [
            Line(self.p1, self.p2),
            Line(self.p2, self.p3),
            Line(self.p3, self.p4),
            Line(self.p4, self.p1),
        ]

        x1, y1, x2, y2 = wall.x1, wall.y1, wall.x2, wall.y2

        for line in car_lines:
            x3, y3 = line.pt1.x, line.pt1.y
            x4, y4 = line.pt2.x, line.pt2.y

            denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denominator == 0:
                continue  # Lines are parallel, no collision

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

            # Check if the intersection point lies on both segments
            if 0 < t < 1 and 0 < u < 1:
                return True

        return False
    
    def score(self, goal):
        forward_vector = rotate(Point(0, 0), Point(0, -50), self.angle)
        forward_line = Line(Point(self.position.x, self.position.y), Point(self.position.x + forward_vector.x, self.position.y + forward_vector.y))

        # Goal's endpoints
        x1, y1, x2, y2 = goal.x1, goal.y1, goal.x2, goal.y2

        # Forward line's endpoints
        x3, y3 = forward_line.pt1.x, forward_line.pt1.y
        x4, y4 = forward_line.pt2.x, forward_line.pt2.y

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if denominator == 0:
            # Parallel or coincident lines
            return False

        # Intersection factors
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

        if 0 < t < 1 and 0 < u < 1:
            # Get intersection point
            intersect_x = x1 + t * (x2 - x1)
            intersect_y = y1 + t * (y2 - y1)
            intersection_point = Point(intersect_x, intersect_y)

            if distance(self.position, intersection_point) < 20:
                self.points += GOALREWARD
                return True

        return False
    
    def reset(self):
        self.position = Point(50, 300)
        self.velocity = 0
        self.angle = math.radians(180)
        self.target_angle = self.angle
        self.points = 0
        self.update_corners()
        self.rect = self.image.get_rect(center=(self.position.x, self.position.y))

    def draw(self, win):
        win.blit(self.image, self.rect)


class RacingEnv:

    def __init__(self):
        pygame.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 36)

        self.fps = 120
        self.width = 1000
        self.height = 600
        self.history = []

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RACING DQN")
        self.screen.fill((0,0,0))
        self.back_image = pygame.image.load("assets/track.png").convert()
        self.back_rect = self.back_image.get_rect().move(0, 0)
        self.action_space = None
        self.observation_space = None
        self.game_reward = 0
        self.score = 0
 
        self.reset()


    def reset(self):
        self.screen.fill((0, 0, 0))

        self.car = Car(50, 300)
        self.walls = getWalls()
        self.goals = getGoals()
        self.game_reward = 0

    def step(self, action):

        done = False
        self.car.action(action)
        self.car.update()
        reward = LIFE_REWARD

        # Check if car passes Goal and scores
        index = 1
        for goal in self.goals:
            
            if index > len(self.goals):
                index = 1
            if goal.isactiv:
                if self.car.score(goal):
                    goal.isactiv = False
                    self.goals[index-2].isactiv = True
                    reward += GOALREWARD

            index = index + 1

        # Check if car crashed in the wall
        for wall in self.walls:
            if self.car.collision(wall):
                reward += PENALTY
                done = True

        new_state = self.car.cast(self.walls)
        # Normalize states
        if done:
            new_state = None

        return new_state, reward, done

    def render(self, action):

        DRAW_WALLS = False
        DRAW_GOALS = True
        DRAW_RAYS = False

        pygame.time.delay(10)

        self.clock = pygame.time.Clock()
        self.screen.fill((0, 0, 0))

        self.screen.blit(self.back_image, self.back_rect)

        if DRAW_WALLS:
            for wall in self.walls:
                wall.draw(self.screen)
        
        if DRAW_GOALS:
            for goal in self.goals:
                goal.draw(self.screen)
                if goal.isactiv:
                    goal.draw(self.screen)
        
        self.car.draw(self.screen)

        if DRAW_RAYS:
            i = 0
            for pt in self.car.closestRays:
                pygame.draw.circle(self.screen, (0,0,255), (pt.x, pt.y), 5)
                i += 1
                if i < 15:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.x, self.car.y), (pt.x, pt.y), 1)
                elif i >=15 and i < 17:
                    pygame.draw.line(self.screen, (255,255,255), ((self.car.p1.x + self.car.p2.x)/2, (self.car.p1.y + self.car.p2.y)/2), (pt.x, pt.y), 1)
                elif i == 17:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.p1.x , self.car.p1.y ), (pt.x, pt.y), 1)
                else:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.p2.x, self.car.p2.y), (pt.x, pt.y), 1)

        #render controll
        pygame.draw.rect(self.screen,(255,255,255),(800, 100, 40, 40),2)
        pygame.draw.rect(self.screen,(255,255,255),(850, 100, 40, 40),2)
        pygame.draw.rect(self.screen,(255,255,255),(900, 100, 40, 40),2)
        pygame.draw.rect(self.screen,(255,255,255),(850, 50, 40, 40),2)

        if action == 4:
            pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40)) 
        elif action == 6:
            pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        elif action == 5:
            pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))
        elif action == 1:
            pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40)) 
        elif action == 8:
            pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        elif action == 7:
            pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))
        elif action == 2:
            pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        elif action == 3:
            pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))

        # Score
        text_surface = self.font.render(f'Points {self.car.points}', True, pygame.Color('green'))
        self.screen.blit(text_surface, dest=(0, 0))
        # Speed
        text_surface = self.font.render(f'Speed {self.car.velocity*-1}', True, pygame.Color('green'))
        self.screen.blit(text_surface, dest=(800, 0))

        self.clock.tick(self.fps)
        pygame.display.update()

    def close(self):
        pygame.quit()



