import pygame
import math
from Walls import get_walls
from goals import getGoals
from utils import Point, Line, Ray, distance, rotate, rotate_rect, line_intersection

GOALREWARD = 10
LIFE_REWARD = 1
PENALTY = -10

starting_point = (665, 320)
control_start = 1300

class Car:
    def __init__(self, x, y, color=(255, 0, 0)):
        self.position = Point(x, y)
        self.width = 6
        self.height = 15
        self.points = 0
        self.color = color


        self.angle = math.radians(180)
        self.target_angle = self.angle
        self.velocity = 0
        self.max_velocity = 15
        self.acceleration = 5

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

        self.update_corners()


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
    
    def collision_color(self, track_surface, cars):
        # Define the car's edges as lines
        car_lines = [
            Line(self.p1, self.p2),
            Line(self.p2, self.p3),
            Line(self.p3, self.p4),
            Line(self.p4, self.p1),
        ]

        for line in car_lines:
            # Sample points along each line
            num_samples = 10  # Number of points to sample along each edge
            for i in range(num_samples + 1):
                # Interpolate between the two points of the line
                x = int(line.pt1.x + i * (line.pt2.x - line.pt1.x) / num_samples)
                y = int(line.pt1.y + i * (line.pt2.y - line.pt1.y) / num_samples)

                # Check if the point is within the bounds of the track surface
                if 0 <= x < track_surface.get_width() and 0 <= y < track_surface.get_height():
                    color = track_surface.get_at((x, y))[:3]  # Get RGB color at the point
                    if color == (255, 255, 255):
                        return True  # Collision detected
        for car in cars:
            # Skip self-check
            if car == self:
                continue
            
            other_car = car

            other_car_lines = [
                Line(other_car.p1, other_car.p2),
                Line(other_car.p2, other_car.p3),
                Line(other_car.p3, other_car.p4),
                Line(other_car.p4, other_car.p1),
            ]

            # Check if any line of this car intersects with any line of the other car
            for line in car_lines:
                for other_line in other_car_lines:
                    if line_intersection(line, other_line):
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
        self.velocity = 0
        self.angle = math.radians(180)
        self.target_angle = self.angle
        self.points = 0
        self.update_corners()
        self.rect = self.image.get_rect(center=(self.position.x, self.position.y))

    def draw(self, screen):
        # Draw the car as a rotated rectangle
        pygame.draw.polygon(
            screen, 
            self.color, 
            [(self.p1.x, self.p1.y), (self.p2.x, self.p2.y), (self.p3.x, self.p3.y), (self.p4.x, self.p4.y)]
        )


class RacingEnv:
    def __init__(self, num_cars=1):
        pygame.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 36)

        self.fps = 120
        self.width = 1600
        self.height = 800

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RACING DQN")
        self.track_surface = pygame.image.load("assets/yas_track.png").convert()
        self.track_surface = pygame.transform.scale(self.track_surface, (self.width, self.height))

        self.num_cars = num_cars
        self.reset()

    def reset(self):
        self.cars = [Car(starting_point[0] + i * 20, starting_point[1] + i * 20, color = (0, 255, 0) if i == 0 else (0, 0, 255)) for i in range(self.num_cars)]  # Position cars slightly apart
        self.walls = get_walls()
        # self.goals = getGoals()

    def step(self, actions):
        """
        Accept a list of actions, one for each car.
        """
        done = [False] * self.num_cars
        rewards = [LIFE_REWARD] * self.num_cars
        new_states = [None] * self.num_cars

        for i, car in enumerate(self.cars):
            car.action(actions[i])
            car.update()
            reward = LIFE_REWARD

            # Check if car passes Goal and scores
            # index = 1
            # for goal in self.goals:
            #     if index > len(self.goals):
            #         index = 1
            #     if goal.isactiv:
            #         if car.score(goal):
            #             goal.isactiv = False
            #             self.goals[index-2].isactiv = True
            #             reward += GOALREWARD

            #     index = index + 1

            # Check if car crashed in the wall
            if car.collision_color(self.track_surface, self.cars):
                reward += PENALTY
                done[i] = True
                car.velocity = 0

            # Update state and reward for this car
            new_states[i] = car.cast(self.walls)
            rewards[i] = reward

            # If done, set state to None
            if done[i]:
                new_states[i] = None

        return new_states, rewards, done

    def render(self, actions):
        DRAW_WALLS = False
        DRAW_GOALS = False
        DRAW_RAYS = False

        pygame.time.delay(10)

        self.clock = pygame.time.Clock()
        self.screen.blit(self.track_surface, (0, 0))

        if DRAW_WALLS:
            for wall in self.walls:
                wall.draw(self.screen)
        
        # if DRAW_GOALS:
        #     for goal in self.goals:
        #         goal.draw(self.screen)
        #         if goal.isactiv:
        #             goal.draw(self.screen)
        
        for i, car in enumerate(self.cars):
            car.draw(self.screen)

            if DRAW_RAYS:
                for pt in car.closestRays:
                    pygame.draw.circle(self.screen, (0, 0, 255), (pt.x, pt.y), 5)
                    pygame.draw.line(self.screen, (255, 255, 255), (car.position.x, car.position.y), (pt.x, pt.y), 1)

            # Draw controls for each car
            if i == 0:
                pygame.draw.rect(self.screen, (255, 255, 255), (control_start + i * 50, 100, 40, 40), 2)
                pygame.draw.rect(self.screen, (255, 255, 255), (control_start + 50 + i * 50, 100, 40, 40), 2)
                pygame.draw.rect(self.screen, (255, 255, 255), (control_start + 100 + i * 50, 100, 40, 40), 2)
                pygame.draw.rect(self.screen, (255, 255, 255), (control_start + 50 + i * 50, 50, 40, 40), 2)

                if actions[i] == 4:
                    pygame.draw.rect(self.screen, (0, 255, 0), (control_start + 50 + i * 50, 50, 40, 40)) 
                elif actions[i] == 6:
                    pygame.draw.rect(self.screen, (0, 255, 0), (control_start + 50 + i * 50, 50, 40, 40))
                    pygame.draw.rect(self.screen, (0, 255, 0), (control_start + i * 50, 100, 40, 40))
                elif actions[i] == 5:
                    pygame.draw.rect(self.screen, (0, 255, 0), (control_start + 50 + i * 50, 50, 40, 40))
                    pygame.draw.rect(self.screen, (0, 255, 0), (control_start + 100 + i * 50, 100, 40, 40))
                elif actions[i] == 1:
                    pygame.draw.rect(self.screen, (0, 255, 0), (control_start + 50 + i * 50, 100, 40, 40)) 
                elif actions[i] == 8:
                    pygame.draw.rect(self.screen, (0, 255, 0), (control_start + 50 + i * 50, 100, 40, 40))
                    pygame.draw.rect(self.screen, (0, 255, 0), (control_start + i * 50, 100, 40, 40))
                elif actions[i] == 7:
                    pygame.draw.rect(self.screen, (0, 255, 0), (control_start + 50 + i * 50, 100, 40, 40))
                    pygame.draw.rect(self.screen, (0, 255, 0), (control_start + 100 + i * 50, 100, 40, 40))
                elif actions[i] == 2:
                    pygame.draw.rect(self.screen, (0, 255, 0), (control_start + i * 50, 100, 40, 40))
                elif actions[i] == 3:
                    pygame.draw.rect(self.screen, (0, 255, 0), (control_start + 100 + i * 50, 100, 40, 40))

                # Draw scores
                text_surface = self.font.render(f'Car {i+1} Points: {car.points}', True, pygame.Color('green'))
                self.screen.blit(text_surface, dest=(10, i * 40))
                text_surface = self.font.render(f'Car {i+1} Speed: {car.velocity * -1}', True, pygame.Color('green'))
                self.screen.blit(text_surface, dest=(control_start, i * 40))

        self.clock.tick(self.fps)
        pygame.display.update()

    def close(self):
        pygame.quit()




