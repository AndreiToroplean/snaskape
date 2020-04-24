import numpy as np

from core import VECTOR_SIZE, RANDOM_POSITION, AI_TRAINING_MODE, WHITE, fill_cell, pick_random_position


class Food:
    def __init__(self, snakes, food, color, position=RANDOM_POSITION):
        self.color = color
        self.position = np.zeros(VECTOR_SIZE, int)
        if np.array_equal(position, RANDOM_POSITION):
            self.randomize_position(snakes, food)
        else:
            self.position = position

    def randomize_position(self, snakes, food):
        good_pick = False
        while not good_pick:
            self.position = pick_random_position()
            good_pick = True
            for snake in snakes:
                for cell_position in snake.positions:
                    if np.array_equal(cell_position, self.position):
                        good_pick = False
            for item in food:
                if np.array_equal(item.position, self.position):
                    good_pick = False

    def draw(self, screen, food_surface):
        if not AI_TRAINING_MODE:
            fill_cell(screen, self.position, self.color)
        food_surface.set_at(self.position, WHITE)
