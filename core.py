import math
import os

import numpy as np
import pygame as pg

TARGET_RESOLUTION = np.array([1280, 720])
UNIT_SIZE = 20
GRID_SIZE = (TARGET_RESOLUTION / UNIT_SIZE).astype(int)
RESOLUTION = UNIT_SIZE * GRID_SIZE
PERSPECTIVE_GRID_SIZE = np.array([np.max(GRID_SIZE) * 2 - 1, np.max(GRID_SIZE) * 2 - 1])
PERSPECTIVE_GRID_MIDDLE = (PERSPECTIVE_GRID_SIZE / 2).astype(int)
H_SCREEN_CENTER = int(RESOLUTION[0] / 2)
V_SCREEN_CENTER = int(RESOLUTION[1] / 2)
MIDDLE_OF_THE_GRID = [(GRID_SIZE / 2).astype(int)]
LEFT_QUARTER_OF_THE_GRID = [(GRID_SIZE * np.array([0.25, 0.5])).astype(int)]
RIGHT_QUARTER_OF_THE_GRID = [(GRID_SIZE * np.array([0.75, 0.5])).astype(int) + [1, 0]]
VECTOR_SIZE = (1, 2)
RANDOM_POSITION = np.array([-1, -1])
RANDOM_VEL = np.array([0, 0])
FONT = "res/SourceCodePro-Regular.ttf"
MAIN_FONT_SIZE = 20
FPS_FONT_SIZE = 12
V_LINE_PADDING = 4
V_LINE_OFFSET = MAIN_FONT_SIZE + V_LINE_PADDING
V_MENU_OFFSET = 15
V_MENU_PADDING = 15
H_MENU_PADDING = 20
DEBUG_MODE = True
DISPLAY_NN_DATA = False
LAYERS_PER_STATE = 5
AI_TRAINING_MODE = False
TRAINING_DATA_PATH = os.path.dirname(__file__) + "/training_data/" \
                     + str(GRID_SIZE[0]) + "_" + str(GRID_SIZE[1]) + "/"
HUMAN_MODEL_MODE = False
RW_HAS_KILLED = 100
RW_HAS_BITTEN = 65
RW_HAS_EATEN = 10
RW_TIME_PASSED = -1
RW_DID_NOT_MOVE = -10
RW_GOT_BITTEN = -50
RW_HAS_BITTEN_HIMSELF = -65
RW_GOT_KILLED = -200
DISCOUNT_FACTOR = 0.9
IGNORE_REWARD_DISCOUNTED_BY = 0.01
TOO_LATE_REWARD = math.ceil(math.log(IGNORE_REWARD_DISCOUNTED_BY, DISCOUNT_FACTOR))
TRAIN_EVERY_NB_DEAD_SNAKES = 10
MIN_NB_SNAKES = 2
MAX_NB_SNAKES = 15
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BLUE = (0, 0, 100)
LIGHT_GREY = (150, 150, 150)
DARK_GREY = (100, 100, 100)
RED = (255, 0, 0)
CYAN = (85, 232, 217)
LEFT = np.array([-1, 0])
RIGHT = np.array([1, 0])
UP = np.array([0, -1])
DOWN = np.array([0, 1])
DO_NOTHING = np.array([True, False, False, False], dtype=bool)
REVERSE = np.array([False, True, False, False], dtype=bool)
TURN_LEFT = np.array([False, False, True, False], dtype=bool)
TURN_RIGHT = np.array([False, False, False, True], dtype=bool)
RQ_MAIN_MENU = 0
RQ_RESTART = 1
RQ_CONTINUE = 2
INITIAL_NB_SNAKES = 2
INITIAL_SNAKE_LENGTH = 10
DEAD_AT_LENGTH = 2
BIRTH_FROM_LENGTH = 3
FOOD_CHANCE_PER_TICK = 0.05
MAX_FOOD = 12
TARGET_FPS = 15
NB_LAST_FRAMES = 2


def fill_cell(screen, cell_position, color):
    screen.fill(color, pg.Rect(*((cell_position * UNIT_SIZE).tolist()), UNIT_SIZE, UNIT_SIZE))


def pick_random_position():
    return (np.random.rand(*VECTOR_SIZE) * GRID_SIZE).astype(int).flatten()
