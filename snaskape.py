import numpy as np
import pygame as pg
import random
import sys
import os
import math

from food import Food
from snake import Snake

pg.init()

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
MAIN_FONT = pg.font.Font(FONT, MAIN_FONT_SIZE)
FPS_FONT = pg.font.Font(FONT, FPS_FONT_SIZE)
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
if not os.path.exists(TRAINING_DATA_PATH):
    os.makedirs(TRAINING_DATA_PATH)
HUMAN_MODEL_MODE = False

# Rewards:
RW_HAS_KILLED = 100
RW_HAS_BITTEN = 65
RW_HAS_EATEN = 10
RW_TIME_PASSED = -1
RW_DID_NOT_MOVE = -10
RW_GOT_BITTEN = -50
RW_HAS_BITTEN_HIMSELF = -65
RW_GOT_KILLED = -200

# Model parameters:
DISCOUNT_FACTOR = 0.9
IGNORE_REWARD_DISCOUNTED_BY = 0.01
TOO_LATE_REWARD = math.ceil(math.log(IGNORE_REWARD_DISCOUNTED_BY, DISCOUNT_FACTOR))
TRAIN_EVERY_NB_DEAD_SNAKES = 10

# AI TRAINING MODE PARAMETERS:
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

# Menu requests:
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


def get_char_width(font):
    return font.render("a", True, BLACK).get_rect().width


def get_popup_size(message, char_width):
    max_line_length = 0
    for line in message:
        if len(line) > max_line_length:
            max_line_length = len(line)
    return (
        max_line_length * char_width + 2 * H_MENU_PADDING,
        len(message) * V_LINE_OFFSET - V_LINE_PADDING + 2 * V_MENU_PADDING
        )


def display_message(screen, message, popup_top):
    i = 0
    for line in message:
        text = MAIN_FONT.render(line, True, BLACK)
        screen.blit(text, (
            H_SCREEN_CENTER - text.get_rect().width / 2,
            popup_top + V_MENU_PADDING + i * V_LINE_OFFSET - V_MENU_OFFSET
            ))
        i += 1


def display_popup(screen, message):
    message.insert(1, "-" * len(message[0]))
    char_width = get_char_width(MAIN_FONT)
    popup_size = get_popup_size(message, char_width)
    popup_top = V_SCREEN_CENTER - popup_size[1] / 2
    screen.fill(WHITE,
        pg.Rect(((H_SCREEN_CENTER - popup_size[0] / 2), popup_top - V_MENU_OFFSET),
            popup_size))
    display_message(screen, message, popup_top)


def pick_random_position():
    return (np.random.rand(*VECTOR_SIZE) * GRID_SIZE).astype(int).flatten()


def screen_to_pers(surface, position, direction):
    pers_surface = pg.Surface(PERSPECTIVE_GRID_SIZE, depth=8)
    pers_surface.fill(BLACK)
    if np.array_equal(direction, UP):
        angle = 0
    elif np.array_equal(direction, LEFT):
        angle = -90
    elif np.array_equal(direction, DOWN):
        angle = 180
    else:
        angle = 90
    pers_surface.blit(surface, PERSPECTIVE_GRID_MIDDLE - position)
    return pg.transform.rotate(pers_surface, angle)


def back_propagate_rewards(rewards):
    bp_rewards = []
    for i in range(len(rewards)):
        bp_reward = 0
        for j in range(len(rewards[i:i + TOO_LATE_REWARD])):
            bp_reward += rewards[i + j] * pow(DISCOUNT_FACTOR, j)
        bp_rewards.append(bp_reward)
    return bp_rewards


def pseudo_predict():
    return random.choice((DO_NOTHING, REVERSE, TURN_LEFT, TURN_RIGHT))


def start_loop(screen, clock):
    screen.fill(LIGHT_GREY)

    message = ["WELCOME TO SNASKAPE!",
        "",
        "Start playing directly by pressing the number of players on the keypad.",
        "Possible choices: 1, 2, and 0 for AI mode.",
        "",
        "Controls:",
        "Player one: arrows",
        "Player two: Z, Q, S, D",
        "Pause: Escape",
        "",
        "Press Enter on the keypad to train the AI."]
    display_popup(screen, message)

    pg.display.update()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    sys.exit()
                elif event.key == pg.K_KP0:
                    return 0
                elif event.key == pg.K_KP1:
                    return 1
                elif event.key == pg.K_KP2:
                    return 2
                elif event.key == pg.K_KP_ENTER:
                    return -1

        clock.tick(TARGET_FPS)


def pause_loop(screen, clock):
    message = ["PAUSE MENU",
        "",
        "Continue: Enter",
        "Restart: Backspace",
        "Main menu: Tab",
        "Exit: Escape"]
    display_popup(screen, message)

    pg.display.update()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    sys.exit()
                elif event.key == pg.K_RETURN:
                    return RQ_CONTINUE
                elif event.key == pg.K_BACKSPACE:
                    return RQ_RESTART
                elif event.key == pg.K_TAB:
                    return RQ_MAIN_MENU

        clock.tick(TARGET_FPS)


def end_loop(screen, clock, winner):
    message = ["GAME OVER, {} WINS!".format(winner),
        "",
        "Restart: Backspace",
        "Main menu: Tab",
        "Exit: Escape"]
    display_popup(screen, message)

    pg.display.update()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    sys.exit()
                elif event.key == pg.K_BACKSPACE:
                    return RQ_RESTART
                elif event.key == pg.K_TAB:
                    return RQ_MAIN_MENU

        clock.tick(TARGET_FPS)


# ---------------------------------- INITIALIZING GAME ----------------------------------

def game_loop(screen, clock, nb_of_players, living_players):
    # Controlling number of snakes:
    if INITIAL_NB_SNAKES < nb_of_players:
        nb_of_snakes = nb_of_players
    else:
        nb_of_snakes = INITIAL_NB_SNAKES

    # Initialising snakes:
    snakes = []
    for i in range(nb_of_snakes):
        if i == 0:
            if nb_of_players == 0:
                snakes.append(Snake(RIGHT_QUARTER_OF_THE_GRID, LEFT))
            elif nb_of_players == 1:
                snakes.append(Snake(MIDDLE_OF_THE_GRID, RANDOM_VEL, BLACK))
            elif nb_of_players == 2:
                snakes.append(Snake(RIGHT_QUARTER_OF_THE_GRID, LEFT, BLACK))
            elif nb_of_players > 2:
                snakes.append(Snake(RANDOM_POSITION, RANDOM_VEL, BLACK))
        elif i == 1:
            if nb_of_players == 0:
                snakes.append(Snake(LEFT_QUARTER_OF_THE_GRID, RIGHT))
            elif nb_of_players == 1:
                snakes.append(Snake(RANDOM_POSITION, RANDOM_VEL))
            elif nb_of_players == 2:
                snakes.append(Snake(LEFT_QUARTER_OF_THE_GRID, RIGHT, DARK_BLUE))
            elif nb_of_players > 2:
                snakes.append(Snake(RANDOM_POSITION, RANDOM_VEL, DARK_BLUE))
        else:
            snakes.append(Snake(RANDOM_POSITION, RANDOM_VEL))

    # Assigning snakes to players:
    players = []
    for i in range(nb_of_players):
        players.append(snakes[i])

    # Initialising food:
    food = []
    for i in range(random.randrange(1, 4)):
        food.append(Food(snakes, food, CYAN))

    # Initialising other variables:
    last_frames = NB_LAST_FRAMES
    slow_mo = 1
    game_loop_cycles = 0
    human_data_processed = False
    dead_snakes = []

    heads_surface = pg.Surface(GRID_SIZE, depth=8)
    food_surface = pg.Surface(GRID_SIZE, depth=8)
    borders_surface = pg.Surface(GRID_SIZE, depth=8)
    borders_surface.blit(screen, (0, 0))
    borders_surface.fill(WHITE)

    # Displaying AI training popup:
    if AI_TRAINING_MODE:
        screen.fill(LIGHT_GREY)
        message = ["TRAINING THE AI",
            ""
            "Check console for details.",
            "Save and exit: Escape"]
        display_popup(screen, message)

        pg.display.update()

    # Loading human data:
    if nb_of_players > 0:
        human_data_exists = False
    # if os.path.isfile(TRAINING_DATA_PATH + "human_states.npy"):
    # 	human_data_exists = True
    # 	human_states = np.load(TRAINING_DATA_PATH + "human_states.npy")
    # 	human_actions = np.load(TRAINING_DATA_PATH + "human_actions.npy")
    # 	human_rewards = np.load(TRAINING_DATA_PATH + "human_rewards.npy")
    # 	print("Human data loaded.")

    # ---------------------------------- ACTUAL GAME LOOP ----------------------------------
    while True:

        # User input handling:
        if not AI_TRAINING_MODE:
            # Controls handling:
            for player in players:
                player.pressed_arrow_key = False

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        # Pause menu:
                        request = pause_loop(screen, clock)
                        if request == RQ_RESTART or request == RQ_MAIN_MENU:
                            return request
                    elif event.key == pg.K_BACKSPACE and DEBUG_MODE:
                        return RQ_RESTART
                    elif event.key == pg.K_KP_ENTER and DEBUG_MODE:
                        slow_mo = 5

                    if nb_of_players > 0:
                        if event.key == pg.K_LEFT:
                            players[0].new_potential_vel = LEFT
                            players[0].pressed_arrow_key = True
                        elif event.key == pg.K_RIGHT:
                            players[0].new_potential_vel = RIGHT
                            players[0].pressed_arrow_key = True
                        elif event.key == pg.K_UP:
                            players[0].new_potential_vel = UP
                            players[0].pressed_arrow_key = True
                        elif event.key == pg.K_DOWN:
                            players[0].new_potential_vel = DOWN
                            players[0].pressed_arrow_key = True
                        elif event.key == pg.K_KP_PLUS and DEBUG_MODE:
                            players[0].length += 1
                        elif event.key == pg.K_KP_MINUS and DEBUG_MODE:
                            players[0].length -= 1

                    if nb_of_players > 1:
                        if event.key == pg.K_a:
                            players[1].new_potential_vel = LEFT
                            players[1].pressed_arrow_key = True
                        elif event.key == pg.K_d:
                            players[1].new_potential_vel = RIGHT
                            players[1].pressed_arrow_key = True
                        elif event.key == pg.K_w:
                            players[1].new_potential_vel = UP
                            players[1].pressed_arrow_key = True
                        elif event.key == pg.K_s:
                            players[1].new_potential_vel = DOWN
                            players[1].pressed_arrow_key = True

                elif event.type == pg.KEYUP:
                    if event.key == pg.K_KP_ENTER:
                        slow_mo = 1

            # Applying and logging controls:
            for i in range(len(players)):
                if players[i].pressed_arrow_key \
                        and players[i].not_going_into_wall(players[i].new_potential_vel) \
                        and players[i].not_going_into_snake_head(snakes) \
                        and not np.array_equal(players[i].vel, players[i].new_potential_vel):
                    if not np.array_equal(- players[i].vel, players[i].new_potential_vel):
                        if players[i].vel.tolist() == LEFT.tolist():
                            if players[i].new_potential_vel.tolist() == UP.tolist():
                                players[i].actions.append(TURN_RIGHT)
                            else:
                                players[i].actions.append(TURN_LEFT)
                        elif players[i].vel.tolist() == UP.tolist():
                            if players[i].new_potential_vel.tolist() == RIGHT.tolist():
                                players[i].actions.append(TURN_RIGHT)
                            else:
                                players[i].actions.append(TURN_LEFT)
                        elif players[i].vel.tolist() == RIGHT.tolist():
                            if players[i].new_potential_vel.tolist() == DOWN.tolist():
                                players[i].actions.append(TURN_RIGHT)
                            else:
                                players[i].actions.append(TURN_LEFT)
                        else:
                            if players[i].new_potential_vel.tolist() == LEFT.tolist():
                                players[i].actions.append(TURN_RIGHT)
                            else:
                                players[i].actions.append(TURN_LEFT)

                        players[i].vel = players[i].new_potential_vel
                    else:
                        players[i].actions.append(REVERSE)

                        players[i].vel = players[i].get_tail_direction()
                        players[i].positions.reverse()
                else:
                    players[i].actions.append(DO_NOTHING)

        else:  # if AI_TRAINING_MODE:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        # Save the model
                        pg.quit()
                        sys.exit()

        # AI input handling:
        for snake in snakes:
            if snake not in players:
                # Ask model for prediction...
                snake.pseudo_prediction = DO_NOTHING
                if random.random() < 2 / TARGET_FPS:
                    snake.pseudo_prediction = pseudo_predict()
                snake.actions.append(snake.pseudo_prediction)

                if snake.vel.tolist() == LEFT.tolist():
                    if snake.pseudo_prediction.tolist() == TURN_RIGHT.tolist():
                        snake.new_potential_vel = UP
                    elif snake.pseudo_prediction.tolist() == TURN_LEFT.tolist():
                        snake.new_potential_vel = DOWN
                    elif snake.pseudo_prediction.tolist() == REVERSE.tolist():
                        snake.new_potential_vel = RIGHT
                elif snake.vel.tolist() == RIGHT.tolist():
                    if snake.pseudo_prediction.tolist() == TURN_RIGHT.tolist():
                        snake.new_potential_vel = DOWN
                    elif snake.pseudo_prediction.tolist() == TURN_LEFT.tolist():
                        snake.new_potential_vel = UP
                    elif snake.pseudo_prediction.tolist() == REVERSE.tolist():
                        snake.new_potential_vel = LEFT
                elif snake.vel.tolist() == UP.tolist():
                    if snake.pseudo_prediction.tolist() == TURN_RIGHT.tolist():
                        snake.new_potential_vel = RIGHT
                    elif snake.pseudo_prediction.tolist() == TURN_LEFT.tolist():
                        snake.new_potential_vel = LEFT
                    elif snake.pseudo_prediction.tolist() == REVERSE.tolist():
                        snake.new_potential_vel = DOWN
                else:
                    if snake.pseudo_prediction.tolist() == TURN_RIGHT.tolist():
                        snake.new_potential_vel = LEFT
                    elif snake.pseudo_prediction.tolist() == TURN_LEFT.tolist():
                        snake.new_potential_vel = RIGHT
                    elif snake.pseudo_prediction.tolist() == REVERSE.tolist():
                        snake.new_potential_vel = UP

                if snake.not_going_into_wall(snake.new_potential_vel) and snake.not_going_into_snake_head(snakes):
                    if snake.pseudo_prediction.tolist() == REVERSE.tolist():
                        snake.vel = snake.get_tail_direction()
                        snake.positions.reverse()
                    elif not snake.pseudo_prediction.tolist() == DO_NOTHING.tolist():
                        snake.vel = snake.new_potential_vel

        # Making snakes birth and die in AI training mode:
        if AI_TRAINING_MODE:
            while len(snakes) < MIN_NB_SNAKES:
                snakes.append(Snake(RANDOM_POSITION, RANDOM_VEL))
            while len(snakes) > MAX_NB_SNAKES:
                random.choice(snakes).is_dead = True

        # Snakes logic:
        for snake in snakes:
            snake.rewards.append(RW_TIME_PASSED)
            snake.move(snakes)

        for snake in snakes:
            snake.check_got_bitten(snakes)
            snake.check_has_eaten(food)

        # Dealing with dead snakes:
        for snake in snakes:
            if snake.is_dead:
                if not AI_TRAINING_MODE:
                    if snake in players:
                        living_players[players.index(snake)] = False
                else:
                    dead_snakes.append(snake)
                snakes.remove(snake)

        # Screen reset:
        screen.fill(LIGHT_GREY)

        # Snakes display and state processing:
        heads_surface.fill(BLACK)
        for snake in snakes:
            snake.body_surface.fill(BLACK)
            snake.draw(screen, heads_surface)

        # Food logic:
        if random.random() < FOOD_CHANCE_PER_TICK * math.pow((MAX_FOOD - len(food)) / MAX_FOOD, 4):
            food.append(Food(snakes, food, CYAN))

        # Food display:
        food_surface.fill(BLACK)
        for item in food:
            item.draw(screen, food_surface)

        # Perspective transformations per snake
        if len(snakes) > 0:
            bodies_surface = snakes[0].body_surface.copy()
        for snake in snakes[1:]:
            bodies_surface.blit(snake.body_surface, (0, 0), special_flags=pg.BLEND_ADD)
        for snake in snakes:
            for i in range(LAYERS_PER_STATE):
                snake.pers_surfaces[i].fill(BLACK)
            snake.pers_surfaces[0] = screen_to_pers(snake.body_surface, snake.positions[-1], snake.vel)
            snake.pers_surfaces[1] = screen_to_pers(bodies_surface, snake.positions[-1], snake.vel)
            snake.pers_surfaces[2] = screen_to_pers(heads_surface, snake.positions[-1], snake.vel)
            snake.pers_surfaces[3] = screen_to_pers(food_surface, snake.positions[-1], snake.vel)
            snake.pers_surfaces[4] = screen_to_pers(borders_surface, snake.positions[-1], snake.vel)
            snake.pers_surfaces[1].blit(snake.pers_surfaces[0], (0, 0), special_flags=pg.BLEND_SUB)

            # Logging state:
            for i in range(LAYERS_PER_STATE):
                snake.states[i].append((pg.surfarray.array2d(snake.pers_surfaces[i]) / 127).astype(bool))

        if DEBUG_MODE:
            stand_offset = 40
            pers_offset = 140
            top = 5
            left = 5
            right = 1145

            # FPS display:
            current_fps = round(clock.get_fps(), 1)
            screen.blit(FPS_FONT.render(str(current_fps) + " / " + str(TARGET_FPS), True, BLACK), (left, top))

            # Display neural net data:
            i = 0.4
            if DISPLAY_NN_DATA and nb_of_players > 0:
                for snake in snakes:
                    screen.blit(snake.body_surface, (left, top + stand_offset * i))
                    i += 1
                screen.blit(heads_surface, (left, top + stand_offset * i))
                i += 1
                screen.blit(food_surface, (left, top + stand_offset * i))
                i += 1
                screen.blit(bodies_surface, (left, top + stand_offset * i))
                i += 1

                if living_players[0]:
                    rw_min = -300
                    rw_max = 300
                    if game_loop_cycles == 0:
                        rw = 0
                    else:
                        rw = players[0].rewards[game_loop_cycles] + rw * DISCOUNT_FACTOR
                    rw_color = max(min(math.floor((rw - rw_min) / (rw_max - rw_min) * 256), 255), 0)
                    screen.fill((rw_color, rw_color, rw_color), pg.Rect((left, top + stand_offset * i), GRID_SIZE))

                    i = 0
                    for pers_surface in players[0].pers_surfaces:
                        screen.blit(pers_surface, (right, top + pers_offset * i))
                        i += 1

        # Display update
        if not AI_TRAINING_MODE:
            pg.display.update()

        # Concatenating dead snakes data and training the AI model:
        if AI_TRAINING_MODE and len(dead_snakes) >= TRAIN_EVERY_NB_DEAD_SNAKES:
            ai_data_created = False
            for dead_snake in dead_snakes:
                ai_bp_rewards = back_propagate_rewards(dead_snake.rewards)
                if ai_data_created:
                    ai_states = np.concatenate((ai_states, np.array(dead_snake.states, dtype=bool)[:, :-1]), 1)
                    ai_actions = np.concatenate((ai_actions, np.array(dead_snake.actions, dtype=bool)[1:]), 0)
                    ai_rewards = np.concatenate((ai_rewards, np.array(ai_bp_rewards, dtype=float)[1:]), 0)
                else:
                    ai_states = np.array(dead_snake.states, dtype=bool)[:, 1:]
                    ai_actions = np.array(dead_snake.actions, dtype=bool)[1:]
                    ai_rewards = np.array(ai_bp_rewards, dtype=float)[1:]
                    ai_data_created = True

            # Train AI model
            print("AI model trained.")

        # Saving human data and training the human model:
        if nb_of_players > 0:
            if sum(living_players) < nb_of_players or len(snakes) == 1:
                # Last frames timer:
                last_frames -= 1

                if not human_data_processed:
                    for player in players:
                        human_bp_rewards = back_propagate_rewards(player.rewards)
                        if human_data_exists:
                            human_states = np.concatenate((human_states, np.array(player.states, dtype=bool)[:, :-1]),
                                1)
                            human_actions = np.concatenate((human_actions, np.array(player.actions, dtype=bool)[1:]), 0)
                            human_rewards = np.concatenate((human_rewards, np.array(human_bp_rewards, dtype=float)[1:]),
                                0)
                        else:
                            human_states = np.array(player.states, dtype=bool)[:, 1:]
                            human_actions = np.array(player.actions, dtype=bool)[1:]
                            human_rewards = np.array(human_bp_rewards, dtype=float)[1:]
                            human_data_exists = True
                    print("Human data saved.")
                    np.save(TRAINING_DATA_PATH + "human_states.npy", human_states)
                    np.save(TRAINING_DATA_PATH + "human_actions.npy", human_actions)
                    np.save(TRAINING_DATA_PATH + "human_rewards.npy", human_rewards)

                    # Train human model
                    print("Human model trained." + ".. I wish...")

                    human_data_processed = True

        # End of game:
        if (nb_of_players > 0 and not DEBUG_MODE) or len(snakes) == 0:
            if sum(living_players) == 0:
                if last_frames == 0:
                    winner = "AI"
                    request = end_loop(screen, clock, winner)
                    return request
            elif sum(living_players) == 1 and (nb_of_players > 1 or len(snakes) == 1):
                if last_frames == 0:
                    winner = "PLAYER " + str(living_players.index(True) + 1)
                    request = end_loop(screen, clock, winner)
                    return request

        # Time flow:
        if not AI_TRAINING_MODE:
            clock.tick(TARGET_FPS / slow_mo)
        for snake in snakes:
            snake.time += 1
        game_loop_cycles += 1


# ---------------------------------- GAME LOOP END ----------------------------------


def main():
    pg.display.set_caption("Snaskape")
    screen = pg.display.set_mode(RESOLUTION)
    clock = pg.time.Clock()

    while True:
        nb_of_players = start_loop(screen, clock)
        if nb_of_players == -1:
            global AI_TRAINING_MODE
            AI_TRAINING_MODE = True
            global HUMAN_MODEL_MODE
            HUMAN_MODEL_MODE = False
            global DEBUG_MODE
            DEBUG_MODE = False
            nb_of_players = 0

        # Loop for restarting the game:
        while True:
            living_players = []
            for i in range(nb_of_players):
                living_players.append(True)
            request = game_loop(screen, clock, nb_of_players, living_players)
            if request == RQ_MAIN_MENU:
                break


# Program start:
if __name__ == '__main__':
    main()
