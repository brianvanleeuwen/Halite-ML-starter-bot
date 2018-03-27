"""Runs the main game loop

The halite game engine requires that the module file is "MyBot.py".  This script loads the trained model,
produces moves using that model, and communicates with the game engine via stdin/stdout.

"""
import os
import sys

import numpy as np

import networking
from hlt import Move, Location

VISIBLE_DISTANCE = 4
INPUT_DIM = 4 * (2 * VISIBLE_DISTANCE + 1) * (2 * VISIBLE_DISTANCE + 1)

myID, init_map = networking.get_init()

with open(os.devnull, 'w') as sys.stderr:  # unfortunately, this hack is necessary to prevent a keras import side effect
    import keras.models
    model = keras.models.load_model('model.h5')
    model.predict(np.random.randn(1, INPUT_DIM))  # make sure model is compiled during init by running it


def stack_to_input(stack, position):
    """Crops out the input for a given position from the feature stack"""
    return np.take(np.take(stack,
                   np.arange(-VISIBLE_DISTANCE, VISIBLE_DISTANCE + 1) + position[0], axis=1, mode='wrap'),
                   np.arange(-VISIBLE_DISTANCE, VISIBLE_DISTANCE + 1) + position[1], axis=2, mode='wrap').flatten()


def frame_to_stack(frame):
    """Converts a game frame into a stack of features"""
    game_map = np.array([[(x.owner, x.production, x.strength) for x in row] for row in frame.contents])
    return np.array([(game_map[:, :, 0] == myID),  # 0 : owner is me
                     ((game_map[:, :, 0] != 0) & (game_map[:, :, 0] != myID)),  # 1 : owner is enemy
                     game_map[:, :, 1] / 20,  # 2 : production
                     game_map[:, :, 2] / 255,  # 3 : strength
                     ]).astype(np.float32)


networking.send_init('brianvanleeuwen')

# Main game loop
while True:
    current_stack = frame_to_stack(networking.get_frame())
    positions = np.transpose(np.nonzero(current_stack[0]))
    output = model.predict(np.array([stack_to_input(current_stack, p) for p in positions]))
    networking.send_frame(
        [Move(Location(positions[i][1], positions[i][0]), output[i].argmax()) for i in range(len(positions))])
