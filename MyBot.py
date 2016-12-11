from networking import *
import os
import sys
import numpy as np

VISIBLE_DISTANCE = 4
input_dim=4*(2*VISIBLE_DISTANCE+1)*(2*VISIBLE_DISTANCE+1)

myID, gameMap = getInit()

with open(os.devnull, 'w') as sys.stderr:
    from keras.models import load_model
    model = load_model('model.h5')

model.predict(np.random.randn(1,input_dim)).shape # make sure model is compiled during init

def stack_to_input(stack, position):
    return np.take(np.take(stack,
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[0],axis=1,mode='wrap'),
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[1],axis=2,mode='wrap').flatten()

def frame_to_stack(frame):
    game_map = np.array([[(x.owner, x.production, x.strength) for x in row] for row in frame.contents])
    return np.array([(game_map[:, :, 0] == myID),  # 0 : owner is me
                      ((game_map[:, :, 0] != 0) & (game_map[:, :, 0] != myID)),  # 1 : owner is enemy
                      game_map[:, :, 1]/20,  # 2 : production
                      game_map[:, :, 2]/255,  # 3 : strength
                      ]).astype(np.float32)

sendInit('brianvanleeuwen')
while True:
    stack = frame_to_stack(getFrame())
    positions = np.transpose(np.nonzero(stack[0]))
    output = model.predict(np.array([stack_to_input(stack, p) for p in positions]))
    sendFrame([Move(Location(positions[i][1],positions[i][0]), output[i].argmax()) for i in range(len(positions))])
