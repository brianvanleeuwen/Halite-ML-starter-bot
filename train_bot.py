import os
import sys

import numpy as np
import json

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

REPLAY_FOLDER = sys.argv[1]
training_input = None

VISIBLE_DISTANCE = 5
input_dim=4*(2*VISIBLE_DISTANCE+1)*(2*VISIBLE_DISTANCE+1)

model = Sequential([Dense(512, activation='relu', input_dim=input_dim),
                    Dense(512, activation='relu'),
                    Dense(512, activation='relu'),
                    Dense(5, activation='softmax')])
model.compile('nadam','categorical_crossentropy', metrics=['accuracy'])

def stack_to_input(stack, position):
    return np.take(np.take(stack,
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)-position[0],axis=1,mode='wrap'),
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)-position[1],axis=2,mode='wrap').flatten()

for replay_name in os.listdir(REPLAY_FOLDER):
    if replay_name[-4:]!='.hlt':continue
    print('Loading {}'.format(replay_name))
    replay = json.load(open('{}/{}'.format(REPLAY_FOLDER,replay_name)))

    frames=np.array(replay['frames'])
    player=frames[:,:,:,0]
    players,counts = np.unique(player[-1],return_counts=True)
    target_id = players[counts.argmax()]
    if target_id == 0: continue

    prod = np.repeat(np.array(replay['productions'])[np.newaxis],replay['num_frames'],axis=0)
    strength = frames[:,:,:,1]

    moves = (np.arange(5) == np.array(replay['moves'])[:,:,:,None]).astype(int)
    stacks = np.array([player==target_id,(player!=target_id) & (player!=0),prod/20,strength/255])
    stacks = stacks.transpose(1,0,2,3)[:len(moves)].astype(np.float32)

    position_indices = stacks[:,0].nonzero()
    sampling_rate = 1/stacks[:,0].mean(axis=(1,2))[position_indices[0]]
    sampling_rate /= sampling_rate.sum()
    sample_indices = np.transpose(position_indices)[np.random.choice(np.arange(len(sampling_rate)),2048,p=sampling_rate)]

    replay_input = np.array([stack_to_input(stacks[i],[j,k]) for i,j,k in sample_indices])
    replay_target = moves[tuple(sample_indices.T)]
        
    if training_input is None:
        training_input = replay_input
        training_target = replay_target
    else:            
        training_input = np.append(training_input,replay_input,axis=0)
        training_target = np.append(training_target,replay_target,axis=0)
    
indices = np.arange(len(training_input))
np.random.shuffle(indices)
model.fit(training_input[indices],training_target[indices],validation_split=0.2,
          callbacks=[EarlyStopping()], batch_size=1024, nb_epoch=100)

model.save('model.h5')
