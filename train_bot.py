"""Trains the bot's model from a replay cache

Usage:

    `python train_bot.py path/to/replay/cache/`

"""
import datetime
import json
import os
import sys

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, load_model

VISIBLE_DISTANCE = 4
INPUT_DIM = 4 * (2 * VISIBLE_DISTANCE + 1) * (2 * VISIBLE_DISTANCE + 1)

replay_folder = sys.argv[1]
training_input = []
training_target = []


def stack_to_input(stack, position):
    """Crops out the input for a given position from the feature stack"""
    return np.take(np.take(stack,
                   np.arange(-VISIBLE_DISTANCE, VISIBLE_DISTANCE + 1) + position[0], axis=1, mode='wrap'),
                   np.arange(-VISIBLE_DISTANCE, VISIBLE_DISTANCE + 1) + position[1], axis=2, mode='wrap').flatten()


if __name__ == '__main__':

    # Defining the model
    np.random.seed(0)  # for reproducible weight initialization
    model = Sequential([Dense(512, input_dim=INPUT_DIM),
                        LeakyReLU(),
                        Dense(512),
                        LeakyReLU(),
                        Dense(512),
                        LeakyReLU(),
                        Dense(5, activation='softmax')])
    model.compile('nadam', 'categorical_crossentropy', metrics=['accuracy'])

    num_replays = len(os.listdir(replay_folder))

    # Loop over replays, parse training data
    for index, replay_name in enumerate(os.listdir(replay_folder)):

        if replay_name[-4:] != '.hlt':
            continue

        print('Loading {} ({}/{})'.format(replay_name, index, num_replays))
        replay = json.load(open('{}/{}'.format(replay_folder, replay_name)))

        # Parse frames and player data
        frames = np.array(replay['frames'])
        player = frames[:, :, :, 0]
        players, counts = np.unique(player[-1], return_counts=True)
        target_id = players[counts.argmax()]

        if target_id == 0:
            continue

        # Build features stacks from game frames
        prod = np.repeat(np.array(replay['productions'])[np.newaxis], replay['num_frames'], axis=0)
        strength = frames[:, :, :, 1]
        moves = (np.arange(5) == np.array(replay['moves'])[:, :, :, None]).astype(int)[:128]
        stacks = np.array([player == target_id, (player != target_id) & (player != 0), prod / 20, strength / 255])
        stacks = stacks.transpose((1, 0, 2, 3))[:len(moves)].astype(np.float32)

        # Apply sampling scheme
        position_indices = stacks[:, 0].nonzero()
        sampling_rate = 1 / stacks[:, 0].mean(axis=(1, 2))[position_indices[0]]
        sampling_rate *= moves[position_indices].dot(np.array([1, 10, 10, 10, 10]))  # weight moves 10 times higher
        sampling_rate /= sampling_rate.sum()
        sample_indices = np.transpose(position_indices)[np.random.choice(np.arange(len(sampling_rate)),
                                                                         min(len(sampling_rate), 2048), p=sampling_rate,
                                                                         replace=False)]

        # Build model input and target
        replay_input = np.array([stack_to_input(stacks[i], [j, k]) for i, j, k in sample_indices])
        replay_target = moves[tuple(sample_indices.T)]

        training_input.append(replay_input.astype(np.float32))
        training_target.append(replay_target.astype(np.float32))

    training_input = np.concatenate(training_input, axis=0)
    training_target = np.concatenate(training_target, axis=0)

    # Shuffle training samples
    indices = np.arange(len(training_input))
    np.random.shuffle(indices)
    training_input = training_input[indices]
    training_target = training_target[indices]

    # Train model
    model.fit(training_input, training_target, validation_split=0.2,
              callbacks=[EarlyStopping(patience=10),
                         ModelCheckpoint('model.h5', verbose=1, save_best_only=True),
                         TensorBoard(log_dir='./logs/' + datetime.datetime.now().strftime('%Y.%m.%d %H.%M'))],
              batch_size=1024, nb_epoch=1000)

    # Evaluate model accuracy
    model = load_model('model.h5')  # overwrites weights to use the checkpointed set that MyBot.py uses
    still_mask = training_target[:, 0].astype(bool)
    print('STILL accuracy:', model.evaluate(training_input[still_mask], training_target[still_mask], verbose=0)[1])
    print('MOVE accuracy:', model.evaluate(training_input[~still_mask], training_target[~still_mask], verbose=0)[1])
