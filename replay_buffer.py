import numpy as np
from environment import Rufspiel
import os
import json

class ReplayBuffer:
    """ Base class for replay buffers. """

    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

class ReplayBufferFromFiles:
    """ Class for experiences from already played games. """

    def __init__(self, filepaths='games', shuffle=True, seed=None):
        super().__init__()
        self.files = os.listdir(filepaths)
        self.dir = filepaths
        if seed is not None:
            np.random.seed(seed)
        if shuffle:
            np.random.shuffle(self.files)
        self.idx = 0
    
    def next(self,):
        """ Yields 32 new decision states. 0-7, 8-15, 16-23, 24-31 form a sequence for one player.
        
        Returns:
        --------
        states : ndarray, shape [32, 8, 9]
            The state of each game to decide upon. A state is represented by cards played so far as a [8, 9] array.
            The 8 axis corresponds to each possible trick that was done before.
            The 9 axis corresponds to cards and players that acted in that trick: [player_i, card_i, player_i+1, card_i+1, ... winner_of_trick]
        actions : ndarray, shape [32]
            Which action (card to play) the player took in a state.
        current_trick : ndarray, shape [32]
            The how many-th trick this state resembles. For example, if there have been 5 tricks preceeding state current_trick[i] = 6.
        agent_cards : ndarray, shape [32, 8]
            The cards the the acting player has in hands for each state. Not available cards are padded with 0 (because they were played already...)
        reward : ndarray, shape [32]
            The reward the player had after that trick.
        color : int
            The color of the game (for all states in the replay buffer).
        player : ndarray, shape [32]
            Who is the player of this game (from the perspective of the player that has to take action.)
        """
        while True:
            with open(os.path.join(self.dir, self.files[self.idx])) as f:
                content = json.load(f)
            self.idx += 1
            try:
                game = Rufspiel(None, None, None)
                game.from_json(content)
            except AssertionError as e:
                continue

            buffer, actions, current_trick, agent_cards, player = game.get_replay_buffer()
            buffer = buffer.reshape([buffer.shape[0] * buffer.shape[1]] + list(buffer.shape[2:]))
            actions = actions.reshape((-1,))
            current_trick = current_trick.reshape((-1,))
            agent_cards = agent_cards.reshape((-1, 8))
            return buffer, actions, current_trick, agent_cards, game.rewards.reshape((-1,)), game.color, player.reshape((-1, ))



if __name__ == '__main__':
    buf = ReplayBufferFromFiles(seed=1337)
    states, actions, current_trick, agent_cards, rewards, color, player = buf.next()
    # sates is 4 x 8 x 8 x 9

    print(player)

    print(states[22], current_trick[22], actions[22], agent_cards[22], rewards[22], color, player[22])
    print(states[23], current_trick[23], actions[23], agent_cards[23], rewards[23], color, player[22])


    print(states[30], current_trick[30], actions[30], agent_cards[30], rewards[30], color, player[31])
    print(states[31], current_trick[31], actions[31], agent_cards[31], rewards[31], color, player[31])


    print(states.shape, states.dtype)
    print(actions.shape, actions.dtype)
    print(current_trick.shape, current_trick.dtype)
    print(rewards.shape, rewards.dtype)