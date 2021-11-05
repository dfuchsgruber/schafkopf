import numpy as np
import json

# 1-hot encodings for each card type, "card-stacks" can be represented as 32-dimensional tensors
card_to_idx, idx_to_card = {}, [None for _ in range(32)]
for c_idx, color in enumerate(('E', 'G', 'H', 'S')):
    for r_idx, rank in enumerate(('7', '8', '9', 'U', 'O', 'K', 'X', 'A')):
        card_to_idx[color + rank] = 8 * c_idx +  r_idx
        idx_to_card[8 * c_idx + r_idx] = color + rank

card_idx_to_value = [
    {3 : 2, 4 : 3, 5 : 4, 6 : 10, 7 : 11}.get(idx % 8, 0) for idx in range(32)
]
assert sum(card_idx_to_value) == 120

standard_trumps = {
    trump : rank for (rank, trump) in enumerate([card_to_idx[card] for card in ['H7', 'H8', 'H9', 'HX', 'HK', 'HA', 'SU', 'HU', 'GU', 'EU', 'SO', 'HO', 'GO', 'EO']])
}

def card_compare(first, second, trump_ranks=standard_trumps):
    """ Compares to cards by their int-representation. 
    
    Parameters:
    -----------
    first : int
        The first card.
    second : int
        The second card.
    trump_ranks : dict
        A dict that ranks all the trumps in the current game.
    
    Returns:
    --------
    winner : int
        Positive sign indicates first card is a winner. Negative sign indicates second card is a winner.
    """
    if first in trump_ranks and second in trump_ranks:
        return trump_ranks[first] - trump_ranks[second]
    elif first in trump_ranks:
        return np.inf
    elif second in trump_ranks:
        return -np.inf
    elif first // 8 != second // 8:
        return np.inf
    else: # No trumps, both of the same color
        return first % 8 - second % 8

def deal_cards():
    deck = np.random.permutation(np.arange(32))
    return deck[:8], deck[8:16], deck[16:24], deck[24:32]

def print_cards(player_card_sets):
    for i, cards in enumerate(player_card_sets):
        print(f'Player {i} cards: {[idx_to_card[card] for card in cards]}')

class Game:
    """ Environment for a Schafkopf-Game """

    def __init__(self, trump_ranks, agents):
        """ Initializes a new game. Player 0 always starts. 
        
        Parameters:
        -----------
        trump_ranks : dict
            A ranking of all trumps in the game, indexed by their number.
        agents : list of Agent
        """
        self.agent_cards = [[cards] for cards in deal_cards()]
        self.card_history = []
        self.agent_history = []
        self.trick_history = []
        self.trump_ranks = trump_ranks

        # Initialize the agents
        self.agents = agents
        self.players = None # Player and his partner
        self.player = None # Player

        # Initialize rewards
        self.rewards = np.zeros((8, 4))

    def _is_player(self, agent_idx):
        return agent_idx in self.players

    def _get_eligible_cards(self, cards, turn_history):
        if len(turn_history) == 0: # Can play any card
            eligible_cards = cards
        elif turn_history[0] in self.trump_ranks: # Must play a trump
            eligible_cards = [card for card in cards if card in self.trump_ranks]
        else:
            eligible_cards = [card for card in cards if (card // 8 == turn_history[0] // 8) and card not in self.trump_ranks]
        if len(eligible_cards) == 0: # Can play any card
            eligible_cards = cards
        return eligible_cards

    def _play_card(self, agent_idx, card):
        self.agent_history.append(agent_idx)
        self.card_history.append(card)

    def _end_turn(self, i, verbose=True):
        # Determine the winner of the round
        winner, best_card = self.agent_history[4 * i], self.card_history[4 * i]
        for j in range(1, 4):
            if card_compare(best_card, self.card_history[4 * i + j], trump_ranks=self.trump_ranks) < 0: # New winner
                winner, best_card = self.agent_history[4 * i + j], self.card_history[4 * i + j]
        return winner

    def _calculate_rewards(self, i, verbose=True):
        winner = self.trick_history[i]
        value = sum(card_idx_to_value[card] for card in self.card_history[4 * i : 4 * (i + 1)])
        if verbose:
            print(f'Player {winner} wins {[idx_to_card[card] for card in self.card_history[4 * i : 4 * (i + 1)]]} with {idx_to_card[best_card]}')
        for agent_idx in range(4):
            if self._is_player(agent_idx) == self._is_player(winner):
                self.rewards[i, agent_idx] = value
                if verbose:
                    print(f'Player {agent_idx} gets a reward of {value}')

    def play(self, verbose=True):
        if verbose:
            for agent_idx in range(4):
                print(f'Player {agent_idx} has cards:  {[idx_to_card[card] for card in self.agent_cards[agent_idx][0]]}')
        first = 0
        for i in range(8):
            if verbose:
                print(f'Round {i} starts. Player {first} goes first.')
            for j in range(4): # Each agent takes a turn
                agent_idx = (first + j) % 4
                turn_history = self.card_history[4 * i : ]
                eligible_cards = self._get_eligible_cards(self.agent_cards[agent_idx][i], turn_history)
                if verbose:
                    print(f'\tPlayer {agent_idx} has eligible cards:  {[idx_to_card[card] for card in eligible_cards]}')
                action = self.agents[agent_idx].act(eligible_cards, self.card_history, self.agent_history, turn_history)
                self.agent_cards[agent_idx].append(np.array([card for card in self.agent_cards[agent_idx][i] if card != action]))
                self._play_card(agent_idx, action)
                if verbose:
                    print(f'\tPlayer {agent_idx} plays {idx_to_card[action]}. Remaining cards: {[idx_to_card[card] for card in self.agent_cards[agent_idx][i + 1]]}')
            first = self._end_turn(i, verbose=verbose)
            self.trick_history.append(first) # This player won the trick
            self._calculate_rewards(i, verbose=verbose)
            if verbose:
                print(f'Current Rewards: {self.rewards.sum(0)}')

    def print_game(self):
        for agent_idx in range(4):
            print(f'Initial cards for agent {agent_idx}: {[idx_to_card[card] for card in self.agent_cards[agent_idx][0]]}')
        for i in range(8):
            print(f'Turn {i} starts. Player {self.agent_history[4 * i]} goes first.')
            for j in range(4):
                print(f'Player {self.agent_history[4 * i + j]} plays {idx_to_card[self.card_history[4 * i + j]]}. Hand was {[idx_to_card[card] for card in self.agent_cards[j][i]]}')
            print(f'Player {self.trick_history[i]} wins the trick: Rewards {self.rewards[i]}')
        
        print(f'Final result : {self.rewards.sum(0)}')

    def from_json(self, json):
        self.card_history = []
        self.trick_history = []
        self.agent_history = []
        self.agent_cards = [[] for _ in range(4)]
        for record in json:
            if record['type'] == 'playedACard':
                card, agent = card_to_idx[record['cardID']], record['position']
                self.card_history.append(card)
                self.agent_history.append(agent)
                self.agent_cards[agent].append([])
                for previous_cards in self.agent_cards[agent]:
                    previous_cards.append(card)
            elif record['type'] == 'wonTheTrick':
                self.trick_history.append(record['position'])
        # At the end no player has cards
        for rounds in self.agent_cards:
            rounds.append([])

    def get_replay_buffer(self, player_shifting=True):
        """ Returns a replay buffer. 
        
        Parameters:
        -----------
        player_shifting : bool
            If True, the players are shifted such that for each sequence, the player that is to act is identified as p0.
            Other players are then referred to by relative positions.

        Returns:
        --------
        buffer : ndarray, shape [4, 8, 8, 9]
            A set of states in the buffer. Each state is a sequence of turns. 
            That is, each state is identified by a [8, 9] array. The first axis is the trick. For the second axis,
            in the first entries we have [card_i, player_i, card_j, player_j, ...]. The 9-th component is the player
            that won the trick.
            In total, a game yields 8 "state-action" pairs for each player, which corresponds to the first two axis (4 and 8).
        action : ndarray, shape [4, 8]
            Which action a player took in a certain state.
        current_trick : ndarray, shape [4, 8]
            The idx of the trick that is not yet complete and requires an action for each state.
        agent_cards : ndarray, shape [4, 8, 8]
            All the cards that a player has in a given state. Cards that are already played are represented by 0.
        player : ndarray, shape [4, 8]
            Who is the player of the game (from perspective of player i).
        """
        # Zero will be padding
        agent_history = np.array(self.agent_history)
        card_history = np.array(self.card_history)
        trick_history = np.array(self.trick_history)

        trick_buffer = np.ones((4, 8, 8, 1)) * np.nan
        actions = np.zeros((4, 8), dtype=np.int)
        buffer = np.ones((4, 8, 32, 2)) * np.nan
        agent_cards = np.zeros((4, 8, 8), dtype=np.int) - 1
        player = np.ones((4, 8), dtype=np.int) * self.player

        for agent in range(4):
            for i, turn in enumerate(np.where(agent_history == agent)[0]):
                buffer[agent, i, : turn, 1] = card_history[ : turn]
                buffer[agent, i, : turn, 0] = agent_history[ : turn]
                trick_buffer[agent, i, : turn // 4, 0] = trick_history[ : turn // 4]
                actions[agent, i] = card_history[turn]
                agent_cards[agent, i, : 8 - i] = np.array(self.agent_cards[agent][i])
        buffer = buffer.reshape((4, 8, 8, 8)) # agent, sequence, trick, card in trick
        buffer = np.concatenate((buffer, trick_buffer), axis=3)

        if player_shifting:
            for agent in range(4): # It is always player 0 who needs to do an action.
                buffer[agent, :, :, ::2] += 4 - agent
                player[agent, :] += 4 - agent
                np.seterr(invalid='ignore')
                buffer[agent, :, :, ::2] %= 4
                player[agent, :] %= 4

        buffer[:, :, :, ::2] += 32 # Player Tokens are offset by 32 
        buffer += 1 # 0 corresponds to padding
        actions += 1
        agent_cards += 1

        return np.nan_to_num(buffer, 0).astype(np.int), actions, np.arange(8).reshape((1, -1)).repeat(4, axis=0), np.array(agent_cards), player

class Rufspiel(Game):
    """ Rufspiel. """

    def __init__(self, agents, color, player):
        super().__init__(standard_trumps, agents)
        self.color = color
        self.partner_ace_searched = False
        self.player = player # The one who called the game
        if color is not None and player is not None:
            self._determine_partner_ace()
            self._determine_players()
        
    def _determine_partner_ace(self):
        self.partner_ace = {0 : card_to_idx['EA'], 1 : card_to_idx['GA'], 3 : card_to_idx['SA']}[self.color]
    
    def _determine_players(self):
        self.players = set(agent_idx for agent_idx in range(4) if self.partner_ace in self.agent_cards[agent_idx][0])
        self.players.add(self.player)
        assert len(self.players) == 2

    def _get_eligible_cards(self, cards, turn_history):
        eligible_cards = super()._get_eligible_cards(cards, turn_history)
        if self.partner_ace in eligible_cards and not self.partner_ace_searched:
            # print(f'\t\tPlayer has partner ace in hand {[idx_to_card[card] for card in cards]}.\n\t\tHistory is {[idx_to_card[card] for card in turn_history]}')
            # "Weglaufen" : If you have 3 other cards of the partner ace's color you may play any of them as long as you are
            # the first player to move
            if len(turn_history) == 0:
                if len([card for card in eligible_cards if card // 8 == self.color and card not in self.trump_ranks]) >= 4:
                    return eligible_cards
                else: # You can not play any other card of this color but the partner ace
                    return [card for card in eligible_cards if card == self.partner_ace or card // 8 != self.color or card in self.trump_ranks]
            else:
                if turn_history[0] // 8 == self.color and turn_history[0] not in self.trump_ranks: # Must play the partner ace
                    return [self.partner_ace]
                elif len(cards) > 1: # You can not play the partner ace
                    return [card for card in eligible_cards if card != self.partner_ace]
        return eligible_cards

    def _end_turn(self, i, verbose=True):
        winner = super()._end_turn(i, verbose=verbose)
        # If the partner ace's color was played, there is no obligation to play the partner ace anymore
        first_card_in_turn = self.card_history[4 * i]
        if first_card_in_turn // 8 == self.color and first_card_in_turn not in self.trump_ranks:
            self.partner_ace_searched = True
            if verbose:
                print(f'The partner ace was searched. The game is no "aufgeklaert". Card that searched for it was {idx_to_card[first_card_in_turn]}')
        return winner

    def from_json(self, json):
        super().from_json(json)
        self.color = None
        for record in json:
            if record['type'] == 'youGotCards':
                assert len(record['cards']) % 4 == 0
            if record['type'] == 'playsTheGame':
                assert record['gameType'] == 1
                self.color = {'E' : 0, 'G' : 1, 'S' : 3}[record['suit']]
                self.player = record['position']
                break
        assert self.color is not None
        self.partner_ace_searched = False
        self._determine_partner_ace()
        self._determine_players()
        for i in range(8):
            self._calculate_rewards(i, verbose=False)

    def print_game(self):
        print(f'Player {self.player} searches for {idx_to_card[self.partner_ace]}.')
        print(f'Players are {self.players}')
        super().print_game()

class Agent:
    """ An agent for a Schafkopf-Game. """

    def __init__(self):
        pass

    def act(self, cards, card_history, player_history, turn_history, trump_ranks):
        raise NotImplementedError

class RandomAgent(Agent):

    def __init__(self):
        super().__init__()

    def act(self, eligible_cards, card_history, player_history, turn_history):
        return np.random.choice(eligible_cards)

def print_replace_buffer(replay_buffer, actions, current_trick):        
    def e_to_str(e):
        if e == 0:
            return '  '
        elif e <= 32:
            return idx_to_card[e - 1]
        else:
            return f'p{e-33}'
    for agent, sequences in enumerate(replay_buffer):
        for sequence_idx, sequence in enumerate(sequences):
            print(f'Sequence {sequence_idx} for player {agent} starts:')
            for trick_idx, trick in enumerate(sequence):
                print(f'\tTrick {trick_idx}, f{[e_to_str(e) for e in trick]}' + f'{" (current) " if trick_idx == current_trick[agent, sequence_idx] else ""}')
            print(f'Action taken: {e_to_str(actions[agent, sequence_idx])}')




"""
agents = [RandomAgent() for _ in range(4)]
#game = Game(standard_trumps, agents)
game = Rufspiel(agents, None, None)
with open('games/game_1000039442.json') as f:
    game.from_json(json.load(f))
# game.play(verbose=True)
# game.print_game()
buffer, actions, current_trick = game.get_replay_buffer()
print_replace_buffer(buffer, actions, current_trick)
"""