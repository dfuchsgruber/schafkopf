import numpy as np
from environment import Rufspiel
from replay_buffer import ReplayBufferFromFiles
import os
import json
import argparse
from tqdm import tqdm

def parse_rufspiel_lange_karte(input_dir, output_dir):
    """ 
        Parses all input files (json) to numpy compressed states into output dir. Every file input_dir/{a}.json is mapped to output_dir/{b}.npy
        if it is a Rufspiel with 8 cards dealt. Each output array contains 32 states (points at which an agent has to make a decision).
    """
    files = [fn for fn in os.listdir(input_dir) if fn.endswith('.json')]
    for fn in tqdm(files):
        with open(os.path.join(input_dir, fn)) as f:
            content = json.load(f)
        game = Rufspiel(None, None, None)
        try:
            game.from_json(content)
        except AssertionError: # The game is either not a rufspiel or doesn't have lange karte
            continue
        buffer, actions, current_trick, agent_cards, player = game.get_replay_buffer()
        np.savez(os.path.join(output_dir, fn.replace('.json', '.npz')),
            state = buffer.reshape((32, 8, 9)),
            actions = actions.reshape((32,)),
            current_trick = current_trick.reshape((32,)),
            agent_cards = agent_cards.reshape((32, 8)),
            player = player.reshape((32,)),
            rewards = game.rewards.reshape((32,)),
            color = game.color,
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                    help='Directory for input files.')
    parser.add_argument('output', type=str,
                    help='Directory for output files (npz format).')
    parser.add_argument('--rufspiel', dest='rufspiel', action='store_true', help='Parse only Sauspiele.')
    parser.add_argument('--lange-karte', dest='lange_karte', action='store_true', help='Parse only games with 8 cards dealt.')

    args = parser.parse_args()
    if args.rufspiel and args.lange_karte:
        parse_rufspiel_lange_karte(args.input, args.output)
    else:
        RuntimeError(f'Unsupported argument settings Rufspiel : {args.rufspiel}, Lange Karte : {args.lange_karte}')
        
       