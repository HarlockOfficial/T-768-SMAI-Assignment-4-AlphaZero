import agents.agent as agent
import random


class RandomAgent(agent.Agent):

    def __init__(self, name, params):
        # random.seed(42)
        super().__init__(params, f'random_agent_{name}')

    def play(self, game):
        # Plays a winning moves if one exists, otherwise a random move.
        winning_moves = []
        moves = game.generate()
        for m in moves:
            game.make(m)
            if game.is_terminal():
                winning_moves.append(m)
            game.retract(m)
        return random.choice(winning_moves) if winning_moves else random.choice(moves), None