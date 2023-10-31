import agents.agent as agent
import random


class GreedyAgent(agent.Agent):
    """
    Plays a random:
        - winning moves if one exists,
        - otherwise a capture move,
        - otherwise other move.
    """
    def __init__(self, name, params):
        super().__init__(params, f'base_greedy_agent_{name}')

    def play(self, game):
        assert not game.is_terminal()
        capture_moves = []
        moves = game.generate(True)  # Randomize move generation order to encourage more diverse games.
        for m in moves:
            game.make(m)
            is_winning_move = game.is_terminal()
            game.retract(m)
            if is_winning_move:
                return m, None
            elif m[2] != game.get_board().NoPce:  # capture move
                capture_moves.append(m)
        return random.choice(capture_moves) if capture_moves else random.choice(moves), None
