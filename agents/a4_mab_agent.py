import copy
import math
import random
import misc.utils as utils
import agents.agent as agent
from misc import game_tensor as gt
import misc.model_learner as model_learner


class A4MABAgent(agent.Agent):
    """
    Multi-Arm-Bandit agent.
    """

    class NodeLabel:

        def __init__(self, moves):
            self.moves = moves
            self.len = len(self.moves)
            self.n = 0
            self.q = [utils.Avg() for _ in range(self.len)]
            self.p = [1.0 for _ in range(self.len)]
            return

    # ------------------------- Some helper routines  (you add as needed(  -------------------------------

    @staticmethod
    def visits(q, i):
        return q[i].n

    # Use this naive playout policy for your experiments.
    # It returns the game outcome from the perspective of the playing having the in the game state where called.
    @staticmethod
    def playout(game):
        g = copy.deepcopy(game)
        player = g.get_to_move()
        while not g.is_terminal():
            move = random.choice(g.generate())
            g.make(move)
        return -1.0 if g.get_to_move() == player else 1.0

    # -------------- Methods -----------------------

    def __init__(self, name, params):
        super().__init__(params, f'mab_agent_{name}')

        # Time management ...
        abort_type = self._params.get('abort')
        if abort_type == 'iterations':
            self._abort_checker = utils.Aborter(utils.AbortType.ITERATIONS, self._params.get('number'))
        elif abort_type == 'time':
            self._abort_checker = utils.Aborter(utils.AbortType.TIME, self._params.get('number'))
        else:
            self._abort_checker = utils.Aborter(utils.AbortType.ITERATIONS, 0)  # 0 disables aborter

        # Play mode.
        self._play_mode = params.get('play_mode', 0)

        # Model
        self._model = params.get('model', None)
        if self._model is not None:
            self._model.eval()
        elif self._play_mode >= 1:
            print('Play mode requires a model, which is missing --- falling back to mode 0')
            self._play_mode = 0
        using_model = self._model is not None
        print(f'A4MABAgent {self.name()} in play_mode number {self._play_mode} (model={using_model}).')

    def play(self, game):
        assert not game.is_terminal()
        self._abort_checker.reset()

        # Use to keep track of root statistics.
        label = self.NodeLabel(game.generate(True))

        # If in a play_node requiring a nn, call the model and initialize the label.p[i] priors
        # with the returned policy.
        # TO DO:
        ...

        num_simulations = 0
        while True:

            # TO DO:
            # Pick a move using UCT/PUCT, simulate it (using the playout(...) routine,
            # and update necessary logistics.
            ...

            # Update simulation count, and check if search resources up.
            num_simulations += 1
            if self._abort_checker.do_abort(num_simulations, 0):
                break

        # Pick and return the move most often visited.
        max_i = utils.argmax(label.q, len(label.q), self.visits)
        best_move = label.moves[max_i]

        # Additionally collect and return information about how often each move was visited.
        root_info = dict()
        for i in range(len(label.moves)):
            root_info[label.moves[i]] = self.visits(label.q, i)

        # Print out information if requested (e.g., for debugging purposes)
        if self._params.get('verbose', 0) > 0:
            print('play mode: ', self._play_mode)
            print(game)
            for m in game.generate():
                print(game.get_board().move_to_str(m), end=' ')
            print()
            print('playing:', game.get_board().move_to_str(best_move))
            print()
            print(num_simulations, max_i, root_info)

        return best_move, root_info
