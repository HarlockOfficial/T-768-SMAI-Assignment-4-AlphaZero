import copy
import math
import random
import misc.utils as utils
import agents.agent as agent
from misc import game_tensor as gt, game_tensor
import misc.model_learner as model_learner


C_uct = 1.0
C_puct = 1.0

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

    # ------------------------- Some helper routines  (you add as needed)  -------------------------------

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

    @staticmethod
    def uct(move: tuple, index: int):
        # TODO test, move might not have q
        return move[1].q[index].avg + C_uct * math.sqrt(2 * math.log(move[1].n) / move[1].q[index].n)

    @staticmethod
    def puct(move: tuple, index: int):
        # TODO test, move might not have q
        return move[1].q[index].avg + C_puct * move[1].p[index] * math.sqrt(move[1].n) / (1 + move[1].q[index].n)
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
        all_moves = game.generate(True)
        label = self.NodeLabel(all_moves)

        # If in a play_node requiring a nn, call the model and initialize the label.p[i] priors
        # with the returned policy.
        if self._play_mode >= 1:
            try:
                game_tensor.assert_initialized()
            except AssertionError:
                game_tensor.init(game)
            g_tensor = game_tensor.state_to_tensor(game)
            _, policy_tensor = self._model(g_tensor)
            policy = game_tensor.move_tensor_to_policy(all_moves, policy_tensor, game.get_to_move())
            for i in range(label.len):
                label.p[i] = policy[all_moves[i]]

        num_simulations = 0
        while True:
            if self._play_mode == 0:
                next_move_index = utils.argmax(label.moves, label.len, A4MABAgent.uct)
            elif self._play_mode >= 1:
                next_move_index = utils.argmax(label.moves, label.len, A4MABAgent.puct)
            else:
                raise ValueError('Invalid play mode')
            move = label.moves[next_move_index]
            game.make(move)
            result = A4MABAgent.playout(game)
            game.unmake(move)
            label.q[next_move_index].add(result)
            label.n += 1

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
