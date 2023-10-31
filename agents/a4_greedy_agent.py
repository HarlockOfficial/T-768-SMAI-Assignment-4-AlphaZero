import random
import agents.agent as agent
import misc.utils as utils
import misc.model_learner as model_learner


class A4GreedyAgent(agent.Agent):

    """
    A4GreedyAgent
    """

    # ------------------------------- Static helper routines ---------------------------------

    @staticmethod
    def is_sqr_on_back_rank(g, sqr):
        return g.get_board().row(sqr) in [0, g.get_board().row(sqr) - 1]

    @staticmethod
    def is_capture_move(g, move):
        (_, _, capture) = move
        return capture != g.get_board().NoPce

    @staticmethod
    def is_winning_move(g, move):
        (_, sqr_to, capture) = move
        opponent = g.Black if g.get_to_move() == g.White else g.White
        if A4GreedyAgent.is_sqr_on_back_rank(g, sqr_to):
            # Reached opponent's back rank.
            return True
        elif A4GreedyAgent.is_capture_move(g, move) and g.get_pce_count()[opponent] == 1:
            # Capturing opponent's last piece.
            return True
        return False

    # -------------------------------- Play mode routines -----------------------------------

    def play_mode_any(self, g):
        return random.choice(g.generate())

    def play_mode_winning_captures_any(self, g):
        # To do ...
        return random.choice(g.generate())

    def play_mode_one_ply_piece_count(self, g):
        # To do ...
        # Note: only use the evaluation function to evaluate non-terminal states.
        return random.choice(g.generate())

    def play_mode_one_ply_model_value_head(self, g):
        # To do ...
        return random.choice(g.generate())

    def play_mode_model_policy_head(self, g):
        # To do ...
        # Note: only use the evaluation function to evaluate non-terminal states.
        return random.choice(g.generate())

    # -------------------------------- Methods -----------------------------------

    def __init__(self, name, params):
        super().__init__(params, f'a4_greedy_agent_{name}')

        # Process the parameters.
        self._verbose = params.get('verbose', 0)
        self._play_mode = params.get('play_mode', 0)
        self._model = params.get('model', None)
        if self._model is not None:
            self._model.eval()
        elif self._play_mode >= 3:
            print('Play mode requires a model, which is missing --- falling back to mode 0')
            self._play_mode = 0
        using_model = self._model is not None
        print(f'A4GreedyAgent {self.name()} in play_mode number {self._play_mode} (model={using_model}).')

    def play(self, game):
        assert not game.is_terminal()

        # Pick a move to play
        if self._play_mode == 4:
            move = self.play_mode_model_policy_head(game)
        elif self._play_mode == 3:
            move = self.play_mode_one_ply_model_value_head(game)
        elif self._play_mode == 2:
            move = self.play_mode_one_ply_piece_count(game)
        elif self._play_mode == 1:
            move = self.play_mode_winning_captures_any(game)
        else:
            move = self.play_mode_any(game)

        # Display extra information if requested
        if self._verbose >= 2:
            print('play mode: ', self._play_mode)
            print(game)
            for m in game.generate():
                print(game.get_board().move_to_str(m), end=' ')
            print()
            print('playing:', game.get_board().move_to_str(move))
            print()

        return move, None  # No additional information (except move to play) returned.
