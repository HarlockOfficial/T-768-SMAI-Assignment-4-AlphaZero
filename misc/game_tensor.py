#
# Contains various helper routines to help with translating game states and moves
# into tensors and back.
#
import torch
import game.breakthrough as bt
import misc.model_learner as model_learner


GameTensor_idx_sqr = None  # Precomputed move indices.
GameTensor_game_board = None    # Use to keep track of the board dimensions.


def init(game):
    """
    Call once to initialize before using any other method in here.
    :param game: The game to play
    :return: None
    """
    global GameTensor_game_board, GameTensor_idx_sqr
    GameTensor_game_board = board = game.get_board()
    GameTensor_idx_sqr = [0] * (board.cols() * (board.rows() - 1))
    idx = 0
    for r in range(board.rows() - 1):
        for c in range(board.cols()):
            GameTensor_idx_sqr[board.sqr(c, r)] = idx
            idx += 2 if c == 0 or c == board.cols() - 1 else 3


def assert_initialized():
    assert GameTensor_idx_sqr is not None, "Must call GameTensor.init(game) once before any use."


def state_to_tensor(game):
    """
    Represents the current game state with 3 channels (white, black, side_to_move).
    :param game: The current game state.
    :return: A tensor
    """
    board = game.get_board()
    tensor = torch.zeros([3, board.cols(), board.rows()])
    for sq in board.get_pce_locations(bt.Breakthrough.White):
        tensor[0][board.col(sq)][board.row(sq)] = 1.0
    for sq in board.get_pce_locations(bt.Breakthrough.Black):
        tensor[1][board.col(sq)][board.row(sq)] = 1.0
    if game.get_to_move() == bt.Breakthrough.Black:
        for c in range(board.cols()):
            for r in range(board.rows()):
                tensor[2][c][r] = 1.0
    return tensor


def state_to_mirror_tensor(game):
    """
    Represents the current game state with 2 channels (white, black).
    States with black to more are mirrored (to be from white perspective)
    :param game: The current game state.
    :return: A tensor
    """
    board = game.get_board()
    tensor = torch.zeros([board.cols(), board.rows()])
    if game.get_to_move() == bt.Breakthrough.White:
        for sq in board.get_pce_locations(bt.Breakthrough.White):
            tensor[0][board.col(sq)][board.row(sq)] = 1.0
        for sq in board.get_pce_locations(bt.Breakthrough.Black):
            tensor[1][board.col(sq)][board.row(sq)] = 1.0
    else:
        for sq in board.get_pce_locations(bt.Breakthrough.White):
            sq = mirror_sqr(sq)
            tensor[1][board.col(sq)][board.row(sq)] = 1.0
        for sq in board.get_pce_locations(bt.Breakthrough.Black):
            sq = mirror_sqr(sq)
            tensor[0][board.col(sq)][board.row(sq)] = 1.0
    return tensor


def visits_to_policy(visits):
    """
    Turn move visit counts into a policy dictionary.
    :param visits: dictionary (move:n) of visit counts of given moves.
    :return: Policy dictionary
    """
    n = 0
    for v in visits.values():
        n += v
    policy = {}
    if n > 0:
        for k, v in visits.items():
            policy[k] = v / n
    return policy


def policy_to_tensor(policy, player):
    """
    Turn a move policy into a tensor (representing all possible board moves).
    :param policy: dictionary (move:float) of the probabilities of available moves.
    :param player: Player to move (used to mirror moves, if black to move)
    :return: A tensor.
    """
    assert player == bt.Breakthrough.White or player == bt.Breakthrough.Black
    tensor = torch.zeros([number_of_idx_moves()])
    for k, v in policy.items():
        idx = move_to_idx(k, player)
        tensor[idx] = v
    return tensor


def move_tensor_to_policy(moves, tensor, player):
    """
    Turn a move tensor into a policy (e.g., to use as PUCT priors)
    :param moves: The state's moves (list)
    :param tensor: A tensor (representing all possible board moves)
    :param player: Player to move (used to mirror moves, if black to move)
    :return: A policy dictionary for the moves in <moves>
    """
    assert_initialized()
    assert player == bt.Breakthrough.White or player == bt.Breakthrough.Black
    policy = dict()
    policy_sum = 0.0
    for move in moves:
        v = tensor[move_to_idx(move, player)].item()
        policy[move] = v
        policy_sum += v
    if policy_sum > 0.01:
        for m in moves:
            policy[m] /= policy_sum
    return policy


def number_of_idx_moves():
    """
    Returns the total number of all possible moves for a board of a given size (e.g. for 6x6 board this is 80).
    :return:
    """
    assert_initialized()
    return GameTensor_idx_sqr[-1] + 2


def move_to_idx(move, player):
    """
    Returns the index a move has in a given move_tensor
    :param move: The move
    :param player: THe side to move (used for mirroring if black to move).
    :return: index
    """
    assert_initialized()
    sqr_from, sqr_to, _ = move
    if player == bt.Breakthrough.Black:
        sqr_from, sqr_to = mirror_sqr(sqr_from), mirror_sqr(sqr_to)
    idx = GameTensor_idx_sqr[sqr_from]
    if GameTensor_game_board.col(sqr_to) == GameTensor_game_board.col(sqr_from):
        idx += 1
    elif GameTensor_game_board.col(sqr_to) > GameTensor_game_board.col(sqr_from):
        idx += 2
    return idx


def moves_to_idx(moves, player):
    """
    Returns the index of all moves in the moves list.
    :param moves: The move list
    :param player: THe side to move (used for mirroring if black to move).
    :return: A list on indices for the moves in the move list
    """
    assert_initialized()
    index = []
    for move in moves:
        index.append(move_to_idx(move, player))
    return index


def mirror_sqr(sqr):
    """
    Returns a mirror square (e.g. on a 6x6 board a1 would become h8)
    :param sqr: Square to mirror
    :return: Square
    """
    assert_initialized()
    col = GameTensor_game_board.cols() - 1 - GameTensor_game_board.col(sqr)
    row = GameTensor_game_board.rows() - 1 - GameTensor_game_board.row(sqr)
    return GameTensor_game_board.sqr(col, row)


def mirror_move(move):
    """
    Returns a mirrored move
    :param move: The move to mirror
    :return: A mirrored move
    """
    assert_initialized()
    sqr_from, sqr_to, cap_pce = move
    return mirror_sqr(sqr_from), mirror_sqr(sqr_to), cap_pce
