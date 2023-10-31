from typing import NamedTuple
import copy
import random
import misc.utils as utils


class MoveRecord(NamedTuple):
    fen: str
    move: tuple[int, int, int]
    info: dict  # Any additional information we want to store with each move (e.g., visit counts)


class GameRecord(NamedTuple):
    players: list[str, str]
    outcome: float
    move_records: list[MoveRecord]


def score_game_records(records: list[GameRecord], agents):
    score_color = {'White': 0, 'Black': 0}
    score_agent = dict()
    for a in agents:
        score_agent[a.name()] = 0
    for record in records:
        if record.outcome == 1:
            score_color['White'] += 1
            score_agent[record.players[0]] += 1
        elif record.outcome == -1:
            score_color['Black'] += 1
            score_agent[record.players[1]] += 1
        else:
            print('Invalid score:', record.outcome)
    return score_color, score_agent


def play_a_game(game, players, disp_state, random_ply=0):
    game.setup()
    players[0].reset()
    players[1].reset()
    n = 0
    move_records = []
    while not game.is_terminal():
        if disp_state:
            print(game)
        if n < random_ply:
            move, info = random.choice(game.generate()), None
        else:
            players[n % 2].reset()
            move, info = players[n % 2].play(copy.deepcopy(game))
        assert(move != utils.NoMove)
        move_records.append(MoveRecord(game.to_fen(), move, info))
        game.make(move)
        n += 1
    outcome = -1 if game.get_to_move() == game.White else 1
    return GameRecord([players[0].name(), players[1].name()], outcome, move_records)


def play_a_match(game, agents, num_games, disp_state, random_ply=0):
    records: list[GameRecord] = []
    for _ in range(num_games):
        record = play_a_game(game, agents, disp_state, random_ply)
        records.append(record)
        record = play_a_game(game, agents[::-1], disp_state, random_ply)
        records.append(record)
    return records


def play_a_tournament(game, agents, num_games, disp_state, random_ply=0):
    tournament_records = []
    for i in range(len(agents)-1):
        for j in range(i+1, len(agents)):
            match_agents = [agents[i], agents[j]]
            match_records = play_a_match(game, match_agents, num_games, disp_state, random_ply)
            tournament_records += match_records
            print(score_game_records(match_records, match_agents))
    return tournament_records


def benchmark_agents(game, agents, num_games=50, random_ply=2, print_agents=False):
    print('Benchmarking agents ...')
    if print_agents:
        print(agents)
    game_records = play_a_match(game, agents, num_games, False, random_ply)
    score_color, score_agents = score_game_records(game_records, agents)
    print('\t', score_color, score_agents)
    return game_records
