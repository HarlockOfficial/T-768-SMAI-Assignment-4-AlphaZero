#
# Please install the following packages:
#    numpy
#    torch (version 2.1.0)
#
import torch
import demo
import game.breakthrough as breakthrough
import agents.a4_greedy_agent as a4_greedy_agent
import agents.a4_mab_agent as a4_mab_agent
from misc import game_tensor as gt, my_model
import misc.play as play


# Create a Breakthrough 6x& game.
# Initialise the game_tensor package for that game. Need to do before any use.
breakthrough_game = breakthrough.Breakthrough(6, 6)
gt.init(breakthrough_game)


# Load the NN (a small ResNets)
model_vp = my_model.MyResNet()
model_vp.load_state_dict(torch.load('models/model_vp.pt'))
my_model = model_vp

#
# Example of running a tournament amongst the A4Greedy agents.
#
params = {'verbose': 0, 'model': my_model}
agents = [a4_greedy_agent.A4GreedyAgent('pm_0', params | {'play_mode': 0}),
          a4_greedy_agent.A4GreedyAgent('pm_1', params | {'play_mode': 1}),
          a4_greedy_agent.A4GreedyAgent('pm_2', params | {'play_mode': 2}),
          a4_greedy_agent.A4GreedyAgent('pm_3', params | {'play_mode': 3}),
          a4_greedy_agent.A4GreedyAgent('pm_4', params | {'play_mode': 4})]
result = play.play_a_tournament(breakthrough_game, agents, 500, False, random_ply=4)
print('Total:')
print(play.score_game_records(result, agents))


#
# Example of benchmarking the two MAB agents.
#
params = {'verbose': 0, 'abort': 'iterations', 'number': 500, 'model': my_model}
agents = [a4_mab_agent.A4MABAgent('pm_0', params | {'play_mode': 0}),
          a4_mab_agent.A4MABAgent('pm_1', params | {'play_mode': 1})]
#play.benchmark_agents(breakthrough_game, agents, num_games=50, random_ply=4)


#
# Example of how to do inference with models in PyTorch.
#
#demo.demo_inference_1(breakthrough_game, my_model)
#demo.demo_inference_2(breakthrough_game, my_model)
result = play.play_a_tournament(breakthrough_game, agents, 500, False, random_ply=4)
print('Total:')
print(play.score_game_records(result, agents))
