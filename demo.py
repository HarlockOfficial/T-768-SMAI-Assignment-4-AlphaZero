import torch
import misc.game_tensor as gt
import misc.model_learner as model_learner


def print_friendly(game, policy):
    """
    Returns the policy as a print-friendly dictionary.
    :param game: A game.
    :param policy: A policy (as a move:float dictionary)
    :return: A print-friendly policy
    """
    print_policy = {}
    for k, v in policy.items():
        print_policy[game.get_board().move_to_str(k)] = round(v, 3)
    return print_policy


def demo_inference_1(game, model):
    """
    Demo of how to do inference in PyTorch.
    """
    model.eval()  # Put in evaluation mode.
    tensor_state = gt.state_to_tensor(game)
    tensor_state = tensor_state[None]  # Wrap one outer dimension (as for a batch)
    with torch.no_grad():  # Disable gradient computation (not needed for inference, only training)
        tensor_value, tensor_policy = model(tensor_state)  # Feed through network
        tensor_value = tensor_value.squeeze()  # Remove outermost 1-dimension (the batch)
        tensor_policy = tensor_policy.squeeze()
    print(game)
    print('Value = ', tensor_value.item())
    legal_moves = game.generate()
    to_move = game.get_to_move()
    policy = gt.move_tensor_to_policy(legal_moves, tensor_policy, to_move)
    print('Policy = ', print_friendly(game, policy))
    print()


def demo_inference_2(game, model):
    """
    Demo of how to do inference in PyTorch using the inference call in the provided model_learning
    """
    value, policy = model_learner.inference(game, model)
    print('Value = ', value)
    print('Policy = ', print_friendly(game, policy))
    print()
