from typing import NamedTuple
import random

import torch
from torch.utils.data import Dataset

from misc import game_tensor as gt
import misc.play as play


class LabeledDataRecord(NamedTuple):
    fen: str
    z: float
    policy: dict[tuple[int, int, int], float]


class LabeledDataRecDataset(Dataset):

    def __init__(self, g, records: list[LabeledDataRecord]):
        self._data = []
        for rec in records:
            g.setup_fen(rec.fen)
            self._data.append(
                (
                    gt.state_to_tensor(g),
                    (
                        torch.tensor([float(rec.z)]),
                        gt.policy_to_tensor(rec.policy, g.get_to_move())
                    )
                )
            )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.data: list[LabeledDataRecord] = []

    def __len__(self):
        return len(self.data)

    def add(self, record: LabeledDataRecord):
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(record)

    def add_many(self, records: list[LabeledDataRecord]):
        for record in records:
            self.add(record)

    def sample_from(self, number_of_samples: int) -> list[LabeledDataRecord]:
        return random.sample(self.data, min(len(self.data), number_of_samples))


def game_record_to_labeled_data(game, record: play.GameRecord, agent_name: str = None) -> list[LabeledDataRecord]:
    labeled_data: list[LabeledDataRecord] = []
    if agent_name:  # Only collect data where this player is to move.
        player = game.White if agent_name == record.players[0] else game.Black
    else:
        player = None
    for move_record in record.move_records:
        game.setup_fen(move_record.fen)
        if (player is None or player == game.get_to_move()):  # and move_record.info is not None:
            z = float(record.outcome) if game.get_to_move() == game.White else float(-record.outcome)
            info = move_record.info if move_record.info is not None else dict()
            policy = gt.visits_to_policy(info)
            labeled_data.append(LabeledDataRecord(move_record.fen, z, policy))
    return labeled_data


def game_records_to_labeled_data(game, game_records: [play.GameRecord],
                                 name: str = None) -> list[LabeledDataRecord]:
    labeled_data: list[LabeledDataRecord] = []
    for game_record in game_records:
        labeled_data += game_record_to_labeled_data(game, game_record, name)
    return labeled_data


def inference(g, model):
    tensor_in = gt.state_to_tensor(g)  # Turn game-state into tensor.
    tensor_in = tensor_in[None]  # Wrap one outer dimension (as for a batch)
    with torch.no_grad():
        tensor_value_out, tensor_policy_out = model(tensor_in)  # Feed through neural network
    tensor_value_out = tensor_value_out.squeeze()  # Remove the outermost batch-size dimension
    tensor_policy_out = tensor_policy_out.squeeze()
    return tensor_value_out.item(), gt.move_tensor_to_policy(g.generate(), tensor_policy_out, g.get_to_move())


def testing_model(model, dl):
    sum_loss = 0.0
    model.eval()  # Put model in evaluation mode.
    with torch.no_grad():
        for X, y in dl:
            pred = model(X)
            loss = model.loss_func(pred, y)
            sum_loss += loss * len(X)
    avg_loss = sum_loss / len(dl.dataset)
    print(f"Average loss in testing: {avg_loss}")


def training_model(model, dl, opt):
    size = len(dl.dataset)
    model.train()  # Tell the model to be in training model.
    for batch, (X, y) in enumerate(dl):
        # Feed-forward pass and compute the prediction loss (error).
        pred = model(X)
        loss = model.loss_func(pred, y)
        loss.backward()  # computing the gradient
        opt.step()  # changing the weights of the network
        opt.zero_grad()  # resetting the gradients

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"\tloss: {loss:>7f} [{current:>5d}/{size:>5d}]")
