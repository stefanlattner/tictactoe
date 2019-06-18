import torch
import torch.nn as nn

import numpy as np
from torch.distributions import Categorical
from torch.optim import Adam


class FFNN(nn.Module):
    def __init__(self, n_in, n_out, n_mid):
        super(FFNN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(n_in, n_mid),
                                     nn.Linear(n_mid, n_mid),
                                     nn.Linear(n_mid, n_mid),
                                     nn.Linear(n_mid, n_mid),
                                     nn.Linear(n_mid, n_out)])

        self.softmax = nn.Softmax()

    def forward(self, x, mask):
        for l in self.layers[:-1]:
            x = nn.SELU()(l(x))

        x = self.layers[-1](x) * mask.float()

        #x[x < 0] = 0
        x[mask] = self.softmax(x[mask])
        # x = x - x.min()
        # x = ((x + 1e-8) * mask)
        #x = x / (x.sum() + 1e-6)
        return x


class World4(object):

    def __init__(self, size=(4, 4)):
        super(World4, self).__init__()
        self.size = size
        self.world = torch.FloatTensor(np.zeros((2, self.size**2))).cuda()
        self.states = []
        self.last_idx0 = -1
        self.last_idx1 = -1

    def get_state(self, player=0):
        idx = [0, 1] if player == 0 else [1, 0]
        return self.world[idx, ...].view((-1))

    def add_to_state(self, choice, player=0):
        self.world[player] = self.world[player] + choice
        if player == 0:
            self.last_idx0 = torch.argmax(choice)
        if player == 1:
            self.last_idx1 = torch.argmax(choice)

        #assert not np.any(self.world > 1)

    def save_state(self, pred):
        # pred = prediction which lead to this state
        self.states.append({"pred": pred.detach(),
                            "world": self.world.detach().clone(),
                            "mask": self.mask(),
                            "state_pl0": self.get_state(0),
                            "state_pl1": self.get_state(1)
                            })

    def mask(self):
        return (1 - self.world.detach().sum(0)).byte()

    def print(self):
        self.print_state(self.world)

    def print_state(self, world_):
        world_tmp = world_.clone()
        world_tmp[0] *= 1
        world_tmp[1] *= 2
        print(world_tmp.sum(0).view((self.size, self.size)))

    def game_full(self):
        return self.size**2 == self.world.sum()

    def game_over(self):
        if False: #len(self.states) > 1:
            return (False, True)
        world = self.world.clone().detach().cpu()
        horizx = np.any(
            world[0].view([self.size]*2).sum(0).data == self.size)
        horizy = np.any(
            world[1].view([self.size]*2).sum(0).data == self.size)
        vertx = np.any(
            world[0].view([self.size]*2).sum(1).data == self.size)
        verty = np.any(
            world[1].view([self.size]*2).sum(1).data == self.size)
        diagdownx = np.any((torch.eye(self.size) * world[0].view(
            [self.size] * 2).data).sum() == self.size)
        diagdowny = np.any((torch.eye(self.size) * world[1].view(
            [self.size] * 2).data).sum() == self.size)
        diagupx = np.any((np.eye(self.size)[::-1] * world[0].view(
            [self.size] * 2).numpy()).sum() == self.size)
        diagupy = np.any((np.eye(self.size)[::-1] * world[1].view(
            [self.size] * 2).numpy()).sum() == self.size)

        return (horizx or vertx or diagdownx or diagupx),\
               (horizy or verty or diagdowny or diagupy)

    def reset(self):
        self.world = self.world.detach().clone()
        self.world *= 0
        self.states = []
        self.last_idx0 = -1
        self.last_idx1 = -1

    def plot_grads(self):
        print(self.world.grad)


def compare_states(player_init=0):
    print("Simulate game..")
    player = player_init
    for i in range(len(world.states)):
        curr_state = world.states[i]
        if player == 1:
            curr_input = curr_state["state_pl0"]
        else:
            curr_input = curr_state["state_pl1"]

        curr_pred = model(curr_input, curr_state["mask"])
        last_pred = curr_state["pred"]
        world.print_state(curr_state["world"])
        print((curr_pred - last_pred).view((4,4)))
        #print(curr_state)
        player = (player + 1) % 2


def evaluate():
    with torch.no_grad():
        print("Evaluate..")
        results = []
        nr_games = 2000
        for j in range(nr_games):
            if j % 500 == 0:
                print(j)
            world.reset()
            player_init = np.random.randint(2)
            world_init = np.random.randint(0, world.size ** 2)
            result = play_game(model, world, player_init, world_init,
                               verbose=False, random_guess=True)
            results.append(result)
            world.reset()
        sums = np.sum(results, axis=0)
        ratio = sums / np.sum(sums)
        print(f"Won/Lost against random guessing in {nr_games} games = {ratio}")


def heat(dist, t):
    return dist ** (1 / t) / (dist ** (1 / t)).sum()


def train(world, model, optimizer, batch_size=100):
    world.reset()
    i = 0
    temp = 1
    while i < 1e6:
        if i % batch_size == 0:
            optimizer.zero_grad()

        player_init = np.random.randint(2)
        world_init = np.random.randint(0, world.size ** 2)

        result = play_game(model, world, player_init, world_init,
                           verbose=i % 1000 == 0,
                           temperature=temp,
                           random_guess=False)

        if result[0]:
            loss0 = world.world[0][world.last_idx0].sum() * -1
        elif result[1]:
            loss0 = world.world[0][world.last_idx0].sum() * 1
        else: #Draw
            loss0 = world.world[0][world.last_idx0].sum() * .5

        loss = loss0
        loss.backward()

        if i % 500 == 0:
            print(f"\n--------------{i}--------------")
            print(result)

        i = i + 1

        if (i+1) % batch_size == 0:
            optimizer.step()

        #print(model.layers[0].weight)

        world.reset()
        temp = np.maximum(1, temp * (1 - 1e-5))

        if i % 5000 == 1 and np.any(result):
            evaluate()


def play_game(model, world, player_init, world_init, verbose=True,
              random_guess=False, temperature=1.):

    t = temperature
    player = player_init
    world.world[player, world_init] = 1

    #if verbose:
        #print(f"initial: player {player}, world {world_rand}")

    while not world.game_full():
        player = (player + 1) % 2
        pred = model(world.get_state(player), world.mask())

        if player == 1:
            pred = pred.detach()
            if random_guess:
                pred = (pred * 0 + world.mask().float()) / torch.sum(world.mask())

        if verbose:
            print(f"Prediction of player {player+1} (t={t}):")
            print(heat(pred, t).view((world.size, world.size)))

        world.save_state(pred)
        s = Categorical(heat(pred, t))
        vals = s.sample()
        #vals = torch.argmax(pred)
        choice = world.world[0].detach().clone() * 0
        #print(vals)
        choice[vals] = 1
        #distance = (choice - pred).detach().abs()
        #add_choice = pred * distance + (choice - pred * distance).detach()
        distance = (choice - pred).detach()
        add_choice = pred + distance
        world.add_to_state(add_choice, player)

        if verbose:
            print(f"New State (chosen: {vals}) =")
            world.print()

        result = world.game_over()
        if np.any(result):
            if verbose:
                print(f"Finished: {result}")
            break

    return result


def save_model(save_key=""):
    torch.save(f"{cache_fn}{save_key}.save", model.get_state_dict())


def load_model(save_key=""):
    return model.load_state_dict(f"{cache_fn}{save_key}.save")


if __name__ == '__main__':
    batch_size = 1
    cache_fn = "model_"
    world = World4(4)
    n_out = world.size**2
    model = FFNN(n_out * 2, n_out, n_mid=256)
    model.cuda()
    optimizer = Adam(model.parameters(), lr=1e-4)
    train(world, model, optimizer, batch_size=batch_size)
