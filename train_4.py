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
                                nn.Linear(n_mid, n_out)])

    def forward(self, x, mask):
        for l in self.layers[:-1]:
            x = nn.SELU()(l(x))

        x = self.layers[-1](x)

        x = x * mask
        x = x - x.min()
        x = ((x + 1e-8) * mask)
        x = x / x.sum()
        return x


class World4(object):

    def __init__(self, size=(4, 4)):
        super(World4, self).__init__()
        self.size = size
        self.world = torch.FloatTensor(np.zeros((2, self.size**2)))

    def get_state(self, player=0):
        idx = [0, 1] if player == 0 else [1, 0]
        return self.world[idx, ...].view((-1))

    def add_to_state(self, choice, player=0):
        self.world[player] = self.world[player] + choice
        assert not np.any(self.world > 1)

    def mask(self):
        return 1 - self.world.sum(0)

    def print(self):
        world_tmp = self.world.clone()
        world_tmp[0] *= 1
        world_tmp[1] *= 2
        print(world_tmp.sum(0).view((self.size, self.size)))

    def game_full(self):
        return self.size**2 == self.world.sum()

    def game_over(self):
        world = self.world.clone().detach()
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
        self.world = self.world.clone().detach()
        self.world *= 0


def run(world, model, optimizer, batch_size=100):
    world.reset()
    i = 0
    to_2 = 0
    while i < 100000:
        if i % batch_size == 0:
            optimizer.zero_grad()

        if to_2 == 0:
            player_init = np.random.randint(2)
            world_init = np.random.randint(0, world.size ** 2)
            to_2 = 1
        else:
            to_2 = 0

        result = play_game(model, optimizer, world, player_init, world_init,
                           verbose=False)

        if np.any(result):
            loss0 = world.world[0].mean() * ((2 * (result[0])) - 1)
            loss1 = world.world[0].mean() * ((2 * (result[1])) - 1)
            loss = loss0 + loss1
            loss.backward()
            if i % batch_size == 0:
                optimizer.step()
                print(f"\n--------------{i}--------------")
                print(result)
            i = i + 1

        world.reset()

        if i % 1000 == 0 and np.any(result):
            print("Evaluate..")
            results = []
            for j in range(1000):
                if j % 100 == 0:
                    print(j)
                world.reset()
                player_init = np.random.randint(2)
                world_init = np.random.randint(0, world.size ** 2)
                result = play_game(model, optimizer, world, player_init, world_init,
                                   verbose=False, random_guess=True)
                results.append(result)
                world.reset()
            sums = np.sum(results, axis=0)
            ratio = sums / np.sum(sums)
            print(f"Evaluation = {ratio}")




def play_game(model, optimizer, world, player_init, world_init, verbose=False,
              random_guess=False):

    player = player_init
    world.world[player, world_init] = 1

    if verbose:
        print(f"initial: player {player}, world {world_rand}")

    while not world.game_full():
        player = (player + 1) % 2
        pred = model(world.get_state(player), world.mask())

        if random_guess and player == 2:
            pred = pred * 0 + world.mask() // np.sum(world.mask())

        if verbose:
            print(np.var(model.layers[0].weight.data.numpy()))
            print(pred.view((world.size, world.size)))

        s = Categorical(pred)
        vals = s.sample()
        choice = world.world[0].clone() * 0
        choice[vals] = 1
        add_choice = pred + (choice - pred).detach()
        world.add_to_state(add_choice, player)

        if verbose:
            world.print()

        result = world.game_over()
        if np.any(result):
            break

    return result


if __name__ == '__main__':
    world = World4(4)
    n_out = world.size**2
    model = FFNN(n_out * 2, n_out, n_mid=128)
    optimizer = Adam(model.parameters(), lr=0.01)
    run(world, model, optimizer)