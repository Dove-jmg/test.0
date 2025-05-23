from torch.utils.data import Dataset, DataLoader
import torch
import os
import pickle
from .state_taop import StateTAOP
from U4_upper.utils.beam_search import beam_search
import numpy as np


class TAOP(object):
    NAME = 'taop'

    # @profile
    @staticmethod
    def LmaskDataset(dataset, tour_1, tour_2, tour_3, tour_4):
        batch_size = dataset['loc'].size()[0]
        dataset_1, dataset_2, dataset_3, dataset_4 = dataset.copy(), dataset.copy(), dataset.copy(), dataset.copy()
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        # construct the mask matrix where 0 is the feasible node
        mask1 = get_mask(tour_1, batch_size)
        mask2 = get_mask(tour_2, batch_size)
        mask3 = get_mask(tour_3, batch_size)
        mask4 = get_mask(tour_4, batch_size)
        dataset_1['mask'], dataset_2['mask'], dataset_3['mask'], dataset_4['mask'] = mask1, mask2, mask3, mask4

        loc_1 = loc_with_depot.gather(1, tour_1[..., None].expand(*tour_1.size(), loc_with_depot.size(-1)))
        loc_2 = loc_with_depot.gather(1, tour_2[..., None].expand(*tour_2.size(), loc_with_depot.size(-1)))
        loc_3 = loc_with_depot.gather(1, tour_3[..., None].expand(*tour_3.size(), loc_with_depot.size(-1)))
        loc_4 = loc_with_depot.gather(1, tour_4[..., None].expand(*tour_4.size(), loc_with_depot.size(-1)))
        dataset_1['loc'], dataset_2['loc'], dataset_3['loc'], dataset_4['loc'] = loc_1, loc_2, loc_3, loc_4

        return dataset_1, dataset_2, dataset_3, dataset_4

    # @profile
    @staticmethod
    def get_costs(Lmodel, Ltrain, dataset_1, dataset_2, dataset_3, dataset_4):
        # calculate the cost
        model = Lmodel
        torch.manual_seed(1234)
        model.cuda()
        if not Ltrain:
            model.eval()
            model.set_decode_type('greedy')
            with torch.no_grad():
                length1, _, pi_1 = model(dataset_1, return_pi=True)
                length2, _, pi_2 = model(dataset_2, return_pi=True)
                length3, _, pi_3 = model(dataset_3, return_pi=True)
                length4, _, pi_4 = model(dataset_4, return_pi=True)

        else:
            model.train()
            model.set_decode_type("sampling")

        loc1_depot = torch.cat((dataset_1['depot'][:, None, :], dataset_1['loc']), 1)
        loc2_depot = torch.cat((dataset_2['depot'][:, None, :], dataset_2['loc']), 1)
        loc3_depot = torch.cat((dataset_3['depot'][:, None, :], dataset_3['loc']), 1)
        loc4_depot = torch.cat((dataset_4['depot'][:, None, :], dataset_4['loc']), 1)
        Ttour_1 = loc1_depot.gather(1, pi_1[..., None].expand(*pi_1.size(), loc1_depot.size(-1))).cpu().numpy()
        Ttour_2 = loc2_depot.gather(1, pi_2[..., None].expand(*pi_2.size(), loc2_depot.size(-1))).cpu().numpy()
        Ttour_3 = loc3_depot.gather(1, pi_3[..., None].expand(*pi_3.size(), loc3_depot.size(-1))).cpu().numpy()
        Ttour_4 = loc4_depot.gather(1, pi_4[..., None].expand(*pi_4.size(), loc4_depot.size(-1))).cpu().numpy()
        total_cost = length1 + length2 + length3 + length4

        return total_cost, None, Ttour_1, Ttour_2, Ttour_3, Ttour_4
    @staticmethod
    def make_dataset(*args, **kwargs):
        return TAOPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTAOP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TAOP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)

def get_mask(tour, batch_size):
    Tour = tour.cpu().numpy()
    index = np.ones(Tour.shape)
    for i in range(batch_size):
        _, index_i = np.unique(Tour[i, :], return_index=True)
        index_i = np.delete(index_i, 0) if Tour[i, :][0] == 0 else index_i
        index[i, index_i] = 0
    return index

def make_instance(args):
    depot, loc, prize, max_length, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'prize': torch.tensor(prize, dtype=torch.float),  # scale demand
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'max_length': torch.tensor(max_length, dtype=torch.float)
    }


class TAOPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=10000, offset=0, distribution=None):
        super(TAOPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:
            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    'depot': torch.FloatTensor(2).uniform_(0, 1),
                    'prize': torch.ones(size),
                    'max_length': torch.tensor(2.)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)  # num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]  # index of sampled data


