import torch
from typing import NamedTuple
from utils.boolmask import mask_long_scatter


class StateTAOP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc, [batch_size, graph_size+1, 2]
    # demand: torch.Tensor
    # capacity: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    veh: torch.Tensor  # numver of vehicles

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    veh_num = 6

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return self.visited_[:, None, :].expand(self.visited_.size(0), 1, -1).type(torch.ByteTensor)

    @property
    def dist(self):  # coords: []
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice) # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            veh=self.veh[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            cur_coord=self.cur_coord[key],
        )


    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['loc']

        batch_size, n_loc, _ = loc.size()  # n_loc = graph_size
        return StateTAOP(
            coords=torch.cat((depot[:, None, :], loc), -2),  # [batch_size, graph_size+1, 2]]
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            veh=torch.arange(StateTAOP.veh_num, dtype=torch.int64, device=loc.device)[:, None],
            prev_a=torch.zeros(batch_size, StateTAOP.veh_num, dtype=torch.long, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            cur_coord=input['depot'][:, None, :].expand(batch_size, StateTAOP.veh_num, -1),
            # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):
        assert self.all_finished()
        # coords: [batch_size, graph_size+1, 2]
        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected, veh):  # [batch_size, num_veh]

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        prev_a = selected  # [batch_size, num_veh]
        batch_size, _ = selected.size()

        # # Add the length, coords:[batch_size, graph_size, 2]
        cur_coord = self.coords.gather(  # [batch_size, num_veh, 2]
            1,
            selected[:, :, None].expand(selected.size(0), self.veh_num, self.coords.size(-1))
        )

        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[torch.arange(batch_size), veh][:, None, None].expand_as(
                self.visited_[:, :, 0:1]), 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a[torch.arange(batch_size), veh])

    
        return self._replace(
            prev_a=prev_a, visited_=visited_,
            cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self):
        return self.visited[:,:,1:].all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    # need to be modified
    def get_mask(self, veh):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        batch_size = self.visited_.size(0)
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]  # [batch_size, 1, n_loc]
        else:
            visited_loc = self.visited_[:, 1:][:, None, :]  # [batch_size, 1, n_loc]

        exceeds_cap = (self.demand[self.ids, 1:] + (self.used_capacity[torch.arange(batch_size), veh].unsqueeze(-1))[
            ..., None].expand_as(self.demand[self.ids, 1:]) >
                       ((torch.tensor(self.VEHICLE_CAPACITY)[None, :].expand(batch_size,
                                                                             len(self.VEHICLE_CAPACITY)).cuda())[
                            self.ids, veh][0].unsqueeze(-1))[..., None].expand_as(
                           self.demand[self.ids, 1:]))  # �ҳ���һ��demand�����node, ids [batch_size,1]

        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap  # [batch_size, 1, n_loc]

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a[torch.arange(batch_size), veh] == 0)[:, None] & (
                    (mask_loc == 0).int().sum(-1) > 0)  # [batch_size, 1]

        return torch.cat((mask_depot[:, :, None], mask_loc), -1)  # [batch_size, 1, graph_size]

    
    def construct_solutions(self, actions):
        return actions
