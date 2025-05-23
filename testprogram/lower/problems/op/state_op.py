import torch
from typing import NamedTuple
from lower.utils.boolmask import mask_long2bool, mask_long_scatter
import torch.nn.functional as F
import numpy as np


class StateOP:
    # 将NamedTuple改为普通类
    def __init__(self, coords, prize, max_length, ids, prev_a, visited_, lengths, cur_coord, cur_total_prize, i):
        self.coords = coords
        self.prize = prize
        self.max_length = max_length
        self.ids = ids
        self.prev_a = prev_a
        self.visited_ = visited_
        self.lengths = lengths
        self.cur_coord = cur_coord
        self.cur_total_prize = cur_total_prize
        self.i = i
        # 用于跟踪递归调用
        self._recursion_guard = False

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(-2))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        # 防止递归
        if self._recursion_guard:
            print("检测到递归调用__getitem__，返回None")
            return None
        
        self._recursion_guard = True
        try:
            if isinstance(key, int):
                # 创建新对象时只传递必要的字段
                result = StateOP(
                    coords=self.coords,
                    prize=self.prize,
                    max_length=self.max_length,
                    ids=self.ids[key:key+1] if self.ids is not None else None,
                    prev_a=self.prev_a[key:key+1] if self.prev_a is not None else None,
                    visited_=self.visited_[key:key+1] if self.visited_ is not None else None,
                    lengths=self.lengths[key:key+1] if self.lengths is not None else None,
                    cur_coord=self.cur_coord[key:key+1] if self.cur_coord is not None else None,
                    cur_total_prize=self.cur_total_prize[key:key+1] if self.cur_total_prize is not None else None,
                    i=self.i  # i通常是标量，不需要索引
                )
                return result
            else:  # 张量或切片
                result = StateOP(
                    coords=self.coords,
                    prize=self.prize,
                    max_length=self.max_length,
                    ids=self.ids[key] if self.ids is not None else None,
                    prev_a=self.prev_a[key] if self.prev_a is not None else None,
                    visited_=self.visited_[key] if self.visited_ is not None else None,
                    lengths=self.lengths[key] if self.lengths is not None else None,
                    cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
                    cur_total_prize=self.cur_total_prize[key] if self.cur_total_prize is not None else None,
                    i=self.i  # i通常是标量，不需要索引
                )
                return result
        except Exception as e:
            print(f"错误在StateOP.__getitem__: {e}, key类型: {type(key)}")
            raise
        finally:
            # 完成后清除标记
            self._recursion_guard = False

    # 替代NamedTuple的_replace方法
    def _replace(self, **kwargs):
        # 创建一个当前对象的拷贝
        new_state = StateOP(
            coords=kwargs.get('coords', self.coords),
            prize=kwargs.get('prize', self.prize),
            max_length=kwargs.get('max_length', self.max_length),
            ids=kwargs.get('ids', self.ids),
            prev_a=kwargs.get('prev_a', self.prev_a),
            visited_=kwargs.get('visited_', self.visited_),
            lengths=kwargs.get('lengths', self.lengths),
            cur_coord=kwargs.get('cur_coord', self.cur_coord),
            cur_total_prize=kwargs.get('cur_total_prize', self.cur_total_prize),
            i=kwargs.get('i', self.i)
        )
        return new_state

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        depot = input['depot']
        loc = input['loc']
        prize = input['prize']
        max_length = input['max_length']
        if not torch.is_tensor(input['mask']):
            mask = np.pad(input['mask'], ((0, 0), (1, 0)), 'constant', constant_values=(0, 0))
        elif input['mask'].any():
            mask = input['mask'].cpu().numpy()
            mask = np.pad(mask, ((0, 0), (1, 0)), 'constant', constant_values=(0, 0))
        else:
            mask = np.zeros([prize.shape[0], prize.shape[1]+1])

        batch_size, n_loc, _ = loc.size()
        coords = torch.cat((depot[:, None, :], loc), -2)
        return StateOP(
            coords=coords,
            prize=F.pad(prize, (1, 0), mode='constant', value=0),  # add 0 for depot
            # max_length is max length allowed when arriving at node, so subtract distance to return to depot
            # Additionally, substract epsilon margin for numeric stability
            max_length=max_length[:, None] - (depot[:, None, :] - coords).norm(p=2, dim=-1) - 1e-6,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently (if there is an action for depot)
                torch.tensor(mask, dtype=torch.uint8).unsqueeze(1).cuda()
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 1 + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            cur_total_prize=torch.zeros(batch_size, 1, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_remaining_length(self):
        # max_length[:, 0] is max length arriving at depot so original max_length
        return self.max_length[self.ids, 0] - self.lengths

    def get_final_cost(self):

        assert self.all_finished()
        # The cost is the negative of the collected prize since we want to maximize collected prize
        return -self.cur_total_prize

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Add the collected prize
        cur_total_prize = self.cur_total_prize + self.prize[self.ids, selected]

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, by check_unset=False it is allowed to set the depot visited a second a time
            visited_ = mask_long_scatter(self.visited_, prev_a, check_unset=False)

        return self._replace(
            prev_a=prev_a, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, cur_total_prize=cur_total_prize, i=self.i + 1
        )

    def all_finished(self):
        # All must be returned to depot (and at least 1 step since at start also prev_a == 0)
        # This is more efficient than checking the mask
        return self.i.item() > 0 and (self.prev_a == 0).all()
        # return self.visited[:, :, 0].all()  # If we have visited the depot we're done

    def get_current_node(self):
        """
        Returns the current node where 0 is depot, 1...n are nodes
        :return: (batch_size, num_steps) tensor with current nodes
        """
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        exceeds_length = (
            self.lengths[:, :, None] + (self.coords[self.ids, :, :] - self.cur_coord[:, :, None, :]).norm(p=2, dim=-1)
            > self.max_length[self.ids, :]
        )
        # Note: this always allows going to the depot, but that should always be suboptimal so be ok
        # Cannot visit if already visited or if length that would be upon arrival is too large to return to depot
        # If the depot has already been visited then we cannot visit anymore
        visited_ = self.visited.to(exceeds_length.dtype)
        mask = visited_ | visited_[:, :, 0:1] | exceeds_length
        # Depot can always be visited
        # (so we do not hardcode knowledge that this is strictly suboptimal if other options are available)
        mask[:, :, 0] = 0
        return mask

    def construct_solutions(self, actions):
        return actions
