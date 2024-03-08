import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from queue import PriorityQueue

class ListNode():
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next
    def __lt__(self, other):
        return True

def new_ax(string = None):
    return plt.figure(string).add_subplot(1,1,1)


def RDP(points, num = 3):
    '''
    Rahmer Douglas Peucker - Algorithm
    '''
    def get_farthest_point(start, stop):
        if start >= stop:
            raise ValueError
        if start == stop + 1:
            return 0, None
        Y = []
        for i in range(start + 1, stop):
            Y.append([points[i][0] - points[start][0], points[i][1] - points[start][1]])
        x_n = [points[stop][0] - points[start][0], points[stop][1] - points[start][1]]
        norm = math.sqrt(x_n[1] ** 2 + x_n[0] ** 2)
        normal_vector = [-x_n[1] / norm, x_n[0] / norm]
        delta = []
        for x in Y:
            delta.append(math.fabs(normal_vector[0] * x[0] + normal_vector[1] * x[1]))
        i = np.argmax(delta)
        delta_i = delta[i]
        return delta_i, start + i + 1
    if len(points) < num:
        raise ValueError()
    if num == 2:
        return [points[0], points[-1]]
    end = ListNode(val = len(points)-1)
    root = ListNode(next = end)


    prioq = PriorityQueue()
    delta, index = get_farthest_point(0, len(points) - 1)
    prioq.put((-delta, (root, index)))
    num_points = 2
    while True:
        obj = prioq.get()[1]
        start = obj[0]
        stop = obj[0].next
        mid = ListNode(obj[1], stop)
        start.next = mid
        num_points += 1
        if num_points == num:
            break
        if start.val != mid.val - 1:
            delta,index = get_farthest_point(start.val,mid.val)
            prioq.put((-delta, (start, index)))
        if mid.val != stop.val - 1:
            delta,index = get_farthest_point(mid.val,stop.val)
            prioq.put((-delta, (mid, index)))
    node = root
    out = []
    while node != None:
        out.append(points[node.val])
        node = node.next
    assert len(out) == num_points
    return out

class Stepper():

    def __init__(self, limit):
        
        self.l = -1
        self.e = 0
        self.limit = limit
        self.cache = None

    def save(self):
        raise NotImplementedError()

    def step(self):
        if self.l == -1:
            self.l = 0
        else:
            self.e += 1
            if self.e == self.limit:
                self.l += 1
                self.e = 0
            if self.l == 0:
                self.l = 1
                self.e = 0
    def next_exists(self):
        if self.l >= len(self.cache):
            return False
        return True

    def reset_counter(self):
        self.l = -1
        self.e = 0

    def element_step(self):
        self.step()
        return self.element()

    def __getitem__(self, key):
        return self.cache.__getitem__(key)

    def element(self):
        if self.l >= len(self.cache):
            raise EOFError()
        return self[self.l][self.e]

class LayerCache(Stepper):
    def __init__(self, model = None, limit=1, cache = None):
        super().__init__(limit)
        if cache is None:
            self.cache = [[None for _ in range(limit)] for _ in model.layer_sizes]
        else:
            self.cache = cache

    def save(self, obj):
        self.step()
        self.cache[self.l][self.e] = obj

class NodeCache(Stepper):
    def __init__(self, model = None, limit=1, cache = None):
        super().__init__(limit)
        if cache is None:
            self.cache = [[[None for _ in range(item)] for _ in range(limit)] for item in model.layer_sizes]
        else:
            self.cache = cache

    def save(self, obj, i):
        if i == 0:
            self.step()
        self.cache[self.l][self.e][i] = obj


if __name__ == "__main__":
    points = [(1,1), (2,5),(3,600), (4,1), (5,1), (6,5),(7,2), (80,1)]
    print(RDP(points,num=6))

