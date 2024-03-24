import torch
import PyCUCOLib

torch.manual_seed(32)

keys = torch.randint(0, 10000, (100, )).int().cuda().unique()
values = keys + 1
print(keys)
print(values)

hashmap = PyCUCOLib.CUCOStaticHashmap(keys, values, 0.5)
print(0)
print(hashmap)

print(hashmap.query(keys))
print(hashmap.query(keys + 1))
