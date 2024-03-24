import torch
import BGHTLib

keys = torch.randint(0, 10000, (100,)).int().cuda().unique()
values = keys + 1
print(keys)
print(values)

hashmap = BGHTLib.BGHTHashmap(keys, values)
print(0)
print(hashmap)

print(hashmap.query(keys))
print(hashmap.query(keys + 1))

print(hashmap.memory_usage())