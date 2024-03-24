import torch
import BGHTLib

keys = torch.randint(0, 10000, (100,)).int().cuda().unique()
values = keys + 1
print(keys)
print(values)

hashmap = BGHTLib.BGHTHashmap(keys, values)

print(hashmap)

print(hashmap.query(keys + 1))