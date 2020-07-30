import torch
sc = 4

a = torch.tensor(list(range(512))).cuda().view(4,-1).unsqueeze(1)
print('a', a)

b = a.view(4, 1, 32, 4).permute(0,1,3,2).contiguous().view(4,4,32)
b_cond = torch.tensor(list(range(128))).cuda().view(4,-1).unsqueeze(1)
print('b', b)
print('b_cond', b_cond.shape)

B, C, T = b.shape
c = b.permute(0,2,1).contiguous().view(B, (C*T)//sc, sc)
c = c.permute(0,2,1).contiguous().view(B*sc, T//sc, C)
c = c.permute(0,2,1).contiguous()
c_cond1 = b_cond.repeat(1, 4 , 1).view(-1, 1, T)
c_cond2 = torch.repeat_interleave(b_cond, dim=0, repeats=sc)
# print('c', c)
print('c_cond1', c_cond1- c_cond2)
print('c_cond2', c_cond2)

# B, C, T = c.shape
# d = c.permute(0,2,1).contiguous().view(B, (C*T)//sc, sc)
# d = d.permute(0,2,1).contiguous().view(B*sc, T//sc, C)
# d = d.permute(0,2,1).contiguous()
# print('d', d)


# B, C, T = d.shape
# cc = d.permute(0,2,1).contiguous()
# cc = cc.view(B//sc, sc, T, C).permute(0,2,3,1).contiguous()
# cc = cc.view(B//sc, T*sc, C).permute(0,2,1).contiguous()

# print('cc', cc)

# B, C, T = cc.shape
# bb = cc.permute(0,2,1).contiguous()
# bb = bb.view(B//sc, sc, T, C).permute(0,2,3,1).contiguous()
# bb = bb.view(B//sc, T*sc, C).permute(0,2,1).contiguous()

# print('bb', bb)
