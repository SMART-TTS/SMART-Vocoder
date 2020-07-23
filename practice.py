        # a = torch.tensor(list(range(256))).cuda().view(2,-1).unsqueeze(1)
        # b = a.view(2, 1, 32, 4).permute(0,1,3,2).contiguous().view(2,4,32)
        # print('b', b)
        # B, C, T = b.shape
        # c = b.permute(0,2,1).contiguous().view(B, (C*T)//sc, sc)
        # c = c.permute(0,2,1).contiguous().view(B*sc, T//sc, C)
        # c = c.permute(0,2,1).contiguous()
        # print('c', c)

        # B, C, T = c.shape
        # d = c.permute(0,2,1).contiguous().view(B, (C*T)//sc, sc)
        # d = d.permute(0,2,1).contiguous().view(B*sc, T//sc, C)
        # d = d.permute(0,2,1).contiguous()

        # print('d', d)

        # B, C, T = d.shape
        # f = d.permute(0,2,1).contiguous()
        # f = f.view(B//sc, sc, T, C).permute(0,2,3,1).contiguous()
        # f = f.view(B//sc, T*sc, C).permute(0,2,1).contiguous()
        # print('f', f)

        # print(c - f)
