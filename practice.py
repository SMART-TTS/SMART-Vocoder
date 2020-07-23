
        a = torch.tensor(list(range(512))).cuda().view(4,-1).unsqueeze(1)
        b = a.view(4, 1, 32, 4).permute(0,1,3,2).contiguous().view(4,4,32)
        print('b', b)

        B, C, T = b.shape
        c = b.permute(0,2,1).contiguous().view(B, (C*T)//sc, sc)
        c = c.permute(0,2,1).contiguous().view(B*sc, T//sc, C)
        c = c.permute(0,2,1).contiguous().view(B*sc, C, T//sc)
        print('c', c)
    
        B, C, T = c.shape
        d = c.permute(0,2,1).contiguous().view(B, (C*T)//sc, sc)
        d = d.permute(0,2,1).contiguous().view(B*sc, T//sc, C)
        d = d.permute(0,2,1).contiguous().view(B*sc, C, T//sc)
        print('d', d)


        B, C, T = d.shape
        cc = d.permute(0,2,1).contiguous()
        cc = cc.view(B//sc, sc, T, C).permute(0,2,3,1).contiguous()
        cc = cc.view(B//sc, T*sc, C).permute(0,2,1).contiguous()

        print('cc', cc-c)

        B, C, T = cc.shape
        bb = cc.permute(0,2,1).contiguous()
        bb = bb.view(B//sc, sc, T, C).permute(0,2,3,1).contiguous()
        bb = bb.view(B//sc, T*sc, C).permute(0,2,1).contiguous()

        print('bb', bb-b)
