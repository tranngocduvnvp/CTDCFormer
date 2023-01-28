from lib import *
from pvit import Feature_extractor_PVT
from mit import Feature_extractor_MIT
class UpDownstream(nn.Module):
    def __init__(self, scale, in_channel, out_channel) -> None:
        super().__init__()
        self.scale = scale
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.ac = nn.GELU()
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.ac(x)
        bn, c, h, w = x.shape
        x = F.interpolate(x, size=(int(h*self.scale), int(w*self.scale)), mode="bilinear")
        return x

class NormMode(nn.Module):
    def __init__(self, scale, in_channel, out_channel) -> None:
        super().__init__()
        """nhận đầu vào là một tensor cxhxw

        Returns:
            - vector key: d
            - Tensor value: c1xh1xw1
        """
        self.norm = UpDownstream(scale, in_channel, out_channel)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.mg = nn.AdaptiveMaxPool2d((1,1))
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.GELU(),
            nn.Conv2d(in_channel, out_channel, 1)
        )

    def forward(self, x):
        v = self.norm(x).unsqueeze(1) # (bs, 1, c, h, w)
        k = self.mlp(self.avg(x)+self.mg(x)).view(x.shape[0], 1, -1) #(bs,1, c)
        return v, k
        

class AttentionDC(nn.Module):
    def __init__(self, scale, in_channel, out_channel) -> None:
        super().__init__()
        self.normfm1 = NormMode(scale[0], in_channel[0], out_channel)
        self.normfm2 = NormMode(scale[1], in_channel[1], out_channel)
        self.normfm3 = NormMode(scale[2], in_channel[2], out_channel)
        self.normfm4 = NormMode(scale[3], in_channel[3], out_channel)
        self.normfmDecode = NormMode(1, out_channel, out_channel)
        self.mlp = nn.Linear(out_channel*2, out_channel)

    
    def forward(self, feature_maps):
        fm1, fm2, fm3, fm4, fmdecode = feature_maps
        v1, k1 = self.normfm1(fm1)
        v2, k2 = self.normfm2(fm2)
        v3, k3 = self.normfm3(fm3)
        v4, k4 = self.normfm4(fm4)
        vd, qd = self.normfmDecode(fmdecode) #(bs, 1, c)
        K = torch.cat([k1, k2, k3, k4], dim=1) #(bs, 4, c)
        K = torch.cat([K, qd.expand_as(K)], dim=2) #(bs, 4, 2c)
        atten = F.softmax(self.mlp(K), dim=1).unsqueeze(-1).unsqueeze(-1)   #(bs, 4, c, 1, 1)
        V = torch.cat([v1,v2,v3,v4], dim=1) #(bs, 4, c, h, w)
        V = V*atten #(bs, 4, c, h, w)
        V = torch.sum(V, dim=1) #(bs, c, h, w)
        V = torch.cat([V, vd.squeeze(1)], dim=1)
        return V

class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.out_layers = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding="same"),
            nn.BatchNorm2d(out_channels, out_channels),
            nn.GELU(),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)

        

class CTDC(nn.Module):
    def __init__(self, args = args) -> None:
        super().__init__()
        if "Mit" in args.backbone:
            self.feature_extractor = Feature_extractor_MIT(args.backbone)
            channel, scale = self.feature_extractor.channel, self.feature_extractor.scale
        elif "Pvt" in args.backbone:
            self.feature_extractor = Feature_extractor_PVT(args.backbone)
            channel, scale = self.feature_extractor.channel, self.feature_extractor.scale
        
        self.attention1 = AttentionDC(scale[:4], channel, channel[-1])
        self.attention2 = AttentionDC(scale[1:5], channel, channel[-2])
        self.attention3 = AttentionDC(scale[2:6], channel, channel[-3])
        self.attention4 = AttentionDC(scale[3:7], channel, channel[-4])
        # self.csp = BottleneckCSP(channel[-1], channel[-1])
        self.rb1 = nn.Sequential(
            RB(channel[-1], channel[-1]),
            RB(channel[-1], channel[-1])
        )
        self.rb2 = nn.Sequential(
            RB(channel[-1]*2, channel[-2]),
            RB(channel[-2], channel[-2]),
        )
        self.rb3 = nn.Sequential(
            RB(channel[-2]*2, channel[-3]),
            RB(channel[-3], channel[-3]),
        )
        self.rb4 = nn.Sequential(
            RB(channel[-3]*2, channel[-4]),
            RB(channel[-4], channel[-4]),
        )
        self.rb5 = nn.Sequential(
            RB(channel[-4]*2, channel[-4]),
            RB(channel[-4], channel[-4])
        )
        self.rb6 = nn.Sequential(
            RB(channel[-4], channel[-4]),
            RB(channel[-4], channel[-4]),
        )
        self.head = nn.Conv2d(channel[-4], 1, 1)

    
    def forward(self, x):
        fm1, fm2, fm3, fm4 = self.feature_extractor(x)
        decode1 = self.rb1(fm4) #328x7x7
        out1 = self.attention1([fm1, fm2, fm3, fm4, decode1]) #756x7x7
        decode2 = F.interpolate(self.rb2(out1), (out1.shape[2]*2, out1.shape[3]*2), mode="bilinear") #192x14x14
        out2 = self.attention2([fm1, fm2, fm3, fm4, decode2]) #384x14x14
        decode3 = F.interpolate(self.rb3(out2), (out2.shape[2]*2, out2.shape[3]*2), mode="bilinear") #80x28x28
        out3 = self.attention3([fm1, fm2, fm3, fm4, decode3]) #160x28x28
        decode4 = F.interpolate(self.rb4(out3), (out3.shape[2]*2, out3.shape[3]*2), mode="bilinear") #56x56x56
        out4 = self.attention4([fm1, fm2, fm3, fm4, decode4]) #112x56x56
        decode5 = F.interpolate(self.rb5(out4), (out4.shape[2]*2, out4.shape[3]*2), mode="bilinear") #32x112x112
        decode6 = F.interpolate(self.rb6(decode5), (decode5.shape[2]*2, decode5.shape[3]*2), mode="bilinear")
        mask_pred = self.head(decode6)
        return mask_pred

if __name__ == "__main__":

    model = CTDC()
    x = torch.rand(2,3,352,352)
    out = model(x)
    print(out.shape)
