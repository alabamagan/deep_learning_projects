import torch
import torch.nn as nn
import torch.nn.functional as F

from .old_stdlayers import ResidualBlock3d, Conv3d, InvertedConv3d, DoubleConv3d

__all__ = ['AttentionResidualGRUNet', 'AttentionResidualNet', 'AttentionResidualNet_SW',
           'AttentionResidualNet_64']

class SoftMaskBranch_aed2d(nn.Module):
    def __init__(self, in_ch, out_ch, r=1):
        super(SoftMaskBranch_aed2d, self).__init__()

        down1 = [ResidualBlock3d(in_ch, out_ch)] * r
        self.down1 = nn.Sequential(*down1)

        down2 = [ResidualBlock3d(out_ch, out_ch)] * (2*r)
        self.down2 = nn.Sequential(*down2)

        self.skip = ResidualBlock3d(out_ch, out_ch)

        up1 = [ResidualBlock3d(out_ch, out_ch)] * r
        self.up1 = nn.Sequential(*up1)

        self.out_conv = nn.Sequential(
            InvertedConv3d(out_ch, out_ch, kern_size=1, padding=0),
            InvertedConv3d(out_ch, out_ch, kern_size=1, padding=0)
        )

        pass

    def forward(self, x):
        mask1 = F.max_pool3d(x, [3, 3, 1], stride=[2, 2, 1])
        mask1 = self.down1(mask1)

        mask2 = F.max_pool3d(mask1, [3, 3, 1], stride=[2, 2, 1])
        mask2 = self.down2(mask2)

        skip = self.skip(mask1)
        mask2 = F.interpolate(mask2, skip.size()[-3:])

        mask3 = self.up1(mask2 + skip)
        mask3 = F.interpolate(mask3, x.size()[-3:])

        out = self.out_conv(mask3)
        out = torch.sigmoid(out)
        return out



class AttentionModule(nn.Module):
    def __init__(self, in_ch, out_ch, p=1, t=2, r=1, save_mask=False):
        super(AttentionModule, self).__init__()

        in_conv = [ResidualBlock3d(in_ch, out_ch)] * p
        self.in_conv = nn.Sequential(*in_conv)

        self.soft_mask_branch = SoftMaskBranch_aed2d(out_ch, out_ch, r)

        trunk_branch = [ResidualBlock3d(out_ch, out_ch)] * t
        self.trunk_branch = nn.Sequential(*trunk_branch)

        out_conv = [ResidualBlock3d(out_ch, out_ch)] * p
        self.out_conv = nn.Sequential(*out_conv)

        self.bool_save_mask = save_mask
        self.saved_mask = None

    def forward(self, x):
        res = F.relu(F.max_pool3d(x, [1, 2, 2]))
        out = self.in_conv(res)
        out += res

        trunk = self.trunk_branch(out)

        mask = self.soft_mask_branch(out)
        mask = trunk * (1 + mask)
        if self.bool_save_mask:
            self.saved_mask = mask.cpu()

        out = self.out_conv(mask)
        return out

    def get_mask(self):
        return self.saved_mask


class AttentionModule_Modified(nn.Module):
    def __init__(self, in_ch, out_ch, p=1, t=2, r=1, save_mask=False):
        super(AttentionModule_Modified, self).__init__()

        in_conv = [ResidualBlock3d(in_ch, out_ch)] * p
        self.in_conv = nn.Sequential(*in_conv)

        self.soft_mask_branch = SoftMaskBranch_aed2d(out_ch, out_ch, r)

        trunk_branch = [ResidualBlock3d(out_ch, out_ch)] * t
        self.trunk_branch = nn.Sequential(*trunk_branch)

        out_conv = [ResidualBlock3d(out_ch, out_ch)] * p
        self.out_conv = nn.Sequential(*out_conv)

        self.bool_save_mask = save_mask
        self.saved_mask = None

    def forward(self, x):
        res = F.relu(F.max_pool3d(x, [2, 2, 1]))
        out = self.in_conv(res)
        # out += res

        trunk = self.trunk_branch(out)

        mask = self.soft_mask_branch(out)
        mask = trunk * (1 + mask)
        if self.bool_save_mask:
            self.saved_mask = mask.cpu()

        out = self.out_conv(mask)
        return out

    def get_mask(self):
        return self.saved_mask



class AttentionResidualNet(nn.Module):
    def __init__(self, in_ch, out_ch, save_mask=False, save_weight=False):
        super(AttentionResidualNet, self).__init__()

        self.save_weight=save_weight
        self.in_conv1 = Conv3d(in_ch, 64, stride=[1, 2, 2], padding=[1, 2, 2])
        self.in_sw = Conv3d(64, 20)
        self.x_w = None

        self.in_conv2 = ResidualBlock3d(64, 256)


        self.att1 = AttentionModule(256, 256, save_mask=save_mask)
        self.r1 = ResidualBlock3d(256, 512)
        self.att2 = AttentionModule(512, 512, save_mask=save_mask)
        self.r2 = ResidualBlock3d(512, 1024)
        self.att3 = AttentionModule(1024, 1024, save_mask=save_mask)

        self.out_conv1 = ResidualBlock3d(1024, 2048)
        self.out_linear = nn.Linear(20, 1)

        self.out_fc1 = nn.Linear(2048, out_ch)

        # initilization
        nn.init.ones_(self.out_linear.weight)
        nn.init.zeros_(self.out_linear.bias)


    def forward(self, x):
        while x.dim() < 5:
            x = x.unsqueeze(0)
        x = self.in_conv1(x)

        # Construct slice weight
        x_w = self.in_sw(x)
        # print("x_w0: {}".format(x_w.shape))
        x_w = F.avg_pool3d(x_w,kernel_size=x_w.shape[-3:]).squeeze()
        # print("x_w1: {}".format(x_w.shape))
        x_w = torch.sigmoid(x_w) + 0.5
        if self.save_weight:
            self.x_w = x_w.data.cpu()
        # print("x_w2: {}".format(x_w.shape))

        # Permute the axial dimension to the last
        x = F.max_pool3d(x, [1, 2, 2], stride=[1, 2, 2]).permute([1, 3, 4, 0, 2])
        # print("x_premult: {}".format(x.shape))
        x_shape = x.shape
        new_shape = list(x_shape[:3]) + [x_shape[-2] * x_shape[-1]]
        x = x.reshape(new_shape)
        # print("x_reshape: {}".format(x.shape))
        x = x * x_w.view([-1]).expand_as(x)
        # print("x_expand: {}".format(x.shape))


        # Resume dimension
        x = x.view(x_shape).permute([3, 0, 4, 1, 2])
        # print("x_resume: {}".format(x.shape))
        x = self.in_conv2(x)
        # print("x_in_conv2: {}".format(x.shape))

        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)

        x = self.out_conv1(x)
        x = F.avg_pool3d(x, kernel_size=[1] + list(x.shape[-2:])).squeeze()
        # while x.dim() < 3:
        #     x = x.unsqueeze(0)
        # x = x.permute([1, 0, 2])
        # x_shape = x.shape
        # new_shape = [x_shape[0]] + [x_shape[1]*x_shape[2]]
        # x = x.reshape(new_shape)
        # x = x * x_w.expand_as(x)
        # x = x.view(x_shape).permute([1, 0, 2])
        #
        if x.dim() < 3:
            x = x.unsqueeze(0)
        x = x.permute([1, 0, 2])
        x = x * x_w.expand_as(x)
        x = x.permute([1, 0, 2])
        x = self.out_linear(x).squeeze()
        x = self.out_fc1(x)
        while x.dim() < 2:
            x = x.unsqueeze(0)
        return x


    def get_mask(self):
        #[[B,H,W,D],[B,H,W,D],[B,H,W,]]
        return [r.get_mask() for r in [self.att1, self.att2, self.att3]]

    def get_slice_attention(self):
        if not self.x_w is None:
            while self.x_w.dim() < 2:
                self.x_w = self.x_w.unsqueeze(0)
            return self.x_w
        else:
            print("Attention weight was not saved!")
            return None



class AttentionResidualNet_64(nn.Module):
    def __init__(self, in_ch, out_ch, save_mask=False, save_weight=False):
        super(AttentionResidualNet_64, self).__init__()

        self.save_weight=save_weight
        self.in_conv1 = Conv3d(in_ch, 64, stride=[1, 2, 2], padding=[1, 1, 1])
        self.in_sw = Conv3d(64, 20)
        self.x_w = None

        self.in_conv2 = ResidualBlock3d(64, 256)


        self.att1 = AttentionModule_Modified(256, 256, save_mask=save_mask)
        self.r1 = ResidualBlock3d(256, 512, p=0.3)
        self.att2 = AttentionModule_Modified(512, 512, save_mask=save_mask)
        self.r2 = ResidualBlock3d(512, 1024, p=0.3)
        self.att3 = AttentionModule_Modified(1024, 1024, save_mask=save_mask)

        self.out_conv1 = ResidualBlock3d(1024, 2048, p=0.3)
        self.out_linear = nn.Linear(20, 1)

        self.out_fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.out_fc2 = nn.Linear(1024, out_ch)

        # initilization
        # nn.init.ones_(self.out_linear.weight)
        # nn.init.zeros_(self.out_linear.bias)


    def forward(self, x):
        while x.dim() < 5:
            x = x.unsqueeze(0)
        x = self.in_conv1(x)

        # Construct slice weight
        x_w = self.in_sw(x)
        # print("x_w0: {}".format(x_w.shape))
        x_w = F.avg_pool3d(x_w,kernel_size=x_w.shape[-3:]).squeeze()
        # print("x_w1: {}".format(x_w.shape))
        x_w = torch.sigmoid(x_w) + 0.5
        if self.save_weight:
            self.x_w = x_w.data.cpu()
        # print("x_w2: {}".format(x_w.shape))

        # Permute the axial dimension to the last
        x = F.max_pool3d(x, [1, 2, 2], stride=[1, 2, 2]).permute([1, 3, 4, 0, 2])
        # print("x_premult: {}".format(x.shape))
        x_shape = x.shape
        new_shape = list(x_shape[:3]) + [x_shape[-2] * x_shape[-1]]
        x = x.reshape(new_shape)
        # print("x_reshape: {}".format(x.shape))
        x = x * x_w.view([-1]).expand_as(x)
        # print("x_expand: {}".format(x.shape))


        # Resume dimension
        x = x.view(x_shape).permute([3, 0, 4, 1, 2])
        # print("x_resume: {}".format(x.shape))
        x = self.in_conv2(x)
        # print("x_in_conv2: {}".format(x.shape))

        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)

        x = self.out_conv1(x)
        x = F.avg_pool3d(x, kernel_size=[1] + list(x.shape[-2:])).squeeze()
        # while x.dim() < 3:
        #     x = x.unsqueeze(0)
        # x = x.permute([1, 0, 2])
        # x_shape = x.shape
        # new_shape = [x_shape[0]] + [x_shape[1]*x_shape[2]]
        # x = x.reshape(new_shape)
        # x = x * x_w.expand_as(x)
        # x = x.view(x_shape).permute([1, 0, 2])
        #
        if x.dim() < 3:
            x = x.unsqueeze(0)
        x = x.permute([1, 0, 2])
        x = x * x_w.expand_as(x)
        x = x.permute([1, 0, 2])
        x = self.out_linear(x).squeeze(dim=-1)
        x = self.out_fc1(x)
        x = self.out_fc2(x)
        while x.dim() < 2:
            x = x.unsqueeze(0)
        return x


    def get_mask(self):
        #[[B,H,W,D],[B,H,W,D],[B,H,W,]]
        return [r.get_mask() for r in [self.att1, self.att2, self.att3]]

    def get_slice_attention(self):
        if not self.x_w is None:
            while self.x_w.dim() < 2:
                self.x_w = self.x_w.unsqueeze(0)
            return self.x_w
        else:
            print("Attention weight was not saved!")
            return None


class AttentionResidualNet_SW(nn.Module):
    def __init__(self, in_ch, out_ch, save_mask=False, save_weight=False):
        super(AttentionResidualNet_SW, self).__init__()

        self.save_weight=save_weight
        self.in_conv1 = Conv3d(in_ch, 64, stride=[1, 2, 2], padding=[1, 2, 2])
        self.in_sw = nn.Sequential(
            Conv3d(64, 128, stride=[1, 2, 2], kern_size=2),
            Conv3d(128, 20, stride=[1, 3, 3], kern_size=3),
        )
        self.x_w = None

        self.in_conv2 = ResidualBlock3d(64, 256)


        self.att1 = AttentionModule_Modified(256, 256, save_mask=save_mask)
        self.r1 = ResidualBlock3d(256, 512, p=0.3)
        self.att2 = AttentionModule_Modified(512, 512, save_mask=save_mask)
        self.r2 = ResidualBlock3d(512, 1024, p=0.3)
        self.att3 = AttentionModule_Modified(1024, 1024, save_mask=save_mask)
        self.out_conv1 = ResidualBlock3d(1024, 2048, p=0.3)
        self.out_linear = nn.Linear(20, 1)

        self.out_fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        self.out_fc2 = nn.Linear(1024, out_ch)

        # initilization
        # nn.init.ones_(self.out_linear.weight)
        # nn.init.zeros_(self.out_linear.bias)


    def forward(self, x):
        while x.dim() < 5:
            x = x.unsqueeze(0)
        x = self.in_conv1(x)

        # Construct slice weight
        x_w = self.in_sw(x)
        x_w = F.avg_pool3d(x_w,kernel_size=x_w.shape[-3:]).squeeze()
        x_w = (torch.sigmoid(x_w) - 0.5) * 10 + 5. # make the range larger.
        if self.save_weight:
            self.x_w = x_w.data.cpu()

        # Permute the axial dimension to the last
        x = F.max_pool3d(x, [1, 2, 2], stride=[1, 2, 2]).permute([1, 3, 4, 0, 2])
        x_shape = x.shape
        new_shape = list(x_shape[:3]) + [x_shape[-2] * x_shape[-1]]
        x = x.reshape(new_shape)
        x = x * x_w.view([-1]).expand_as(x)


        # Resume dimension
        x = x.view(x_shape).permute([3, 0, 4, 1, 2])
        x = self.in_conv2(x)

        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)

        x = self.out_conv1(x)
        x = F.max_pool3d(x, kernel_size=[1] + list(x.shape[-2:])).squeeze()

        if x.dim() < 3:
            x = x.unsqueeze(0)
        x = x.permute([1, 0, 2])
        x = x * x_w.expand_as(x)
        x = x.permute([1, 0, 2])
        x = self.out_linear(x).squeeze(dim=-1)
        x = self.out_fc1(x)
        x = torch.sigmoid(x)
        x = self.out_fc2(x)
        while x.dim() < 2:
            x = x.unsqueeze(0)
        return x


    def get_mask(self):
        #[[B,H,W,D],[B,H,W,D],[B,H,W,]]
        return [r.get_mask() for r in [self.att1, self.att2, self.att3]]

    def get_slice_attention(self):
        if not self.x_w is None:
            while self.x_w.dim() < 2:
                self.x_w = self.x_w.unsqueeze(0)
            return self.x_w
        else:
            print("Attention weight was not saved!")
            return None
