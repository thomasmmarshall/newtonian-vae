import torch as tc
import torch.nn as nn
import torch.nn.functional as F

device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, inp):
        return inp.reshape([inp.shape[0]]+self.shape)

def coord_grid(shape):
    x = tc.linspace(-1, 1, shape[0]).to(device)
    y = tc.linspace(-1, 1, shape[1]).to(device)
    x_grid, y_grid = tc.meshgrid(x, y, indexing='ij')
    grid = tc.cat([x_grid[None,...], y_grid[None,...]], dim=0)
    return grid


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_hidden_layers,
                 hidden_dim,
                 hidden_act,
                 output_act=None):
        super(MLP, self).__init__()
        if output_act is None:
            output_act = nn.Identity()
        seq = [nn.Linear(input_dim, hidden_dim), hidden_act, nn.LazyBatchNorm1d()]+\
            [nn.Linear(hidden_dim, hidden_dim), hidden_act, nn.LazyBatchNorm1d()]*(n_hidden_layers-1)+\
            [nn.Linear(hidden_dim, output_dim), output_act]
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class ConvNet(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch,
                 kernel_size,
                 n_hidden_layers,
                 hidden_ch,
                 hidden_act,
                 output_act=None,
                 input_kernel_size=None):
        super(ConvNet, self).__init__()
        if input_kernel_size is None:
            input_kernel_size = kernel_size
        if output_act is None:
            output_act = nn.Identity()
        seq = [nn.Conv2d(input_ch, hidden_ch, input_kernel_size, 1, input_kernel_size//2), hidden_act]+\
            [nn.Conv2d(hidden_ch, hidden_ch, kernel_size, 1, kernel_size//2), hidden_act]*(n_hidden_layers-1)+\
            [nn.Conv2d(hidden_ch, output_ch, kernel_size, 1, kernel_size//2), output_act]
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class DeconvDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 output_ch,
                 kernel_size,
                 n_hidden_layers,
                 hidden_ch,
                 hidden_act,
                 output_act=None,
                 input_kernel_size=None,
                 broadcast=True):
        super(DeconvDecoder, self).__init__()
        if input_kernel_size is None:
            input_kernel_size = kernel_size
        if output_act is None:
            output_act = nn.Identity

        self.broadcast = broadcast
        if self.broadcast:
            self.width = int(2**(n_hidden_layers+3))
            seq = [nn.Conv2d(input_dim+2, hidden_ch, kernel_size, padding=kernel_size//2), hidden_act()]+\
                [nn.Conv2d(hidden_ch, hidden_ch, kernel_size, padding=kernel_size//2), hidden_act()]*(n_hidden_layers-1)+\
                [nn.Conv2d(hidden_ch, output_ch, kernel_size, padding=kernel_size//2), output_act()]
            self.conv_seq = nn.Sequential(*seq)
        else:
            self.fc_seq = nn.Sequential(nn.Linear(input_dim, hidden_ch*4), nn.ReLU())
            seq = [nn.ConvTranspose2d(hidden_ch, hidden_ch, input_kernel_size, 2, padding=1), hidden_act()]+\
                [nn.ConvTranspose2d(hidden_ch, hidden_ch, kernel_size, 2, padding=1), hidden_act()]*n_hidden_layers+\
                [nn.ConvTranspose2d(hidden_ch, output_ch, kernel_size, 2, padding=1), output_act()]
            self.conv_seq = nn.Sequential(*seq)

    def forward(self, inp):
        if self.broadcast:
            x, y = tc.linspace(-1., 1., self.width).to(device), tc.linspace(-1., 1., self.width).to(device)
            x, y = tc.meshgrid(x, y, indexing='ij')
            x = x[None,None,:,:]
            y = y[None,None,:,:]
            x = x.repeat(inp.shape[0], 1, 1, 1)
            y = y.repeat(inp.shape[0], 1, 1, 1)
            inp = inp[:,:,None,None]
            inp = inp.repeat(1, 1, self.width, self.width)
            h = tc.cat([inp, x, y], dim=1)
            h = self.conv_seq(h)
        else:
            h = self.fc_seq(inp)
            h = h.reshape([h.shape[0], -1, 2, 2])
            h = self.conv_seq(h)
        return h


class HaConvNet(nn.Module):
    def __init__(self,
                 input_ch,
                 divider=1):
        super(HaConvNet, self).__init__()
        d = divider
        seq = [nn.Conv2d(input_ch+2, 32//d, 4, 2, 0), nn.ReLU(), #nn.LazyBatchNorm2d(),
               nn.Conv2d(32//d, 64//d, 4, 2, 0), nn.ReLU(), #nn.LazyBatchNorm2d(),
               nn.Conv2d(64//d, 128//d, 4, 2, 0), nn.ReLU(), #nn.LazyBatchNorm2d(),
               nn.Conv2d(128//d, 256//d, 4, 2, 0), nn.ReLU(), #nn.LazyBatchNorm2d(),
               nn.Flatten()]
        self.seq = nn.Sequential(*seq)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        grid = coord_grid([64,64])
        inputs = tc.cat([inputs, grid[None,...].repeat(batch_size, 1, 1, 1)], dim=1)
        return self.seq(inputs)


class HaDeconvNet(nn.Module):
    def __init__(self,
                 input_dim,
                 output_ch,
                 divider=1):
        super(HaDeconvNet, self).__init__()
        self.d = d = divider
        seq = [nn.ConvTranspose2d(1024//d, 128//d, 5, 2, 0), nn.ReLU(), #nn.LazyBatchNorm2d(),
               nn.ConvTranspose2d(128//d, 64//d, 5, 2, 0), nn.ReLU(), #nn.LazyBatchNorm2d(),
               nn.ConvTranspose2d(64//d, 32//d, 6, 2, 0), nn.ReLU(), #nn.LazyBatchNorm2d(),
               nn.ConvTranspose2d(32//d, output_ch, 6, 2, 0)]
        self.linear = nn.Sequential(nn.Linear(input_dim, 1024//d))
        self.deconv_seq = nn.Sequential(*seq)

    def forward(self, x):
        x = self.linear(x.float())
        x = x.reshape([x.shape[0], 1024//self.d, 1, 1])
        x = self.deconv_seq(x)
        return x


class JakabConvNet(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch):
        super().__init__()
        seq = [nn.Conv2d(input_ch, 32, 7, 1, 3), nn.ReLU(),
               nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
               nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
               nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(128, output_ch, 1, 1, 0)]
        self.seq = nn.Sequential(*seq)

    def forward(self, inputs):
        return self.seq(inputs)


class JakabDeconvNet(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch):
        super().__init__()
        seq = [nn.Conv2d(input_ch, 128, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
               nn.Upsample([32,32], mode="bilinear"),
               nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
               nn.Upsample([64,64], mode="bilinear"),
               nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(32, output_ch, 3, 1, 1)]
        self.seq = nn.Sequential(*seq)

    def forward(self, inputs):
        return self.seq(inputs)


class BroadcastEncoder(nn.Module):
    def __init__(self,
                 input_ch,
                 divider=1):
        super(BroadcastEncoder, self).__init__()
        d = divider
        seq = [nn.Conv2d(input_ch+2, 64//d, 4, 2, 0), nn.ReLU(),
               nn.Conv2d(64//d, 64//d, 4, 2, 0), nn.ReLU(),
               nn.Conv2d(64//d, 64//d, 4, 2, 0), nn.ReLU(),
               nn.Conv2d(64//d, 64//d, 4, 2, 0), nn.ReLU(),
               nn.Flatten()]
        self.seq = nn.Sequential(*seq)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        grid = coord_grid([64,64])
        inputs = tc.cat([inputs, grid[None,...].repeat(batch_size, 1, 1, 1)], dim=1)
        return self.seq(inputs)


class BroadcastDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 output_ch,
                 divider=1):
        super(BroadcastDecoder, self).__init__()
        self.d = d = divider
        seq = [nn.Conv2d(input_dim+2, 64//d, 5, 1, 2), nn.ReLU(),
               nn.Conv2d(64//d, 64//d, 5, 1, 2), nn.ReLU(),
               nn.Conv2d(64//d, 64//d, 5, 1, 2), nn.ReLU(),
               nn.Conv2d(64//d, output_ch, 5, 1, 2)]
        self.deconv_seq = nn.Sequential(*seq)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs = inputs[:,:,None,None]
        inputs = inputs.repeat(1,1,64,64)

        grid = coord_grid([64,64])
        inputs = tc.cat([inputs, grid[None,...].repeat(batch_size, 1, 1, 1)], dim=1)
        return self.deconv_seq(inputs)


class STN(nn.Module):
    def __init__(self, trg_size, ratio):
        super(STN, self).__init__()
        self.trg_size = trg_size
        self.ratio = ratio
        self.src_size = [s//ratio for s in trg_size] 

    def forward(self, x, pos, angle, scale):
        return self.stn(x, pos, angle, scale)

    def stn(self, x, pos, angle, scale):
        if angle.dim() == 2:
            angle = angle[:,0]
        if scale.dim() == 2:
            scale = scale[:,0]

        cos = tc.cos(angle)
        sin = tc.sin(angle)

        theta0 = cos/scale
        theta1 = -sin/scale
        theta2 = (cos*(self.trg_size[0]/2-pos[:,0])-\
            sin*(self.trg_size[0]/2-pos[:,1]))/self.src_size[0]/scale
        theta3 = sin/scale
        theta4 = cos/scale
        theta5 = (sin*(self.trg_size[0]/2-pos[:,0])+\
            cos*(self.trg_size[0]/2-pos[:,1]))/self.src_size[0]/scale
        theta = tc.stack([theta0, theta1, theta2, theta3, theta4, theta5], dim=1)

        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.shape[:2]+tuple(self.trg_size))
        x = F.grid_sample(x, grid)
        return x


class ReadSTN(nn.Module):
    # Angle not supported yet! Might not be correct

    def __init__(self, trg_size, ratio):
        super(ReadSTN, self).__init__()
        self.trg_size = trg_size
        self.ratio = ratio
        self.src_size = [s*ratio for s in trg_size] 

    def forward(self, x, pos, angle, scale):
        return self.stn(x, pos, angle, scale)

    def stn(self, x, pos, angle, scale):
        if angle.dim() == 2:
            angle = angle[:,0]
        if scale.dim() == 2:
            scale = scale[:,0]

        cos = tc.cos(angle)
        sin = tc.sin(angle)

        theta0 = cos*scale
        theta1 = -sin*scale
        theta2 = -(self.src_size[0]/2-pos[:,0])/self.trg_size[0]
        theta3 = sin*scale
        theta4 = cos*scale
        theta5 = -(self.src_size[0]/2-pos[:,1])/self.trg_size[0]
        theta = tc.stack([theta0, theta1, theta2, theta3, theta4, theta5], dim=1)

        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.shape[:2]+tuple(self.trg_size))
        x = F.grid_sample(x, grid)
        return x