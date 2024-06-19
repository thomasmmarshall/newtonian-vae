import numpy as np
import torch as tc
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from lib.blocks import *
from lib.schedulers import *
from lib.utils import tc2np, bvecmat, frame_seq_to_ch

from abc import abstractmethod
device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")


class Embedder(nn.Module):

    @abstractmethod
    def forward(self, inputs, train=False):
        # Compute action and collect other outputs
        pass

    @abstractmethod
    def compute_losses(self, input_data, target_data, outputs):
        pass


class VAE(Embedder):
    def __init__(self, img_shape, action_dim, state_dim, 
                 window=1, kl_reg=1.0, divider=1, decoder="ha"):

        super().__init__()

        self.state_dim = state_dim
        self.img_shape = img_shape # [C,W,H]
        self.img_size = np.prod(self.img_shape)
        self.window = window
        if type(kl_reg) == str:
            self.kl_scheduler = LinearScheduler(kl_reg)
        else:
            self.kl_scheduler = ConstantScheduler(kl_reg)

        self._encoder = nn.Sequential(
                            HaConvNet(3*self.window, divider=divider),
                            nn.Linear(1024//divider, 2*state_dim)
                           )
        if decoder == "ha":
            self._decoder = HaDeconvNet(state_dim, 3*self.window, divider=divider)
        elif decoder == "broadcast":
            self._decoder = BroadcastDecoder(state_dim, 3*self.window, divider=divider)

    def prior(self, shape):
        shape = tuple(shape)
        mean = tc.zeros(shape).to(device)
        std = tc.ones(shape).to(device)
        return Normal(mean, std)

    def posterior(self, frame):
        output = self._encoder(frame)
        mean, logstd = tc.chunk(output, 2, 1)
        return Normal(mean, logstd.exp())

    def forward(self, input_data, train=False):
        # frames: [B,T,C,W,H]
        # actions: [B,T,act_dim]
        frames = input_data["img"]
        actions = input_data["act"]

        # Last index is the current frame
        frame_window = frame_seq_to_ch(frames[:,-self.window:])
        posterior_dist = self.posterior(frame_window)
        if train:
            posterior_sample = posterior_dist.rsample()
        else:
            posterior_sample = posterior_dist.mean

        prior = self.prior([frames.shape[0], self.state_dim])

        recons = self.decode(posterior_sample)

        return {"posterior": posterior_dist,
                "posterior_samples": posterior_sample,
                "prior": prior,
                "recons": recons}

    def decode(self, state):
        return self._decoder(state)

    def compute_losses(self, input_data, outputs, epoch):
        frames = input_data["img"][:,-self.window:]
        frames = frame_seq_to_ch(frames)
        rec_loss = 0.5*((frames-outputs["recons"])**2).sum(dim=[-1,-2,-3]).mean()

        _kl = kl_divergence(outputs["posterior"], outputs["prior"])
        prior_kl = _kl.sum(dim=1).mean() if len(_kl.shape) == 2 else _kl.mean()
        kl_reg = self.kl_scheduler.get_value(epoch)
        return {"rec": rec_loss, "prior_kl": prior_kl*kl_reg} # multiply kl_reg

    def process_outputs(self, input_data, outputs, epoch, path):
        pass


class BaseWindowE2C(VAE):
    # Assumes posterior inference doesn't depend on previous state
    # For now also assumes inference is made based on a single frame

    def __init__(self, img_shape, action_dim, state_dim, 
                 window=1, kl_reg=1.0, divider=1, decoder="ha"): 
        super().__init__(img_shape, action_dim, state_dim, window, kl_reg, divider, decoder)
        self.action_dim = action_dim

    def prior_transition(self, prev_dist, action):
        raise NotImplementedError

    def forward(self, input_data, train=False):
        # frames: [B,window,C,W,H]
        # actions: [B,window,act_dim]
        frames = input_data["img"]
        actions = input_data["act"]

        vae_outputs = super().forward(input_data, train)

        prev_frames = frame_seq_to_ch(frames[:,-self.window-1:-1])
        prev_posterior_dist = self.posterior(prev_frames)
        trans_prior = self.prior_transition(prev_posterior_dist, actions[:,-2])

        return {"trans_prior": trans_prior, **vae_outputs}

    def compute_losses(self, input_data, outputs, epoch):
        vae_losses = super().compute_losses(input_data, outputs, epoch)
        trans_kl = kl_divergence(outputs["posterior"], outputs["trans_prior"]).sum(dim=1).mean()

        return {"trans_kl": trans_kl, "rec": vae_losses["rec"]}


class NewtonianVAE(BaseWindowE2C):
    def __init__(self, img_shape, action_dim, state_dim, 
                 rank="diag", dt=0.1, steps=1, window=1, kl_reg=1.0, divider=1, decoder="ha"): 
        super().__init__(img_shape, action_dim, state_dim, window, kl_reg, divider, decoder)
        self.rank = rank
        self.steps = steps

        self._transition_matrices = MLP(state_dim+action_dim, state_dim*(2*state_dim+action_dim), 2, 16, nn.ReLU())

        self._prior_vel_logstd = nn.Parameter(tc.tensor(0.0))
        self._post_vel_logstd = nn.Parameter(tc.tensor(0.0))
        self._prior_pos_logstd = nn.Parameter(tc.tensor(np.log(0.1)))
        self._logdt = tc.tensor(np.log(dt)).to(device)

    def forward(self, input_data, train=False):
        # frames: [B,window,C,W,H]
        # actions: [B,window,act_dim]
        
        frames = input_data["img"]
        actions = input_data["act"]
        steps = self.steps
        
        #print('Steps', steps)
        #print('Frame shape', frames.shape)
        #print('Action shape', actions.shape)

        curr_frame = frames[:,-1]
        curr_pos_post = self.posterior(curr_frame)
        curr_pos_post_sample = curr_pos_post.rsample()

        prev_frame = frames[:,-1-steps]
        prev_pos_post = self.posterior(prev_frame)
        prev_pos_post_sample = prev_pos_post.rsample()
        
        pprev_frame = frames[:,-2-steps]
        pprev_pos_post = self.posterior(pprev_frame)
        pprev_pos_post_sample = pprev_pos_post.rsample()

        prev_vel = (prev_pos_post_sample - pprev_pos_post_sample)/self._logdt.exp()

        _pos = prev_pos_post_sample
        _vel = prev_vel
        for t in range(steps):
            trans_pos_prior, trans_vel = self.prior_transition(_pos, _vel, actions[:,-1-steps+t])
            _pos = trans_pos_prior.mean
            _vel = trans_vel

        prior = self.prior(curr_pos_post_sample.shape)
        recons = self.decode(prev_pos_post_sample) 
        next_recons = self.decode(trans_pos_prior.rsample())

        return {"trans_pos_prior": trans_pos_prior,
                "next_pos_posterior": curr_pos_post,
                "pos_posterior": prev_pos_post,
                "prior": prior,
                "recons": recons,
                "next_recons": next_recons}

    def prior_transition(self, pos, vel, action):
        
        dt = self._logdt.exp()
        matrices = self._transition_matrices(tc.cat([pos, action], dim=1))

        A, B, C = tc.split(matrices, [self.state_dim**2, self.action_dim*self.state_dim, self.state_dim**2], dim=1)
        A = A.reshape([-1, self.state_dim, self.state_dim])
        B = B.reshape([-1, self.action_dim, self.state_dim])
        C = C.reshape([-1, self.state_dim, self.state_dim]) 

        if self.rank == "diag":
            next_vel = vel + dt*(-vel*C[:,0].exp()-pos*A[:,0].exp()+action*B[:,0].exp())
        elif self.rank == "full":
            next_vel = vel + dt*(bvecmat(vel,C)+bvecmat(pos,A)+bvecmat(action,B))
        else:
            raise NotImplementedError

        next_pos = pos + dt*next_vel
        return Normal(next_pos, self._prior_pos_logstd.exp()), next_vel

    def compute_losses(self, input_data, outputs, epoch):
        frames = input_data["img"][:,-2]
        next_frames = input_data["img"][:,-1]

        #prior_kl = kl_divergence(outputs["pos_posterior"], outputs["prior"]).sum(dim=1).mean()
        trans_pos_kl = kl_divergence(outputs["next_pos_posterior"], outputs["trans_pos_prior"]).sum(dim=1).mean()
        #trans_vel_kl = kl_divergence(outputs["vel_posterior"], outputs["trans_vel_prior"]).sum(dim=1).mean()
        #rec_loss = 0.5*((frames-outputs["recons"])**2).sum(dim=[-1,-2,-3]).mean()
        next_rec_loss = 0.5*((next_frames-outputs["next_recons"])**2).sum(dim=[-1,-2,-3]).mean()

        kl_reg = self.kl_scheduler.get_value(epoch)
        return {"trans_kl": kl_reg*trans_pos_kl,  #pos
                #"trans_vel_kl": trans_vel_kl,
                #"prior_kl": 0.0*prior_kl,
                #"rec": 0.0*rec_loss,
                "next_rec": next_rec_loss}
        

    def __init__(self, img_shape, action_dim, state_dim, window=1, kl_reg=1.0, divider=1, decoder="ha"): 
        super().__init__(img_shape, action_dim, state_dim, window, kl_reg, divider, decoder)

        self._transition_matrices = MLP(state_dim+action_dim, state_dim*(state_dim+action_dim+1), 2, 16, nn.ReLU())
        self._prior_logstd = nn.Parameter(tc.tensor(np.log(0.1)))

    def forward(self, input_data, train=False):
        # frames: [B,window,C,W,H]
        # actions: [B,window,act_dim]
        frames = input_data["img"].float()
        actions = input_data["act"].float()

        curr_frames = frame_seq_to_ch(frames[:,-self.window:])
        curr_post = self.posterior(curr_frames)
        curr_post_sample = curr_post.rsample()

        prev_frames = frame_seq_to_ch(frames[:,-self.window-1:-1])
        prev_post = self.posterior(prev_frames)
        prev_post_sample = prev_post.rsample()

        #pprev_frames = frame_seq_to_ch(frames[:,-self.window-2:-2])
        #pprev_post = self.posterior(pprev_frames)
        #pprev_post_sample = pprev_post.rsample()


        trans_prior = self.prior_transition(prev_post_sample, actions[:,-2])
        #trans_prior = self.prior_transition(pprev_post_sample, actions[:,-3])
        #trans_prior = self.prior_transition(trans_prior.mean, actions[:,-2])

        prior = self.prior(curr_post_sample.shape)
        recons, next_recons = tc.chunk(self.decode(tc.cat([prev_post_sample, trans_prior.rsample()], dim=0)), 2, 0)
        #recons = self.decode(curr_pos_post_sample)
        return {"trans_prior": trans_prior,
                "next_posterior": curr_post,
                "posterior": prev_post,
                "prior": prior,
                "recons": recons,
                "next_recons": next_recons}

    def prior_transition(self, state, action):
        
        matrices = self._transition_matrices(tc.cat([state, action], dim=1))

        A, B, C = tc.split(matrices, [self.state_dim**2, self.action_dim*self.state_dim, self.state_dim], dim=1)
        A = A.reshape([-1, self.state_dim, self.state_dim])
        B = B.reshape([-1, self.action_dim, self.state_dim])
        C = C.reshape([-1, self.state_dim])

        next_state = bvecmat(state,A)+bvecmat(action,B) + C
        return Normal(next_state, self._prior_logstd.exp())

    def compute_losses(self, input_data, outputs, epoch):
        frames = frame_seq_to_ch(input_data["img"][:,-self.window-1:-1])
        next_frames = frame_seq_to_ch(input_data["img"][:,-self.window:])

        prior_kl = kl_divergence(outputs["posterior"], outputs["prior"]).sum(dim=1).mean()
        trans_kl = kl_divergence(outputs["next_posterior"], outputs["trans_prior"]).sum(dim=1).mean()
        rec_loss = 0.5*((frames-outputs["recons"])**2).sum(dim=[-1,-2,-3]).mean()
        next_rec_loss = 0.5*((next_frames-outputs["next_recons"])**2).sum(dim=[-1,-2,-3]).mean()

        kl_reg = self.kl_scheduler.get_value(epoch)
        return {"trans_kl": kl_reg*trans_kl, 
                "prior_kl": kl_reg*prior_kl,
                "rec": rec_loss,
                "next_rec": next_rec_loss}