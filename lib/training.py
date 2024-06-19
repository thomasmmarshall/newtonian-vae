# Import dependencies
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import PIL.Image

toTensor = transforms.ToTensor()


from lib.utils import load_dataset, parse_arglist, plot_latent_spaces
from lib.models import *

from dm_control.suite.wrappers import pixels
from control.dm_control import suite

import matplotlib.pyplot as plt
from matplotlib import rc
import moviepy.editor as mpy
from torchvision.utils import save_image
from importlib import import_module

# Enable TeX
rc("font", **{"size": 14, "family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EmbedderAgent(nn.Module):
    def __init__(self, img_shape, action_dim, state_dim, window=1):
        super().__init__()
        self.img_shape = img_shape
        self.img_size = np.prod(self.img_shape)
        self.state_dim = state_dim

    def predict(self, input_data):
        raise NotImplementedError

    def forward(self, input_data, train=False):
        embedder_outputs = self.embedder(input_data, train=train)
        return embedder_outputs

    def compute_losses(self, input_data, outputs, epoch):
        embedder_losses = self.embedder.compute_losses(input_data, outputs, epoch)
        return embedder_losses

    def process_outputs(self, input_data, outputs, epoch, path):
        self.embedder.process_outputs(input_data, outputs, epoch, path)
        plot_recons_frames(outputs["recons"][:100, -3:], path)
        plot_frames(input_data["img"][:100, -1], path)


def plot_recons_frames(frames, path):
    save_image(
        frames, os.path.join(path, "recs.jpg"), nrow=int(np.sqrt(frames.shape[0]))
    )


def plot_frames(frames, path):
    save_image(
        frames, os.path.join(path, "frames.jpg"), nrow=int(np.sqrt(frames.shape[0]))
    )


def get_embedder_class(name):
    return getattr(import_module("lib.models"), name)


def plot_samples(state_dim, decode, path):
    # Assumes the prior is always Normal(0.0, 1.0)
    sample_states = tc.randn(25, state_dim).to(device)
    generated_images = decode(sample_states)[:, -3:]
    save_image(generated_images, os.path.join(path, "vae_samples.jpg"), nrow=5)


class VAEAgent(EmbedderAgent):
    def __init__(self, img_shape, action_dim, state_dim, embedder, **kwargs):
        super().__init__(img_shape, action_dim, state_dim)
        self.embedder = get_embedder_class(embedder)(
            img_shape, action_dim, state_dim, **kwargs
        )

    def process_outputs(self, input_data, outputs, epoch, path):
        super().process_outputs(input_data, outputs, epoch, path)
        plot_samples(self.state_dim, self.embedder.decode, path)


class Experiment:
    def __init__(self, args, train=False):

        # Interpret args
        self.args = args
        self.static = True
        self.pad = 3
        model_args = (
            "action_dim=%s,embedder=%s,state_dim=%s,steps=%s,kl_reg=%s,divider=%s"
            % (
                self.args["action_dim"],
                self.args["embedder"],
                self.args["state_dim"],
                self.args["steps"],
                self.args["kl_reg"],
                self.args["divider"],
            )
        )
        if self.args["embedder"] == "NewtonianVAE":
            model_args += ",rank=%s" % self.args["rank"] + ",dt=%s" % self.args["dt"]
        self.model_args = parse_arglist(model_args)

        if train:
            # Load the dataset
            train_set, valid_set, test_set = load_dataset(
                self.args["dataset"], transform=transforms.ToTensor()
            )

            # Create the dataloaders
            self.train_loader = DataLoader(
                train_set,
                batch_size=self.args["batch_size"],
                shuffle=True,
                persistent_workers=True,
                num_workers=2,
            )
            self.valid_loader = DataLoader(
                valid_set, batch_size=self.args["batch_size"], shuffle=True
            )
            self.test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

        # Load the model to train
        self.model = VAEAgent(self.args["input_shape"], **self.model_args).to(device)

        # Set up the optimiser
        self.optimiser = Adam(self.model.parameters(), lr=self.args["lr"])

        print("\nModel Description\n", self.model, "\n")

    def load_model(self):
        if device != "cuda:0":
            checkpoint = torch.load(
                "experiments/" + self.args["experiment"] + ".pth",
                map_location=torch.device("cpu"),
            )
        else:
            checkpoint = torch.load("experiments/" + self.args["experiment"] + ".pth")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]

    def save_model(self):
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimiser.state_dict(),
            },
            "experiments/" + self.args["experiment"] + ".pth",
        )

    def _process_time_series(self, data):
        if self.static:
            state_seq = data["state"]
            action_seq = data["act"]

            state_dim = state_seq.shape[-1]
            action_dim = action_seq.shape[-1]
            batch_size = state_seq.shape[0]
            seq_len = state_seq.shape[1]

            state_list = []
            action_list = []

            for t in range(seq_len - self.pad):
                state_list.append(state_seq[:, t : t + self.pad + 1])
                action_list.append(action_seq[:, t : t + self.pad + 1])

            state = torch.stack(state_list, dim=1).reshape(
                [batch_size * (seq_len - self.pad), self.pad + 1, state_dim]
            )
            action = torch.stack(action_list, dim=1).reshape(
                [batch_size * (seq_len - self.pad), self.pad + 1, action_dim]
            )
            data["state"] = state
            data["act"] = action

            if "img" in data.keys():
                frame_seq = data["img"]
                img_shape = list(frame_seq.shape[-3:])
                frame_list = []

                for t in range(seq_len - self.pad):
                    frame_list.append(frame_seq[:, t : t + self.pad + 1])
                frame = torch.stack(frame_list, dim=1).reshape(
                    [batch_size * (seq_len - self.pad), self.pad + 1] + img_shape
                )
                data["img"] = frame

            return data
        else:
            return data

    def train(self):

        # Loop for epochs
        for epoch in range(self.args["epochs"]):
            self.epoch = epoch

            # Loop for every sample in the dataset
            epoch_loss = [0.0, 0.0]
            for i, data in enumerate(self.train_loader):

                # Get the sequence
                data = {key: data[key].to(device).float() for key in data.keys()}

                # Calculate the loss from a forward pass
                data = self._process_time_series(data)
                outputs = self.model.forward(data, train=True)
                losses = self.model.compute_losses(data, outputs, epoch)
                loss, train_metrics = sum(losses.values()), losses

                # Backward on the computational graph to find the gradients
                self.optimiser.zero_grad()
                loss.backward()

                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000.0)

                # Apply the gradient update using the optimiser
                self.optimiser.step()

                metrics = tc2np(train_metrics)
                if self.args["embedder"] == "VAE":
                    rec, kl = metrics["rec"], metrics["prior_kl"]
                else:
                    rec, kl = metrics["next_rec"], metrics["trans_kl"]

                epoch_loss[0] += rec / len(self.train_loader)
                epoch_loss[1] += kl / len(self.train_loader)

                if i == len(self.train_loader) - 1:
                    print(
                        "Epoch {:03d}".format(epoch)
                        + "\tRec: {:.4f}".format(epoch_loss[0])
                        + "\tKL: {:.6f}".format(epoch_loss[1])
                        + "\t NELBO: {:.4f}".format(epoch_loss[0] + epoch_loss[1]),
                        end="\r",
                    )

                elif i % 100 == 0:
                    # Update the progress bar
                    # metrics = tc2np(train_metrics)

                    print(
                        "Epoch {:03d}".format(epoch)
                        + ":\tBatch {:04d}".format(i)
                        + "/"
                        + "{:04d}".format(len(self.train_loader))
                        + "\tRec: {:.4f}".format(rec)
                        + "\tKL: {:.6f}".format(kl),
                        end="\r",
                    )

            if epoch % 10 == 0:
                print("")

            # Save the progress
            if self.epoch % 10 == 0:
                self.save_model()
                # self.visualise()

    def visualiseLatentSpace(self):

        # Create the dataloaders
        self.model.eval()

        # Prepare the test iter
        data = next(iter(self.test_loader))
        data["img"] = data["img"].float()
        data["state"] = data["state"].float()
        data["act"] = data["act"].float()

        # Calculate the state vectors for each frame
        with torch.no_grad():
            true_states = data["state"][:, :, :2].reshape([-1, 2])
            frames = data["img"].reshape([-1, 3, 64, 64]).to(device)
            states = self.model.embedder.posterior(frames).mean

        # Plot correlation with the true states
        plot_latent_spaces(
            states, true_states, name="latentspace_%s" % (self.args["experiment"])
        )

    def verifyControl(self, domain, goal):

        self.model.eval()

        # Start environment

        env = suite.load(
            domain_name=domain, task_name="easy", task_kwargs=dict(random=3)
        )

        env = pixels.Wrapper(
            env,
            pixels_only=False,
            render_kwargs=dict(camera_id=0, height=240, width=240),
        )

        # Store and return a single episode
        episode_frames = []
        episode_states = []
        time_step = env.reset()
        goal_state = np.array(goal)

        with torch.no_grad():

            for t in range(100):
                frame = time_step.observation["pixels"]
                im = PIL.Image.fromarray(frame)
                frame = np.array(im.resize((64, 64), PIL.Image.Resampling.BICUBIC))

                state = self.model.embedder.posterior(toTensor(frame)[None, :]).mean
                state = tc2np(state)[0]
                action = 0.2 * (goal_state - state)

                # Collect
                episode_frames.append(frame)
                episode_states.append(state)

                time_step = env.step(action)
                # print(time_step)

            clip = mpy.ImageSequenceClip(episode_frames, fps=10)
            clip.write_gif("test.gif")

        episode_frames = np.array(episode_frames)
        episode_states = np.array(episode_states)

        plt.scatter(
            episode_states[:, 0],
            episode_states[:, 1],
            c=np.stack([np.arange(100) / 100, np.zeros(100), np.ones(100)], axis=1),
            label="Trajectory",
        )
        plt.scatter(goal_state[:1], goal_state[1:], c="green", label="Goal state")
        plt.xlabel(r"$X_1$")
        plt.ylabel(r"$X_2$")
        plt.legend()
        plt.savefig(
            "experiments/figures/" + self.args["experiment"] + "PControlTrajectory.pdf"
        )
        plt.show()
