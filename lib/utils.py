# Import dependencies 
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Set root path
root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Pytorch custom dataset class
class PytorchDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform if transform is not None else lambda x: x

    def __len__(self):
        return len(self.data[list(self.data)[0]])

    def __getitem__(self, idx):
        items_dict = {}
        for label in self.data.keys():
            item = self.data[label][idx]
            if label == "img":
                if item.ndim == 4: # video
                    item = torch.stack([self.transform(elem) for elem in item], dim=0)
                else: # single image
                    item = self.transform(item)
            item = torch.Tensor(item)
            items_dict[label] = item # ADDITION
        return items_dict

# Turn the npz dataset into three ytorch datasets: train, validation and test
def load_dataset(file, transform=None, datapoints=0):
    npz_file = np.load(os.path.join(root_path, "datasets", file))
    data = {key: npz_file[key] for key in npz_file.files}

    _key = [key for key in list(data) if "train_" in key][0]
    train_size = len(data[_key])
    if datapoints == 0:
        datapoints = train_size

    train_set = PytorchDataset({key[6:]: data[key][:datapoints] for key in data.keys() if "train_" in key}, transform=transform)
    valid_set = PytorchDataset({key[6:]: data[key] for key in data.keys() if "valid_" in key}, transform=transform)
    test_set  = PytorchDataset({key[5:]: data[key] for key in data.keys() if "test_" in key}, transform=transform)
    return train_set, valid_set, test_set


def bvecmat(x, y):
    assert x.dim() == 2 and y.dim() == 3
    return torch.bmm(x.unsqueeze(1), y).squeeze(1)


def tc2np(data):
    if type(data) == torch.Tensor:
        return data.detach().cpu().numpy()
    elif type(data) == dict:
        return {k:(v.detach().cpu().numpy() if type(v) == torch.Tensor else v) for k, v in data.items()}
    elif type(data) == list:
        return [v.detach().cpu().numpy() if type(v) == torch.Tensor else v for v in data]
    else:
        return data


def frame_seq_to_ch(frames):
    # frames has shape [B,T,C,W,H]
    assert len(frames.shape) == 5
    return torch.cat(torch.unbind(frames, dim=1), dim=1)


def collect_variables(names, obj, local_vars):
    var_dict = {}
    for name in names:
        if name[:4] == "self":
            try: 
                var_dict[name[5:]] = vars(obj)[name[5:]].detach().cpu().numpy()
            except KeyError:
                var_dict[name[5:]] = vars(obj)["_parameters"][name[5:]].detach().cpu().numpy()
            
        else:
            var_dict[name] = local_vars[name].detach().cpu().numpy()
    return var_dict 


def collect_by_name(names, vars):
    var_dict = {name: vars[name] for name in names}
    return var_dict 


def np2tc(array):
    return torch.from_numpy(array).float().to(device)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temp):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temp, dim=-1)


def gumbel_softmax(logits, temp, hard=True):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temp)
    if not hard:
        return y, y
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1]).to(device)
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y, y


def parse_arglist(argstr):
    def parse_elem(elem):
        if elem.lower() in ["true", "false"]:
            return elem.lower() == "true"
        else:
            try:
                return int(elem)
            except ValueError:
                try:
                    return float(elem)
                except ValueError:
                    return elem
    if argstr == "":
        return {}
    args = [a.split("=") for a in argstr.split(",")]
    args = {k: parse_elem(v) for k, v in args}
    return args


def load_encoder(model, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.embedder._encoder.load_state_dict({k[len('embedder._encoder.'):]:v 
                                             for k,v in checkpoint["model_state_dict"].items() if "_encoder" in k})

    
def load_pytorch_datasets(file, transform=None, datapoints=0):
    npz_file = np.load(os.path.join("datasets", file))
    data = {key: npz_file[key] for key in npz_file.files}

    _key = [key for key in list(data) if "train_" in key][0]
    train_size = len(data[_key])
    if datapoints == 0:
        datapoints = train_size

    train_set = PytorchDataset({key[6:]: data[key][:datapoints] for key in data.keys() if "train_" in key}, transform=transform)
    valid_set = PytorchDataset({key[6:]: data[key] for key in data.keys() if "valid_" in key}, transform=transform)
    test_set  = PytorchDataset({key[5:]: data[key] for key in data.keys() if "test_" in key}, transform=transform)
    return train_set, valid_set, test_set


def plot_latent_spaces(states, true_states, name):
    true_angle = tc2np(true_states)
    states = states.cpu()

    true_angle = (true_angle-true_angle.min(axis=0, keepdims=True))/(true_angle.max(axis=0, keepdims=True)-true_angle.min(axis=0, keepdims=True))
    true_angle = (true_angle-0.5)*0.8 + 0.5
    colors = np.concatenate([true_angle, 0.2*np.ones_like(true_angle)[:,:1]], axis=1)

    fig, ax = plt.subplots()
    ax.scatter(states[:,0], states[:,1], c=colors, s=16, label="Manifold")
    ax.set_aspect(1 / ax.get_data_ratio())
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(width=2, color="grey", labelsize=14, labelcolor="#484848")
    for axis in ['left','bottom']: ax.spines[axis].set_linewidth(2); ax.spines[axis].set_color("grey")
    plt.xlabel(r'$X_1$'); plt.ylabel(r'$X_2$')
    fig.savefig('./experiments/figures/%s.pdf'%name, bbox_inches='tight')
    plt.show()


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape

    bordered = 0.5*np.ones([nindex, height+2, width+2, intensity])
    for i in range(nindex):
        bordered[i,1:-1,1:-1,:] = array[i]

    array = bordered
    nindex, height, width, intensity = array.shape

    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def style_plot(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(width=2, color="grey", labelsize=14, labelcolor="#484848")
    for axis in ['left','bottom']: 
        ax.spines[axis].set_linewidth(2); 
        ax.spines[axis].set_color("grey")
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))


def dict_dataset_split(train_set_size, valid_set_size, test_set_size, data):

    dataset = {}
    for key in data.keys():
        dataset["train_"+key] = data[key][:train_set_size]
        dataset["valid_"+key] = data[key][train_set_size:train_set_size+valid_set_size]
        dataset["test_"+key] = data[key][train_set_size+valid_set_size:]

    return dataset