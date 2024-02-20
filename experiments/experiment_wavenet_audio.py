import argparse
import logging
import os
from numpy import savez_compressed

import torch
import wandb
import rich

from torch.utils.data import DataLoader

import blvm.models

from blvm.data import BaseDataset
from blvm.data.batchers import DynamicTensorBatcher, ListBatcher
from blvm.data.datasets import DATASETS
from blvm.data.loaders import AudioLoader
from blvm.data.samplers.length_samplers import LengthEvalSampler, LengthTrainSampler
from blvm.data.transforms import Compose, Denormalize, MuLawDecode, Normalize, Quantize, RandomSegment, MuLawEncode, StackTensor
from blvm.evaluation.tracker import Tracker
from blvm.modules.distributions import CategoricalDense, DiagonalGaussianDense, DiagonalGaussianMixtureDense, DiscretizedLogisticMixtureDense
from blvm.training.restore import load_run
from blvm.utils.argparsing import str2bool
from blvm.utils.argparsers import parser
from blvm.utils.device import get_device
from blvm.utils.rand import set_seed, get_random_seed
from blvm.utils.wandb import is_run_resumed, restore_run


LOGGER = logging.getLogger(name=__file__)

parser.set_defaults(
    dataset="timit",
    lr=3e-4,
    epochs=3000,
    num_workers=8,
    save_checkpoints=True,
    project=blvm.WANDB_PROJECT,
    entity=None,
)

parser.add_argument("--n_layers", default=10, type=int, help="number of layers per stack")
parser.add_argument("--n_stacks", default=4, type=int, help="number of stacks")
parser.add_argument("--res_channels", default=64, type=int, help="number of channels in residual connections")
parser.add_argument("--kernel_size", default=2, type=int, help="kernel size for convolutions")
parser.add_argument("--base_dilation", default=2, type=int, help="base dilation for dilated convolutions (d**(l-1))")
parser.add_argument("--input_coding", default="mu_law", type=str, choices=["mu_law", "linear"], help="input encoding")
parser.add_argument("--input_embedding_dim", default=1, type=int, help="if not 1, embed input as vector of this dim")
parser.add_argument("--num_bits", default=16, type=int, help="number of bits for mu_law enc (note the data bits depth)")
parser.add_argument("--likelihood", default="DMoL", type=str, help="likelihood for the output p(x_t|x_t-1)")
parser.add_argument("--random_segment_size", default=None, type=int, help="timesteps to randomly subsample per example")
parser.add_argument("--n_stack_frames", default=1, type=int, help="frames to stack as input if input_coding is frames")
parser.add_argument("--split_eval", default=False, type=str2bool, help="If true, split evaluation sequences")
parser.add_argument("--generate_every", default=25, type=int, help="test every N epochs")

args = parser.parse_args()

if args.seed is None:
    args.seed = get_random_seed()

set_seed(args.seed)

args.batch_len = 16000 * args.batch_len if isinstance(args.batch_len, float) else args.batch_len

device = get_device() if args.device == "auto" else torch.device(args.device)
torch.cuda.set_device(device)

dataset = DATASETS[args.dataset]

rich.print(vars(args))


encode_transform = []
decode_transform = []
if args.input_coding == "mu_law":
    encode_transform.append(MuLawEncode(bits=args.num_bits))
    decode_transform.append(MuLawDecode(bits=args.num_bits))

if args.likelihood == "gaussian" or args.likelihood[:4] == "gmm-":
    ds = BaseDataset(source=dataset.train, modalities=[(AudioLoader(dataset.audio_ext), Compose(*encode_transform), ListBatcher())], sort=False)
    mean, variance = ds.compute_statistics(num_workers=args.num_workers)
    encode_transform.append(Normalize(mean=mean, std=variance.sqrt()))
    decode_transform.append(Denormalize(mean=mean, std=variance.sqrt()))

if args.input_embedding_dim > 1:
    encode_transform.append(Quantize(bits=args.num_bits))

if args.likelihood == "categorical":
    decode_transform.append()  # TODO Opposite of Quantize operation (Scale?)

if args.random_segment_size is not None:
    encode_transform_train = Compose(*[RandomSegment(args.random_segment_size), *encode_transform])
else:
    encode_transform_train = Compose(*encode_transform)

encode_transform = Compose(*encode_transform)
decode_transform = Compose(*decode_transform)

modalities_train = [(AudioLoader(dataset.audio_ext), encode_transform_train, DynamicTensorBatcher())]
modalities_test = [(AudioLoader(dataset.audio_ext), encode_transform, DynamicTensorBatcher())]

train_dataset = BaseDataset(source=dataset.train, modalities=modalities_train)
if args.batch_len:
    train_sampler = LengthTrainSampler(
        source=dataset.train,
        field=dataset.audio_length,
        batch_len=args.batch_len,
        max_pool_difference=16000 * 0.3,
        min_pool_size=512,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=train_dataset.collate,
        num_workers=args.num_workers,
        batch_sampler=train_sampler,
        pin_memory=True,
    )
else:
    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=train_dataset.collate,
        num_workers=args.num_workers,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
    )

valid_test_dataloaders = dict()
if args.split_eval and args.batch_size > 0:
    batch_kwarg = dict(batch_size=args.batch_size * 3)  # set batch size instead of length if random segmenting
else:
    batch_kwarg = dict(batch_len=args.batch_len or "max")
for source_name in [*dataset.valid_sets, *dataset.test_sets]:
    valid_dataset = BaseDataset(source=source_name, modalities=modalities_test)
    valid_sampler = LengthEvalSampler(
        source=source_name,
        field=dataset.audio_length,
        shuffle=True,
        **batch_kwarg
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        collate_fn=valid_dataset.collate,
        num_workers=args.num_workers,
        batch_sampler=valid_sampler,
        pin_memory=True,
    )
    valid_test_dataloaders[source_name] = valid_loader


if args.likelihood == "DMoL":
    likelihood = DiscretizedLogisticMixtureDense(args.res_channels, 1, num_mix=10, num_bins=2 ** args.num_bits)
elif args.likelihood == "Categorical":
    likelihood = CategoricalDense(args.res_chanels, 2 ** args.num_bits)
elif args.likelihood == "Gaussian":
    likelihood = DiagonalGaussianDense(args.res_channels, 1, initial_sd=1, epsilon=1e-4)
elif args.likelihood[:4] == "GMM-":
    num_mix = int(args.likelihood.split("-")[-1])
    likelihood = DiagonalGaussianMixtureDense(args.res_channels, 1, num_mix=num_mix, initial_sd=1, epsilon=1e-4)
else:
    raise ValueError(f"Unknown likelihood: {args.likelihood}")

model = blvm.models.WaveNet(
    likelihood=likelihood,
    n_layers=args.n_layers,
    n_stacks=args.n_stacks,
    in_channels=args.input_embedding_dim,
    res_channels=args.res_channels,
    base_dilation=args.base_dilation,
    kernel_size=args.kernel_size,
    num_bins=2 ** args.num_bits,
    n_stack_frames=args.n_stack_frames,
)

print(model)
with torch.cuda.amp.autocast(enabled=args.use_amp):
    (x, x_sl), metadata = next(iter(train_loader))
    model.summary(input_data=x, x_sl=x_sl)
model = model.to(device)
rich.print(model.receptive_field)

wandb.init(**vars(parser.parse_args_by_group().wandb), config=args)
wandb.save("*.pt")
wandb.watch(model, log="all", log_freq=len(train_loader))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

if is_run_resumed():
    directory = restore_run(wandb.run.id, wandb.run.project, wandb.run.entity, exclude="media")
    model, checkpoint, optimizer,  scaler = load_run(directory, model, optimizer, scaler)


tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl), metadata in tracker.steps(train_loader):
        x = x.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            loss, metrics, outputs = model(x, x_sl)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tracker.update(metrics)

    model.eval()
    with torch.no_grad():
        extra = dict()
        if ((epoch - 1) % args.test_every) == 0:
            for loader in valid_test_dataloaders.values():
                if not args.split_eval:
                    for (x, x_sl), metadata in tracker.steps(loader):
                        x = x.to(device, non_blocking=True)
                        with torch.cuda.amp.autocast(enabled=args.use_amp):
                            loss, metrics, outputs = model(x, x_sl)
                        tracker.update(metrics)
                else:
                    for (x, x_sl), metadata in tracker.steps(loader):
                        x = x.to(device, non_blocking=True)
                        splits_x, splits_x_sl = model.split_sequence(x, x_sl, length=args.random_segment_size)
                        for i, (xs, xs_sl) in enumerate(zip(splits_x, splits_x_sl)):
                            with torch.cuda.amp.autocast(enabled=args.use_amp):
                                loss, metrics, outputs = model.forward_split(xs, xs_sl, i_split=i)
                            tracker.update(metrics)

            pred = decode_transform(outputs.predictions)
            pred = [
                wandb.Audio(pred[i].cpu().flatten().numpy(), 16000, f"Reconstruction {i}")
                for i in range(min(pred.shape[0], 2))
            ]
            extra["predictions"] = pred

            if (
                args.save_checkpoints
                and wandb.run is not None
                and wandb.run.dir != "/"
                and dataset.test in tracker.accumulated_values
                and tracker.accumulated_values[dataset.test]["loss"][-1] == tracker.best_values[dataset.test]["best_loss"]
            ):
                model.save(wandb.run.dir)
                checkpoint = dict(
                    epoch=epoch,
                    best_loss=tracker.accumulated_values[dataset.test]["loss"][-1],
                    optimizer_state_dict=optimizer.state_dict(),
                    scaler_state_dict=scaler.state_dict(),
                )
                torch.save(checkpoint, os.path.join(wandb.run.dir, "checkpoint.pt"))
                print(f"Saved model checkpoint at {wandb.run.dir}")

        if (epoch % args.generate_every) == 0:
            x = model.generate(n_samples=2, n_frames=16000 * 8 // args.n_stack_frames)
            x = decode_transform(x)
            samples = [wandb.Audio(x[i].flatten().cpu().numpy(), 16000, f"Sample {i}") for i in range(2)]
            extra["samples"] = samples

        tracker.log(**extra)
