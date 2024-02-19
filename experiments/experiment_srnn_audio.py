import os

import torch
import wandb
import rich

from torch.utils.data import DataLoader

import blvm
import blvm.data
import blvm.models
import blvm.utils
import blvm.utils.device

from blvm.data import BaseDataset
from blvm.data.batchers import DynamicTensorBatcher, ListBatcher
from blvm.data.datasets import DATASETS
from blvm.data.loaders import AudioLoader, NumpyLoader
from blvm.data.samplers.length_samplers import LengthEvalSampler, LengthTrainSampler
from blvm.data.transforms import Compose, Denormalize, MuLawDecode, MuLawEncode, Normalize, RandomSegment
from blvm.evaluation import Tracker
from blvm.utils.argparsing import str2bool
from blvm.utils.argparsers import parser
from blvm.utils.optimization import get_learning_rates_dict
from blvm.utils.rand import set_seed, get_random_seed
from blvm.training.annealers import CosineAnnealer
from blvm.training.restore import load_run, save_run
from blvm.utils.wandb import is_run_resumed, restore_run


parser.set_defaults(
    epochs=1000,
    batch_size=64,
    save_checkpoints=True,
    test_every=10,
    length_sampler=False,
    optimizer="Adam",
    lr=3e-4,
    lr_scheduler="MultiStepLR",
    lr_scheduler_kwargs=dict(milestones=[1500, 3000, 4500], gamma=0.1),
    max_grad_norm=3000.0,
    max_grad_value=1000.0,
    dataset="timit",
    project="benchmarks",
    entity=None,
)

model_group = parser.add_argument_group("model")
model_group.add_argument("--stack_frames", default=200, type=int, help="Number of audio frames to stack in feature vector")
model_group.add_argument("--hidden_size", default=512, type=int, help="dimensionality of hidden state in VRNN")
model_group.add_argument("--latent_size", default=256, type=int, help="dimensionality of latent state in VRNN")
model_group.add_argument("--residual_posterior", default=True, type=str2bool, help="residual parameterization of posterior")
model_group.add_argument("--smoothing", default=True, type=str2bool, help="smoothing or filtering inference model")
model_group.add_argument("--dropout", default=0.0, type=float, help="dropout")
model_group.add_argument("--input_coding", default="linear", type=str, choices=["mu_law", "linear"], help="input encoding")
model_group.add_argument("--num_bits", default=8, type=int, help="number of bits for DML and input")
model_group.add_argument("--random_segment_size", default=None, type=int, help="timesteps to subsample per training example")
model_group.add_argument("--likelihood", default="GMM", type=str, help="likelihood module")
model_group.add_argument("--num_mix", default=10, type=int, help="number of logistic mixture components")

model_group.add_argument("--beta_anneal_steps", default=50000, type=int, help="number of steps to anneal beta")
model_group.add_argument("--beta_start_value", default=0, type=float, help="initial beta annealing value")
model_group.add_argument("--free_nats_steps", default=0, type=int, help="number of steps to constant/anneal free bits")
model_group.add_argument("--free_nats_start_value", default=0.0625, type=float, help="free bits per timestep")

model_group.add_argument("--debug_steps", default=float("inf"), type=int, help="iters within each dataset for debugging")
model_group.add_argument("--split_eval", default=False, type=str2bool, help="If true, split evaluation sequences")

args = parser.parse_args()


if args.seed is None:
    args.seed = get_random_seed()

set_seed(args.seed)

args.batch_len = 16000 * args.batch_len if isinstance(args.batch_len, float) else args.batch_len

device = blvm.utils.device.get_device() if args.device == "auto" else torch.device(args.device)
torch.cuda.set_device(device)

dataset = DATASETS[args.dataset]

rich.print(vars(args))


tracker = Tracker(debug_epoch_break_steps=args.debug_steps)


encode_transform = []
decode_transform = []
if args.input_coding == "mu_law":
    encode_transform.append(MuLawEncode(bits=args.num_bits))
    decode_transform.append(MuLawDecode(bits=args.num_bits))

audio_loader = NumpyLoader("npz", key="samples", allow_pickle=True) if dataset.name == "timit_srnn" else AudioLoader(dataset.audio_ext)
if args.likelihood == "Gaussian" or args.likelihood == "GMM":
    # Global normalization of waveform with train set statistics
    ds = BaseDataset(source=dataset.train, modalities=[(audio_loader, Compose(*encode_transform), ListBatcher())], sort=False)
    mean, variance = ds.compute_statistics(num_workers=args.num_workers)
    encode_transform.append(Normalize(mean=mean, std=variance.sqrt()))
    decode_transform.append(Denormalize(mean=mean, std=variance.sqrt()))

if args.random_segment_size is not None:
    encode_transform_train = Compose(*[RandomSegment(args.random_segment_size), *encode_transform])
else:
    encode_transform_train = Compose(*encode_transform)

encode_transform = Compose(*encode_transform)
decode_transform = Compose(*decode_transform)

modalities_train = [(audio_loader, encode_transform_train, DynamicTensorBatcher())]
modalities_test = [(audio_loader, encode_transform, DynamicTensorBatcher())]


train_dataset = BaseDataset(source=dataset.train, modalities=modalities_train)
rich.print(train_dataset)
if args.batch_len:
    train_sampler = LengthTrainSampler(
        source=dataset.train,
        field=dataset.audio_length,
        batch_len=args.batch_len,
        max_pool_difference=16000 * 0.3,
        min_pool_size=512,
    )
    rich.print(train_sampler)
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
        source=valid_dataset if dataset.name == "timit_srnn" else source_name,
        field=dataset.audio_length,
        **batch_kwarg,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        collate_fn=valid_dataset.collate,
        num_workers=args.num_workers,
        batch_sampler=valid_sampler,
        pin_memory=True,
    )
    valid_test_dataloaders[source_name] = valid_loader
    rich.print(valid_dataset)
    rich.print(valid_sampler)


wandb.init(**vars(parser.parse_args_by_group().wandb), config=args)
wandb.save("*.pt")

model = blvm.models.SRNNAudio(
    likelihood=args.likelihood,
    input_size=args.stack_frames,
    hidden_size=args.hidden_size,
    latent_size=args.latent_size,
    dropout=args.dropout,
    residual_posterior=args.residual_posterior,
    num_mix=args.num_mix,
    num_bins=2 ** args.num_bits,
    smoothing=args.smoothing,
)

print(model)
x, x_sl = next(iter(train_loader))[0]
model.summary(input_data=x[:, :2*args.stack_frames], x_sl=torch.LongTensor([2*args.stack_frames] * x.size(0)), device="cpu")
model = model.to(device)
wandb.watch(model, log="all", log_freq=len(train_loader))

optimizer_class = getattr(torch.optim, args.optimizer)
optimizer = optimizer_class(model.parameters(), lr=args.lr, **args.optimizer_kwargs)

lr_scheduler_class = getattr(torch.optim.lr_scheduler, args.lr_scheduler)
lr_scheduler = lr_scheduler_class(optimizer, **args.lr_scheduler_kwargs)

scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

if is_run_resumed():
    directory = restore_run(os.environ.get("WANDB_RUN_ID"), "srnn", "blvm", exclude="media")
    model, optimizer, scaler, checkpoint = load_run(directory, model, optimizer, scaler, device=device)


beta_annealer = CosineAnnealer(anneal_steps=args.beta_anneal_steps, start_value=args.beta_start_value, end_value=1)
free_nats_annealer = CosineAnnealer(
    anneal_steps=args.free_nats_steps // 2,
    constant_steps=args.free_nats_steps // 2,
    start_value=args.free_nats_start_value,
    end_value=0,
)


for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl), metadata in tracker(train_loader):
        x = x.to(device)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            loss, metrics, outputs = model(x, x_sl, beta=beta_annealer.step(), free_nats=free_nats_annealer.step())

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_value_(model.parameters(), args.max_grad_value)

        scaler.step(optimizer)
        scaler.update()

        tracker.update(metrics)

    extra_log = get_learning_rates_dict(optimizer)
    lr_scheduler.step()

    model.eval()
    with torch.no_grad():
        if (epoch % args.test_every) == 0:
            for loader in valid_test_dataloaders.values():
                if not args.split_eval:
                    for (x, x_sl), metadata in tracker.steps(loader):
                        x = x.to(device, non_blocking=True)
                        with torch.cuda.amp.autocast(enabled=args.use_amp):
                            loss, metrics, output = model(x, x_sl)
                        tracker.update(metrics)
                else:
                    for (x, x_sl), metadata in tracker.steps(loader):
                        x = x.to(device, non_blocking=True)
                        d_0, a_0, z_0 = None, None, None
                        splits_x, splits_x_sl = model.split_sequence(x, x_sl, length=args.random_segment_size,)
                        for i, (xs, xs_sl) in enumerate(zip(splits_x, splits_x_sl)):
                            with torch.cuda.amp.autocast(enabled=args.use_amp):
                                loss, metrics, output = model.forward_split(xs, xs_sl, d_0=d_0, a_0=a_0, z_0=z_0)
                            tracker.update(metrics)
                            d_0, a_0, z_0 = output.d_n, output.a_n, output.z_n

            # Log reconstructions and samples
            mode = decode_transform(outputs.reconstructions_mode)
            mode = mode[:2].flatten(1).cpu().numpy().astype(float)
            mode = [wandb.Audio(r, caption=f"Reconstruction mode {i}", sample_rate=16000) for i, r in enumerate(mode)]

            sample = decode_transform(outputs.reconstructions)
            sample = sample[:2].flatten(1).cpu().numpy().astype(float)
            sample = [wandb.Audio(r, caption=f"Reconstruction {i}", sample_rate=16000) for i, r in enumerate(sample)]

            (x, x_sl), output = model.generate(n_samples=2, max_timesteps=128000 // args.stack_frames)
            x = decode_transform(x)
            x = x.flatten(1).cpu().numpy().astype(float)
            x = [wandb.Audio(r, caption=f"Sample {i}", sample_rate=16000) for i, r in enumerate(x)]

            extra_log.update(samples=x, reconstructions=sample, reconstructions_mode=mode)

            # Save model checkpoints
            if (
                args.save_checkpoints
                and wandb.run is not None
                and wandb.run.dir != "/"
                and dataset.test in tracker.accumulated_values
                and tracker.accumulated_values[dataset.test]["elbo"][-1] == tracker.best_values[dataset.test]["best_elbo"]
            ):
                save_run(
                    wandb.run.dir,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    tracker=tracker,
                )

        tracker.log(**extra_log)
