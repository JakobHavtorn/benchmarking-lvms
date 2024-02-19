import os

import torch
import wandb
import rich

from torch.utils.data import DataLoader

import blvm.models

from blvm.data import BaseDataset
from blvm.data.batchers import DynamicTensorBatcher
from blvm.data.datasets import DATASETS
from blvm.data.loaders import AudioLoader
from blvm.data.samplers import LengthEvalSampler, LengthTrainSampler
from blvm.data.transforms import Compose, MuLawDecode, RandomSegment, MuLawEncode
from blvm.evaluation.tracker import Tracker
from blvm.training.restore import load_run, save_run
from blvm.utils.argparsing import str2bool
from blvm.utils.argparsers import parser
from blvm.utils.device import get_device
from blvm.utils.operations import split_sequence
from blvm.utils.rand import set_seed, get_random_seed
from blvm.utils.wandb import is_run_resumed, restore_run


parser.set_defaults(
    epochs=2000,
    batch_size=40,
    save_checkpoints=True,
    test_every=5,
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
model_group.add_argument("--stack_size", default=64, type=int, help="number of waveform frames to stack to input vector")
model_group.add_argument("--hidden_size", default=256, type=int, help="dimensionality of hidden state in CWVAE")
model_group.add_argument("--num_layers", default=1, type=int, help="number of lstm layers")
model_group.add_argument("--dropout", default=0, type=float, help="dropout after lstm layers")
model_group.add_argument("--input_coding", default="mu_law", type=str, choices=["mu_law", "linear"], help="input encoding")
model_group.add_argument("--num_bits", default=16, type=int, help="number of bits for DML and input")
model_group.add_argument("--num_mix", default=10, type=int, help="number of logistic mixture components")
model_group.add_argument("--random_segment_size", default=None, type=int, help="timesteps to subsample per training example")
model_group.add_argument("--split_eval", default=False, type=str2bool, help="If true, split evaluation sequences")

args = parser.parse_args()

if args.seed is None:
    args.seed = get_random_seed()
set_seed(args.seed)

device = get_device() if args.device == "auto" else torch.device(args.device)
torch.cuda.set_device(device)

dataset = DATASETS[args.dataset]


model = blvm.models.lstm.LSTMAudio(
    stack_size=args.stack_size,
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout=args.dropout,
    num_mix=args.num_mix,
    num_bins=2 ** args.num_bits,
)

encode_transform = []
decode_transform = []
if args.input_coding == "mu_law":
    encode_transform.append(MuLawEncode(bits=args.num_bits))
    decode_transform.append(MuLawDecode(bits=args.num_bits))

if args.random_segment_size is not None:
    encode_transform_train = Compose(*[RandomSegment(args.random_segment_size), *encode_transform])
else:
    encode_transform_train = Compose(*encode_transform)

encode_transform = Compose(*encode_transform)
decode_transform = Compose(*decode_transform)

modalities_train = [(AudioLoader(dataset.audio_ext), encode_transform_train, DynamicTensorBatcher())]
modalities_test = [(AudioLoader(dataset.audio_ext), encode_transform, DynamicTensorBatcher())]

train_dataset = BaseDataset(source=dataset.train, modalities=modalities_train)
rich.print(train_dataset)
if args.batch_len:
    train_sampler = LengthTrainSampler(
        source=dataset.train,
        field=dataset.audio_length,
        batch_len=16000 * args.batch_len if isinstance(args.batch_len, float) else args.batch_len,
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
    batch_kwarg = dict(batch_size=args.batch_size)  # set batch size instead of length if random segmenting
else:
    batch_kwarg = dict(batch_len=100 * 16000)
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


print(model)
with torch.cuda.amp.autocast(enabled=args.use_amp):
    (x, x_sl), metadata = next(iter(train_loader))
    model.summary(input_data=x, x_sl=x_sl)
model = model.to(device)

wandb.init(**vars(parser.parse_args_by_group().wandb), config=args)
wandb.save("*.pt")
wandb.watch(model, log="all", log_freq=len(train_loader))

optimizer_class = getattr(torch.optim, args.optimizer)
optimizer = optimizer_class(model.parameters(), lr=args.lr, **args.optimizer_kwargs)

lr_scheduler_class = getattr(torch.optim.lr_scheduler, args.lr_scheduler)
lr_scheduler = lr_scheduler_class(optimizer, **args.lr_scheduler_kwargs)

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
            loss, metrics, output = model(x, x_sl)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tracker.update(metrics)

    model.eval()
    with torch.no_grad():
        extra = dict()
        if (epoch % args.test_every) == 0:
            for loader in valid_test_dataloaders.values():
                for (x, x_sl), metadata in tracker.steps(loader):
                    x = x.to(device, non_blocking=True)
                    if args.split_eval:
                        s0 = None
                        splits_x, splits_x_sl = split_sequence(x, x_sl, length=args.random_segment_size, overlap=0)
                        for i, (xs, xs_sl) in enumerate(zip(splits_x, splits_x_sl)):
                            with torch.cuda.amp.autocast(enabled=args.use_amp):
                                loss, metrics, output = model(xs, xs_sl, s0=s0)
                            tracker.update(metrics)
                            s0 = [output.sn[h_or_c][:xs.shape[0]] for h_or_c in range(2)]  # remove done examples
                    else:
                        with torch.cuda.amp.autocast(enabled=args.use_amp):
                            loss, metrics, outputs = model(x, x_sl)
                        tracker.update(metrics)

            rec_mode = decode_transform(outputs.reconstruction_mode)
            rec_mode = [
                wandb.Audio(rec_mode[i].to(torch.float32).flatten().cpu().numpy(), caption=f"Reconstruction mode {i}", sample_rate=16000)
                for i in range(min(2, rec_mode.shape[0]))
            ]

            rec_sample = decode_transform(outputs.reconstruction_sample)
            rec_sample = [
                wandb.Audio(rec_sample[i].to(torch.float32).flatten().cpu().numpy(), caption=f"Reconstruction sample {i}", sample_rate=16000)
                for i in range(min(2, rec_sample.shape[0]))
            ]

            # (x, x_sl), outputs = model.generate(n_samples=2, max_timesteps=128000)
            # x = decode_transform(x)
            # samples = [
            #     wandb.Audio(x[i].flatten().cpu().numpy(), caption=f"Sample {i}", sample_rate=16000) for i in range(2)
            # ]

            extra = dict(
                # samples=samples,
                reconstructions_mode=rec_mode,
                reconstructions_sample=rec_sample,
            )

            if (
                args.save_checkpoints
                and wandb.run is not None
                and wandb.run.dir != "/"
                and epoch > args.test_every
                and dataset.test in tracker.accumulated_values
                and len(tracker.accumulated_values[dataset.test]["loss"]) > 1
                and min(tracker.accumulated_values[dataset.test]["loss"][:-1])
                > tracker.accumulated_values[dataset.test]["loss"][-1]
            ):
                save_run(
                    wandb.run.dir,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    tracker=tracker,
                )

        tracker.log(learning_rate=lr_scheduler.get_last_lr()[0], **extra)

        lr_scheduler.step()