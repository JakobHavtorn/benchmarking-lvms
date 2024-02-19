import argparse

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
from blvm.data.batchers import DynamicTensorBatcher
from blvm.data.datasets import DATASETS
from blvm.data.loaders import AudioLoader
from blvm.data.transforms import Compose, MuLawDecode, MuLawEncode
from blvm.data.samplers.length_samplers import LengthEvalSampler
from blvm.evaluation import Tracker
from blvm.evaluation.metrics import BitsPerDimMetric, LLMetric, RunningMeanMetric, RunningVarianceMetric
from blvm.utils.argparsers import parser
from blvm.utils.log_likelihoods import discretized_logistic_mixture_ll
from blvm.utils.operations import sequence_mask
from blvm.utils.rand import set_seed, get_random_seed


parser.set_defaults(
    batch_size=256,
    dataset="timit",
    num_workers=4,
    seed=42,
    project=blvm.WANDB_PROJECT,
)
parser.add_argument("--input_coding", default="mu_law", type=str, choices=["mu_law", "linear"], help="input encoding")
parser.add_argument("--num_bits", default=16, type=int, help="number of bits for DML and input")

args = parser.parse_args()


if args.seed is None:
    args.seed = get_random_seed()

set_seed(args.seed)

device = blvm.utils.device.get_device() if args.device == "auto" else torch.device(args.device)

dataset = DATASETS[args.dataset]

wandb.init(
    entity=None,
    project="benchmarks",
    group=args.wandb_group,
    notes=args.wandb_notes,
    tags=args.wandb_tags,
    mode=args.wandb_mode,
    config=args,
)
rich.print(vars(args))


decode_transform = []
encode_transform = []
if args.input_coding == "mu_law":
    encode_transform.append(MuLawEncode(bits=args.num_bits))
    decode_transform.append(MuLawDecode(bits=args.num_bits))

encode_transform = Compose(*encode_transform)
decode_transform = Compose(*decode_transform)

batcher = DynamicTensorBatcher()
loader = AudioLoader(dataset.audio_ext, cache=False)
modalities = [(loader, encode_transform, batcher)]

train_dataset = BaseDataset(source=dataset.train, modalities=modalities)
valid_dataset = BaseDataset(source=dataset.test, modalities=modalities)
rich.print(train_dataset)

train_sampler = LengthEvalSampler(
    source=dataset.train,
    field=dataset.audio_length,
    batch_len=16000 * args.batch_size if args.batch_size > 0 else "max",
    shuffle=True,
)
valid_sampler = LengthEvalSampler(
    source=dataset.test,
    field=dataset.audio_length,
    batch_len=16000 * args.batch_size if args.batch_size > 0 else "max",
    shuffle=True,
)

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    batch_sampler=train_sampler,
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    collate_fn=valid_dataset.collate,
    num_workers=args.num_workers,
    batch_sampler=valid_sampler,
)

tracker = Tracker()

if args.input_coding == "mu_law":
    logit_probs = torch.ones(2)
    loc = torch.tensor([[-0.551, 0.551]])
    log_scales = torch.tensor([[0.11, 0.11]]).log()
    num_mix = 2
else:
    logit_probs = torch.ones(1)
    loc = torch.tensor([[0.0]])
    log_scales = torch.tensor([[0.005]]).log()
    num_mix = 1


num_bins = 2 ** args.num_bits

with torch.no_grad():

    for loader in [train_loader, valid_loader]:
        for (x, x_sl), metadata in tracker(loader):
            x = x.unsqueeze(-1)  # create data dimension

            seq_mask = sequence_mask(x_sl, dtype=bool, device=x.device).unsqueeze(-1)
            log_prob = discretized_logistic_mixture_ll(x, logit_probs, loc, log_scales, num_bins)
            log_prob = log_prob * seq_mask
            log_likelihoods = log_prob.sum(1)

            x_abs = x.abs()

            metrics = [LLMetric(log_likelihoods), BitsPerDimMetric(log_likelihoods, reduce_by=x_sl)]
            for i in range(x.size(0)):
                metrics.append(RunningMeanMetric(x[i, :x_sl[i]], name="mean(x)", reduce_by=x_sl[i]))
                metrics.append(RunningVarianceMetric(x[i, :x_sl[i]], name="var(x)", reduce_by=x_sl[i]))
                metrics.append(RunningMeanMetric(x_abs[i, :x_sl[i]], name="mean(abs(x))", reduce_by=x_sl[i]))
                metrics.append(RunningVarianceMetric(x_abs[i, :x_sl[i]], name="var(abs(x))", reduce_by=x_sl[i]))

            tracker.update(metrics, check_unique=False)

    tracker.log()
