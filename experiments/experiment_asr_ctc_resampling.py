import os
import torch
import wandb
import rich

from torch.utils.data import DataLoader

import blvm
import blvm.data
import blvm.models
from blvm.settings import SOURCE_DIRECTORY
import blvm.utils
import blvm.utils.device

from blvm.data import BaseDataset
from blvm.data.batchers import DynamicTensorBatcher, TextBatcher
from blvm.data.datasets import DATASETS
from blvm.data.loaders import TextLoader, AudioLoader
from blvm.data.samplers import LengthTrainSampler, LengthEvalSampler
from blvm.data.text_cleaners import clean_timit
from blvm.data.token_map import TokenMap
from blvm.data.tokenizers import char_tokenizer, word_tokenizer
from blvm.data.tokens import BLANK_TOKEN, TIMIT_ALPHABET, TIMIT_PHONESET
from blvm.data.transforms import Compose, EncodeInteger, TextCleaner
from blvm.evaluation import Tracker
from blvm.models.lstm_asr import SimpleLSTMASR
from blvm.training.restore import load_run, save_run
from blvm.utils.argparsing import str2bool
from blvm.utils.argparsers import parser
from blvm.utils.rand import set_seed, get_random_seed
from blvm.utils.wandb import get_run, is_run_resumed, restore_run


parser.set_defaults(
    epochs=40,
    batch_len=120,
    optimizer="Adam",
    lr=3e-4,
    lr_scheduler="MultiStepLR",
    lr_scheduler_kwargs=dict(milestones=[100, 200, 300], gamma=0.5),
    max_grad_norm=3000.0,
    max_grad_value=1000.0,
    save_checkpoints=False,
    use_amp=True,
    dataset="timit",
    project=blvm.WANDB_PROJECT,
    entity=None,
    num_workers=12,
)


model_group = parser.add_argument_group("model")

model_group.add_argument("--train_source", default=None, type=str, help="training data source file name")
model_group.add_argument("--model_id", type=str, help="id of the model to use for representation extraction.")
model_group.add_argument("--model_project", default=blvm.WANDB_PROJECT, type=str, help="project of the model to use for representation extraction.")
model_group.add_argument("--z_index", type=int, help="index of the latent representation layer")
model_group.add_argument("--text_type", default="phon", choices=["word", "char", "phon"], help="text type")

model_group.add_argument("--sample_rate", default=16000, type=int, help="sample rate")

model_group.add_argument("--hidden_size", default=128, type=int, help="size of the LSTM layers")
model_group.add_argument("--num_layers", default=1, type=int, help="number of LSTM layers")
model_group.add_argument("--bidirectional", default=False, type=str2bool, help="bidirectional LSTM layers")
model_group.add_argument("--sum_directions", default=False, type=str2bool, help="sum directions instead of concat")
model_group.add_argument("--dropout_prob", default=0.30, type=float, help="dropout rate")
model_group.add_argument("--temporal_dropout", default=True, type=str2bool, help="use temporal dropout")

model_group.add_argument("--num_batches_per_epoch", default=1000, type=int, help="number of batches per epoch")


args = parser.parse_args()
args_groups = parser.parse_args_by_group()

if args.seed is None:
    args.seed = get_random_seed()

if args_groups.wandb.name is None:
    args_groups.wandb.name = f"h={args.hidden_size}"
    if args.num_layers > 1:
        args_groups.wandb.name += f" l={args.num_layers}"
    if args.bidirectional:
        args_groups.wandb.name += " bidir"
    args_groups.wandb.name += f" d={args.dropout_prob} id={args.model_id} {args.model_project} z={args.z_index} resampling"
    if args.train_source:
        args_groups.wandb.name += " " + args.train_source

rich.print(vars(args))

set_seed(args.seed)

device = blvm.utils.device.get_device() if args.device == "auto" else torch.device(args.device)
torch.cuda.set_device(device)


run = get_run(args.model_id, args.model_project, args.entity)
directory = restore_run(run, exclude="media")
rep_model, checkpoint = load_run(directory, device=device)


dataset = DATASETS[args.dataset]

train_source = dataset.train if args.train_source is None else os.path.join(SOURCE_DIRECTORY, dataset.name, args.train_source)

text_exts = dict(
    word="TXT",
    char="TXT",
    phon="PHN",
)
text_ext = text_exts[args.text_type]

tokenizer = char_tokenizer if args.text_type == "char" else word_tokenizer
alphabet = TIMIT_PHONESET if args.text_type == "phon" else TIMIT_ALPHABET

token_map = TokenMap(tokens=alphabet, add_blank=True)
blank_token_idx = token_map.token2index[BLANK_TOKEN]
output_size = len(token_map)

text_loader = TextLoader(text_ext, cache=True)
text_transform = Compose(
    TextCleaner(clean_timit, lambda s: s.replace("h#", "").strip()),
    EncodeInteger(token_map=token_map, tokenizer=tokenizer)
)
text_batcher = TextBatcher()


loader = AudioLoader(dataset.audio_ext, cache=False, sum_channels=True)
transform = None
batcher = DynamicTensorBatcher()

modalities = [
    (loader, transform, batcher),
    (text_loader, text_transform, text_batcher)
]


train_dataset = BaseDataset(source=train_source, modalities=modalities)
if args.batch_len:
    train_sampler = LengthTrainSampler(
        source=train_source,
        field=dataset.audio_length,
        batch_len=args.sample_rate * args.batch_len,
        max_pool_difference=args.sample_rate * 0.3,
        min_pool_size=512,
        num_batches=args.num_batches_per_epoch,
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
for source_name in [*dataset.valid_sets, *dataset.test_sets]:
    valid_dataset = BaseDataset(source=source_name, modalities=modalities)
    valid_sampler = LengthEvalSampler(
        source=source_name,
        field=dataset.audio_length,
        shuffle=True,
        batch_size=args.batch_size * 3,
        batch_len=args.sample_rate * args.batch_len,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        collate_fn=valid_dataset.collate,
        num_workers=args.num_workers,
        batch_sampler=valid_sampler,
        pin_memory=True,
    )
    valid_test_dataloaders[source_name] = valid_loader

def get_representation(x, x_sl, rep_model):
    """Extract and return representation from represnetation model.
    
    Args:
        x (torch.Tensor): input tensor (B, T)

    Returns:
        Tuple[torch.Tensor]: representation (B, D, T), representation length (B)
    """
    with torch.inference_mode():
        _, _, output = rep_model(x, x_sl)
        output.z = [output.z] if isinstance(output.z, torch.Tensor) else output.z
        output.z_sl = [output.z_sl] * len(output.z) if isinstance(output.z_sl, torch.Tensor) else output.z_sl
        z, z_sl = output.z[args.z_index], output.z_sl[args.z_index]
        z = z.permute(0, 2, 1)  # (B, T, D) -> (B, D, T)
        return z, z_sl


((x, x_sl), (y, y_sl)), (x_m, t_m) = next(iter(train_loader))
x = x.to(device)
z, z_sl = get_representation(x, x_sl, rep_model)

model = SimpleLSTMASR(
    input_size=z.size(1),
    hidden_size=args.hidden_size,
    token_map=token_map,
    num_layers=args.num_layers,
    bidirectional=args.bidirectional,
    sum_directions=args.sum_directions,
    temporal_dropout=args.temporal_dropout,
    dropout_prob=args.dropout_prob,
)
print(model)
model.summary(input_data=z, x_sl=z_sl, y=y, y_sl=y_sl, device="cpu")
model.to(device)

optimizer_class = getattr(torch.optim, args.optimizer)
optimizer = optimizer_class(model.parameters(), lr=args.lr, **args.optimizer_kwargs)

lr_scheduler_class = getattr(torch.optim.lr_scheduler, args.lr_scheduler)
lr_scheduler = lr_scheduler_class(optimizer, **args.lr_scheduler_kwargs)

scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

if is_run_resumed():
    directory = restore_run(wandb.run.id, wandb.run.project, wandb.run.entity, exclude="media")
    model, optimizer, checkpoint = load_run(directory, model, optimizer, device=device)


wandb.init(**vars(args_groups.wandb), config=args)
wandb.save("*.pt")


tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    # training
    model.train()
    for ((x, x_sl), (y, y_sl)), metadata in tracker.steps(train_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            z, z_sl = get_representation(x, x_sl, rep_model)
            loss, metrics, outputs = model(z, z_sl, y, y_sl)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_value_(model.parameters(), args.max_grad_value)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        tracker.update(metrics)

    # evaluation
    model.eval()
    with torch.inference_mode():
        extra = dict()
        for loader in valid_test_dataloaders.values():
            for ((x, x_sl), (y, y_sl)), metadata in tracker.steps(loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    z, z_sl = get_representation(x, x_sl, rep_model)
                    loss, metrics, output = model(z, z_sl, y, y_sl)
                tracker.update(metrics)

    tracker.log(learning_rate=lr_scheduler.get_last_lr()[0])

    for r, h in zip(output.refs[:5], output.hyps[:5]):
        print(f"Ref: {r}\nHyp: {h}")

    # update hyperparams
    lr_scheduler.step()

    # save model
    if (
        args.save_checkpoints
        and wandb.run is not None
        and wandb.run.dir != "/"
        and dataset.test in tracker.accumulated_values
        and tracker.accumulated_values[dataset.test]["wer"][-1] == tracker.best_values[dataset.test]["best_wer"]
    ):
        save_run(
            wandb.run.dir,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            tracker=tracker,
        )
