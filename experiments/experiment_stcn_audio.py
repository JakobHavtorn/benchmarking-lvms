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
from blvm.data.loaders import AudioLoader
from blvm.data.samplers import LengthEvalSampler, LengthTrainSampler
from blvm.data.transforms import Compose, Denormalize, MuLawDecode, MuLawEncode, Normalize, RandomSegment
from blvm.evaluation import Tracker
from blvm.utils.argparsing import str2bool
from blvm.utils.argparsers import parser
from blvm.utils.optimization import get_learning_rates_dict
from blvm.utils.rand import set_seed
from blvm.utils.wandb import is_run_resumed, restore_run
from blvm.training.annealers import CosineAnnealer
from blvm.training.restore import load_run, save_run


def main():
    parser.set_defaults(
        epochs=1000,
        batch_size=20,
        save_checkpoints=True,
        test_every=20,
        optimizer="Adam",
        lr=3e-4,
        lr_scheduler="MultiStepLR",
        lr_scheduler_kwargs=dict(milestones=[2500, 3500, 4500], gamma=0.1),
        max_grad_norm=3000.0,
        max_grad_value=1000.0,
        dataset="timit",
        project=blvm.WANDB_PROJECT,
        entity=None,
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--hidden_size", default=256, type=int, help="dimensionality of hidden state")
    model_group.add_argument("--latent_size", default=[256, 128, 64, 32, 16], type=int, nargs="+", help="dimensionality of latent state")
    model_group.add_argument("--num_layers", default=5, type=int, help="WaveNet blocks per stack")
    model_group.add_argument("--num_stacks", default=None, type=int, help="WaveNet stacks (defaults to number of latents)")
    model_group.add_argument("--input_coding", default="linear", type=str, choices=["mu_law", "linear"], help="input encoding")
    model_group.add_argument("--num_bits", default=16, type=int, help="number of bits for DML and input")
    model_group.add_argument("--num_mix", default=10, type=int, help="number of logistic mixture components")
    model_group.add_argument("--dense", default=False, type=str2bool, help="dense connectivity to latent variables")
    model_group.add_argument("--precision_posterior", default=True, type=str2bool, help="precision weighted parameterization of posterior")
    model_group.add_argument("--top_down", default=True, type=str2bool, help="top-down or bottom-up inference")
    model_group.add_argument("--num_stack_frames", default=200, type=int, help="frames to stack as input if input_coding is frames")
    model_group.add_argument("--random_segment_size", default=16000, type=int, help="timesteps to randomly subsample per training example")
    model_group.add_argument("--likelihood", default="DMoL", type=str, help="likelihood module")

    model_group.add_argument("--beta_anneal_steps", default=50000, type=int, help="number of steps to anneal beta")
    model_group.add_argument("--beta_start_value", default=0, type=float, help="initial beta annealing value")
    model_group.add_argument("--free_nats_steps", default=0, type=int, help="number of steps to constant/anneal free bits")
    model_group.add_argument("--free_nats_start_value", default=4, type=float, help="free bits per timestep")
    model_group.add_argument("--split_eval", default=False, type=str2bool, help="If true, split evaluation sequences")

    args = parser.parse_args()

    run(args)


def run(args):

    set_seed(args.seed)

    device = blvm.utils.device.get_device() if args.device == "auto" else torch.device(args.device)
    torch.cuda.set_device(device)

    dataset = DATASETS[args.dataset]

    model = blvm.models.STCN(
        likelihood=args.likelihood,
        in_channels=1,
        n_layers=args.num_layers,
        n_stacks=args.num_stacks,
        latent_size=args.latent_size,
        res_channels=args.hidden_size,
        kernel_size=2,
        base_dilation=2,
        n_stack_frames=args.num_stack_frames,
        precision_posterior=args.precision_posterior,
        dense=args.dense,
        top_down=args.top_down,
    )
    model = model.to(device)

    decode_transform = []
    encode_transform = []
    if args.input_coding == "mu_law":
        encode_transform.append(MuLawEncode(bits=args.num_bits))
        decode_transform.append(MuLawDecode(bits=args.num_bits))

    if args.likelihood == "Gaussian" or args.likelihood == "GMM":
        # Global normalization of waveform with train set statistics
        modalities = [(AudioLoader(dataset.audio_ext), Compose(*encode_transform), ListBatcher())]
        ds = BaseDataset(source=dataset.train, modalities=modalities)
        mean, variance = ds.compute_statistics(num_workers=args.num_workers)
        std = variance.sqrt()
        print(mean, std, variance)
        encode_transform.append(Normalize(mean=mean, std=std))
        decode_transform.append(Denormalize(mean=mean, std=std))

    if args.random_segment_size:
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
    if args.split_eval:
        if args.batch_size:
            batch_kwarg = dict(batch_size=args.batch_size)
        else:
            batch_kwarg = dict(batch_len=args.batch_len if args.batch_len else "max")
    else:
        batch_kwarg = dict(batch_len=args.batch_len * 3)

    for source_name in [*dataset.valid_sets, *dataset.test_sets]:
        valid_dataset = BaseDataset(source=source_name, modalities=modalities_test)
        valid_sampler = LengthEvalSampler(
            source=source_name,
            field=dataset.audio_length,
            shuffle=True,
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

    rich.print(vars(args))
    rich.print(train_dataset)
    if args.length_sampler:
        rich.print(train_sampler)
        rich.print(valid_sampler)
    print(model)
    model = model.to(device)
    (x, x_sl), metadata = next(iter(train_loader))
    x = x.to(device)
    model(x, x_sl)
    model.summary(input_data=x, x_sl=x_sl)
    print(model.receptive_fields)

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
        model, optimizer, checkpoint = load_run(directory, model, optimizer, device=device)

    beta_annealer = CosineAnnealer(
        anneal_steps=args.beta_anneal_steps,  # // 2,
        # constant_steps=args.beta_anneal_steps // 2,
        start_value=args.beta_start_value,
        end_value=1,
    )
    free_nats_annealer = CosineAnnealer(
        anneal_steps=args.free_nats_steps // 2,
        constant_steps=args.free_nats_steps // 2,
        start_value=args.free_nats_start_value,
        end_value=0,
    )

    tracker = Tracker()

    for epoch in tracker.epochs(args.epochs):
        model.train()
        for (x, x_sl), metadata in tracker(train_loader):
            x = x.to(device, non_blocking=True)

            beta = beta_annealer.step()
            free_nats = free_nats_annealer.step()

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                loss, metrics, outputs = model(x, x_sl, beta=beta, free_nats=free_nats)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_value_(model.parameters(), args.max_grad_value)

            scaler.step(optimizer)
            scaler.update()

            tracker.update(metrics)

        extra_log = dict()
        model.eval()
        with torch.no_grad():
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

                rec = decode_transform(outputs.reconstructions)
                rec = rec[:2].flatten(1).cpu().numpy().astype(float)
                rec = [wandb.Audio(r, caption=f"Reconstruction {i}", sample_rate=16000) for i, r in enumerate(rec)]

                mod = decode_transform(outputs.reconstructions_mode)
                mod = mod[:2].flatten(1).cpu().numpy().astype(float)
                mod = [wandb.Audio(r, caption=f"Reconstruction mode {i}", sample_rate=16000) for i, r in enumerate(mod)]

                # (x, x_sl), output = model.generate(n_samples=2, max_timesteps=128000)
                # x = decode_transform(x)
                # x = x.flatten(1).cpu().numpy().astype(float)
                # x = [wandb.Audio(r, caption=f"Sample {i}", sample_rate=16000) for i, r in enumerate(x)]

                # extra_log.update(samples=samples, reconstructions=rec, reconstructions_mode=rec_mode)
                extra_log.update(reconstructions=rec, reconstructions_mode=mod)

                if (
                    args.save_checkpoints
                    and wandb.run is not None
                    and wandb.run.dir != "/"
                    and epoch > args.test_every
                    and dataset.test in tracker.accumulated_values
                    and tracker.accumulated_values[dataset.test]["elbo (bpx)"][-1]
                    == tracker.best_values[dataset.test]["best_elbo (bpx)"]
                ):
                    save_run(
                        wandb.run.dir,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        tracker=tracker,
                    )

        lr_scheduler.step()

        extra_log.update(get_learning_rates_dict(optimizer))
        tracker.log(**extra_log)


if __name__ == "__main__":
    main()
