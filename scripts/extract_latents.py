import argparse
import os
import pickle

from ffrecord import FileWriter
from slickconf import load_arg_config, instantiate
import torch
from torch import distributed as dist
from torch.utils import data
from tqdm import tqdm
from imm.config import IMMConfig
from imm.data import ImageDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--output", type=str)

    conf, args = load_arg_config(IMMConfig, parser=parser)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    torch.cuda.set_device(torch.cuda.device(rank))
    dist.init_process_group(backend="nccl")

    dataset = ImageDataset(
        args.input, transform=instantiate(conf.training.image_transform)
    )
    sampler = data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=8,
        multiprocessing_context="fork",
    )

    pbar = loader
    if rank == 0:
        total_samples = len(dataset)
        output_writer = FileWriter(args.output, total_samples)
        pbar = tqdm(loader)

    encoder = instantiate(conf.model.latent)

    saved_indexes = set()
    for images, labels, indexes in pbar:
        latents = encoder.encode_images(images.to("cuda"))
        outputs = [None for _ in range(world_size)]
        dist.all_gather_object(outputs, (latents, labels, indexes))
        records = []
        for output in outputs:
            for mean, std, label, index in zip(
                output[0][0].cpu().unbind(0),
                output[0][1].cpu().unbind(0),
                output[1].tolist(),
                output[2].tolist(),
            ):
                records.append((mean.numpy(), std.numpy(), label, index))

        records = sorted(records, key=lambda x: x[3])

        for record in records:
            if record[3] in saved_indexes:
                continue

            saved_indexes.add(record[3])

            if rank == 0:
                output_writer.write_one(pickle.dumps(record))

                pbar.set_description(f"{len(saved_indexes)}/{total_samples}")

    if rank == 0:
        output_writer.close()


if __name__ == "__main__":
    main()

    torch.distributed.destroy_process_group()
