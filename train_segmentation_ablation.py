import argparse
from pathlib import Path

from train_segmentation import train


EXPERIMENTS = {
    "baseline": ("unet", "unet_baseline"),
    "fdconv": ("fdconv", "unet_fdconv"),
    "fdconv_capsule": ("fdconv_capsule", "unet_fdconv_capsule"),
    "fdconv_rfapm": ("fdconv_rfapm", "unet_fdconv_rfapm"),
    "full": ("caps_fdrnet_lite", "caps_fdrnet_lite"),
}

ORDER = ("baseline", "fdconv", "fdconv_capsule", "fdconv_rfapm", "full")


def train_one(args: argparse.Namespace, experiment: str) -> None:
    model_type, run_name = EXPERIMENTS[experiment]
    if args.prefix:
        run_name = f"{args.prefix}_{run_name}"
    output = Path(args.project) / run_name
    train_args = argparse.Namespace(
        data=args.data,
        output=str(output),
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        workers=args.workers,
        base=args.base,
        model_type=model_type,
        cpu=args.cpu,
    )
    print(f"[{experiment}] model_type={model_type} output={output}")
    train(train_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train segmentation ablations.")
    parser.add_argument("--experiment", choices=[*ORDER, "all"], default="baseline")
    parser.add_argument("--data", default="datasets/segmentation")
    parser.add_argument("--project", default="runs/segmentation_ablation")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    experiments = ORDER if parsed.experiment == "all" else (parsed.experiment,)
    for exp in experiments:
        train_one(parsed, exp)
