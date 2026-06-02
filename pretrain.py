from training.trainer import build_parser, run_training


def main():
    args = build_parser("pretrain").parse_args()
    run_training(args, stage="pretrain")


if __name__ == "__main__":
    main()
