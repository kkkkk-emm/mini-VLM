from training.trainer import build_parser, run_training


def main():
    args = build_parser("sft").parse_args()
    run_training(args, stage="sft")


if __name__ == "__main__":
    main()
