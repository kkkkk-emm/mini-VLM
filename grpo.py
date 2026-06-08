from training.grpo_trainer import build_parser, run_grpo_training


def main():
    args = build_parser().parse_args()
    run_grpo_training(args)


if __name__ == "__main__":
    main()
