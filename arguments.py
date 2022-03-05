# coding=utf-8


"""argparser configuration"""

import argparse
import os
import torch
import deepspeed


def add_model_config_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group("model", "model configuration")

    group.add_argument("--model-config", type=str, default=None,
                       help="the configuration of the base model")
    group.add_argument("--cpu-optimizer", action="store_true",
                       help="Run optimizer on CPU")
    group.add_argument("--cpu_torch_adam", action="store_true",
                       help="Use Torch Adam as optimizer on CPU.")

    return parser


def add_fp16_config_args(parser: argparse.ArgumentParser):
    """Mixed precision arguments."""

    group = parser.add_argument_group("fp16", "fp16 configurations")

    group.add_argument("--fp16", action="store_true",
                       help="Run model in fp16 mode")
    group.add_argument("--fp32-embedding", action="store_true",
                       help="embedding in fp32")
    group.add_argument("--fp32-layernorm", action="store_true",
                       help="layer norm in fp32")
    group.add_argument("--fp32-tokentypes", action="store_true",
                       help="embedding token types in fp32")
    group.add_argument("--fp32-allreduce", action="store_true",
                       help="all-reduce in fp32")
    group.add_argument("--hysteresis", type=int, default=2,
                       help="hysteresis for dynamic loss scaling")
    group.add_argument("--loss-scale", type=float, default=None,
                       help="Static loss scaling, positive power of 2 "
                       "values can improve fp16 convergence. If None, dynamic loss scaling is used.")
    group.add_argument("--loss-scale-window", type=float, default=1000,
                       help="Window over which to raise/lower dynamic scale")
    group.add_argument("--min-scale", type=float, default=1,
                       help="Minimum loss scale for dynamic loss scale")

    return parser


def add_training_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    group = parser.add_argument_group("train", "training configurations")

    group.add_argument("--do-train", action="store_true",
                       help="whether do training")
    group.add_argument("--do-valid", action="store_true",
                       help="whether do validation")
    group.add_argument("--do-eval", action="store_true",
                       help="whether do testing")
    group.add_argument("--do-infer", action="store_true",
                       help="whether do inference (testing without labels)")
    group.add_argument("--train-ratio",type=float, default=1.0,
                       help="the ratio of the training set used for training")
    group.add_argument("--train-num",type=int, default=-1,
                       help="the number of training samples, -1 for all sample")
    group.add_argument("--dev-ratio",type=float, default=1.0,
                       help="the ratio of the training set used for validation")
    group.add_argument("--dev-num",type=int, default=-1,
                       help="the number of validation samples, -1 for all sample")
    group.add_argument("--test-ratio",type=float, default=1.0,
                       help="the ratio of the training set used for testing")
    group.add_argument("--test-num",type=int, default=-1,
                       help="the number of testing samples, -1 for all sample")
    group.add_argument("--epochs", type=int, default=1,
                       help="the epochs for training")
    group.add_argument("--batch-size", type=int, default=4,
                       help="Data Loader batch size")
    group.add_argument("--dev-batch-size", type=int, default=2,
                       help="Data Loader batch size")
    group.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="gradient accumulation steps")
    group.add_argument("--weight-decay", type=float, default=0.01,
                       help="weight decay coefficient for L2 regularization")
    group.add_argument("--checkpoint-activations", action="store_true",
                       help="checkpoint activation to allow for training "
                       "with larger models and sequences")
    group.add_argument("--checkpoint-num-layers", type=int, default=1,
                       help="chunk size (number of layers) for checkpointing")
    group.add_argument("--num-checkpoints", type=int, default=24, 
                       help="For activation checkpointing")
    group.add_argument("--deepspeed-activation-checkpointing", action="store_true",
                       help="uses activation checkpointing from deepspeed")
    group.add_argument("--clip-grad", type=float, default=1.0,
                       help="gradient clipping")
    group.add_argument("--train-iters", type=int, default=1000000,
                       help="total number of iterations to train over all training runs")
    group.add_argument("--log-interval", type=int, default=100,
                       help="report interval")
    group.add_argument("--max-save", type=int, default=-1,
                       help="max checkpoints to save")
    group.add_argument("--seed", type=int, default=1234, 
                       help="random seed")

    # Learning rate.
    group.add_argument("--lr-decay-iters", type=int, default=None,
                       help="number of iterations to decay LR over,"
                       " If None defaults to `--train-iters`*`--epochs`")
    group.add_argument("--lr-decay-style", type=str, default="linear",
                       choices=["constant", "linear", "cosine", "exponential", "noam"],
                       help="learning rate decay function")
    group.add_argument("--lr", type=float, default=1.0e-4,
                       help="initial learning rate")
    group.add_argument("--warmup", type=float, default=0.0,
                       help="percentage of data to warmup on (.01 = 1% of all "
                       "training iters). Default 0.01")
    group.add_argument("--warmup-iter", type=int, default=0)
    
    # save
    group.add_argument("--save", type=str, default=None,
                       help="Output directory to save checkpoints to.")
    group.add_argument("--save-interval", type=int, default=5000,
                       help="number of iterations between saves")
    group.add_argument("--no-save-optim", action="store_true",
                       help="Do not save current optimizer.")

    # load
    group.add_argument("--load", type=str, default=None,
                       help="Path to a directory containing a model checkpoint.")
    group.add_argument("--load-oprimizer-states", action="store_true",
                       help="whether to load optimizer states")
    group.add_argument("--load-lr-scheduler-states", action="store_true",
                       help="whether to load learning rate scheduler states")
    group.add_argument("--no-load-optim", action="store_true",
                       help="Do not load optimizer when loading checkpoint.")
    group.add_argument("--log-file", type=str, default=None,
                        help="the path to save log.txt file")

    # distributed training args
    group.add_argument("--distributed-backend", default="nccl",
                       help="which backend to use for distributed training. One of [gloo, nccl]")
    group.add_argument("--local_rank", type=int, default=None,
                       help="local rank passed from distributed launcher")

    return parser


def add_prompt_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("prompt", "prompt configurations")

    group.add_argument("--load_prompt", type=str, default=None,
                       help="the path to load prompt from")
    group.add_argument("--prompt-tune", action="store_true",
                       help="whether to do prompt tuning")
    group.add_argument("--prompt-config", type=str, default=None,
                       help="the path of the prompt configuration")
    group.add_argument("--save-prompt-only", action="store_true",
                       help="whether to save the prompt only. If true, only prompts will be saved otherwise, "
                       "the whole model together with the prompt will be saved.")

    return parser


def add_evaluation_args(parser: argparse.ArgumentParser):
    """Evaluation arguments."""

    group = parser.add_argument_group("validation", "validation configurations")

    group.add_argument("--eval-batch-size", type=int, default=None,
                       help="Data Loader batch size for evaluation datasets. Defaults to `--batch-size`")
    group.add_argument("--eval-iters", type=int, default=100,
                       help="number of iterations to run for evaluation validation/test for")
    group.add_argument("--eval-interval", type=int, default=1000,
                       help="interval between running evaluation on validation set")

    return parser


def add_data_args(parser: argparse.ArgumentParser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group("data", "data configurations")
    group.add_argument("--model-parallel-size", type=int, default=1,
                       help="size of the model parallel.")
    group.add_argument("--data-path", type=str, default=None,
                       help="Path to combined dataset to split.")
    group.add_argument("--data-ext", type=str, default=".json",
                       help="the extension of the data file")
    group.add_argument("--data-name", type=str, default=None,
                       help="the name of the dataset")
    group.add_argument("--data-prefix", type=str, default=None,
                       help="the prefix to add before each data sample")
    group.add_argument("--num-workers", type=int, default=2,
                       help="Number of workers to use for dataloading")
    group.add_argument("--tokenizer-path", type=str, default="tokenizer.model",
                       help="path used to save/load sentencepiece tokenization models")
    group.add_argument("--enc-seq-length", type=int, default=512,
                       help="Maximum sequence length to process")
    group.add_argument("--dec-seq-length", type=int, default=512,
                       help="Maximum sequence length to process")

    return parser


def get_args():
    """Parse all the args."""

    parser = argparse.ArgumentParser(description="PyTorch BERT Model")
    parser = add_model_config_args(parser)
    parser = add_fp16_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_data_args(parser)
    parser = add_prompt_args(parser)

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    if not args.data_path:
        print("WARNING: No training data specified")

    args.cuda = torch.cuda.is_available()

    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    args.model_parallel_size = min(args.model_parallel_size, args.world_size)
    if args.rank == 0:
        print("using world size: {} and model-parallel size: {} ".format(
            args.world_size, args.model_parallel_size))

    args.dynamic_loss_scale = False
    if args.loss_scale is None:
        args.dynamic_loss_scale = True
        if args.rank == 0:
            print(" > using dynamic loss scaling")

    return args
