import argparse


def get_args():
    parser = argparse.ArgumentParser(description='QA')

    """Optimization related arguments"""
    optim_args = parser.add_argument_group('Optimization related arguments')
    optim_args.add_argument('--train_batch_size', type=int,  default= 32, help='Training batch Size')
    optim_args.add_argument('--eval_batch_size', type=int,  default= 32, help='Evalutatino batch Size')
    optim_args.add_argument('--adam_epsilon', type=float,  default= 1e-8, help='Adam epsilon')
    optim_args.add_argument('--warmup_steps', type=int,  default= 100, help='Warmup Steps')
    optim_args.add_argument('--weight_decay', type=float,  default= 0.04, help='Warmup Steps')
    optim_args.add_argument('--learning_rate', type=float,  default=5e-5, help='Initial Learning rate')
    optim_args.add_argument( "--gradient_accumulation_steps",type=int, default=1, 
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
    optim_args.add_argument( "--val_check_interval",type=float, default=0.5, 
                            help="validation check interval ratio")
    optim_args.add_argument( "--gradient_cliping",type=float, default=0.5, 
                            help=" The value at which to clip gradients ")
    optim_args.add_argument( "--temperature",type=int, default=0, help=" Temperature ")
    optim_args.add_argument('--requires_grad', action="store_true", help='requiring gradients')
    optim_args.add_argument('--num_workers', type=int, default=1, help='The number of workers')
    
    
    """Data related arguments"""
    data_args = parser.add_argument_group('Data related arguments')
    data_args.add_argument('--input_max_seq_length', type=int, default= 128, help='Max sequence legnth')
    data_args.add_argument('--output_max_seq_length', type=int, default= 128, help='Max sequence legnth')
    data_args.add_argument('--nle_data_dir', type=str, default= '.', help='Directory to NLE load dataset')
    data_args.add_argument('--annotation_data_dir', type=str, default= '.', help='Directory to annotation load dataset')   
    data_args.add_argument('--coco_data_dir', type=str, default= '.', help='Directory to coco dataset') 
    data_args.add_argument('--img_size', type=int, default= 0, help='Image size')
    data_args.add_argument('--tokenizer_name', type=str, default= None, help='Pretrained Tokenizer name')
    data_args.add_argument('--filename', type=str, default= None, help='Optimizer file name')
    data_args.add_argument('--task_A', action="store_true", help='Doing task A')
    data_args.add_argument("--fewshot", type=float, default=None, help="few shot percentage")
    
    """Model related arguments"""
    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument('--max_epochs', type=int, default=10, help='Max epoch size')
    model_args.add_argument('--output_dir', type=str, default=".", help='Max epoch size')    
    model_args.add_argument('--load_from_epoch', type=str, default=None, help='Loading from epoch')
    model_args.add_argument("--prompting", action="store_true")
    model_args.add_argument("--finetuning", action="store_true")

    """Logging related arguments"""
    misc_args = parser.add_argument_group('Logging related & Misc arguments')
    misc_args.add_argument('--seed', type=int, default=42, help='Random Seed')
    misc_args.add_argument('--experiment_name', type=str, default='experiment', help='Experiment name for wandb')
    misc_args.add_argument('--ngpu', type=int, default=-1, help='Number of gpu')
    misc_args.add_argument('--ckpt_path', type=str, default="/media/storage/checkpoints", help='Checkpoint directory')
    misc_args.add_argument('--checkpoints_dir', type=str, default=None, help='Checkpoint file directory')




    args = parser.parse_args()
    return args
