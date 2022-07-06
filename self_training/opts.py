import argparse


def get_args():
    parser = argparse.ArgumentParser(description='QA')

    """Optimization related arguments"""
    optim_args = parser.add_argument_group('Optimization related arguments')
    optim_args.add_argument("--mode", type=str, default="train", help="train | test")
    optim_args.add_argument('--train_batch_size', type=int,  default= 32, help='Training batch Size')
    optim_args.add_argument('--eval_batch_size', type=int,  default= 1, help='Evalutatino batch Size')
    optim_args.add_argument('--adam_epsilon', type=float,  default= 1e-8, help='Adam epsilon')
    optim_args.add_argument('--warmup_steps', type=int,  default=-1, help='Warmup Steps')
    optim_args.add_argument("--warmup_ratio", type=float, default=0.0)
    optim_args.add_argument('--weight_decay', type=float,  default= 0.04, help='Weight Decay')
    optim_args.add_argument('--learning_rate', type=float,  default=5e-5, help='Initial Learning rate')
    optim_args.add_argument( "--gradient_accumulation_steps",type=int, default=1, 
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
    optim_args.add_argument( "--val_check_interval",type=float, default=0.5, 
                            help="validation check interval ratio")
    optim_args.add_argument( "--gradient_cliping",type=float, default=1.0, 
                            help=" The value at which to clip gradients ")
    optim_args.add_argument('--max_epochs', type=int, default=30, help='Max epoch size')
    optim_args.add_argument('--requires_grad', action="store_true", help='requiring gradients')
    optim_args.add_argument('--max_iter', type=int, default=10, help='The number of iteration for self-training')
    
    
    """Data related arguments"""
    data_args = parser.add_argument_group('Data related arguments')
    data_args.add_argument("--img_encoded", action="store_true")
    # NLE dataset
    data_args.add_argument('--nle_train_anno_path', type=str, default=None, help='Path to nle annotation for training')
    data_args.add_argument('--nle_valid_anno_path', type=str, default=None, help='Path to nle annotation for validation')
    data_args.add_argument("--nle_test_anno_path", type=str, default=None, help='Path to nle annotation for test')
    data_args.add_argument('--nle_train_image_dir', type=str, default=None, help='Directory to nle image train dataset')
    data_args.add_argument('--nle_valid_image_dir', type=str, default=None, help='Directory to nle image valid dataset')
    data_args.add_argument("--nle_test_image_dir", type=str, default=None, help="Directory to nle image test dataset")
    # Captioning datasets
    data_args.add_argument("--captioning_anno_paths", nargs="+")
    data_args.add_argument("--captioning_image_dirs", nargs="+")
    # External datasets
    data_args.add_argument("--external_data", type=bool, default=False)
    data_args.add_argument("--max_seq_len", type=int, default=40)
    data_args.add_argument('--img_size', type=int, default=224, help='Image size')        
    data_args.add_argument("--fewshot_ratio", type=float, default=-1, help="Ratio of few-shot data")
    data_args.add_argument("--fewshot_num", type=int, default=None, help="The number of few-shot data")
    data_args.add_argument("--cached_dir", type=str, default="cached/nlx_gpt_base", help="Path to cached file")
    data_args.add_argument("--vis_rep_len", type=int, default=7*7, help="visual representation length")
    data_args.add_argument("--n_train_workers", type=int, default=8)
    data_args.add_argument("--n_valid_workers", type=int, default=8)
    data_args.add_argument('--pseudo_data', action="store_true", help='Except for few shot data')
    data_args.add_argument('--output_dir', type=str, default=None, help='Directory to store generated dataset')
    data_args.add_argument('--vqax_test_anno_path', type=str, default=None, help='Directory of vqax test annotation path')
    data_args.add_argument('--pseudo_labels_pth', type=str, default=None, help='Directory of pseudo labels dataset')
    
    
    """Inference related arguments"""
    infer_args = parser.add_argument_group('Inference related arguments')
    infer_args.add_argument("--load_ckpt_path", type=str, default=None)
    infer_args.add_argument("--top_k", type=float, default=0.0, help="top_k for generation")
    infer_args.add_argument("--top_p", type=float, default=0.9, help="top_p for generation")
    optim_args.add_argument( "--temperature", type=float, default=1.0, help=" Temperature ")
    

    """Logging related arguments"""
    misc_args = parser.add_argument_group('Logging related & Misc arguments')
    misc_args.add_argument('--seed', type=int, default=42, help='Random Seed')
    misc_args.add_argument('--experiment_name', type=str, default='experiment', help='Experiment name for wandb')
    misc_args.add_argument('--ngpu', type=int, default=1, help='Number of gpu')
    misc_args.add_argument('--checkpoints_dir', type=str, default="./ckpts", help='Checkpoint directory')

    args = parser.parse_args()
    return args
