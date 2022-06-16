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
    data_args.add_argument('--enc_max_len', type=int, default=256, help='Max sequence legnth')
    data_args.add_argument('--dec_max_len', type=int, default=128, help='Max sequence legnth')
    data_args.add_argument('--train_anno_path', type=str, default=None, help='Path to annotation for training')
    data_args.add_argument('--valid_anno_path', type=str, default=None, help='Path to annotation for validation')
    data_args.add_argument('--image_dir', type=str, default=None, help='Directory to image dataset') 
    data_args.add_argument('--img_size', type=int, default=224, help='Image size')        
    data_args.add_argument("--fewshot_ratio", type=float, default=-1, help="Ratio of few-shot data")
    data_args.add_argument("--fewshot_num", type=int, default=500, help="The number of few-shot data")
    data_args.add_argument("--cached_dir", type=str, default="cached", help="Directory with cached file")
    data_args.add_argument("--vis_rep_len", type=int, default=7*7, help="visual representation length")
    data_args.add_argument("--n_train_workers", type=int, default=8)
    data_args.add_argument("--n_valid_workers", type=int, default=4)
    data_args.add_argument("--pseudo_data", action="store_true", help="Pseudo dataset generation")
    
    """Model related arguments"""
    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument("--lm_backbone", type=str, default="t5-large", help="Pretrained language model")
    model_args.add_argument("--visual_backbone", type=str, default="microsoft/swin-base-patch4-window7-224-in22k")
    model_args.add_argument('--max_epochs', type=int, default=10, help='Max epoch size')
    model_args.add_argument('--load_from_epoch', type=str, default=None, help='Loading from epoch')
    model_args.add_argument("--finetuning", action="store_true")
    model_args.add_argument("--no_prompt_proj", action="store_true", help="Do not project prefix")
    model_args.add_argument("--prefix_hidden_size", type=int, default=768)
    model_args.add_argument("--prefix_len", type=int, default=80)
    model_args.add_argument("--prefix_dropout", type=float, default=0.1)


    """Logging related arguments"""
    misc_args = parser.add_argument_group('Logging related & Misc arguments')
    misc_args.add_argument('--seed', type=int, default=42, help='Random Seed')
    misc_args.add_argument('--experiment_name', type=str, default='experiment', help='Experiment name for wandb')
    misc_args.add_argument('--ngpu', type=int, default=1, help='Number of gpu')
    misc_args.add_argument('--ckpt_dir', type=str, default="./ckpts", help='Checkpoint directory')




    args = parser.parse_args()
    return args
