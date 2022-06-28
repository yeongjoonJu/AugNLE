import os
import torch
from pytorch_lightning import Trainer, seed_everything
from dataloader import VQAX_ST_DataModule
from models.lightning import Self_training
import opts
import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import json
from models.utils import pseudo_labeling


AVAIL_GPUS = torch.cuda.device_count()

def self_training(args, selfe_mode, iteration, checkpoint_cb, wan_logger):
    args.selfe_mode = selfe_mode
    dm = VQAX_ST_DataModule(args, mode = selfe_mode, turn = iteration, data_aug = iteration>0)
    model = Self_training(hparams =args)

    trainer = Trainer(max_epochs=args.max_epochs,
                    accelerator = "gpu",
                    gpus= args.ngpu,
                    strategy = "ddp",
                    val_check_interval=args.val_check_interval,
                    accumulate_grad_batches = args.gradient_accumulation_steps,
                    gradient_clip_val=args.gradient_cliping,
                    check_val_every_n_epoch = 1,
                    callbacks=[checkpoint_cb],
                    logger=wan_logger,
                    )
    
    trainer.fit(model, datamodule=dm)
    results = trainer.predict(model, dataloaders=dm)
    # trainer.test(model,dataloaders=dm)
    pseudo_data_pth = pseudo_labeling(selfe_mode, results, args.pseudo_labels_pth, iteration)
    with open(f"{args.output_dir}/{selfe_mode}-k-{args.top_k}_p-{args.top_p}_t-{args.temperature}_itr-{iteration}.json", "w") as fout:
        json.dump(results, fout, indent=2)
        
if __name__ == '__main__':
    seed_everything(42)
    args = opts.get_args()
    args.filename = 'ckpt_stats_' + str(args.load_from_epoch) + '.tar'
    wandb_logger = WandbLogger(project="Self-training_teacher", name =args.experiment_name)

    # Checkpoint call back    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d_%H:%M:%S')
    checkpoint_dir = args.checkpoints_dir + "/"+ nowDatetime + '/'
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_callback = checkpoint_callback = ModelCheckpoint(
        monitor=f'val_loss',
        mode = "min",
        save_top_k = 2,
        dirpath=checkpoint_dir,
        filename='{epoch:02d}-{val_loss:.2f}',
    )    
    

            
    itr = 0 
    while itr < args.iteration:
        # teacher
        print("************************ Teacher *****************************")
        self_training(args, selfe_mode= "teacher", iteration = itr, checkpoint_cb = checkpoint_callback, wan_logger = wandb_logger)
        itr += 1
        print("************************ Student *****************************")
        self_training(args, selfe_mode= "student", iteration = itr, checkpoint_cb = checkpoint_callback, wan_logger = wandb_logger)
        itr += 1