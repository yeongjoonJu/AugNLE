import os
import torch
from pytorch_lightning import Trainer, seed_everything
from dataloader import VQAXDataModule
from lightning import PromptTuning
import opts
import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


AVAIL_GPUS = torch.cuda.device_count()

if __name__ == '__main__':
 
  args = opts.get_args()
  seed_everything(args.seed)
  
  args.filename = 'ckpt_stats_' + str(args.load_from_epoch) + '.tar'
  wandb_logger = WandbLogger(project="Aug_NLX", name =args.experiment_name)
  dm = VQAXDataModule(args)
  dm.setup()

  model = PromptTuning(hparams =args)

  # Checkpoint call back
  
  now = datetime.datetime.now()
  nowDatetime = now.strftime('%Y-%m-%d_%H:%M:%S')
  checkpoint_dir = args.checkpoints_dir + nowDatetime + "/"
  if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
    
  checkpoint_callback = checkpoint_callback = ModelCheckpoint(
      monitor='val_loss',
      mode = "min",
      save_top_k = 2,
      dirpath=checkpoint_dir,
      filename='{epoch:02d}-{val_loss:.2f}',
  )
  
  trainer = Trainer(max_epochs=args.max_epochs,
                  accelerator = "gpu",
                  gpus= args.ngpu,
                  strategy = "deepspeed",
                  val_check_interval=args.val_check_interval,
                  accumulate_grad_batches = args.gradient_accumulation_steps,
                  gradient_clip_val=args.gradient_cliping,
                  check_val_every_n_epoch = 1,
                  callbacks=[checkpoint_callback],
                  logger=wandb_logger,
                  )
  trainer.fit(model, datamodule=dm)