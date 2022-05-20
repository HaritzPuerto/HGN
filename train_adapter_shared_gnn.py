import argparse
import numpy as np
import logging
import sys

from os.path import join
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from csr_mhqa.argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from csr_mhqa.data_processing import Example, InputFeatures, DataHelper
from csr_mhqa.utils import *

from transformers import AdapterGraphQA
from transformers import get_linear_schedule_with_warmup

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_training_params(graphqa, print_stats=False):
    params = []
    params_name = []
    params_name_frozen = []

    num_training_params = 0
    num_fronzen_params = 0
    num_params_hgn = 0
    training_params = ['adapter', 'hgn', 'predict_layer']

    for n, p in graphqa.named_parameters():
        trained = False
        for trained_param in training_params:
            if trained_param in n:
                num_training_params += p.numel()
                trained = True
                params.append(p)
                params_name.append(n)
        if not trained:
            num_fronzen_params += p.numel()
            params_name_frozen.append(n)
        if 'encoder' in n:
            num_params_hgn += p.numel()
        if 'hgn' in n:
            num_params_hgn += p.numel()
        if 'predict_layer' in n:
            num_params_hgn += p.numel()
        if 'adapter' in n:
            num_params_hgn -= p.numel()
    if print_stats:
        num_total_params = num_training_params + num_fronzen_params
        print(f"Number of training parameters: {num_training_params/1e6:.2f}M")
        print(f"Number of frozen parameters: {num_fronzen_params/1e6:.2f}M")
        print(f"Number of total parameters: {num_total_params/1e6:.2f}M")
        print(f"Number of adapter parameters: {(num_total_params - num_params_hgn)/1e6:.2f}M")
        print(f"-----------------------")
        print(f"Number of training parameters in original HGN: {num_params_hgn/1e6:.2f}M")
        
    return params_name, params


#########################################################################
# Initialize arguments
##########################################################################
parser = default_train_parser()

logger.info("IN CMD MODE")
args_config_provided = parser.parse_args(sys.argv[1:])
if args_config_provided.config_file is not None:
    argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
else:
    argv = sys.argv[1:]
args = parser.parse_args(argv)
args = complete_default_train_parser(args)

logger.info('-' * 100)
logger.info('Input Argument Information')
logger.info('-' * 100)
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

#########################################################################
# Read Data
##########################################################################
helper = DataHelper(gz=True, config=args)

# Set datasets
train_dataloader = helper.train_loader
dev_example_dict = helper.dev_example_dict
dev_feature_dict = helper.dev_feature_dict
dev_dataloader = helper.dev_loader

#########################################################################
# Initialize Model
##########################################################################
cached_config_file = join(args.exp_name, 'cached_config.bin')
if os.path.exists(cached_config_file):
    cached_config = torch.load(cached_config_file)
    encoder_path = join(args.exp_name, cached_config['encoder'])
    model_path = join(args.exp_name, cached_config['model'])
    learning_rate = cached_config['lr']
    start_epoch = cached_config['epoch']
    best_joint_f1 = cached_config['best_joint_f1']
    logger.info("Loading encoder from: {}".format(encoder_path))
    logger.info("Loading model from: {}".format(model_path))
else:
    encoder_path = None
    model_path = None
    start_epoch = 0
    best_joint_f1 = 0
    learning_rate = args.learning_rate

model = AdapterGraphQA('roberta-large', args)
model.to(args.device)

params_name, params = get_training_params(model, print_stats=True)


_, _, tokenizer_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(args.encoder_name_or_path,
                                            do_lower_case=args.do_lower_case)


#########################################################################
# Get Optimizer
##########################################################################
if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

no_decay = ["bias", "LayerNorm.weight"]
weight_decay = 0
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in zip(params_name, params) if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {"params": [p for n, p in zip(params_name, params) if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)


scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_steps,
                                            num_training_steps=t_total)

#########################################################################
# launch training
##########################################################################
global_step = 0
loss_name = ["loss_total", "loss_span", "loss_type", "loss_sup", "loss_ent", "loss_para"]
tr_loss, logging_loss = [0] * len(loss_name), [0]* len(loss_name)


model.zero_grad()

train_iterator = trange(start_epoch, start_epoch+int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
for epoch in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    train_dataloader.refresh()
    dev_dataloader.refresh()

    for step, batch in enumerate(epoch_iterator):
        model.train()

        batch['context_mask'] = batch['context_mask'].float().to(args.device)
        start, end, q_type, paras, sents, ents, yp1, yp2 = model(batch, input_ids=batch['context_idxs'], attention_mask=batch['context_mask'])

        loss_list = compute_loss(args, batch, start, end, paras, sents, ents, q_type)
        del batch

        loss_list[0].backward()
        params_name, params = get_training_params(model)
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        
        for idx in range(len(loss_name)):
            if not isinstance(loss_list[idx], int):
                tr_loss[idx] += loss_list[idx].data.item()
            else:
                tr_loss[idx] += loss_list[idx]

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                avg_loss = [ (_tr_loss - _logging_loss) / (args.logging_steps*args.gradient_accumulation_steps)
                             for (_tr_loss, _logging_loss) in zip(tr_loss, logging_loss)]

                loss_str = "step[{0:6}] " + " ".join(['%s[{%d:.5f}]' % (loss_name[i], i+1) for i in range(len(avg_loss))])
                logger.info(loss_str.format(global_step, *avg_loss))

                
                logging_loss = tr_loss.copy()
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
    
    output_pred_file = os.path.join(args.exp_name, f'pred.epoch_{epoch+1}.json')
    output_eval_file = os.path.join(args.exp_name, f'eval.epoch_{epoch+1}.txt')
    metrics, threshold = eval_model(args, model,
                                    dev_dataloader, dev_example_dict, dev_feature_dict,
                                    output_pred_file, output_eval_file, args.dev_gold_file)

    if metrics['joint_f1'] >= best_joint_f1:
        best_joint_f1 = metrics['joint_f1']
        torch.save({'epoch': epoch+1,
                    'lr': scheduler.get_lr()[0],
                    'encoder': 'encoder.pkl',
                    'model': 'model.pkl',
                    'best_joint_f1': best_joint_f1,
                    'threshold': threshold},
                    join(args.exp_name, f'cached_config.bin')
        )
        logger.info(f'Saving model at epoch {epoch+1} with best joint f1 {best_joint_f1}')
    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                join(args.exp_name, f'model_{epoch+1}.pkl'))


