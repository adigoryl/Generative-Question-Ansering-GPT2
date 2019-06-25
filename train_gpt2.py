import numpy as np
import datetime
import shutil

import torch
from torch import nn
import argparse
import torch.nn.functional as F

from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2DoubleHeadsModel, OpenAIAdam
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from tqdm import tqdm, trange

import utils.utils as u
from utils.parallel import DataParallelModel, DataParallelCriterion
import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_des", default="LM + pos_id", type=str, help="Description to help identify the run")
    parser.add_argument("--model", default="gpt2", type=str, help="Model name i.e.: gpt2, gpt2-medium")
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument('--train_dataset', type=str, default='', required=True)
    parser.add_argument('--multi_class_len', type=int, default=2)
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help="This is equivalent to batch size, if the GPU has limited memory")
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.000625)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument("--amp_opt_lvl", type=str, default=None, help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    args = parser.parse_args()
    return args


def format_data(dataset, special_tokens, device, multi_class_len, max_a, max_q, max_s):
    dataset = u.filter_len(dataset, max_a, max_q, max_s)
    inputs = []
    pos_ids = []
    token_types = []
    mc_labels = []
    mc_tok_ids = []

    # dataset = np.load("/Users/aw678/PycharmProjects/gpt2_QA/dataset_filtered.npy").tolist()

    for i in range(0, len(dataset)):
        story = dataset[i][0]
        quest = dataset[i][1]
        answ = dataset[i][2]

        r = np.random.randint(0, len(dataset)-1)
        f_quest = dataset[r][1]
        f_answ = dataset[r][2]

        for idx_qa in range(0, len(answ)):
            fake_qa_idx = np.random.randint(len(f_answ))

            for k in range(0, multi_class_len):
                option = np.random.rand()

                if k == 0:
                    full_input, full_pos, full_tok = u.prep_pad(max_q, max_a, max_s, story, quest[idx_qa], answ[idx_qa], special_tokens, 0)
                    inputs.append([np.array(full_input)])
                    pos_ids.append([np.array(full_pos)])
                    token_types.append([np.array(full_tok)])
                    ### Multi Class TOKEN IDS
                    if option > 0.8:
                        mc_tok_id = [np.where(np.array(full_input)==special_tokens[1])[0][0]]
                    elif option <= 0.8:
                        mc_tok_id = [np.where(np.array(full_input)==special_tokens[2])[0][0]]

                    mc_tok_ids.append(mc_tok_id)

                elif k != 0 and option > 0.8:
                    full_input, full_pos, full_tok = u.prep_pad(max_q, max_a, max_s, story, f_quest[fake_qa_idx], answ[idx_qa], special_tokens, 1)
                    inputs[-1].append(np.array(full_input))
                    pos_ids[-1].append(np.array(full_pos))
                    token_types[-1].append(np.array(full_tok))
                    mc_tok_ids[-1].append(np.where(np.array(full_input)==special_tokens[1])[0][0])

                elif k != 0 and option <= 0.8:
                    full_input, full_pos, full_tok = u.prep_pad(max_q, max_a, max_s, story, quest[idx_qa], f_answ[fake_qa_idx], special_tokens, 2)
                    inputs[-1].append(np.array(full_input))
                    pos_ids[-1].append(np.array(full_pos))
                    token_types[-1].append(np.array(full_tok))
                    mc_tok_ids[-1].append(np.where(np.array(full_input)==special_tokens[2])[0][0])

            ### Multi Class LABEL
            mc_label = np.zeros((1))
            mc_labels.append(mc_label)

    ### Lang Model LABEL
    # Replace the padding of 0 to -1
    lm_labels = np.copy(inputs)
    lm_labels[np.where(lm_labels == 0)] = -1

    tensor_dataset = []
    input_tuple = (np.array(inputs), np.array(mc_tok_ids), np.array(lm_labels),  np.array(mc_labels), np.array(token_types), np.array(pos_ids))
    tensor_dataset.append(tuple(torch.tensor(t, dtype=torch.int64, device=device) for t in input_tuple))

    return tensor_dataset[0]


class discriminator(nn.Module):
    def __init__(self, in_features, out_features):
        super(discriminator, self).__init__()
        # Bias by default = True
        self.fwd = nn.Linear(in_features, out_features)

    def forward(self, x, args):
        if args.classifier_mean:
            x = torch.mean(x, 1).squeeze()
        else:
            x = x[0, -1, :]
        x = self.fwd(x)
        return F.softmax(x, dim=0)


def main():
    args = init_args()
    # ======================== PATH AND FILES CONSTRUCTION
    now = datetime.datetime.now().strftime("%d-%m-%Y@%H'%M")
    current_dir = os.path.dirname(__file__)
    log_path = "{}/finetuned_models/test1/{}_{}".format(current_dir, args.model, now)
    u.makedir(log_path)
    run_details_file = os.path.join(log_path, "run_details.txt")

    # Prepare model files
    model_dir = "{}/model".format(log_path)
    u.makedir(model_dir)
    model_file = os.path.join(model_dir, "pytorch_model.bin")
    config_file = os.path.join(model_dir, "config_file.bin")

    # ======================== COPY OF CURRENT FILE
    shutil.copy2(__file__, "{}/copy_of_code_that_run_this_experiment.py".format(log_path))

    # ======================== NOTE ARGUMENTS
    # Open a file and appends to a file. If doesn't exists (+) means to create it.
    d_file = open(run_details_file, "a+")
    d_file.write("@" * 30 + " RUN INFO " + "@" * 30)
    d_file.write("\n\nDATE: {}".format(now))
    d_file.write("\n\nUSING THE FOLLOWING ARGS:\n{}".format(args))

    special_tokens = ['<_S_STORY_>', '<_S_QUE_>', '<_S_ANS_>',
                      '<_E_STORY_>', '<_E_QUE_>', '<_E_ANS_>']

    d_file.write("\n\nSPECIAL TOKENS: {}".format(special_tokens))
    d_file.close()

    # ======================== LOAD PRE-TRAINED MODEL AND TOKENIZER (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model, special_tokens=special_tokens)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    model = GPT2DoubleHeadsModel.from_pretrained(args.model, num_special_tokens=len(special_tokens_ids))

    # Prepare the device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))
    cwd = os.path.dirname(__file__)

    raw_data = u.load_json_dataset(args.train_dataset, cwd)
    token_data = u.tokenize_and_encode(raw_data, tokenizer)
    # max_s, max_q, max_a = u.get_max_lengths(token_data)
    if args.model == "gpt2-medium":
        new_data = format_data(token_data, special_tokens_ids, device, multi_class_len=args.multi_class_len, max_a=200, max_q=48, max_s=770)
    elif args.model == "gpt2":
        new_data = format_data(token_data, special_tokens_ids, device, multi_class_len=args.multi_class_len, max_a=150, max_q=48, max_s=564)


    # ======================== Use the pytorch's dataloader to load the input
    train_data = TensorDataset(*new_data)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # ======================== Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = len(new_data[0]) * args.num_train_epochs // args.train_batch_size
    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                           lr=args.learning_rate,
                           warmup=args.warmup_proportion,
                           max_grad_norm=args.max_grad_norm,
                           weight_decay=args.weight_decay,
                           t_total=num_train_optimization_steps)


    # Automatic mixed precision - speeds up the process and shrinks the model size
    # while maintaining full precision accuracy
    # Requirements cuda
    model = model.to(device)
    if args.amp_opt_lvl:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_lvl)

    if n_gpu > 1: model = torch.nn.DataParallel(model, output_device=device, device_ids=range(torch.cuda.device_count()))


    if args.do_train:
        all_tr_losses = []
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch_losses = []
            past = None
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            # for input_ids, position_ids, lm_labels in train_tensor_data:
            for step, batch in enumerate(tqdm_bar):

                loss, _hidden_states, _ = model(*batch, past=past)
                loss = args.lm_coef * (loss[0] + loss[1])
                # Normalise the loss (Simulates average of a batch)
                loss = loss / args.grad_accumulation_steps
                if args.amp_opt_lvl:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        if n_gpu >1: scaled_loss.sum().backward()
                        else: scaled_loss.backward()
                else:
                    loss.backward()

                if (step + 1) % args.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if n_gpu > 1: epoch_losses.append(loss.sum().item())
                    else: epoch_losses.append(loss.item())

            all_tr_losses.append(epoch_losses)

        # Get a note of the losses for future visualisations
        file = open(run_details_file, "a+")
        file.write("\n\nTraining losses of every n = (BATCH_SIZE * ACCUMULATION STEPS):\n{}".format(str(all_tr_losses)))
        file.close()

    # ======================== Save a trained model
    if args.do_train:
        # Save model, config and vocab files
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), model_file)
        model_to_save.config.to_json_file(config_file)
        tokenizer.save_vocabulary(model_dir)

if __name__ == '__main__':
    main()
