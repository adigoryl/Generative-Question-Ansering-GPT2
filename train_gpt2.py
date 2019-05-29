import numpy as np
import argparse
import torch
import os
import csv
import datetime
import shutil
import math
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2DoubleHeadsModel, OpenAIAdam
import logging
from tqdm import tqdm, trange

import csv
import os
import json

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


def makedir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def load_json_dataset(dataset_path):
    """ Given dataset: (index, song_name, year, artist, genre, lyrics)
        Output a list of tuples(genre, lyrics) """
    cwd = os.path.dirname(__file__)

    with open(cwd + "/dataset/" + dataset_path) as data_file:
        data = json.load(data_file)

    dataset = []
    for i in range(data['data'].__len__()):
        a = []
        for a_idx in range(0, data['data'][i]['answers'].__len__()):
            a.append(data['data'][i]['answers'][a_idx]['span_text'])

        q = []
        for q_idx in range(0, data['data'][i]['questions'].__len__()):
            q.append(data['data'][i]['questions'][q_idx]['input_text'])

        s = data['data'][i]['story']

        dataset.append((s, q, a))

    return dataset


def get_max_lengths(dataset):
    max_a = 0
    max_s = 0
    max_q = 0

    for data in dataset:

        curr_max_s = len(data[0])

        curr_max_q = max(len(q) for q in data[1])

        curr_max_a = max(len(a) for a in data[2])

        if curr_max_s > max_s:
            max_s = curr_max_s

        if curr_max_a > max_a:
            max_a = curr_max_a

        if curr_max_q > max_q:
            max_q = curr_max_q

    print(max_s, max_q, max_a)

    return max_s, max_q, max_a


def format_data(dataset, special_tokens, max_a=200, max_q=50, max_s=770):
    dataset_filter = []

    pos_ids = []
    token_types = []
    mc_labels = []
    mc_tok_ids = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(0, len(dataset)):
        story = dataset[i][0]
        quest = dataset[i][1]
        answ = dataset[i][2]

        story_len = len(story)

        # If story greater in len -> escape the loop / get rid of this data input
        if story_len > max_s:
            continue

        q_idx_arr = []
        a_idx_arr = []

        # Check if Questions or Answers length satisfies
        for j in range(0, len(answ)):

            q_pad = max_q - len(quest[j])
            if q_pad >= 0:
                q_idx_arr.append(j)

            a_pad = max_a - len(answ[j])
            if a_pad >= 0:
                a_idx_arr.append(j)

        # Get only overlapping indexes -> this means an input index satisfied the story, question and answer maximum length
        all_idx = np.intersect1d(q_idx_arr, a_idx_arr)

        for curr_idx in all_idx:
            ### DATA
            q = np.zeros(max_q)
            a = np.zeros(max_a)
            s = np.zeros(max_s)

            q[0: len(np.array(quest)[curr_idx])] = np.array(quest)[curr_idx]
            a[0: len(np.array(answ)[curr_idx])] = np.array(answ)[curr_idx]
            s[0: story_len] = np.array(story)

            full_input = [special_tokens[0]] + s.tolist() + [special_tokens[2]] + q.tolist() + [special_tokens[3]] + a.tolist() + [special_tokens[1]]
            dataset_filter.append(np.array(full_input))

            ### POSITION IDS
            q_pos = np.arange(max_q) + 1
            a_pos = np.arange(max_a) + 1
            s_pos = np.arange(max_s) + 1

            full_pos = [0] + s_pos.tolist() + [0] + q_pos.tolist() + [0] + a_pos.tolist() + [0]
            pos_ids.append(np.array(full_pos))

            ### TOKEN TYPES
            q_tok = np.zeros(max_q)
            a_tok = np.zeros(max_a)
            s_tok = np.zeros(max_s)

            full_tok = [0] + ([1] * len(s_tok.tolist())) + [0] + ([2] * len(q_tok.tolist())) + [0] + ([3] * len(a_tok.tolist())) + [0]
            token_types.append(np.array(full_tok))

            ### Multi Class LABEL
            mc_label = np.zeros((1))
            mc_labels.append(mc_label)

            ### Multi Class TOKEN IDS
            mc_tok_id = np.zeros(1)
            mc_tok_ids.append(mc_tok_id)

    dataset_filter = np.expand_dims(dataset_filter, axis=1)
    token_types = np.expand_dims(token_types, axis=1)
    pos_ids = np.expand_dims(pos_ids, axis=1)

    ### Lang Model LABEL
    # Replace the padding of 0 to -1
    lm_labels = np.copy(dataset_filter)
    lm_labels[np.where(lm_labels == 0)] = -1

    tensor_dataset = []
    inputs = (np.array(dataset_filter), np.array(mc_tok_ids), np.array(lm_labels),  np.array(mc_labels), np.array(token_types), np.array(pos_ids))
    tensor_dataset.append(tuple(torch.tensor(t, dtype=torch.int64, device=torch.device(device)) for t in inputs))

    return tensor_dataset[0]


def tokenize_and_encode(obj, tokenizer):
    """ Tokenize and encode a nested object """
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    elif isinstance(obj, int):
        return obj
    return list(tokenize_and_encode(o, tokenizer) for o in obj)


def main():
    args = init_args()
    # ======================== PATH AND FILES CONSTRUCTION
    now = datetime.datetime.now().strftime("%d-%m-%Y@%H'%M")
    current_dir = os.path.dirname(__file__)
    log_path = "{}/finetuned_models/test/{}_{}_z1".format(current_dir, args.model, now)
    makedir(log_path)
    run_details_file = os.path.join(log_path, "run_details.txt")

    # Prepare model files
    model_dir = "{}/model".format(log_path)
    makedir(model_dir)
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

    special_tokens = ['<_STR_>', '<_END_>', '<_QUE_>', '<_ANS_>']

    d_file.write("\n\nSPECIAL TOKENS: {}".format(special_tokens))
    d_file.close()

    # ======================== LOAD PRE-TRAINED MODEL AND TOKENIZER (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model, special_tokens=special_tokens)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    model = GPT2DoubleHeadsModel.from_pretrained(args.model, num_special_tokens=len(special_tokens_ids))

    raw_data = load_json_dataset(args.train_dataset)
    token_data = tokenize_and_encode(raw_data, tokenizer)
    # max_s, max_q, max_a = get_max_lengths(token_data)
    new_data = format_data(token_data, special_tokens_ids)

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

    # Prepare the device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    # Automatic mixed precision - speeds up the process and shrinks the model size
    # while maintaining full precision accuracy
    # Requirements cuda
    if args.amp_opt_lvl:
        from apex import amp  # Apex is only required if we use fp16 training
        model.to(device)
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_lvl)

    if n_gpu > 1: model = torch.nn.DataParallel(model, output_device=device, device_ids=range(torch.cuda.device_count()))

    if args.do_train:
        all_tr_losses = []
        model.train()
        model.to(device)
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch_losses = []
            past = None
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            # for input_ids, position_ids, lm_labels in train_tensor_data:
            for step, batch in enumerate(tqdm_bar):
                lm_loss, _mc_loss, _ = model(*batch, past=past)
                loss = args.lm_coef * lm_loss[0]
                # Normalise the loss (Simulates average of a batch)
                loss = loss / args.grad_accumulation_steps
                if args.amp_opt_lvl:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (step + 1) % args.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_losses.append(loss.item())

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
