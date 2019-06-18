#!/usr/bin/env python3

import argparse
import logging
import os
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2DoubleHeadsModel, GPT2Config, GPT2Tokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def format_data6(dataset, special_tokens, max_a=200, max_q=50, max_s=770):
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
            local_story = story

            ### DATA
            q = np.zeros(max_q)
            a = np.zeros(max_a)
            s = np.zeros(max_s)

            q[0: len(np.array(quest)[curr_idx])] = np.array(quest[curr_idx])
            a[0: len(np.array(answ)[curr_idx])] = np.array(answ[curr_idx])
            s[0: story_len] = np.array(local_story)

            full_input = [special_tokens[0]] + s.tolist() + [special_tokens[2]] + q.tolist() + [special_tokens[3]] + a.tolist()
            dataset_filter.append(np.array(full_input))

            ### ANSWER IN STORY SPAN PADDING
            q_pos = np.zeros(max_q)
            a_pos = np.zeros(max_a)
            s_pos = np.zeros(max_s)

            q_pos[0: len(np.array(quest)[curr_idx])] = 5
            a_pos[0: len(np.array(answ)[curr_idx])] = 10
            # # s_pos[span_idxs[curr_idx][0][0]:span_idxs[curr_idx][0][1]] = 10
            #
            # s_pos[int(span_idxs[curr_idx][0]*story_len):int(span_idxs[curr_idx][1]*story_len)] = 10
            # # s_pos[int(span_idxs[curr_idx][0]*story_len)] = 5

            full_pos = [1] + s_pos.tolist() + [1] + q_pos.tolist() + [1] + a_pos.tolist()
            pos_ids.append(np.array(full_pos))

            ### TOKEN TYPES
            q_tok = np.zeros(max_q)
            a_tok = np.zeros(max_a)
            s_tok = np.zeros(max_s)

            q_tok[0: len(np.array(quest)[curr_idx])] = 6
            a_tok[0: len(np.array(answ)[curr_idx])] = 7
            s_tok[0: story_len] = 5

            full_tok = [1] + s_tok.tolist() + [2] + q_tok.tolist() + [3] + a_tok.tolist()
            token_types.append(np.array(full_tok))


    dataset_filter = np.expand_dims(dataset_filter, axis=1)
    token_types = np.expand_dims(token_types, axis=1)
    pos_ids = np.expand_dims(pos_ids, axis=1)

    tensor_dataset = []
    inputs = (np.array(dataset_filter), np.array(token_types), np.array(pos_ids))
    tensor_dataset.append(tuple(torch.tensor(t, dtype=torch.int64, device=torch.device(device)) for t in inputs))
    new_word_index = 1 + max_s + 1 + max_q + 1 + len(answ[0])-1

    return tensor_dataset[0], new_word_index


def format_data4(dataset, special_tokens, max_a=200, max_q=50, max_s=770):
    dataset_filter = []

    pos_ids = []
    token_types = []

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
        new_word_index = 0
        for curr_idx in all_idx:
            ### DATA
            q = np.zeros(len(np.array(quest)[curr_idx]))
            a = np.zeros(len(np.array(answ)[curr_idx]))
            s = np.zeros(story_len)

            q[0: len(np.array(quest)[curr_idx])] = np.array(quest)[curr_idx]
            a[0: len(np.array(answ)[curr_idx])] = np.array(answ)[curr_idx]
            s[0: story_len] = np.array(story)

            full_input = [special_tokens[0]] + s.tolist() + [special_tokens[2]] + q.tolist() + [special_tokens[3]] + a.tolist()
            dataset_filter.append(np.array(full_input))

            ### POSITION IDS
            q_pos = np.arange(len(np.array(quest)[curr_idx])) + 2
            a_pos = np.arange(len(np.array(answ)[curr_idx])) + 2
            s_pos = np.arange(story_len) + 2

            # q_pos[len(np.array(quest)[curr_idx]):max_q] = 0
            # a_pos[len(np.array(answ)[curr_idx]):max_a] = 0
            # s_pos[story_len:max_s] = 0

            full_pos = [1] + s_pos.tolist() + [1] + q_pos.tolist() + [1] + a_pos.tolist()
            pos_ids.append(np.array(full_pos))

            ### TOKEN TYPES
            q_tok = np.zeros(len(np.array(quest)[curr_idx]))
            a_tok = np.zeros(len(np.array(answ)[curr_idx]))
            s_tok = np.zeros(story_len)

            q_tok[0: len(np.array(quest)[curr_idx])] = 6
            a_tok[0: len(np.array(answ)[curr_idx])] = 7
            s_tok[0: story_len] = 5

            full_tok = [1] + s_tok.tolist() + [2] + q_tok.tolist() + [3] + a_tok.tolist()
            token_types.append(np.array(full_tok))

            new_word_index = 1 + story_len + 1 + len(np.array(quest)[curr_idx]) + 1 +len(np.array(answ)[curr_idx])-1

    dataset_filter = np.expand_dims(dataset_filter, axis=1)
    token_types = np.expand_dims(token_types, axis=1)
    pos_ids = np.expand_dims(pos_ids, axis=1)

    tensor_dataset = []
    inputs = (np.array(dataset_filter), np.array(token_types), np.array(pos_ids))
    tensor_dataset.append(tuple(torch.tensor(t, dtype=torch.int64, device=torch.device(device)) for t in inputs))

    # new_word_index = 1 + max_s + 1 + max_q + 1 + len(answ[0])-1

    return tensor_dataset[0], new_word_index


def format_data3(dataset, special_tokens, max_a=200, max_q=50, max_s=770):
    dataset_filter = []

    pos_ids = []
    token_types = []

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
            a = np.zeros(len(np.array(answ)[curr_idx]))
            s = np.zeros(max_s)

            q[0: len(np.array(quest)[curr_idx])] = np.array(quest)[curr_idx]
            a[0: len(np.array(answ)[curr_idx])] = np.array(answ)[curr_idx]
            s[0: story_len] = np.array(story)

            full_input = [special_tokens[0]] + s.tolist() + [special_tokens[2]] + q.tolist() + [special_tokens[3]] + a.tolist()
            dataset_filter.append(np.array(full_input))

            ### POSITION IDS
            q_pos = np.arange(max_q) + 2
            a_pos = np.arange(len(np.array(answ)[curr_idx])) + 2
            s_pos = np.arange(max_s) + 2

            q_pos[len(np.array(quest)[curr_idx]):max_q] = 0
            a_pos[len(np.array(answ)[curr_idx]):max_a] = 0
            s_pos[story_len:max_s] = 0

            full_pos = [1] + s_pos.tolist() + [1] + q_pos.tolist() + [1] + a_pos.tolist()
            pos_ids.append(np.array(full_pos))

            ### TOKEN TYPES
            q_tok = np.zeros(max_q)
            a_tok = np.zeros(len(np.array(answ)[curr_idx]))
            s_tok = np.zeros(max_s)

            q_tok[0: len(np.array(quest)[curr_idx])] = 6
            a_tok[0: len(np.array(answ)[curr_idx])] = 7
            s_tok[0: story_len] = 5

            full_tok = [1] + s_tok.tolist() + [2] + q_tok.tolist() + [3] + a_tok.tolist()
            token_types.append(np.array(full_tok))


    dataset_filter = np.expand_dims(dataset_filter, axis=1)
    token_types = np.expand_dims(token_types, axis=1)
    pos_ids = np.expand_dims(pos_ids, axis=1)

    tensor_dataset = []
    inputs = (np.array(dataset_filter), np.array(token_types), np.array(pos_ids))
    tensor_dataset.append(tuple(torch.tensor(t, dtype=torch.int64, device=torch.device(device)) for t in inputs))

    new_word_index = 1 + max_s + 1 + max_q + 1 + len(answ[0])-1

    return tensor_dataset[0], new_word_index


def format_data5(dataset, special_tokens, max_a=200, max_q=50, max_s=770):
    dataset_filter = []

    pos_ids = []
    token_types = []

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
            a = np.zeros(len(np.array(answ)[curr_idx]))
            s = np.zeros(max_s)

            q[0: len(np.array(quest)[curr_idx])] = np.array(quest)[curr_idx]
            a[0: len(np.array(answ)[curr_idx])] = np.array(answ)[curr_idx]
            s[0: story_len] = np.array(story)

            full_input = [special_tokens[0]] + s.tolist() + [special_tokens[2]] + q.tolist() + [special_tokens[3]] + a.tolist()
            dataset_filter.append(np.array(full_input))

            ### POSITION IDS
            q_pos = np.zeros(max_q)
            a_pos = np.zeros(len(np.array(answ)[curr_idx]))
            s_pos = np.zeros(max_s)

            # q_pos[len(np.array(quest)[curr_idx]):max_q] = 0
            # a_pos[len(np.array(answ)[curr_idx]):max_a] = 0
            # s_pos[story_len:max_s] = 0

            full_pos = [1] + s_pos.tolist() + [2] + q_pos.tolist() + [3] + a_pos.tolist()
            pos_ids.append(np.array(full_pos))

            ### TOKEN TYPES
            q_tok = np.zeros(max_q)
            a_tok = np.zeros(len(np.array(answ)[curr_idx]))
            s_tok = np.zeros(max_s)

            q_tok[0: len(np.array(quest)[curr_idx])] = 2
            a_tok[0: len(np.array(answ)[curr_idx])] = 3
            s_tok[0: story_len] = 1

            full_tok = [5] + s_tok.tolist() + [5] + q_tok.tolist() + [5] + a_tok.tolist()
            token_types.append(np.array(full_tok))


    dataset_filter = np.expand_dims(dataset_filter, axis=1)
    token_types = np.expand_dims(token_types, axis=1)
    pos_ids = np.expand_dims(pos_ids, axis=1)

    tensor_dataset = []
    inputs = (np.array(dataset_filter), np.array(token_types), np.array(pos_ids))
    tensor_dataset.append(tuple(torch.tensor(t, dtype=torch.int64, device=torch.device(device)) for t in inputs))

    new_word_index = 1 + max_s + 1 + max_q + 1 + len(answ[0])-1

    return tensor_dataset[0], new_word_index


def format_data2(dataset, special_tokens, max_a=200, max_q=50, max_s=770):
    dataset_filter = []
    pos_ids = []
    token_types = []

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
            q_pos = np.arange(max_q) + 2
            a_pos = np.arange(max_a) + 2
            s_pos = np.arange(max_s) + 2

            q_pos[len(np.array(quest)[curr_idx]):max_q] = 0
            a_pos[len(np.array(answ)[curr_idx]):max_a] = 0
            s_pos[story_len:max_s] = 0

            full_pos = [1] + s_pos.tolist() + [1] + q_pos.tolist() + [1] + a_pos.tolist() + [1]
            pos_ids.append(np.array(full_pos))

            ### TOKEN TYPES
            q_tok = np.zeros(max_q)
            a_tok = np.zeros(max_a)
            s_tok = np.zeros(max_s)

            q_tok[0: len(np.array(quest)[curr_idx])] = 6
            a_tok[0: len(np.array(answ)[curr_idx])] = 7
            s_tok[0: story_len] = 5

            full_tok = [1] + s_tok.tolist() + [2] + q_tok.tolist() + [3] + a_tok.tolist() + [4]
            token_types.append(np.array(full_tok))

    dataset_filter = np.expand_dims(dataset_filter, axis=1)
    token_types = np.expand_dims(token_types, axis=1)
    pos_ids = np.expand_dims(pos_ids, axis=1)

    tensor_dataset = []
    inputs = (np.array(dataset_filter), np.array(token_types), np.array(pos_ids))
    tensor_dataset.append(tuple(torch.tensor(t, dtype=torch.int64, device=torch.device(device)) for t in inputs))

    new_word_index = 1 + max_s + 1 + max_q + 1 + len(answ[0])-1

    return tensor_dataset[0], new_word_index


def format_data(dataset, special_tokens, max_a=200, max_q=50, max_s=770):

    dataset_filter = []
    pos_ids = []
    token_types = []

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
        for j in range(0, len(quest)):

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

            q_pos[len(np.array(quest)[curr_idx]):max_q] = 0
            a_pos[len(np.array(answ)[curr_idx]):max_a] = 0
            s_pos[story_len:max_s] = 0

            full_pos = [0] + s_pos.tolist() + [0] + q_pos.tolist() + [0] + a_pos.tolist() + [0]
            pos_ids.append(np.array(full_pos))

            ### TOKEN TYPES
            q_tok = np.zeros(max_q)
            a_tok = np.zeros(max_a)
            s_tok = np.zeros(max_s)

            q_tok[len(np.array(quest)[curr_idx]):max_q] = 2
            a_tok[len(np.array(answ)[curr_idx]):max_a] = 3
            s_tok[story_len:max_s] = 1

            full_tok = [0] + s_tok.tolist() + [0] + q_tok.tolist() + [0] + a_tok.tolist() + [0]
            token_types.append(np.array(full_tok))

    dataset_filter = np.expand_dims(dataset_filter, axis=1)
    token_types = np.expand_dims(token_types, axis=1)
    pos_ids = np.expand_dims(pos_ids, axis=1)

    tensor_dataset = []
    inputs = (np.array(dataset_filter), np.array(token_types), np.array(pos_ids))
    tensor_dataset.append(tuple(torch.tensor(t, dtype=torch.int64, device=torch.device(device)) for t in inputs))

    new_word_index = 1 + max_s + 1 + max_q + 1 + len(answ[0])-1

    return tensor_dataset[0], new_word_index


def sample_word(model, inputs=None, special_tokens=None, temperature=1, top_k=0, sample=True):

    past = None
    with torch.no_grad():
        context, new_word_index = format_data6(inputs, special_tokens)
        hid_states, presents = model(*context, past=past)
        hid_states = hid_states[0, 0, new_word_index, :] / temperature
        hid_states = top_k_logits(hid_states, k=top_k)
        log_probs = F.softmax(hid_states, dim=-1)

        if sample:
            new_word = torch.multinomial(log_probs, num_samples=1)
        else:
            _, new_word = torch.topk(log_probs, k=5, dim=-1)

    return new_word.item()


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# # Here is how to use this function for top-p sampling
# temperature = 1.0
# top_k = 0
# top_p = 0.9
#
# # Get logits with a forward pass in our model (input is pre-defined)
# logits = model(input)
#
# # Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
# logits = logits[0, -1, :] / temperature
# filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
#
# # Sample from the filtered distribution
# probabilities = F.softmax(filtered_logits, dim=-1)
# next_token = torch.multinomial(probabilities, 1)


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="gpt2", help='pretrained model name or path to local checkpoint')
    parser.add_argument('--load_model_path', type=str, default="/Users/aw678/PycharmProjects/gpt2_QA/finetuned_models/test/gpt2_02-06-2019@15'34_z1/model/")
    # parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=np.random.randint(0,100))
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', type=bool, default=False, help='If true, unconditional generation.')
    parser.add_argument('--special_tag', type=str, default='<_ROCK_>', help='If unconditional, this tag will be used to initiate the generation')

    args = parser.parse_args()
    print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model_dict = model.state_dict()

    # Prepare paths and file names
    output_model_file = os.path.join(args.load_model_path, "pytorch_model.bin")
    output_config_file = os.path.join(args.load_model_path, "config_file.bin")

    # Load fine-tuned model and used volcabulary
    special_tokens = ['<_STR_>', '<_END_>', '<_QUE_>', '<_ANS_>']
    # enc = GPT2Tokenizer(output_vocab_file, output_merges_file, special_tokens=special_tokens)
    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path, special_tokens=special_tokens)

    # Filter out the multi-classification weights
    config = GPT2Config.from_json_file(output_config_file)
    pretrained_model = GPT2LMHeadModel(config)
    # load the pretrained dict
    pretrained_dict = torch.load(output_model_file, map_location='cpu')
    # filter out pretrained_weights so that the content is the same as of standard LM model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_model.load_state_dict(pretrained_dict)

    pretrained_model.to(device)
    pretrained_model.eval()

    if args.length == -1:
        args.length = model.config.n_ctx
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    while True:
        context_tokens = []
        if not args.unconditional:
            # story_text = input("Story input >>> ")
            # while not story_text:
            #     print('Prompt should not be empty!')
            #     story_text = input("Story input >>> ")
            #
            # question_text = input("Question input >>> ")
            # while not question_text:
            #     print('Prompt should not be empty!')
            #     question_text = input("Question input >>> ")
            # question_tokens = enc.encode(question_text)
            # story_tokens = enc.encode(story_text)

            # story_text = "New York (CNN) -- More than 80 Michael Jackson collectibles -- including the late pop star's famous rhinestone-studded glove from a 1983 performance -- were auctioned off Saturday, reaping a total $2 million. Profits from the auction at the Hard Rock Cafe in New York's Times Square crushed pre-sale expectations of only $120,000 in sales. The highly prized memorabilia, which included items spanning the many stages of Jackson's career, came from more than 30 fans, associates and family members, who contacted Julien's Auctions to sell their gifts and mementos of the singer. Jackson's flashy glove was the big-ticket item of the night, fetching $420,000 from a buyer in Hong Kong, China. Jackson wore the glove at a 1983 performance during \"Motown 25,\" an NBC special where he debuted his revolutionary moonwalk. Fellow Motown star Walter Clyde Orange of the Commodores, who also performed in the special 26 years ago, said he asked for Jackson's autograph at the time, but Jackson gave him the glove instead. The legacy that [Jackson] left behind is bigger than life for me, Orange said. I hope that through that glove people can see what he was trying to say in his music and what he said in his music. Orange said he plans to give a portion of the proceeds to charity. Hoffman Ma, who bought the glove on behalf of Ponte 16 Resort in Macau, paid a 25 percent buyer's premium, which was tacked onto all final sales over $50,000. Winners of items less than $50,000 paid a 20 percent premium. "
            # question_text = "Where was the Auction held?"
            # answer_text = "Adrian lived in"

            # story_text = "The laptop is red because it is made of bricks. The laptop is broken because it has fallen down on the floor."
            # question_text = "Why is the laptop broken?"
            # # question_text = "What is the laptop made of?"
            # # question_text = "What is broken?"
            # answer_text = "It is broken because"
            # # answer_text = "It is"
            # answer_text = ""

            story_text = "Once upon a time, in a barn near a farm house, there lived a little white kitten named Cotton. Cotton lived high up in a nice warm place above the barn where all of the farmer's horses slept. But Cotton wasn't alone in her little home above the barn, oh no. She shared her hay bed with her mommy and 5 other sisters."
            question_text = "Where did Cotton live?"
            answer_text = ""

            special_token_ids = enc.convert_tokens_to_ids(special_tokens)

            story_text = enc.convert_tokens_to_ids(enc.tokenize(story_text))
            question_text = enc.convert_tokens_to_ids(enc.tokenize(question_text))
            answer_text = enc.convert_tokens_to_ids(enc.tokenize(answer_text))

            inputs = []
            inputs.append((story_text, [question_text], [answer_text]))

            for _ in trange(args.length):
                new_word_ids = sample_word(
                    model=pretrained_model,
                    inputs=inputs,
                    special_tokens=special_token_ids,
                    temperature=args.temperature, top_k=args.top_k
                )
                inputs[0][2][0].append(new_word_ids)
                # print(enc.decode(new_word_ids.tolist()))

            answer = enc.decode(inputs[0][2][0], skip_special_tokens=False)
            print(answer)
            print("=" * 80)
            break

        if args.unconditional:
            generated = 0
            for _ in trange(args.nsamples // args.batch_size):
                out = sample_word(
                    model=pretrained_model,
                    inputs=None,
                    start_token=enc.convert_tokens_to_ids(args.special_tag),
                    temperature=args.temperature, top_k=args.top_k
                )
                for i in range(args.batch_size):
                    generated += 1
                    text = enc.decode(out[i], skip_special_tokens=False)
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)
            if args.unconditional:
                  break

if __name__ == '__main__':
    run_model()


