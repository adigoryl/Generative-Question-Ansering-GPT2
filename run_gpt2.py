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
        context, new_word_index = format_data(inputs, special_tokens)
        hid_states, presents = model(*context, past=past)
        hid_states = hid_states[0,0, new_word_index, :] / temperature
        hid_states = top_k_logits(hid_states, k=top_k)
        log_probs = F.softmax(hid_states, dim=-1)

        if sample:
            new_word = torch.multinomial(log_probs, num_samples=1)
        else:
            _, new_word = torch.topk(log_probs, k=1, dim=-1)

    return new_word.item()


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="gpt2", help='pretrained model name or path to local checkpoint')
    parser.add_argument('--load_model_path', type=str, default="/Users/aw678/PycharmProjects/gpt2_QA/finetuned_models/test/gpt2_24-05-2019@23'53_z1/model/")
    parser.add_argument("--seed", type=int, default=0)
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
    # output_vocab_file = os.path.join(args.load_model_path, "vocab.json")
    # output_merges_file = os.path.join(args.load_model_path, "merges.txt")
    # output_vocab_dir = args.load_model_path

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
            story_text = "Once upon a time, in a barn near a farm house, there lived a little white kitten named Cotton. Cotton lived high up in a nice warm place above the barn where all of the farmer's horses slept. But Cotton wasn't alone in her little home above the barn, oh no. She shared her hay bed with her mommy and 5 other sisters. All of her sisters were cute and fluffy, like Cotton. But she was the only white one in the bunch. The rest of her sisters were all orange with beautiful white tiger stripes like Cotton's mommy. Being different made Cotton quite sad. She often wished she looked like the rest of her family. So one day, when Cotton found a can of the old farmer's orange paint, she used it to paint herself like them. When her mommy and sisters found her they started laughing. \n\n\"What are you doing, Cotton?!\" \n\n\"I only wanted to be more like you\". \n\nCotton's mommy rubbed her face on Cotton's and said \"Oh Cotton, but your fur is so pretty and special, like you. We would never want you to be any other way\". And with that, Cotton's mommy picked her up and dropped her into a big bucket of water. When Cotton came out she was herself again. Her sisters licked her face until Cotton's fur was all all dry. \n\n\"Don't ever do that again, Cotton!\" they all cried. \"Next time you might mess up that pretty white fur of yours and we wouldn't want that!\" \n\nThen Cotton thought, \"I change my mind. I like being special\"."
            question_text = "Where does Cotton live?"
            answer_text = ""

            special_token_ids = enc.convert_tokens_to_ids(special_tokens)

            story_text = enc.convert_tokens_to_ids(enc.tokenize(story_text))
            answer_text = enc.convert_tokens_to_ids(enc.tokenize(answer_text))
            question_text = enc.convert_tokens_to_ids(enc.tokenize(question_text))

            inputs = []
            inputs.append((story_text, [question_text], [answer_text]))

            # last_token_inx = len(story_text) + len(question_tokens) + 3 - 1
            generated = 0
            for _ in range(args.length):
                new_word_id = sample_word(
                    model=pretrained_model,
                    inputs=inputs,
                    special_tokens=special_token_ids,
                    temperature=args.temperature, top_k=args.top_k
                )
                # out = out[:, len(context_tokens):].tolist()
                inputs[0][2][0].append(new_word_id)
                # for i in range(args.batch_size):
                #     generated += 1
                #     text = enc.decode(out[0][2][0], skip_special_tokens=False)
                #     print(text)

            answer = enc.decode(inputs[0][2][0], skip_special_tokens=False)
            print(answer)
            print("=" * 80)
            break

        if args.unconditional:
            generated = 0
            for _ in range(args.nsamples // args.batch_size):
                out = sample_word(
                    model=pretrained_model,
                    inputs=None,
                    start_token=enc.convert_tokens_to_ids(args.special_tag),
                    temperature=args.temperature, top_k=args.top_k
                )
                # out = out[:,1:].tolist()
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


