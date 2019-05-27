#!/usr/bin/env python3

import argparse
import logging
import os
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2DoubleHeadsModel, GPT2Config, GPT2Tokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
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


def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)

    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="gpt2", help='pretrained model name or path to local checkpoint')
    parser.add_argument('--load_model_path', type=str, default="/Users/aw678/PycharmProjects/gpt2_answers/finetuned_models/test/gpt2_24-05-2019@23'53_z1/model/")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--nsamples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--length", type=int, default=5)
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

    # Prepare paths and file names
    output_model_file = os.path.join(args.load_model_path, "pytorch_model.bin")
    output_config_file = os.path.join(args.load_model_path, "config_file.bin")
    output_vocab_file = os.path.join(args.load_model_path, "vocab.json")
    output_merges_file = os.path.join(args.load_model_path, "merges.txt")
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
    model_dict = model.state_dict()
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
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            last_token_inx = len(context_tokens) - 1
            generated = 0
            for _ in range(args.nsamples // args.batch_size):
                out = sample_sequence(
                    model=pretrained_model, length=args.length,
                    context=context_tokens,
                    start_token=None,
                    batch_size=args.batch_size,
                    temperature=args.temperature, top_k=args.top_k, device=device
                )
                out = out[:, len(context_tokens):].tolist()
                for i in range(args.batch_size):
                    generated += 1
                    text = enc.decode(out[i], skip_special_tokens=False)
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)

            print("=" * 80)

        if args.unconditional:
            generated = 0
            for _ in range(args.nsamples // args.batch_size):
                out = sample_sequence(
                    model=pretrained_model, length=args.length,
                    context=None,
                    start_token=enc.convert_tokens_to_ids(args.special_tag),
                    batch_size=args.batch_size,
                    temperature=args.temperature, top_k=args.top_k, device=device
                )
                out = out[:,1:].tolist()
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


