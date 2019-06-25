#!/usr/bin/env python3

import argparse
import logging
import os
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np
import utils.utils as u

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2DoubleHeadsModel, GPT2Config, GPT2Tokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def top_k_top_p_filtering(logits, top_k=100, top_p=0.90, filter_value=-float('Inf')):
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


def sample_word(model, context, new_word_index=0, temperature=1):

    past = None
    with torch.no_grad():
        lm_logits, _presents, _hid_states = model(*context, past=past)
        # lm_logits, _presents, _hid_states = model(context)
        lm_logits = lm_logits[new_word_index, :] / temperature
        # lm_logits2 = lm_logits[new_word_index+1, :] / temperature
        # lm_logits = lm_logits + lm_logits2
        lm_logits = top_k_top_p_filtering(lm_logits)
        log_probs = F.softmax(lm_logits, dim=-1)
        new_word = torch.multinomial(log_probs, num_samples=1)

    return new_word.item()


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt2", help='pretrained model name or path to local checkpoint')
    parser.add_argument('--load_model_path', type=str, default="/Users/aw678/PycharmProjects/gpt2_QA/finetuned_models/test1/gpt2_20-06-2019@22'02/model/")
    # parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=np.random.randint(0, 100))
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument('--no_padding', type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', type=bool, default=False, help='If true, unconditional generation.')
    parser.add_argument('--special_tag', type=str, default='<_ROCK_>', help='If unconditional, this tag will be used to initiate the generation')

    args = parser.parse_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2LMHeadModel.from_pretrained(args.model)
    model_dict = model.state_dict()

    # Prepare paths and file names
    output_model_file = os.path.join(args.load_model_path, "pytorch_model.bin")
    output_config_file = os.path.join(args.load_model_path, "config_file.bin")

    special_tokens = ['<_S_STORY_>', '<_S_QUE_>', '<_S_ANS_>',
                      '<_E_STORY_>', '<_E_QUE_>', '<_E_ANS_>']

    enc = GPT2Tokenizer.from_pretrained(args.model, special_tokens=special_tokens)

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

    KEEP_GENERATING = True
    while KEEP_GENERATING:
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
            question_text = "Was cotton alone in the house?"
            # question_text = "How many sisters does cotton have?"
            answer_text = ""

            # story_text = "Adrian has five cats, where two are white and three and black."
            # question_text = "How many white cats does Adrian have?"

            story_text = "Staten Island is one of the five boroughs of New York City in the U.S. state of New York. In the southwest of the city, Staten Island is the southernmost part of both the city and state of New York, with Conference House Park at the southern tip of the island and the state. The borough is separated from New Jersey by the Arthur Kill and the Kill Van Kull, and from the rest of New York by New York Bay. With a 2016 Census-estimated population of 476,015, Staten Island is the least populated of the boroughs but is the third-largest in area at . Staten Island is the only borough of New York with a non-Hispanic White majority. The borough is coextensive with Richmond County, and until 1975 was the Borough of Richmond. Its flag was later changed to reflect this. Staten Island has been sometimes called \"the forgotten borough\" by inhabitants who feel neglected by the city government. \n\nThe North Shore\u2014especially the neighborhoods of St. George, Tompkinsville, Clifton, and Stapleton\u2014is the most urban part of the island; it contains the designated St. George Historic District and the St. Paul's Avenue-Stapleton Heights Historic District, which feature large Victorian houses. The East Shore is home to the F.D.R. Boardwalk, the fourth-longest in the world. The South Shore, site of the 17th-century Dutch and French Huguenot settlement, developed rapidly beginning in the 1960s and 1970s and is now mostly suburban in character. The West Shore is the least populated and most industrial part of the island."
            question_text = "Where do Quinton and Kendra travel to and from every day?"
            question_text = "What does Kendra not want to miss?"
            question_text = "How many burroughs are there in New York City?"

            special_token_ids = enc.convert_tokens_to_ids(special_tokens)

            story_ids = enc.convert_tokens_to_ids(enc.tokenize(story_text))
            question_ids = enc.convert_tokens_to_ids(enc.tokenize(question_text))
            answer_ids = enc.convert_tokens_to_ids(enc.tokenize(answer_text))
            if args.model == "gpt2-medium":
                full_input, full_pos, full_tok = u.prep_pad(48, 200, 770, story_ids, question_ids, answer_ids, special_token_ids, 0)
            elif args.model == "gpt2":
                full_input, full_pos, full_tok = u.prep_pad(len(question_ids), len(answer_ids), len(story_ids), story_ids, question_ids, answer_ids, special_token_ids, 0)

            flag = np.where(np.array(full_input) == special_token_ids[2])[0][0]
            full_input = full_input[0:flag+1]
            full_pos = full_pos[0:flag+1]
            full_tok = full_tok[0:flag+1]

            if args.no_padding:
                full_tok = np.zeros(len(full_tok)).tolist()
                full_pos = np.zeros(len(full_pos)).tolist()

            for word_idx in trange(args.length):
                # in_tuple = (full_input, None, None)
                new_word_id = sample_word(
                    model=pretrained_model,
                    # context=torch.tensor(full_input, dtype=torch.int64),
                    context=torch.tensor((full_input, full_pos, full_tok), dtype=torch.int64),
                    # new_word_index=flag+word_idx,
                    new_word_index=flag+word_idx,
                    temperature=args.temperature,
                )

                full_input.append(new_word_id)
                if args.no_padding:
                    full_pos.append(0)
                    full_tok.append(0)
                else:
                    full_pos.append(word_idx+2)
                    full_tok.append(7)

                if new_word_id == special_token_ids[5]:
                    KEEP_GENERATING = False

            full_model_input = enc.decode(full_input, skip_special_tokens=False)
            answer = enc.decode(full_input[flag+1:-1], skip_special_tokens=False)
            # print(answer)
            print("Full_model_input: {}\nStory: {}\nQuestion: {}\nAnswer: {}".format(full_model_input, story_text, question_text, answer))
            # print("=" * 80)
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


