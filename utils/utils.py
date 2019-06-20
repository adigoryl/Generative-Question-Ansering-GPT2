import numpy as np
import torch
import os
import json

def makedir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def load_json_dataset(dataset_path, cwd):
    """ Given dataset: (index, song_name, year, artist, genre, lyrics)
        Output a list of tuples(genre, lyrics) """


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


def prep_pad(max_q, max_a, max_s, story, question, answer, special_tokens, multi_class_tag):

    ### DATA
    q = np.zeros(max_q)
    a = np.zeros(max_a)
    s = np.zeros(max_s)

    q[0: len(question)] = question
    a[0: len(answer)] = answer
    s[0: len(story)] = story

    full_input = [special_tokens[0]] + s.tolist() + [special_tokens[3]] + \
                 [special_tokens[1]] + q.tolist() + [special_tokens[4]] + \
                 [special_tokens[2]] + a.tolist() + [special_tokens[5]]

    ### POSITION IDS
    q_pos = np.arange(max_q) + 2
    a_pos = np.arange(max_a) + 2
    s_pos = np.arange(max_s) + 2

    q_pos[len(question):max_q] = 0
    a_pos[len(answer):max_a] = 0
    s_pos[len(story):max_s] = 0

    full_pos = [1] + s_pos.tolist() + [1] + \
               [1] + q_pos.tolist() + [1] + \
               [1] + a_pos.tolist() + [1]

    ### TOKEN TYPES
    q_tok = np.zeros(max_q)
    a_tok = np.zeros(max_a)
    s_tok = np.zeros(max_s)

    if multi_class_tag == 0:
        q_tok[0: len(question)] = 6
        a_tok[0: len(answer)] = 7
    elif multi_class_tag == 1:
        a_tok[0: len(answer)] = 7
    elif multi_class_tag == 2:
        q_tok[0: len(question)] = 6

    # a_tok[0: len(answer)] = 7
    # q_tok[0: len(question)] = 6
    s_tok[0: len(story)] = 5

    full_tok = [1] + s_tok.tolist() + [1] + \
               [2] + q_tok.tolist() + [2] + \
               [3] + a_tok.tolist() + [3]

    return full_input, full_pos, full_tok


def filter_len(dataset, max_a, max_q, max_s):

    filtered_data = []
    for i in range(0, len(dataset)):
        story = dataset[i][0]
        quest = dataset[i][1]
        answ = dataset[i][2]

        # If story greater in len -> escape the loop / get rid of this data input
        if len(story) > max_s:
            continue

        q_idx_arr = []
        a_idx_arr = []

        # Check if Questions or Answers length satisfies
        for j in range(0, len(answ)):

            q_pad = max_q - len(quest[j])
            if q_pad >= 0:
                q_idx_arr.append(j)
            # else:
            #     print(q_pad)

            a_pad = max_a - len(answ[j])
            if a_pad >= 0:
                a_idx_arr.append(j)
            # else:
            #     print(a_pad)

        # Get only overlapping indexes -> this means an input index satisfied the story, question and answer maximum length
        all_idx = np.intersect1d(q_idx_arr, a_idx_arr)

        if len(all_idx) == 0:
            continue

        filtered_data.append((story, np.array(quest)[all_idx], np.array(answ)[all_idx]))

    return filtered_data


def format_data(dataset, special_tokens, device, multi_class_len, max_a, max_q, max_s):
    dataset = filter_len(dataset, max_a, max_q, max_s)
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
                    full_input, full_pos, full_tok = prep_pad(max_q, max_a, max_s, story, quest[idx_qa], answ[idx_qa], special_tokens, 0)
                    inputs.append([np.array(full_input)])
                    pos_ids.append([np.array(full_pos)])
                    token_types.append([np.array(full_tok)])
                    ### Multi Class TOKEN IDS
                    mc_tok_id = [0]
                    mc_tok_ids.append(mc_tok_id)

                elif k != 0 and option > 0.8:
                    full_input, full_pos, full_tok = prep_pad(max_q, max_a, max_s, story, f_quest[fake_qa_idx], answ[idx_qa], special_tokens, 1)
                    inputs[-1].append(np.array(full_input))
                    pos_ids[-1].append(np.array(full_pos))
                    token_types[-1].append(np.array(full_tok))
                    mc_tok_ids[-1].append(0)

                elif k != 0 and option <= 0.8:
                    full_input, full_pos, full_tok = prep_pad(max_q, max_a, max_s, story, quest[idx_qa], f_answ[fake_qa_idx], special_tokens, 2)
                    inputs[-1].append(np.array(full_input))
                    pos_ids[-1].append(np.array(full_pos))
                    token_types[-1].append(np.array(full_tok))
                    mc_tok_ids[-1].append(0)

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


def tokenize_and_encode(obj, tokenizer):
    """ Tokenize and encode a nested object """
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    elif isinstance(obj, int):
        return obj
    return list(tokenize_and_encode(o, tokenizer) for o in obj)