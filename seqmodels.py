import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.ticker as ticker
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from nltk.translate.bleu_score import sentence_bleu
from main import Config, load_data
from tabulate import tabulate
from tqdm import tqdm


torch.manual_seed(2)

MAX_LENGTH = 100
teacher_forcing_ratio = 0.5
NMIN = 4
NMAX = 12
KMAX = 12

DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'
# DEVICE = "cpu"


def gen_masks(seq_len: int, n_min: int, n_max: int, k_max: int):
    """generate indices for input and targets from a sequence
    input: length of the input sequence
    inp_len_min: minimum length of input
    out_len: length of target
    Returns: all possible input and target indices


    k_max not implemented
    """
    inps = []
    targs = []
    # assert n_min +  <= seq_len, "Sizes not compatible! Please check input and output sizes"
    n_max = min(seq_len, n_max)
    n = range(n_min, n_max)

    for j in n:
        inps.append(list(range(0, j)))
        targs.append(list(range(j, n_max)))
    return inps, targs


def load_config(DOMAIN):
    CONFIG_PATH = "config/" + DOMAIN
    global config
    config = Config()
    config.load_from_file(CONFIG_PATH)
    return config, DEVICE


"""
def gen_masks(seq_len:int, inp_len:int, out_len:int):
    # generate indices for input and targets from a sequence
    # input: length of the input sequence
    # inp_len: length of input
    # out_len: length of target
    # Returns: all possible input and target indices
    #
    inps = []
    targs = []
    assert inp_len + \
        out_len <= seq_len, "Sizes not compatible! Please check input and output sizes"

    for i in range(0, seq_len - inp_len):
        if i + inp_len + out_len > seq_len:
            break
        inps.append(list(range(i, i + inp_len)))
        targs.append(list(range(i + inp_len, i + inp_len + out_len)))
    return inps, targs
"""


def tensors_from_paths(paths, EOS, SOS):
    """return torch tensors of input and target sequences from trajectory paths"""
    all_inps = []
    all_targs = []
    input_lens = []
    output_lens = []
    for i in random.choices(range(len(paths)), k=len(paths)):  # sample randomly
        inp, targ, ilens, olens = tensors_from_pair(paths[i], EOS, SOS)
        all_targs.extend(targ)
        all_inps.extend(inp)
        input_lens.extend(ilens)
        output_lens.extend(olens)

    # print(len(all_inps), "\n", len(all_targs), "\n",
    #      len(input_lens), "\n", len(output_lens))
    return torch.cat(all_inps), torch.cat(all_targs), input_lens, output_lens
    # return (all_inps, all_targs)


def tensors_from_pair(input_path, EOS, SOS):
    """returns input and output tensors from a single trajectory path"""
    # obs, targs = gen_masks(len(input_path), n_obs, n_targ)
    input_tensor = []
    target_tensor = []
    inp_lens = []
    out_lens = []
    input_path = input_path.argmax(dim=1).long()  # convert one-hot to index
    obs, targs = gen_masks(len(input_path), n_min=NMIN, n_max=NMAX, k_max=KMAX)
    for inp, out in zip(obs, targs):
        # input_tensor.append(torch.cat((torch.tensor([SOS_TOKEN], device=DEVICE),
        #                               input_path[inp], torch.tensor([EOS_TOKEN], device=DEVICE))))
        input_tensor.append(input_path[inp])
        inp_lens.append(len(inp))
        target_tensor.append(
            torch.cat((input_path[out], torch.tensor([EOS], device=DEVICE))))
        out_lens.append(len(out)+1)
    # print(input_tensor)
    # print(target_tensor, out_lens)
    # return (torch.stack(input_tensor), torch.stack(target_tensor))
    # return (torch.cat(input_tensor), torch.cat(target_tensor), torch.tensor(inp_lens, device=DEVICE), torch.tensor(out_lens, device=DEVICE))
    return (input_tensor, target_tensor, inp_lens, out_lens)

# Encoder


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

# Decoder


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

# Attention Decoder


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(paths, encoder, decoder, test_set, EOS, SOS, n_epoch=5, tf=0.9, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    print_loss = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    obs, targs, obs_lens, targ_lens = tensors_from_paths(paths, EOS, SOS)
    criterion = nn.NLLLoss()

    n_sample = len(obs_lens)
    for epoch in range(n_epoch):
        obs_ix = 0
        targ_ix = 0

        for ix in tqdm(range(n_sample)):
            # for iter, (input_tensor, target_tensor) in enumerate(zip(training_pairs[0], training_pairs[1]), 1):
            input_tensor = obs[obs_ix:obs_ix+obs_lens[ix]]
            target_tensor = targs[targ_ix:targ_ix+targ_lens[ix]]
            obs_ix += obs_lens[ix]
            targ_ix += targ_lens[ix]
            loss = train(input_tensor.unsqueeze(1), target_tensor.unsqueeze(1), encoder,
                         decoder, EOS, SOS, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=tf)
            print_loss_total += loss
            print_loss += loss

        print_loss_avg = print_loss_total / n_sample
        print_loss_total = 0
        plot_losses.append(print_loss_total)
        print(f'Epoch: {epoch} \t Loss: {print_loss_avg}')
        bleu, node_accuracy, k_accuracy = evaluateRandomly(encoder, decoder, test_set, EOS, SOS,
                                                           "fancy_grid", encoder.embedding.num_embeddings)
        print(bleu, node_accuracy, k_accuracy)
    showPlot(plot_losses)

    return dict({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "optim_encoder": encoder_optimizer.state_dict(), "optim_decoder": decoder_optimizer.state_dict()})


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

    return


def test_plot():
    plt.plot(np.random.randn(10))


def train(input_tensor, target_tensor, encoder, decoder, EOS, SOS, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=0.7, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=DEVICE)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_input = torch.tensor(
        [[SOS]], device=DEVICE, dtype=torch.long)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def evaluate(encoder, decoder, input_tensor, EOS, SOS, max_length=MAX_LENGTH):
    with torch.no_grad():
        # input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS]], device=DEVICE)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, training_pairs, EOS, SOS, fmt, n_node):

    headers = ["User Path Taken", "User Path Predicted", "Bleu Score"]
    rows = []
    bleu = 0.
    # print("-----------------------------------")

    node_accuracy = np.zeros(n_node+2)
    node_counts = np.zeros(n_node+2)
    obs, targs, obs_lens, targ_lens = tensors_from_paths(
        training_pairs, EOS, SOS)

    k_max = 12
    k_accuracy = np.zeros(k_max)

    n_sample = len(obs_lens)
    obs_ix = 0
    targ_ix = 0
    for ix in tqdm(range(n_sample)):
        input_tensor = obs[obs_ix:obs_ix+obs_lens[ix]]
        target = targs[targ_ix:targ_ix+targ_lens[ix]]

        obs_ix += obs_lens[ix]
        targ_ix += targ_lens[ix]
        output_words, attentions = evaluate(
            encoder, decoder, input_tensor, EOS, SOS)

        pred_length = min(len(output_words), len(target))
        corrects = [i == j for (i, j) in zip(
            target[:pred_length-1], output_words[:pred_length-1])]

        right_nodes = [t.item() for i, t in enumerate(
            target[:pred_length-1]) if corrects[i]]

        #print(right_nodes, target)
        node_accuracy[right_nodes] += 1
        node_counts[target.cpu().numpy()] += 1
        k_accuracy[np.argwhere(corrects)] += 1

        target = [str(i) for i in target.tolist()]
        output = [str(i) for i in output_words[:len(target)]]

        rows.append([' '.join(target), ' '.join(output),
                    sentence_bleu([target], output)])
        bleu += sentence_bleu([target], output)

    bleu = bleu/n_sample
    node_accuracy = node_accuracy/node_counts
    k_accuracy = k_accuracy/n_sample
    # return tabulate(rows, headers=headers, tablefmt=fmt),
    return bleu, node_accuracy, k_accuracy
