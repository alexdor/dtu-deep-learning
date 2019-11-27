import numpy as np
import torch

target_to_text = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

EOS = "#"

PAD = "0"

input_characters = " ".join(target_to_text.values())
valid_characters = [
    PAD,
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    EOS,
] + list(set(input_characters))


def print_valid_characters():
    l = ""
    for i, c in enumerate(valid_characters):
        l += "'%s'=%i,\t" % (c, i)
    print("Number of valid characters:", len(valid_characters))
    print(l)


ninput_chars = len(valid_characters)


def generate(
    num_batches=10, batch_size=100, min_len=3, max_len=3, invalid_set=set()
):
    """
    Generates random sequences of integers and translates them to text i.e. 1->'one'.

    :param batch_size: number of samples to return
    :param min_len: minimum length of target
    :param max_len: maximum length of target
    :param invalid_set: set of invalid 'text targets out', e.g. if we want to avoid samples
    that are already in the training set
    """

    text_inputs = []
    int_inputs = []
    text_targets_in = []
    text_targets_out = []
    int_targets_in = []
    int_targets_out = []
    inputs = []
    targets_in = []
    targets_out = []
    batch_target_max_len = np.zeros((1, num_batches))
    targets_mask = []
    batch_input_max_len = np.zeros((1, num_batches))
    inputs_len = []
    _printed_warning = False
    # loop through number of batches
    for i in range(num_batches):
        temp_text_inputs = []
        temp_int_inputs = []
        temp_text_targets_in = []
        temp_text_targets_out = []
        temp_int_targets_in = []
        temp_int_targets_out = []
        iterations = 0
        # while loop until number of rows per batch is reached
        while len(temp_text_inputs) < batch_size:
            iterations += 1

            # choose random sequence length
            tar_len = np.random.randint(min_len, max_len + 1)

            # list of text digits
            text_target = inp_str = "".join(
                map(str, np.random.randint(1, 10, tar_len))
            )
            text_target_in = EOS + text_target
            text_target_out = text_target + EOS

            # generate the targets as a list of integers
            int_target_in = map(
                lambda c: valid_characters.index(c), text_target_in
            )
            int_target_in = list(int_target_in)

            int_target_out = map(
                lambda c: valid_characters.index(c), text_target_out
            )
            int_target_out = list(int_target_out)

            # generate the text input
            text_input = " ".join(map(lambda k: target_to_text[k], inp_str))

            # generate the inputs as a list of integers
            int_input = map(lambda c: valid_characters.index(c), text_input)
            int_input = list(int_input)

            if not _printed_warning and iterations > 5 * batch_size:
                print(
                    "WARNING: doing a lot of iterations because I'm trying to generate a batch that does not"
                    " contain samples from 'invalid_set'."
                )
                _printed_warning = True

            if text_target_out in invalid_set:
                continue
            # append created row to temp arrays
            temp_text_inputs.append(text_input)
            temp_int_inputs.append(int_input)
            temp_text_targets_in.append(text_target_in)
            temp_text_targets_out.append(text_target_out)
            temp_int_targets_in.append(int_target_in)
            temp_int_targets_out.append(int_target_out)
        # turn the temp arrays into tensors and pad them to max length

        # append completed temp batch array to full array of batches
        text_inputs.append(temp_text_inputs)
        int_inputs.append(temp_int_inputs)
        text_targets_in.append(temp_text_targets_in)
        text_targets_out.append(temp_text_targets_out)
        int_targets_in.append(temp_int_targets_in)
        int_targets_out.append(temp_int_targets_out)

        max_target_out_len = max(map(len, int_targets_out[-1]))
        max_input_len = max(map(len, int_inputs[-1]))
        targets_mask_tmp = np.zeros((batch_size, max_target_out_len))
        add_targets_out = np.full((batch_size, max_target_out_len), int(PAD))
        len_arr = [-len(thing) for thing in temp_int_inputs]
        sorted_arr = np.argsort(len_arr)
        add_targets_in = np.full((batch_size, max_target_out_len), int(PAD))
        add_inputs = np.full((batch_size, max_input_len), int(PAD))
        tmp_inputs_len = np.zeros(len(sorted_arr))
        for short_index, row in enumerate(sorted_arr):
            tmp_element = temp_int_inputs[row]
            add_inputs[short_index, : len(tmp_element)] = tmp_element
            tmp_inputs_len[short_index] = len(tmp_element)

            tmp_element = temp_int_targets_in[row]
            add_targets_in[short_index, : len(tmp_element)] = tmp_element

            tmp_element = temp_int_targets_out[row]
            add_targets_out[short_index, : len(tmp_element)] = tmp_element
            targets_mask_tmp[short_index, : len(tmp_element)] = 1
        inputs_len.append(tmp_inputs_len)
        targets_mask.append(targets_mask_tmp)
        inputs.append(add_inputs.astype("int32"))
        targets_in.append(add_targets_in.astype("int32"))
        targets_out.append(add_targets_out.astype("int32"))
        target_in_seq_lengths = torch.LongTensor(
            list(map(len, temp_int_targets_in))
        )
        input_seq_lengths = torch.LongTensor(list(map(len, temp_int_inputs)))

        batch_target_max_len[0, i] = target_in_seq_lengths.max()
        batch_input_max_len[0, i] = input_seq_lengths.max()
    return (
        inputs,
        batch_input_max_len.astype("int32"),
        targets_in,
        targets_out,
        batch_target_max_len.astype("int32"),
        targets_mask,
        text_inputs,
        text_targets_in,
        text_targets_out,
        inputs_len,
    )


def main():
    batch_size = 3
    (
        inputs,
        inputs_seqlen,
        targets_in,
        targets_out,
        targets_seqlen,
        targets_mask,
        text_inputs,
        text_targets_in,
        text_targets_out,
        inputs_len,
    ) = generate(8, 10, min_len=1, max_len=2)

    print_valid_characters()
    print("Stop/start character = #")

    for i in range(batch_size):
        print("\nSAMPLE", i)
        print("TEXT INPUTS:\t\t\t", text_inputs[i])
        print("ENCODED INPUTS:\t\t\t\n", inputs[i])
        print("INPUTS SEQUENCE LENGTH:\t\n", inputs_seqlen)
        print("TEXT TARGETS INPUT:\t\t", text_targets_in[i])
        print("TEXT TARGETS OUTPUT:\t", text_targets_out[i])
        print("ENCODED TARGETS INPUT:\t\n", targets_in[i])
        print("ENCODED TARGETS OUTPUT:\t\n", targets_out[i])
        print("TARGETS SEQUENCE LENGTH:", targets_seqlen)
        print("TARGETS MASK:\t\t\t\n", targets_mask[i])
        print("INPUTS LEN:\t\t\t\n", inputs_len[i])


if __name__ == "__main__":
    main()
