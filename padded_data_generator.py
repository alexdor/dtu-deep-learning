from __future__ import absolute_import, division, print_function

import numpy as np
target_to_text = {
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine',
}

EOS = '#'

input_characters = " ".join(target_to_text.values())
valid_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', EOS] + \
                   list(set(input_characters))


def print_valid_characters():
    l = ''
    for i, c in enumerate(valid_characters):
        l += "\'%s\'=%i,\t" % (c, i)
    print("Number of valid characters:", len(valid_characters))
    print(l)


ninput_chars = len(valid_characters)


def generate(num_batches=10,batch_size=100, min_len=3, max_len=3, invalid_set=set()):
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
    inputs=[]
    targets_in=[]
    targets_out=[]
    batch_target_max_len=np.zeros((1,num_batches))
    batch_input_max_len=np.zeros((1,num_batches))
    _printed_warning = False
    # loop through number of batches
    for i in range(num_batches):
        print('batch ',i,' starting')
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
            text_target = inp_str = "".join(map(str, np.random.randint(0, 10, tar_len)))
            text_target_in = EOS + text_target
            text_target_out = text_target + EOS

            # generate the targets as a list of integers
            int_target_in = map(lambda c: valid_characters.index(c), text_target_in)
            int_target_in = list(int_target_in)
            int_target_out = map(lambda c: valid_characters.index(c), text_target_out)
            int_target_out = list(int_target_out)

            # generate the text input
            text_input = " ".join(map(lambda k: target_to_text[k], inp_str))

            # generate the inputs as a list of integers
            int_input = map(lambda c: valid_characters.index(c), text_input)
            int_input = list(int_input)

            if not _printed_warning and iterations > 5 * batch_size:
                print("WARNING: doing a lot of iterations because I'm trying to generate a batch that does not"
                      " contain samples from 'invalid_set'.")
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
        text_inputs.append(temp_text_inputs)
        int_inputs.append(temp_int_inputs)
        text_targets_in.append(temp_text_targets_in)
        text_targets_out.append(temp_text_targets_out)
        int_targets_in.append(temp_int_targets_in)
        int_targets_out.append(temp_int_targets_out)
        
                # append completed temp batch array to full array of batches

   

        max_target_out_len = max(map(len, int_targets_out[-1]))
        targets_mask = np.zeros((batch_size, max_target_out_len))
#         targets_out = np.zeros((batch_size, max_target_in_len))
        for (j, tar) in enumerate(int_targets_out[-1]):
            cur_len = len(tar)
            targets_mask[j, :cur_len] = 1
            
   
        input_seq_lengths=torch.LongTensor(list(map(len,temp_int_inputs)))
        almost_inputs=Variable(torch.zeros((len(temp_int_inputs), input_seq_lengths.max()))).long()
        for idx, (seq, seqlen) in enumerate(zip(temp_int_inputs, input_seq_lengths)):
            cur_len=len(seq)
            almost_inputs[idx, :cur_len] = torch.LongTensor(seq)
        inputs.append(almost_inputs)    
        
        target_in_seq_lengths=torch.LongTensor(list(map(len, temp_int_targets_in)))
        almost_targets_in=Variable(torch.zeros((len(temp_int_targets_in), target_in_seq_lengths.max()))).long()
        for idx, (seq, seqlen) in enumerate(zip(temp_int_targets_in, target_in_seq_lengths)):
            cur_len=len(seq)
            almost_targets_in[idx, :cur_len] = torch.LongTensor(seq)
        targets_in.append(almost_targets_in)
        
        target_out_seq_lengths= torch.LongTensor(list(map(len, temp_int_targets_out)))    
        almost_targets_out=Variable(torch.zeros((len(temp_int_targets_out), target_out_seq_lengths.max()))).long()
        for idx, (seq, seqlen) in enumerate(zip(temp_int_targets_out, target_out_seq_lengths)):
            cur_len=len(seq)
            almost_targets_out[idx, :cur_len] = torch.LongTensor(seq)
        targets_out.append(almost_targets_out)
        
        text_inputs.append(temp_text_inputs)
        int_inputs.append(temp_int_inputs)
        text_targets_in.append(temp_text_targets_in)
        text_targets_out.append(temp_text_targets_out)
        int_targets_in.append(temp_int_targets_in)
        int_targets_out.append(temp_int_targets_out)

        batch_target_max_len[0,i]=target_in_seq_lengths.max()
        batch_input_max_len[0,i]=input_seq_lengths.max()
        print('batch ',i,' ended with ',len(text_targets_in[-1]))

   
    print('max input lengths',batch_input_max_len)
    print('max target lengths',batch_target_max_len)
    print("Generated batch length {} from {} iterations".format(len(text_inputs), iterations))

    return inputs, \
        batch_input_max_len.astype('int32'), \
        targets_in, \
        targets_out, \
        batch_target_max_len.astype('int32'), \
        targets_mask.astype('float32'), \
        text_inputs, \
        text_targets_in, \
        text_targets_out


def main():
    batch_size = 3
    inputs, inputs_seqlen, targets_in, targets_out, targets_seqlen, targets_mask, \
        text_inputs, text_targets_in, text_targets_out = \
        generate(batch_size=batch_size, max_len=4, min_len=2)

    print("input types:", inputs.dtype, inputs_seqlen.dtype, targets_in.dtype, targets_out.dtype, targets_seqlen.dtype)
    print_valid_characters()
    print("Stop/start character = #")

    for i in range(batch_size):
        print("\nSAMPLE", i)
        print("TEXT INPUTS:\t\t\t", text_inputs[i])
        print("ENCODED INPUTS:\t\t\t", inputs[i])
        print("INPUTS SEQUENCE LENGTH:\t", inputs_seqlen[i])
        print("TEXT TARGETS INPUT:\t\t", text_targets_in[i])
        print("TEXT TARGETS OUTPUT:\t", text_targets_out[i])
        print("ENCODED TARGETS INPUT:\t", targets_in[i])
        print("ENCODED TARGETS OUTPUT:\t", targets_out[i])
        print("TARGETS SEQUENCE LENGTH:", targets_seqlen[i])
        print("TARGETS MASK:\t\t\t", targets_mask[i])

if __name__ == '__main__':
    main()