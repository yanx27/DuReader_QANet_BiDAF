import os

'''Path parameter'''
path = os.getcwd()
vocab_dir = path + '/data/vocab/'
train_files = [ './data/demo//search.train.json']
dev_files = ['./data/demo//search.dev.json']
test_files = ['./data/demo//search.test.json']
pretrained_word_path = None
pretrained_char_path = None
model_dir = path + '/data/models/'
summary_dir = path + '/data/'

# --train --batch_size 16 --learning_rate 1e-3 --optim adam --decay 0.9999 --weight_decay 1e-5
#  --max_norm_grad 5.0 --dropout 0.0 --head_size 1 --hidden_size 64 --epochs 20 --gpu 0
'''Train settings'''
algo = 'qanet'
epochs = 20
loss_type = 'cross_entropy'
batch_size = 16
optim = 'adam'
fix_pretrained_vector = True

learning_rate = 1e-3
dropout = 0.5
weight_decay = 1e-5
l2_norm = 3e-7
clip_weight = True
max_norm_grad = 5.0
decay = 0.9999
gpu  = '0'


'''Model setting'''
max_p_num = 5
max_p_len = 400
max_q_len = 60
max_ch_len =20
max_a_len = 200
word_embed_size = 150
char_embed_size = 32
head_size = 1
hidden_size = 64
use_position_attn = False

