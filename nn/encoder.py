# -*- coding: utf-8 -*-
# Some code in this file is from https://github.com/jiesutd/LatticeLSTM
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from kblayer import GazLayer
from .charbilstm import CharBiLSTM
from .charcnn import CharCNN
from .mglattice import LatticeLSTM
from torch.nn import LSTM
from torch.nn import GRU
from my_bert.modeling_bert import BertModel
import configure

class BiLstmEncoder(nn.Module):
    def __init__(self, data):
        super(BiLstmEncoder, self).__init__()
        print("build batched bilstm-based encoder...")
        self.use_bigram = data.use_bigram #False
        self.gpu = data.HP_gpu#True
        self.use_char = data.HP_use_char #False
        self.use_gaz = data.HP_use_gaz #True
        self.batch_size = data.HP_batch_size #1
        self.bert_flag = data.bert_flag
        self.char_hidden_dim = 0
        if self.use_char:
            self.char_hidden_dim = data.HP_char_hidden_dim #50
            self.char_embedding_dim = data.char_emb_dim #30

            # Character features via CNN or LSTM
            if data.char_features == "CNN":
                self.char_feature = CharCNN(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_features == "LSTM":
                self.char_feature = CharBiLSTM(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            else:
                print("Error char feature selection, please check parameter data.char_features (either CNN or LSTM).")
                exit(0)
        
        self.embedding_dim = data.word_emb_dim #100
        self.linear = nn.Linear(in_features=768, out_features=self.embedding_dim)
        # Size fo hidden vectors
        self.hidden_dim = data.HP_hidden_dim #200
        # Dropout mechanism
        self.drop = nn.Dropout(data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_dropout)
        # Word embeddings, x^w_{b,e} in the paper
        self.word_embeddings = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        # print('embedding type', type(self.word_embeddings))
        self.biword_embeddings = nn.Embedding(data.biword_alphabet.size(), data.biword_emb_dim)
        
        # Position embeddings
        # x^{p1} in the paper
        self.pos1_embeddings = nn.Embedding(data.pos_size,data.pos_emb_dim)
        # x^{p2} in the paper
        self.pos2_embeddings = nn.Embedding(data.pos_size,data.pos_emb_dim)

        # BERT embeddings
        self.bert_model = BertModel.from_pretrained(configure.bert_floder_path, output_hidden_states=True,
                                                    output_attentions=True)
        self.bert_dropout = nn.Dropout(configure.DROPOUT_RATE)

        # freeze the embedding
        for param in list(self.bert_model.embeddings.parameters()):
            param.requires_grad = False
        print("Froze Embedding Layer")
        layer_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for layer_idx in layer_indexes:
            for param in list(self.bert_model.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False
            print("Froze Layer: ", layer_idx)
        # To control if the model is bidrectional, the default value is True
        self.bilstm_flag = data.HP_bilstm
        # To control the layer of LSTM, the default value is 1
        self.lstm_layer = data.HP_lstm_layer
        try:
            self.encoder = data.Encoder
        except:
            self.encoder = 'MGLattice'

        if self.encoder == 'MGLattice':
            self.bilstm_flag = True
        else:
            self.bilstm_flag = False

        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding)) #char embedding
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))
            
        if data.pretrain_biword_embedding is not None: #None
            self.biword_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_biword_embedding))
        else:
            self.biword_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.biword_alphabet.size(), data.biword_emb_dim)))

        # Ramdom initializa pos embeddings
        self.pos1_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.pos_size,data.pos_emb_dim)))
        self.pos2_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.pos_size,data.pos_emb_dim)))
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim #200
        lstm_input = self.embedding_dim + self.char_hidden_dim + data.pos_emb_dim*2

        if self.use_bigram:
            lstm_input += data.biword_emb_dim
        
        if self.encoder == 'MGLattice':
            #print('Using MG-Lattice Encoder')
            self.forward_lstm = LatticeLSTM(lstm_input, lstm_hidden, data.gaz_dropout, data.gaz_alphabet.size(), data.gaz_emb_dim, data.pretrain_gaz_embedding, True, data.HP_fix_gaz_emb, self.gpu)
            self.backward_lstm = LatticeLSTM(lstm_input, lstm_hidden, data.gaz_dropout, data.gaz_alphabet.size(), data.gaz_emb_dim, data.pretrain_gaz_embedding, False, data.HP_fix_gaz_emb, self.gpu)
        elif self.encoder == 'GRU':
            #print('Using Bi-GRU Encoder')
            self.forward_lstm = GRU(lstm_input, lstm_hidden // 2, 1, bias = True, bidirectional = True)
        else:
            print("Error: the configure of encoder is illegal:%s"%(self.encoder))

        if self.gpu:
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.biword_embeddings = self.biword_embeddings.cuda()
            self.pos1_embeddings = self.pos1_embeddings.cuda()
            self.pos2_embeddings = self.pos2_embeddings.cuda()
            self.forward_lstm = self.forward_lstm.cuda()
            if self.bilstm_flag:
                self.backward_lstm = self.backward_lstm.cuda()



    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def get_seq_features(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, pos1_inputs, pos2_inputs,sent_ids):
        """
            input:
                word_inputs: (batch_size, sent_len)
                gaz_list:
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output: 
                Variable(sent_len, batch_size, hidden_dim)
        """
        # print('gaz.size:',len(gaz_list))
        # print('gaz[0]',len(gaz_list[0]))
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        char_seq_embeds = self.bert_model(sent_ids)[-2][-2]

        # print('char_seq_embeds2:',char_seq_embeds2)
        # char_equal_embeds = F.relu(self.linear(char_seq_embeds))
        # print("char_seq_embeds:", char_seq_embeds)# torch.Size([1, 226, 768]
        index_ids = list(range(char_seq_embeds.size(1)))
        char_equal_embeds = torch.index_select(char_seq_embeds, 1, torch.tensor(index_ids[1:-1]).cuda())

        word_embs =  self.word_embeddings(word_inputs)
        if self.bert_flag:
            word_embs = word_embs + torch.tanh(self.linear(char_equal_embeds))
        pos1_embs = self.pos1_embeddings(pos1_inputs)
        pos2_embs = self.pos2_embeddings(pos2_inputs)
        word_embs = torch.cat([word_embs, pos1_embs, pos2_embs],2)



        if self.use_bigram:
            biword_embs = self.biword_embeddings(biword_inputs)
            word_embs = torch.cat([word_embs, biword_embs],2)
        if self.use_char:
            ## Calculate char CNN or LSTM features
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size,sent_len,-1)
            
            ## Concat word and char together, combine the word and character info
            word_embs = torch.cat([word_embs, char_features], 2)
        word_embs = self.drop(word_embs)
        # print("word embs shape:", word_embs.shape) #torch.Size([1, 86, 110])
        hidden = None
        if self.encoder == 'MGLattice':
            lstm_out, hidden = self.forward_lstm(word_embs, gaz_list, hidden)
        else:
            lstm_out, hidden = self.forward_lstm(word_embs, hidden)
        # print("lstm_out shape:", lstm_out.shape)#torch.Size([1, 86, 100])
        # print("hidden shape:", hidden.shape)#torch.Size([1, 86, 100])
        
        # If backward
        if self.bilstm_flag:
            backward_hidden = None 
            backward_lstm_out, backward_hidden = self.backward_lstm(word_embs, gaz_list, backward_hidden)
            lstm_out = torch.cat([lstm_out, backward_lstm_out],2)

        return lstm_out





