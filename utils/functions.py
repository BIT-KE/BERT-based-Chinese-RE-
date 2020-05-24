import sys
import numpy as np
from .alphabet import Alphabet
import re
NULLKEY = "-null-"

# normalize number to 0
def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

# transform string to list
# '第<N>天' -> ['第','<N>','天']
def str2list(str,spec=None):
    str = re.sub('\s+','',str)
    res = []
    i = 0
    while i<len(str):
        match = False
        if spec:
            for sp in spec:
                splen = len(sp)
                if i+splen<=len(str) and str[i:i+splen]==sp:
                    res.append(sp)
                    i += splen
                    match = True
                    break
        if not match:
            res.append(str[i])
            i += 1
    return res
def get_trigger_index(str):
    str_len  = len(str)
    trigger_id = []
    trigger_start =0
    trigger_end = 0
    if '{' in str and '}' in str and '[' not in str and ']' not in str:
        for i in range(str_len):
            if str[i] == '{':
                trigger_start = i
            if str[i] == '}':
                trigger_end = i-1
                break

        for j in range(trigger_start,trigger_end):
            trigger_id.append(j)
        trigger_start = 0
        trigger_end = 0

    if '[' in str and ']' in str and '{' not in str and '}' not in str:
        for i in range(str_len):

            if str[i] == '[':
                trigger_start = i
            if str[i] == ']':
                trigger_end = i-1

            if (trigger_start != 0 or i==0)and trigger_end != 0:
                for j in range(trigger_start, trigger_end):
                    trigger_id.append(j)
                trigger_start = 0
                trigger_end = 0

    return trigger_id





def sent_clean(str):
    str = str.replace('[','')
    str = str.replace(']','')
    str = str.replace('{','')
    str = str.replace('}','')
    return str
# whether ls2 in ls1 (list), return index, similar to function find()
def list_find(ls1,ls2):
    ind = -1
    for i in range(len(ls1)-len(ls2)+1):
        match = True
        for j in range(len(ls2)):
            if ls1[i+j] != ls2[j]:
                match = False
                break
        if match:
            ind = i
            break
    return ind

def trigger_one_hot(trigger_ids, e1_lf,e1_rg, e2_lf, e2_rg, max_sen_length):
    # print(max_sen_length)
    trigger_oh = [0] * max_sen_length
    for i in range(e1_lf, e1_rg+1):
        if i < max_sen_length and i>=0:
            trigger_oh[i] = 1
    for i in range(e2_lf, e2_rg+1):
        if i < max_sen_length and i >= 0:
            trigger_oh[i] =1
    if len(trigger_ids) != 0:
        for i in trigger_ids:
            if i < max_sen_length and i >= 0:
                trigger_oh[i] = 1

    return trigger_oh



MODE_INSTANCE = 0      # One batch contains batch_size instances. [for both training and testing]
MODE_ENTPAIR_BAG = 1   # One batch contains batch_size bags, instances in which have the same entity pair [for testing].
MODE_RELFACT_BAG = 2   # One batch contains batch size bags, instances in which have the same relation fact. [for training].
MODE_INSTANCE_MULTI_LABEL = 3 # One batch contains batch_size instances, and each answer is multi-label (there might be more than 1 correct answer) [for testing].

# read data from file for training/testing
def read_instance_with_gaz_mode(name, input_file, gaz, word_alphabet, biword_alphabet, char_alphabet, gaz_alphabet, label_alphabet, number_normalized, max_sent_length, num_classes, mode, word_sense_map, words_larger_than_one = set(), char_padding_size=-1, char_padding_symbol = '</pad>'):
    fr = open(input_file,'r',encoding='utf-8')

    instence_texts = []
    instence_Ids = []
    
    bags = []
    ori_data = []
    freq = dict()

    # clean data
    for line in fr:
        line = line.strip().split('\t')
        if len(line)==5:
            sentid,ent1,ent2,label,sent = line
        elif len(line)==4:
            ent1,ent2,label,sent = line
            # print(line)#['约旦河', '约旦河西岸', 'PHYS/Near', '约旦河西岸']
        else:
            continue

        if ent1 not in sent_clean(sent):
            print(ent1+" not found in "+sent)
            continue
        if ent2 not in sent_clean(sent):
            print(ent2+" not found in "+sent)
            continue

        if label not in freq:
            freq[label] = 1
        else:
            freq[label] += 1

        ori_data.append({'head':ent1,'tail':ent2,'relation':label,'sentence':sent})
    # print(ori_data)#{'tail': '中央', 'relation': 'ORG-AFF/Employment', 'sentence': '中央领导同志', 'head': '中央领导同志'}
    # deal with multi-label answer
    if mode == MODE_INSTANCE_MULTI_LABEL:
        print("Merging data with same (head,tail,sent)...")
        tmp_data = ori_data
        # print(tmp_data)
        tmp_data.sort(key=lambda a: a['head'] + '#' + a['tail'] + '#' + a['sentence'])
        # print(tmp_data)
        ori_data = []
        prekey = ''
        curins = None
        for ins in tmp_data:
            curkey = ins['head'] + '#' + ins['tail'] + '#' + ins['sentence']
            # print("c:",curkey,"pre:",prekey)
            if curkey != prekey:
                if curins is None:
                    pass
                else:
                    ori_data.append(curins)
                curins = ins
                curins['mult-rel'] = [ins['relation']]
                prekey = curkey
            else:
                curins['mult-rel'].append(ins['relation'])
        # print(curins['mult-rel'])
        if not curins is None:
            ori_data.append(curins)
        # print(ori_data)#将重复的句子和实体进行了归类
        print("Finish merging")
        
    # Sort data by entities and relations
    print("Sort data...")
    ori_data.sort(key=lambda a: a['head'] + '#' + a['tail'] + '#' + a['relation'])
    print("Finish sorting")
    
    bags = []
    prekey = ''
    curbag = []
    maxlen = 0
    # organize data into bags: each bag contains one or more instances according to the mode
    for didx, data in enumerate(ori_data):
        if mode == MODE_ENTPAIR_BAG:
            curkey = data['head'] + '#' + data['tail']
        elif mode == MODE_RELFACT_BAG:
            curkey = data['head'] + '#' + data['tail'] + '#' + data['relation']
        else:
            #curkey = data['head'] + '#' + data['tail'] + '#' + data['relation'] + '#' + data['sentence']
            curkey = str(didx)
            # print("cu:",curkey,"pre:", prekey)
        if curkey!=prekey:
            if len(curbag)>0:
                bags.append(curbag)
            curbag = [data]
            prekey = curkey
        else:
            curbag.append(data)
    if len(curbag)>0:
        bags.append(curbag)
    # print(bags)#[[{'sentence': '5名印度士兵', 'tail': '印度', 'head': '5名印度士兵', 'relation': 'ORG-AFF/Employment'}]]
    ent_cnt = 0
    ent_multi_cnt = 0
    UNK_id = gaz_alphabet.get_index(gaz_alphabet.UNKNOWN)
    for bag in bags:

        bag_texts = []
        bag_Ids = []

        for data in bag:
            words = []
            biwords = []
            chars = []

            word_Ids = []
            biword_Ids = []
            char_Ids = []
            pos1s = []
            pos2s = []
            triggers_id = []
            
            ent1 = data['head']
            ent2 = data['tail']
            sent = data['sentence']
            if 'mult-rel' in data:
                temp_labels = list(set(data['mult-rel']))
            else:
                temp_labels = [data['relation']]
            # print('temp_labels:', temp_labels)
            ent1 = str2list(ent1,words_larger_than_one)
            ent2 = str2list(ent2,words_larger_than_one)
            sent = str2list(sent,words_larger_than_one)
            # print("words_larger_than_one:", words_larger_than_one)
            # print("sent:", sent)#['巴', '勒', '斯', '坦', '权', '力', '机', '构', '同', '埃', '及', '和', '约', '旦', '的', '{', '边', '境', '}']
            # print("sent1:", sent)
            #trigger_id = get_trigger_index(sent)
            # print(trigger_id) #[15, 16]
            # if '[' in sent and ']' in sent:
            #     print(sent)
            # print(triggers_id)

            sent = str2list(sent_clean(''.join(sent)))
            # print(sent)#['高', '雄', '市', '的', '一', '家', '大', '卖', '场']
            # ent1 = str2list(sent_clean((''.join(ent1))))
            # ent2= str2list(sent_clean((''.join(ent2))))
            lf1 = list_find(sent,ent1)
            assert(lf1 != -1)
            rg1 = lf1+len(ent1)-1
            lf2 = list_find(sent,ent2)
            assert(lf2 != -1)
            rg2 = lf2+len(ent2)-1
            # print("sent2:",len(sent), sent)
            # print("ent1:", ent1, "ent2:", ent2)
            #triggers_id = trigger_one_hot(trigger_id, lf1, rg1, lf2, rg2,max_sent_length)
            # print("triggers_id:", triggers_id)
            ent1 = ''.join(ent1)
            ent2 = ''.join(ent2)
            
            if sent[-1] not in  ['。'] and len(sent)<max_sent_length:
                sent.append('。')
            # print(sent)
            for widx,word in enumerate(sent):
                if number_normalized:
                    word = normalize_word(word)
                if widx < len(sent) -1 and len(sent) > 2:
                    biword = word + sent[widx+1]
                else:
                    biword = word + NULLKEY

                # words and bi-gram (not used)
                biwords.append(biword)
                words.append(word)
                word_Ids.append(word_alphabet.get_index(word))
                biword_Ids.append(biword_alphabet.get_index(biword))

                # character features (not uesd)
                char_list = []
                char_Id = []
                for char in word:
                    char_list.append(char)
                if char_padding_size > 0:
                    char_number = len(char_list)
                    if char_number < char_padding_size:
                        char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                    assert(len(char_list) == char_padding_size)
                for char in char_list:
                    char_Id.append(char_alphabet.get_index(char))
                chars.append(char_list)
                char_Ids.append(char_Id)
                # print(chars)
                # print(char_Ids)#[[517], [404], [97]]

                # relative position
                pos1,pos2 = get_pos_embeded(widx,lf1,rg1,lf2,rg2,max_sent_length)
                pos1s.append(pos1)
                pos2s.append(pos2)
                # print(pos1)
                # print(pos1s)#[87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]

            # deal with lexicon (sense-level)
            if ((max_sent_length < 0) or (len(words) <= max_sent_length)) and (len(words)>0):
                gazs = []
                gaz_Ids = []
                w_length = len(words)
                # print(words)#['0', '0', '多', '岁', '胡', '姓', '体', '育', '用', '品', '社', '老', '板', '在', '刚', '刚', '把', '车', '开', '进', '家', '门', '口', '的', '车', '库', '，',
                maxlen = max(maxlen,w_length)
                for widx in range(w_length):
                    # get all potential words that start from index widx
                    matched_list = gaz.enumerateMatchList(words[widx:])
                    # print("match list:", matched_list)#match list: ['用 品']
                    matched_Id = []
                    matched_length = []
                    for entity in matched_list:
                        if gaz.space:
                            entity = entity.split(gaz.space)
                        entlen = len(entity)
                        entity = ''.join(entity)
                        ent_ind = gaz_alphabet.get_index(entity)
                        ent_cnt += 1
                        if ent_ind == UNK_id: # current word is polysemous word with more than one sense
                            if word_sense_map and entity in word_sense_map:
                                ent_multi_cnt += 1
                                # get all senses of current word
                                for cur_ent in word_sense_map[entity]:
                                    cur_ind = gaz_alphabet.get_index(cur_ent)
                                    matched_Id.append(cur_ind)
                                    matched_length.append(entlen) 
                        else:
                            matched_Id.append(ent_ind)
                            matched_length.append(entlen)
                        gazs.append(entity)

                    if matched_Id:
                        gaz_Ids.append([matched_Id, matched_length])
                    else:
                        gaz_Ids.append([])
                    # print(gazs)#['胡姓', '体育', '用品']
                    # print(gaz_Ids)#[[], [], [], [], [[4098], [2]], [], [[520, 521], [2, 2]], [], [[4097], [2]]]
                sentence = ''.join(sent)
                bag_texts.append([words, biwords, chars, gazs, ent1, ent2, temp_labels])
                bag_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, [label_alphabet.get_index(label) for label in temp_labels], pos1s, pos2s,sentence, triggers_id])
            else:
                continue
        
        if len(bag_texts)>0:
            instence_texts.append(bag_texts)
            instence_Ids.append(bag_Ids)   

    print('Total entities:',ent_cnt,' Entities with multi-sense:',ent_multi_cnt,' Ratio:',str(100.0*ent_multi_cnt/ent_cnt)+'%')
    return instence_texts, instence_Ids, freq

# Generate position embeddings
def get_pos_embeded(i,lf1,rg1,lf2,rg2,maxlen=80):

    # scale to [0,2*max-length+2]
    def pos_embed(x):
        if x < -1*maxlen:
            return 0
        if -1*maxlen <= x <= maxlen:
            return x + maxlen + 1
        if x > maxlen:
            return 2*maxlen+2

    # corresponding to Eq. 1 in paper      
    def pos_embed2(i,l,r):
        if i>=l and i<=r:
            x = 0
        elif i<l:
            x = i-l
        else:
            x = i-r
        return pos_embed(x)
                                
    return pos_embed2(i,lf1,rg1),pos_embed2(i,lf2,rg2) 

def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):    
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim
      
def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

# Load pre-trained embeddings from embedding_path
def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path,'r',encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) <= 1:
                continue
            tokens = line.split()
            if len(tokens) <= 3:
                continue
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim
