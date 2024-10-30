#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import argparse
import copy
import random
import socket
from contextlib import suppress
from functools import partial
from tqdm import tqdm
import torch.nn as nn
import numpy as np

import time
import torch.distributed
import torch.distributed as dist
import torch.utils.data
from torch.utils.data import Dataset

# from utils import DEV, get_loaders, get_c4_enc, get_wikitext2_enc

DEV = torch.device('cuda:0')


HOST_NAME = socket.getfqdn(socket.gethostname())

def get_wikitext2_enc(nsamples, seed, seqlen, model):
    from datasets import load_dataset, load_from_disk
    # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    # testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    traindata = load_from_disk('data/wikitext2/wikitext-2-train.hf' )
    testdata = load_from_disk('data/wikitext2/wikitext-2-test.hf' )


    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    final_trainenc = {}
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        # print(tar.shape)
        trainloader.append( tar)
    # print(testenc)
    # print(type(testenc))
    final_trainenc['input_ids'] = torch.cat(trainloader, dim=1)
    print(final_trainenc)
    return final_trainenc, testenc


def get_c4_enc(nsamples, seed, seqlen, model):
    from datasets import load_dataset, load_from_disk
    traindata = load_from_disk('data/c4/traindata.hf' )
    valdata = load_from_disk('data/c4/valdata.hf' )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    final_trainenc = {}
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        trainloader.append(tar)
    final_trainenc['input_ids'] = torch.cat(trainloader, dim=1)
    print(final_trainenc)

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return final_trainenc, valenc 

torch.backends.cudnn.benchmark = True




def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def llama_diff_block_pca(model1, model2, testenc, dev):
    print('Evaluation...')
    testenc = testenc['input_ids']
    print(testenc.shape)
    nsamples = testenc.numel() // model1.seqlen

    use_cache = model1.config.use_cache
    model1.config.use_cache = False
    layers1 = model1.model.layers

    model1.model.embed_tokens = model1.model.embed_tokens.to(dev)
    layers1[0] = layers1[0].to(dev)
    dtype1 = next(iter(model1.parameters())).dtype
    inps1 = torch.zeros((nsamples, model1.seqlen, model1.config.hidden_size), dtype=dtype1, device=dev)

    use_cache = model2.config.use_cache
    model2.config.use_cache = False
    layers2 = model2.model.layers

    model2.model.embed_tokens = model2.model.embed_tokens.to(dev)
    layers2[0] = layers2[0].to(dev)
    dtype2 = next(iter(model2.parameters())).dtype
    inps2 = torch.zeros((nsamples, model2.seqlen, model2.config.hidden_size), dtype=dtype2, device=dev)

    cache1 = {'i': 0, 'attention_mask': None}
    cache2 = {'i': 0, 'attention_mask': None}

    class Catcher1(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps1[cache1['i']] = inp
            cache1['i'] += 1
            cache1['attention_mask'] = kwargs['attention_mask']
            cache1['position_ids'] = kwargs['position_ids']
            raise ValueError
    class Catcher2(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps2[cache2['i']] = inp
            cache2['i'] += 1
            cache2['attention_mask'] = kwargs['attention_mask']
            cache2['position_ids'] = kwargs['position_ids']

            raise ValueError

    layers1[0] = Catcher1(layers1[0])
    layers2[0] = Catcher2(layers2[0])

    for i in range(nsamples):
        batch1 = testenc[:, (i * model1.seqlen):((i + 1) * model1.seqlen)].to(dev)
        try:
            model1(batch1)
        except ValueError:
            pass
    
    for i in range(nsamples):
        batch2 = testenc[:, (i * model2.seqlen):((i + 1) * model2.seqlen)].to(dev)
        try:
            model2(batch2)
        except ValueError:
            pass

    print(torch.dist(inps1, inps2))
    layers1[0] = layers1[0].module
    layers1[0] = layers1[0].cpu()

    model1.model.embed_tokens = model1.model.embed_tokens.cpu()
    torch.cuda.empty_cache()


    layers2[0] = layers2[0].module
    layers2[0] = layers2[0].cpu()
    model2.model.embed_tokens = model2.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs1 = torch.zeros_like(inps1)
    outs2 = torch.zeros_like(inps2)

    attention_mask1 = cache1['attention_mask']
    attention_mask2 = cache2['attention_mask']

    position_ids1 = cache2['position_ids']
    position_ids2 = cache2['position_ids']

    new_dict = {}

    for i in range(len(layers1)):
        print(i)

        layer1 = layers1[i].to(dev)
        layer2 = layers2[i].to(dev)

        input_previous = inps2.detach().flatten(0,1).cpu().to(torch.float32)

        for j in range(nsamples):
            outs1[j] = layer1(inps1[j].unsqueeze(0), attention_mask=attention_mask1, position_ids=position_ids1,use_cache=False)[0]
            outs2[j] = layer2(inps2[j].unsqueeze(0), attention_mask=attention_mask2, position_ids=position_ids2,use_cache=False)[0]
        # outs1 = layer1(inps1, attention_mask=attention_mask1, position_ids=position_ids1,use_cache=False)[0]        
        # outs2 = layer2(inps2, attention_mask=attention_mask2, position_ids=position_ids2,use_cache=False)[0]

        output = (outs1 - outs2).flatten(0,1).detach().cpu().to(torch.float32)
        new_w = torch.linalg.lstsq(input_previous, output).solution
        new_b = torch.mean(output - input_previous @  new_w, dim=0)

        Y_pred = input_previous @ new_w + new_b

        abs_loss = (output - Y_pred).abs().mean()

        ss_tot = torch.sum((output - output.mean(dim=0)).pow(2))
        ss_res = torch.sum((output - Y_pred).pow(2))
        r2_score = 1 - ss_res / ss_tot

        if r2_score > 0:
            print(abs_loss, r2_score,new_w, new_b)
            new_dict['model.layers.' + str(i) + '.lora_weight'] = new_w
            new_dict['model.layers.' + str(i) + '.lora_bias'] = new_b

            # torch.save(new_w, str(i) + '_w.pt')
            # torch.save(new_b, str(i) + '_b.pt')
            print(torch.dist(outs2, outs1))
            outs2 += Y_pred.reshape(outs2.shape).to(torch.bfloat16).to(dev)
            print(torch.dist(outs2, outs1))

            print(input_previous.shape, output.shape)
            print(input_previous, output)

        layers1[i] = layer1.cpu() 
        layers2[i] = layer2.cpu() 

        del layer1
        del layer2
        torch.cuda.empty_cache()
        inps1, outs1 = outs1, inps1
        inps2, outs2 = outs2, inps2

    torch.save(new_dict, 'lora_weight.pt')
    if model1.model.norm is not None:
        model1.model.norm = model1.model.norm.to(dev)
    model1.lm_head = model1.lm_head.to(dev)

    if model2.model.norm is not None:
        model2.model.norm = model2.model.norm.to(dev)
    model2.lm_head = model2.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls1 = []
    nlls2 = []

    for i in range(nsamples):
        loss_fct = nn.CrossEntropyLoss()

        hidden_states1 = inps1[i].unsqueeze(0)
        if model1.model.norm is not None:
            hidden_states1 = model1.model.norm(hidden_states1)
        lm_logits1 = model1.lm_head(hidden_states1)

        shift_logits1 = lm_logits1[:, :-1, :].contiguous()

        hidden_states2 = inps2[i].unsqueeze(0)
        if model2.model.norm is not None:
            hidden_states2 = model2.model.norm(hidden_states2)
        lm_logits2 = model2.lm_head(hidden_states2)

        shift_logits2 = lm_logits2[:, :-1, :].contiguous()

        shift_labels = testenc[:, (i * model1.seqlen):((i + 1) * model1.seqlen)][:, 1:]

        loss1 = loss_fct(shift_logits1.view(-1, shift_logits1.size(-1)), shift_labels.view(-1))
        neg_log_likelihood1 = loss1.float() * model1.seqlen
        nlls1.append(neg_log_likelihood1)

        loss2 = loss_fct(shift_logits2.view(-1, shift_logits2.size(-1)), shift_labels.view(-1))
        neg_log_likelihood2 = loss2.float() * model2.seqlen
        nlls2.append(neg_log_likelihood2)

    ppl1 = torch.exp(torch.stack(nlls1).sum() / (nsamples * model1.seqlen))
    print(ppl1.item())

    ppl2 = torch.exp(torch.stack(nlls2).sum() / (nsamples * model2.seqlen))
    print(ppl2.item())

    model1.config.use_cache = use_cache
    model2.config.use_cache = use_cache

    return ppl1, ppl2

def merge_model(model, lora):
    new_dict = model.state_dict()
    new_dict.update(lora)
    torch.save(new_dict, 'temp.pt')

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model")
parser.add_argument("--quant_model", type=str, help="model")

parser.add_argument('--dataset', default="imagenet")

parser.add_argument('--start_block', default=0, type=int)

parser.add_argument("--batch_size", default=32, type=int, help="batchsize of validation set")

parser.add_argument('--save_files', default=True, action='store_true')
parser.add_argument("--seed", default=0, type=int, help="seed")

args = parser.parse_args()


def get_llama(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from model.modeling_llama import LlamaForCausalLM
    # from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model

def get_quant_llama(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from quant_model.modeling_llama import LlamaForCausalLM
    # from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model


def main():


    seed(args.seed)

    ###############################################
    #这一部分的代码是ptq模型初始化的代码，不同的ptq方法初始化的过程不一样，因此需要根据使用的ptq方法进行修改
    base_model = get_llama(args.model)
    base_model.eval()

    q_model = get_quant_llama(args.quant_model)
    q_model.eval()
    ##################################################

    # dataloader, testloader = get_loaders(args.dataset, seed=args.seed, model=args.model, seqlen=base_model.seqlen)

    #调用函数生成不同block的W和b
    # with torch.no_grad():
    #     q_model = generate_compensation_model(q_model, dataloader, args)

    trainloader, testloader = get_c4_enc(nsamples=128, seed=args.seed, seqlen=base_model.seqlen, model=args.model)
    # trainloader, testloader = get_wikitext2_enc(nsamples=128, seed=args.seed, seqlen=base_model.seqlen, model=args.model)

    with torch.no_grad():
        ppl1,ppl2 = llama_diff_block_pca(base_model, q_model, trainloader ,DEV)
    print(ppl1, ppl2)
    lora = torch.load('lora_weight.pt')
    merge_model(q_model, lora)

if __name__ == '__main__':
    main()
