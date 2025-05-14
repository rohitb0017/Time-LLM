from math import sqrt
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
import transformers

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        self.d_llm = d_llm
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = 16  # Can be parameterized
        self.dropout = nn.Dropout(attention_dropout)

        self.query_projection = nn.Linear(self.d_llm, self.n_heads * self.d_keys)
        self.key_projection = nn.Linear(self.d_llm, self.n_heads * self.d_keys)
        self.value_projection = nn.Linear(self.d_llm, self.n_heads * self.d_keys)
        self.out_projection = nn.Linear(self.n_heads * self.d_keys, self.d_llm)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H, E = self.n_heads, self.d_keys

        target = self.query_projection(target_embedding).view(B, L, H, E)
        source = self.key_projection(source_embedding).view(S, H, E)
        value = self.value_projection(value_embedding).view(S, H, E)

        scores = torch.einsum("blhe,she->bhls", target, source)
        attn_weights = torch.softmax(scores / sqrt(E), dim=-1)
        attn_weights = self.dropout(attn_weights)

        reprogrammed = torch.einsum("bhls,she->blhe", attn_weights, value)
        out = reprogrammed.reshape(B, L, H * E)
        return self.out_projection(out)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = GPT2Model.from_pretrained('gpt2', config=self.gpt2_config)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            raise NotImplementedError(f"Model {configs.llm_model} not supported")

        self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.description = configs.content if configs.prompt_domain else (
            'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.')

        self.dropout = nn.Dropout(configs.dropout)
        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout)
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]

        self.mapping_layer = nn.Linear(configs.d_model, self.d_llm)

        self.reprogramming_layer = ReprogrammingLayer(d_model=configs.d_model, n_heads=configs.n_heads, d_llm=self.d_llm, attention_dropout=0.1)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, head_dropout=configs.dropout)
        else:
            raise NotImplementedError("Task not supported")

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()

        x_enc = x_enc.permute(0, 2, 1).reshape(B * N, T, 1)
        min_vals, max_vals = torch.min(x_enc, dim=1)[0], torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            prompt.append(
                f"<|start_prompt|>Dataset description: {self.description} Task description: forecast the next {self.pred_len} steps given the previous {self.seq_len} steps information; "
                f"Input statistics: min value {min_vals[b].item()}, max value {max_vals[b].item()}, median value {medians[b].item()}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, top 5 lags are : {lags[b].tolist()}<|<end_prompt>|>")

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids.to(x_enc.device))

        source_embeddings = self.word_embeddings.to(x_enc.device)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)

        enc_out = self.mapping_layer(enc_out)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        gpt2_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=gpt2_enc_out).last_hidden_state
        dec_out = dec_out[:, -enc_out.size(1):, :self.d_ff]

        dec_out = dec_out.view(B, n_vars, self.patch_nums, self.d_ff).permute(0, 1, 3, 2)
        out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        out = out.permute(0, 2, 1).contiguous()

        return self.normalize_layers(out, 'denorm')

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1), dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), dim=-1)
        mean_corr = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_corr, self.top_k, dim=-1)
        return lags
