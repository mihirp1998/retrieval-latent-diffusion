import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
import kornia
import ipdb
st = ipdb.set_trace


from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))




class RlBenchContextEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.clip_image = FrozenClipImageEmbedder(model)
        self.clip_text = FrozenCLIPTextEmbedder(model)
        self.clip_image.eval()
        self.clip_text.eval()
        for param in self.clip_text.parameters():
            param.requires_grad = False
        for param in self.clip_image.parameters():
            param.requires_grad = False
        # print('hello')
        self.support_embed = nn.Parameter(torch.randn(1))
        self.target_embed = nn.Parameter(torch.randn(1))
        self.text_embed = nn.Parameter(torch.randn(1))


    def forward(self, x):
        image,text = x
        with torch.no_grad():
            text_encode = self.clip_text(text).unsqueeze(1)
            B,C,H,W = image.shape
            image_concat = torch.concat(image.split(3,1),0)
            image_encode = self.clip_image(image_concat)
            image_encodings = torch.stack(image_encode.split(B,0),1)
        encodings = torch.cat([text_encode,image_encodings],1)
        id_embed = torch.cat([self.text_embed, self.support_embed,self.target_embed],0)
        id_embed = id_embed.unsqueeze(-1).unsqueeze(0).repeat(B,1,1)
        encodings = torch.cat([encodings,id_embed],-1)
        # st()
        return encodings



class RlBenchMultiContextEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.clip_image = FrozenClipImageEmbedder(model)
        self.clip_text = FrozenCLIPTextEmbedder(model)
        self.clip_image.eval()
        self.clip_text.eval()
        for param in self.clip_text.parameters():
            param.requires_grad = False
        for param in self.clip_image.parameters():
            param.requires_grad = False
        # print('hello')
        self.support_embed = nn.Parameter(torch.randn(128))
        self.target_embed = nn.Parameter(torch.randn(128))
        self.text_embed = nn.Parameter(torch.randn(128))
        self.no_time_embed = nn.Parameter(torch.randn(128))        
        self.no_target_image_embed = nn.Parameter(torch.randn(896))
        self.time_encode = nn.Linear(1,128)



    def forward(self, x):
        support_images,target_images,support_idxs,target_idx, text_captions = x
        # st()
        # image,text = x
        with torch.no_grad():
            text_encode = self.clip_text(text_captions).unsqueeze(1)
            B,C,H,W = support_images.shape
            support_image_concat = torch.concat(support_images.split(3,1),0)
            support_image_encode = self.clip_image(support_image_concat)
            target_encodings = self.clip_image(target_images).unsqueeze(1)
            support_encodings = torch.stack(support_image_encode.split(B,0),1)
            
            B,N_support, _ = support_encodings.shape
            _,N_target, _ = target_encodings.shape


        support_embed = self.support_embed.unsqueeze(0).unsqueeze(0).repeat(B,N_support,1)
        target_embed = self.target_embed.unsqueeze(0).unsqueeze(0).repeat(B,N_target,1)
        no_time_embed = self.no_time_embed.unsqueeze(0).unsqueeze(0).repeat(B,1,1)
        no_target_image_embed = self.no_target_image_embed.unsqueeze(0).unsqueeze(0).repeat(B,1,1)


        text_embed = self.text_embed.unsqueeze(0).unsqueeze(0).repeat(B,1,1)

        support_idxs_embed = self.time_encode(support_idxs.unsqueeze(-1))
        target_idxs_embed = self.time_encode(target_idx.unsqueeze(-1).unsqueeze(-1))


        support_encodings = torch.cat([support_encodings,support_embed,support_idxs_embed],-1)
        target_encodings = torch.cat([target_encodings,target_embed,no_time_embed],-1)
        target_topredict_encodings = torch.cat([no_target_image_embed,target_idxs_embed],-1)
        text_encode = torch.cat([text_encode,text_embed,no_time_embed],-1)
        # st()        
        # st()
        encodings = torch.cat([text_encode,support_encodings,target_encodings, target_topredict_encodings],1)
        return encodings
