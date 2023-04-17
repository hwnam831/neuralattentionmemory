import torch.nn
import torch.nn.functional as F
from transformer_generalization.interfaces import ModelInterface
from transformer_generalization.tasks.task import Task
from transformer_generalization.layers import TiedEmbedding
from transformer_generalization import framework
import NAM
from transformer_generalization import dataset
from transformer_generalization.interfaces.encoder_decoder import EncoderDecoderResult
from transformer_generalization.models.encoder_decoder import add_eos
from transformer_generalization.models.transformer_enc_dec import TransformerResult

class NAMTMEncDecModel(torch.nn.Module):
    def __init__(self, n_input_tokens: int, n_out_tokens: int, state_size: int = 512, ff_multiplier: float = 4,
                 nhead: int=8, tapelen: int=64,max_len: int=5000, 
                 encoder_sos: bool = True, same_enc_dec_embedding: bool = False):
        '''
        Transformer encoder-decoder.

        :param n_input_tokens: Number of channels for the input vectors
        :param n_out_tokens: Number of channels for the output vectors
        :param state_size: The size of the internal state of the transformer
        '''
        super().__init__()


        assert (not same_enc_dec_embedding) or (n_input_tokens == n_out_tokens)

        self.register_buffer('int_seq', torch.arange(max_len, dtype=torch.long))

        self.decoder_sos_eos = n_out_tokens
        self.encoder_eos = n_input_tokens
        self.encoder_sos = n_input_tokens + 1 if encoder_sos else None
        self.state_size = state_size

        self.ff_multiplier = ff_multiplier
        self.n_input_tokens = n_input_tokens
        self.n_out_tokens = n_out_tokens
        self.nhead = nhead
        self.tapelen = tapelen
        self.same_enc_dec_embedding = same_enc_dec_embedding
        
        self.input_embedding = torch.nn.Embedding(self.n_input_tokens + 1 + int(self.encoder_sos is not None), 
                                                  self.state_size)
        self.output_embedding = self.input_embedding if self.same_enc_dec_embedding else \
                                torch.nn.Embedding(self.n_out_tokens+1, self.state_size)

        self.output_map = TiedEmbedding(self.output_embedding.weight)


        self.encoder = NAM.NAMTuring(self.state_size, self.nhead, self.tapelen, self.state_size//self.nhead)
        self.decoder = NAM.NAMTuringDecoder(self.state_size, self.nhead, self.tapelen, self.state_size//self.nhead)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, self.state_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.state_size, self.state_size)
        )
    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[: max_len] >= len.unsqueeze(-1)

    def output_embed(self, x: torch.Tensor) -> torch.Tensor:
        o = self.output_embedding(x)
        return o

    def run_greedy(self, src: torch.Tensor, src_len: torch.Tensor, max_len: int) -> TransformerResult:
        batch_size = src.shape[0]
        n_steps = src.shape[1]

        _, tape_in = self.encoder(src.permute(1,0,2))

        running = torch.ones([batch_size], dtype=torch.bool, device=src.device)
        out_len = torch.zeros_like(running, dtype=torch.long)

        next_tgt = self.output_embed(torch.full([batch_size], self.decoder_sos_eos, dtype=torch.long,
                                                            device=src.device))

        all_outputs = []
        state = self.decoder.init_states(tape_in)
        tape, rpos, wpos, cntl_hidden = state
        for i in range(max_len):
            output = self.decoder.forward_step(next_tgt, tape, rpos, wpos, cntl_hidden)
            rval, tape, rpos, wpos, cntl_hidden = output
            output = self.output_map(rval)
            all_outputs.append(output)

            out_token = torch.argmax(output, -1)
            running &= out_token != self.decoder_sos_eos

            out_len[running] = i + 1
            next_tgt = self.output_embed(out_token)

        return TransformerResult.create(torch.stack(all_outputs, 1), out_len)

    def run_teacher_forcing(self, src: torch.Tensor, src_len: torch.Tensor, target: torch.Tensor,
                            target_len: torch.Tensor) -> TransformerResult:
        target = self.output_embed(F.pad(target[:, :-1], (1, 0), value=self.decoder_sos_eos).long())

        _, tape_in = self.encoder(src.permute(1,0,2))
        res = self.decoder(target.permute(1,0,2), tape_in).permute(1,0,2)

        return TransformerResult.create(self.output_map(res), target_len)

    def input_embed(self, x: torch.Tensor) -> torch.Tensor:
        src = self.input_embedding(x.long())
        return src

    def forward(self, src: torch.Tensor, src_len: torch.Tensor, target: torch.Tensor,
                target_len: torch.Tensor, teacher_forcing: bool, max_len = None) -> TransformerResult:
        '''
        Run transformer encoder-decoder on some input/output pair

        :param src: source tensor. Shape: [N, S], where S in the in sequence length, N is the batch size
        :param src_len: length of source sequences. Shape: [N], N is the batch size
        :param target: target tensor. Shape: [N, S], where T in the in sequence length, N is the batch size
        :param target_len: length of target sequences. Shape: [N], N is the batch size
        :param teacher_forcing: use teacher forcing or greedy decoding
        :param max_len: overwrite autodetected max length. Useful for parallel execution
        :return: prediction of the target tensor. Shape [N, T, C_out]
        '''

        if self.encoder_sos is not None:
            src = F.pad(src, (1, 0), value=self.encoder_sos)
            src_len = src_len + 1
            
        src = self.input_embed(src)

        if teacher_forcing:
            return self.run_teacher_forcing(src, src_len, target, target_len)
        else:
            return self.run_greedy(src, src_len, max_len or target_len.max().item())

class NAMTMEncDecInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0):
        self.model = model
        self.label_smoothing = label_smoothing

    def loss(self, outputs: TransformerResult, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        l = framework.layers.cross_entropy(outputs.data, ref, reduction='none', smoothing=self.label_smoothing)
        l = l.reshape_as(ref) * mask
        l = l.sum() / mask.sum()
        return l

    def decode_outputs(self, outputs: EncoderDecoderResult):
        return outputs.outputs, outputs.out_lengths

    def __call__(self, data, train_eos: bool = True) -> EncoderDecoderResult:
        in_len = data["in_len"].long()
        out_len = data["out_len"].long()
        in_with_eos = add_eos(data["in"], data["in_len"], self.model.encoder_eos)
        out_with_eos = add_eos(data["out"], data["out_len"], self.model.decoder_sos_eos)
        in_len += 1
        out_len += 1

        res = self.model(in_with_eos.transpose(0, 1), in_len, out_with_eos.transpose(0, 1),
                         out_len, teacher_forcing=self.model.training, max_len=out_len.max().item())

        res.data = res.data.transpose(0, 1)
        len_mask = ~self.model.generate_len_mask(out_with_eos.shape[0], out_len if train_eos else (out_len - 1)).\
                                                 transpose(0, 1)

        loss = self.loss(res, out_with_eos, len_mask)
        return EncoderDecoderResult(res.data, res.length, loss)

class NAMTMMixin:
    def create_model(self) -> torch.nn.Module:
        
        '''(self, n_input_tokens: int, n_out_tokens: int, state_size: int = 512, ff_multiplier: float = 4,
                 nhead: int=8, tapelen: int=64,
                 encoder_sos: bool = True, same_enc_dec_embedding: bool = False)'''
        return NAMTMEncDecModel(len(self.train_set.in_vocabulary),
                                      len(self.train_set.out_vocabulary), self.helper.args.state_size,
                                      nhead=self.helper.args.transformer.n_heads,
                                      ff_multiplier=self.helper.args.transformer.ff_multiplier)

    def create_model_interface(self):
        self.model_interface = NAMTMEncDecInterface(self.model, label_smoothing=self.helper.args.label_smoothing)

class ScanNAMTM(NAMTMMixin, Task):
    VALID_NUM_WORKERS = 0
    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.ScanLengthResplit("train", (0, self.helper.args.scan.length_cutoff))
        self.valid_sets.val = dataset.ScanLengthResplit("all", (self.helper.args.scan.length_cutoff+1, 9999))
        self.valid_sets.iid = dataset.ScanLengthResplit("test", (0, self.helper.args.scan.length_cutoff))