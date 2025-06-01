import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.max_length = dataset.max_length
        self.vocab_size = dataset.vocab_size

        self.bos_id = dataset.bos_id
        self.eos_id = dataset.eos_id
        self.pad_id = dataset.pad_id
        self.unk_id = dataset.unk_id
        
        self.embedding = nn.Embedding(dataset.vocab_size, 
                                      embed_size, 
                                      padding_idx=self.pad_id)
        self.rnn = rnn_type(embed_size, 
                            hidden_size, 
                            num_layers=rnn_layers, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, self.dataset.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        embedded = self.embedding(indices)
        packed = pack_padded_sequence(embedded, 
                                      lengths, 
                                      batch_first=True, 
                                      enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, 
                                        batch_first=True)
        logits = self.linear(output)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        device = next(self.parameters()).device

        bos_id = self.dataset.bos_id
        eos_id = self.dataset.eos_id

        if prefix:
            prefix_tokens = self.dataset.text2ids(prefix)
            tokens = [bos_id] + prefix_tokens
        else:
            tokens = [bos_id]

        hidden_state = None

        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

        for _ in range(self.max_length - len(tokens)):
            embeddings = self.embedding(input_tensor)
            output, hidden_state = self.rnn(embeddings, hidden_state)
            last_output = output[:, -1, :]
            logits = self.linear(last_output) / temp

            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()

            tokens.append(next_token)

            if next_token == eos_id:
                break

            input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)

        generated_text = self.dataset.sp_model.decode(tokens[1:])

        return generated_text