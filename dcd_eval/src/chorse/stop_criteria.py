import torch
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
from transformers.utils import add_start_docstrings

STOPPING_CRITERIA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
            or scores for each vocabulary token after SoftMax.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional stopping criteria specific kwargs.

    Return:
        `bool`. `False` indicates we should continue, `True` indicates we should stop.

"""


class LLamaQaStoppingCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the model generates '\nQ:' tokens. It means that the model has finished generating the answer and start generating a new question.
    """
    def __init__(self, list_token_ids_sequence: list = [[29984, 29901]]):
        self.token_ids_sequences = []
        self.lengths = []
        for token_ids_sequence in list_token_ids_sequence:
            self.token_ids_sequences.append(torch.tensor(token_ids_sequence, dtype=torch.long))
            self.lengths.append(len(token_ids_sequence))
        
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # check the final {self.length} tokens
        stop = False
        for token_ids_sequence, length in zip(self.token_ids_sequences, self.lengths):
            if input_ids.shape[-1] < length:
                continue
            else:
                if bool(torch.all(input_ids[0, -length:] == token_ids_sequence.to(input_ids.device))):
                    stop = True
                    break
        return stop


def set_stop_words(tokenizer, stop_words):
    stopping_criteria = StoppingCriteriaList()
    list_stop_word_ids = []
    for stop_word in stop_words:
        stop_word_ids = tokenizer.encode('\n' + stop_word)[3:]
        list_stop_word_ids.append(stop_word_ids)
        print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
    stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))
    return stopping_criteria