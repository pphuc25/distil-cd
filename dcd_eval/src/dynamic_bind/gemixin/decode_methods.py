import warnings
from typing import List, Optional, Union

import torch
import torch.distributed as dist

from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GreedySearchEncoderDecoderOutput
from src.dynamic_bind.generation.utils import GreedySearchDecoderOnlyOutput

import transformers.generation.utils as gu
gu.GreedySearchDecoderOnlyOutput = GreedySearchDecoderOnlyOutput


GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput,
                           GreedySearchDecoderOnlyOutput]


@torch.no_grad()
def greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
    used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>


    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.

        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

    >>> input_prompt = "It might be possible to"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> outputs = model.greedy_search(
    ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
    ```"""

    use_dola = model_kwargs.get("dola_decoding", None)
    if use_dola:
        return self.dola_greedy_decode(
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(
            stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(
        input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )
    use_constrative_student = True if model_kwargs.get(
        "input_ids_student", None) != None else None
    use_repe = True if model_kwargs.get("use_repe", None) != None else None
    use_student_lm = True if model_kwargs.get(
        "student_lm", None) != None else None

    if use_constrative_student:
        input_ids_student = model_kwargs['input_ids_student']

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (
        return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get(
            "attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get(
                "hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    model_kwargs['teacher_student'] = model_kwargs.get(
        'teacher_student', False)

    if model_kwargs['teacher_student']:
        model_kwargs_student = model_kwargs.get("model_kwargs_student", {})

    this_peer_finished = False  # used by synced_gpus only
    i_th_token_gen = 0  # Used for constrastive prompting

    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # ===== Contrastive Decoding ====
        if model_kwargs['teacher_student']:
            if use_constrative_student and not use_student_lm:
                model_inputs_student = self.prepare_inputs_for_generation(
                    input_ids_student, **model_kwargs_student)
                
                if model_kwargs["dropout_rate"] != None:
                    # print("Set the current student model (use the same) as enable dropout")
                    self.attention_dropout=model_kwargs["dropout_rate"]
                    self.train()

                outputs_student = self(
                    **model_inputs_student,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                if model_kwargs["dropout_rate"] != None:
                    # print("Set the current student model (use the same) as enable dropout")
                    self.attention_dropout=0.0
                    self.eval()
            elif use_constrative_student and use_student_lm:
                model_inputs_student = model_kwargs['student_lm'].prepare_inputs_for_generation(
                    input_ids_student, **model_kwargs_student)

                outputs_student = model_kwargs['student_lm'](
                    **model_inputs_student,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
            else:
                if use_repe:
                    model_inputs_student = self.prepare_inputs_for_generation(
                        input_ids, **model_kwargs_student)
                else:
                    model_inputs_student = model_kwargs['student_lm'].prepare_inputs_for_generation(
                        input_ids, **model_kwargs_student)
                outputs_student = model_kwargs['student_lm'](
                    **model_inputs_student,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
            next_token_logits_student = outputs_student.logits[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (
                        outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # argmax
        if model_kwargs['teacher_student']:
            import numpy as np
            sorted_logits, sorted_indices = torch.sort(
                next_tokens_scores, descending=True)
            min_thresh = sorted_logits[:,
                                       model_kwargs['min_tokens_to_keep'] - 1]
            cutoff = np.log(
                model_kwargs['alpha_coef']) + torch.max(next_tokens_scores, dim=-1).values
            cutoff = torch.min(min_thresh, cutoff).unsqueeze(-1)
            indices_to_remove = next_tokens_scores < cutoff
            next_tokens_scores = (
                1+model_kwargs['beta_coef']) * next_tokens_scores - model_kwargs['beta_coef'] * next_token_logits_student
            next_tokens_scores = next_tokens_scores.masked_fill(
                indices_to_remove, -float("Inf"))

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError(
                    "If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + \
                pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if use_constrative_student:
            input_ids_student = torch.cat(
                [input_ids_student, next_tokens[:, None]], dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs['teacher_student']:
            model_kwargs_student = self._update_model_kwargs_for_generation(
                outputs_student, model_kwargs_student, is_encoder_decoder=self.config.is_encoder_decoder
            )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(
                    eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

        i_th_token_gen += 1

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids


@torch.no_grad()
def dola_greedy_decode(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
    used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>


    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.

        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

    >>> input_prompt = "It might be possible to"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> outputs = model.greedy_search(
    ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
    ```"""
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(
            stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(
        input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (
        return_dict_in_generate and output_hidden_states) else None
    
    mature_layer = model_kwargs.get("mature_layer", None)
    base_layer = model_kwargs.get("base_layer", None)
    candidate_premature_layers = model_kwargs.get("candidate_premature_layers", None)
    relative_top = model_kwargs.get("relative_top", 0.1)


    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get(
            "attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get(
                "hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    if base_layer is not None and candidate_premature_layers is None:
        early_exit_layers = [base_layer, mature_layer]
        num_base_layers = 1
        premature_layer_dist = {}
    elif candidate_premature_layers is not None:
        early_exit_layers = candidate_premature_layers + [mature_layer]
        num_base_layers = len(candidate_premature_layers)
        premature_layer_dist = {l: 0 for l in candidate_premature_layers}
    else:
        raise ValueError(
            "You must specify either `base_layer` or `candidate_premature_layers`")

    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, **model_kwargs)

        # forward pass to get next token
        dict_outputs, outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            early_exit_layers=early_exit_layers,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need
        if base_layer is not None:
            base_logits = dict_outputs[base_layer][:, -1, :]
            final_logits = dict_outputs[mature_layer][:, -1, :]
            if relative_top > 0.0:
                final_logits = self.relative_top_filter(
                    final_logits, relative_top)
                base_logits = base_logits.log_softmax(dim=-1)
                mask = final_logits[0] < -1e3
                base_logits[0][mask] = -1e3

            logits = final_logits - base_logits
            next_token_logits = logits
        else:
            # 1. Stacking all premature_layers into a new dimension
            stacked_premature_layers = torch.stack(
                [dict_outputs[i][:, -1, :] for i in candidate_premature_layers], dim=0)

            # 2. Calculate the softmax values for mature_layer and all premature_layers
            # shape: (batch_size, num_features)
            softmax_mature_layer = torch.nn.functional.softmax(
                dict_outputs[mature_layer][:, -1, :], dim=-1)
            # shape: (num_premature_layers, batch_size, num_features)
            softmax_premature_layers = torch.nn.functional.softmax(
                stacked_premature_layers, dim=-1)

            # 3. Calculate M, the average distribution
            # shape: (num_premature_layers, batch_size, num_features)
            M = 0.5 * \
                (softmax_mature_layer[None, :,
                                      :] + softmax_premature_layers)

            # 4. Calculate log-softmax for the KL divergence
            log_softmax_mature_layer = torch.nn.functional.log_softmax(
                dict_outputs[mature_layer][:, -1, :], dim=-1)  # shape: (batch_size, num_features)
            # shape: (num_premature_layers, batch_size, num_features)
            log_softmax_premature_layers = torch.nn.functional.log_softmax(
                stacked_premature_layers, dim=-1)

            # 5. Calculate the KL divergences and then the JS divergences
            # shape: (num_premature_layers, batch_size)
            kl1 = torch.nn.functional.kl_div(
                log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)
            # shape: (num_premature_layers, batch_size)
            kl2 = torch.nn.functional.kl_div(log_softmax_premature_layers,
                                             M, reduction='none').mean(-1)
            # shape: (num_premature_layers, batch_size)
            js_divs = 0.5 * (kl1 + kl2)

            # 6. Reduce the batchmean
            # shape: (num_premature_layers,)
            js_divs = js_divs.mean(-1)
            premature_layer = candidate_premature_layers[int(
                js_divs.argmax().cpu().item())]
            premature_layer_dist[premature_layer] += 1

            base_logits = dict_outputs[premature_layer][:, -1, :]
            final_logits = dict_outputs[mature_layer][:, -1, :]
            if relative_top > 0.0:
                final_logits = self.relative_top_filter(
                    final_logits, relative_top)
                base_logits = base_logits.log_softmax(dim=-1)
                mask = final_logits[0] < -1e3
                base_logits[0][mask] = -1e3
            # if theta_coef != None:
            #     last_layer_student = dict_outputs_student[mature_layer][:, -1, :]
            #     last_layer_student[0][mask] = -1e3
            #     # print(f"Using theta coef as {theta_coef}")
            #     logits = (1+theta_coef)*final_logits - theta_coef*base_logits
            #     # logits = (1+theta_coef)*(final_logits - base_logits) - theta_coef*last_layer_student
            # else:
            logits = final_logits - base_logits
            next_token_logits = logits

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (
                        outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError(
                    "If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + \
                pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(
                    eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                premature_layer_dist=premature_layer_dist,
            )
    else:
        return input_ids


def relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
    import numpy as np
    scores_normalized = scores.log_softmax(dim=-1)
    sorted_logits, sorted_indices = torch.sort(
        scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep-1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    scores_normalized[scores_normalized < probs_thresh] = filter_value
    return scores_normalized
