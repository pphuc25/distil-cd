import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
    """Validates model kwargs for generation. Generate argument typos will also be caught here."""
    # Excludes arguments that are handled before calling any model function
    if self.config.is_encoder_decoder:
        for key in ["decoder_input_ids"]:
            model_kwargs.pop(key, None)

    unused_model_args = []
    model_args = set(inspect.signature(
        self.prepare_inputs_for_generation).parameters)
    # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
    # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
    if "kwargs" in model_args or "model_kwargs" in model_args:
        model_args |= set(inspect.signature(self.forward).parameters)

    # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
    if self.config.is_encoder_decoder:
        base_model = getattr(self, self.base_model_prefix, None)

        # allow encoder kwargs
        encoder = getattr(self, "encoder", None)
        # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
        # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
        # TODO: A better way to handle this.
        if encoder is None and base_model is not None:
            encoder = getattr(base_model, "encoder", None)

        if encoder is not None:
            encoder_model_args = set(
                inspect.signature(encoder.forward).parameters)
            model_args |= encoder_model_args

        # allow decoder kwargs
        decoder = getattr(self, "decoder", None)
        if decoder is None and base_model is not None:
            decoder = getattr(base_model, "decoder", None)

        if decoder is not None:
            decoder_model_args = set(
                inspect.signature(decoder.forward).parameters)
            model_args |= {f"decoder_{x}" for x in decoder_model_args}

    for key, value in model_kwargs.items():
        if value is not None and key not in model_args:
            unused_model_args.append(key)