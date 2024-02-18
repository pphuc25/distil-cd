from dcd.generate import greedy_search, _validate_model_kwargs
import transformers.generation.utils as gu

def dcd_pipeline_registry():
    attribute_assign = {
        "greedy_search": greedy_search,
        "_validate_model_kwargs": _validate_model_kwargs
    }
    for name, value in attribute_assign.items():
        setattr(gu.GenerationMixin, name, value)    
