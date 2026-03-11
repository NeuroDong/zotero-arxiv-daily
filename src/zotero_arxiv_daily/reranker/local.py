from .base import BaseReranker, register_reranker
import logging
import warnings
import numpy as np

# Avoid TypeError in stdlib logging when transformers emits (msg, FutureWarning).
# See: transformers.modeling_attn_mask_utils AttentionMaskConverter deprecation.
warnings.filterwarnings("ignore", message=".*attention mask API.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

@register_reranker("local")
class LocalReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        # Silence transformers/hf before import to avoid logging TypeError (msg, FutureWarning).
        from transformers.utils import logging as transformers_logging
        from huggingface_hub.utils import logging as hf_logging
        transformers_logging.set_verbosity_error()
        hf_logging.set_verbosity_error()
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=FutureWarning)
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(self.config.reranker.local.model, trust_remote_code=True)
        if self.config.reranker.local.encode_kwargs:
            encode_kwargs = self.config.reranker.local.encode_kwargs
        else:
            encode_kwargs = {}
        s1_feature = encoder.encode(s1,**encode_kwargs,show_progress_bar=True)
        s2_feature = encoder.encode(s2,**encode_kwargs,show_progress_bar=True)
        sim = encoder.similarity(s1_feature, s2_feature)
        return sim.numpy()