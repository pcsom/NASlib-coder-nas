import torch
import torch.nn as nn
import numpy as np
import logging
import traceback
import gc
from typing import List

from sklearn.decomposition import PCA

# Transformers for CodeLlama
from transformers import AutoTokenizer, AutoModel

# NASLib Utilities for 101
from naslib.search_spaces.core.graph import Graph

logger = logging.getLogger(__name__)

# ==========================================
# GLOBAL DEBUG FLAG
# ==========================================
DEBUG_LOGGING = False

def debug_log(msg):
    if DEBUG_LOGGING:
        print(f"[DEBUG] {msg}")
        logger.debug(msg)

# ==========================================
# 1. THE SINGLETON MODEL HOLDER
# ==========================================
class CodeLlamaEmbedder:
    _instance = None
    cache = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CodeLlamaEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        
        self.model_name = "codellama/CodeLlama-7b-Python-hf"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 8
        self._debug_printed = False
        
        print(f"[LLM Singleton] Initializing {self.model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                device_map="auto",
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            self._initialized = True
            print("[LLM Singleton] Model loaded successfully.")
            
        except Exception as e:
            print(f"[LLM Singleton] CRITICAL ERROR loading model: {e}")
            traceback.print_exc()
            raise e

    @torch.no_grad()
    def get_embeddings(self, code_strings: List[str]) -> np.ndarray:
        all_embeddings = []
        
        for i in range(0, len(code_strings), self.batch_size):
            if not self._debug_printed:
                print("\n" + "="*40)
                print(">>> DEBUG: PREVIEW OF ARCHITECTURE (NB101) <<<")
                print("="*40)
                print(code_strings[i])
                print("="*40 + "\n")
                self._debug_printed = True

            batch_texts = code_strings[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
            ).to(self.model.device)

            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state 
            
            # Mask padding
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
            all_embeddings.append(mean_pooled.cpu().numpy())
            
        return np.vstack(all_embeddings)

# ==========================================
# 2. NB101 ARCHITECTURE STRINGIFIER (ADAPTED)
# ==========================================
class NB101Stringifier:
    def __init__(self):
        # Maps internal NB101 labels to PyTorch code
        # self.OP_MAP = {
        #     'conv3x3-bn-relu': 'nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())',
        #     'conv1x1-bn-relu': 'nn.Sequential(nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(16), nn.ReLU())',
        #     'maxpool3x3': 'nn.MaxPool2d(kernel_size=3, stride=1, padding=1)',
        # }
        self.OP_MAP = {
            'conv3x3-bn-relu': 'Conv2d_BatchNorm_ReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)',
            'conv1x1-bn-relu': 'Conv2d_BatchNorm_ReLU(out_channels, out_channels, kernel_size=1, stride=1, padding=0)',
            'maxpool3x3': 'nn.MaxPool2d(kernel_size=3, stride=1, padding=1)',
        }

    def stringify(self, arch):
        # --- 1. EXTRACT MATRIX AND OPS ---
        matrix, ops = None, None
        
        try:
            # Case A: NASLib Graph object (Hollow or Standard)
            if hasattr(arch, 'spec'):
                if isinstance(arch.spec, dict):
                    # Hollow Patch uses dict
                    matrix = arch.spec['matrix']
                    ops = arch.spec['ops']
                else:
                    # Standard NASLib uses ModelSpec object
                    matrix = arch.spec.matrix
                    ops = arch.spec.ops
            
            # Case B: Direct Dictionary (e.g. from dataset API)
            elif isinstance(arch, dict) and 'matrix' in arch:
                matrix = arch['matrix']
                ops = arch['ops']
                
            # Case C: Tuple/List (matrix, ops)
            elif isinstance(arch, (tuple, list)) and len(arch) == 2:
                matrix, ops = arch
            
            # Case D: Direct ModelSpec Object (Fallback for unpatched code)
            elif hasattr(arch, 'matrix') and hasattr(arch, 'ops'):
                matrix = arch.matrix
                ops = arch.ops
                
            else:
                raise TypeError(f"Unknown NB101 architecture object type: {type(arch)}")
                
        except Exception as e:
            print(f"[Stringifier Error] Could not convert {type(arch)} to (matrix, ops).")
            raise e

        # --- 2. GENERATE STRING ---
        return self._spec_to_pytorch_code(matrix, ops)

    def _get_op_string(self, op_label):
        return self.OP_MAP.get(op_label, None)

    def _spec_to_pytorch_code(self, matrix, ops):
        num_vertices = len(ops)
        
        lines = []
        lines.append("class Cell(nn.Module):")
        lines.append("    def __init__(self, in_channels, out_channels, stride):")
        lines.append("        super().__init__()")
        # lines.append("        self.input_projection = nn.Sequential(nn.Conv2d(C_in, 16, kernel_size=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())")
        lines.append("        self.input_projection = Conv2d_BatchNorm_ReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0)")
        
        for t in range(1, num_vertices - 1):
            op_label = ops[t]
            op_str = self._get_op_string(op_label)
            if op_str:
                lines.append(f"        self.op_{t} = {op_str}")

        lines.append("")
        lines.append("    def forward(self, x):")
        lines.append("        node_0 = self.input_projection(x)")
        
        for t in range(1, num_vertices - 1):
            incoming_indices = [src for src in range(t) if matrix[src][t] == 1]
            
            if not incoming_indices:
                lines.append(f"        node_{t} = torch.zeros_like(node_0)")
            else:
                inputs = [f"node_{src}" for src in incoming_indices]
                sum_str = " + ".join(inputs)
                
                op_label = ops[t]
                op_str = self._get_op_string(op_label)
                
                if op_str:
                    lines.append(f"        node_{t} = self.op_{t}({sum_str})")
                else:
                    lines.append(f"        node_{t} = {sum_str}")
        
        output_idx = num_vertices - 1
        incoming_indices = [src for src in range(output_idx) if matrix[src][output_idx] == 1]
        
        if not incoming_indices:
             lines.append("        return torch.zeros_like(node_0)")
        else:
            cat_parts = [f"node_{src}" for src in incoming_indices]
            cat_str = ", ".join(cat_parts)
            lines.append(f"        return torch.cat([{cat_str}], dim=1)")

        return "\n".join(lines)

# ==========================================
# 3. THE LLM PREDICTOR WRAPPER (ADAPTED FOR NB101)
# ==========================================
class LLM_NB101_Predictor:
    def __init__(self, base_predictor_cls, use_pca=True, pca_components=128, **kwargs):
        """
        Args:
            use_pca (bool): Whether to apply PCA dimensionality reduction.
            pca_components (int): Number of components.
        """
        debug_log(f"LLM_NB101_Predictor init")
        self.llm = CodeLlamaEmbedder()
        self.stringifier = NB101Stringifier() # No include_primitives arg needed for 101
        
        self.predictor = base_predictor_cls(encoding_type=None, **kwargs)
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None
        self._debug_printed = False

    def _get_hash_key(self, arch):
        """Helper to ensure consistent hashing for Read and Write"""
        try:
            # 1. Check if it's a NASLib Graph (Patched or Standard)
            if hasattr(arch, 'spec'):
                if isinstance(arch.spec, dict):
                    # Fast hash for Hollow Patch dicts
                    # Convert numpy matrix to string for hashing
                    m_str = np.array2string(arch.spec['matrix'], separator=',')
                    o_str = str(arch.spec['ops'])
                    return m_str + o_str
                elif hasattr(arch.spec, 'hash_spec'):
                    # Standard ModelSpec object
                    return arch.spec.hash_spec(arch.spec.ops)
            
            # 2. Check if it's a raw Dict
            if isinstance(arch, dict) and 'matrix' in arch:
                m_str = np.array2string(arch['matrix'], separator=',')
                o_str = str(arch['ops'])
                return m_str + o_str

            # 3. Fallback
            return str(arch)
        except Exception:
            return str(arch)

    def _get_llm_embeddings_online(self, architectures):
        embeddings = []
        indices_to_compute = []
        strings_to_compute = []
        shared_cache = self.llm.cache
        
        # --- LOOKUP LOOP ---
        for i, arch in enumerate(architectures):
            arch_hash = self._get_hash_key(arch)
            
            if arch_hash in shared_cache:
                embeddings.append(shared_cache[arch_hash])
            else:
                embeddings.append(None)
                indices_to_compute.append(i)
                code_str = self.stringifier.stringify(arch)
                strings_to_compute.append(code_str)

        # --- COMPUTE & UPDATE LOOP ---
        if strings_to_compute:
            print(f"[LLM Predictor] Encoding {len(strings_to_compute)} new architectures...")
            new_embs = self.llm.get_embeddings(strings_to_compute)
            
            for idx_in_batch, original_idx in enumerate(indices_to_compute):
                emb_vector = new_embs[idx_in_batch]
                embeddings[original_idx] = emb_vector
                
                # Update Cache using SAME HELPER
                arch = architectures[original_idx]
                arch_hash = self._get_hash_key(arch) 
                shared_cache[arch_hash] = emb_vector
                
        return np.vstack(embeddings)

    def fit(self, xtrain, ytrain, train_info=None, **kwargs):
        xtrain_emb = self._get_llm_embeddings_online(xtrain)
        if self.use_pca:
            n_samples = len(xtrain_emb)
            current_dims = min(self.pca_components, n_samples)
            if current_dims < self.pca_components and not self._debug_printed:
                print(f"[LLM] PCA: {current_dims} dims.")
            self.pca = PCA(n_components=current_dims)
            xtrain_emb = self.pca.fit_transform(xtrain_emb)
        return self.predictor.fit(xtrain_emb, ytrain, train_info, **kwargs)

    def query(self, xtest, info=None, *args, **kwargs):
        xtest_emb = self._get_llm_embeddings_online(xtest)
        if self.use_pca and self.pca is not None:
            xtest_emb = self.pca.transform(xtest_emb)
        return self.predictor.query(xtest_emb, info, *args, **kwargs)
    
    def get_model(self, **kwargs):
        return self.predictor.get_model(**kwargs)
        
    def __getattr__(self, name):
        return getattr(self.predictor, name)