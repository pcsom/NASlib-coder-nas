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

# NASLib Utilities for 301
from naslib.search_spaces.nasbench301.conversions import convert_naslib_to_genotype, Genotype
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
                print(">>> DEBUG: PREVIEW OF ARCHITECTURE (SMART IDENTITY + TOGGLE) <<<")
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
# 2. NB301 ARCHITECTURE STRINGIFIER
# ==========================================
class NB301Stringifier:
    def __init__(self, include_primitives=False):
        self.include_primitives = include_primitives

    def stringify(self, arch):
        try:
            if isinstance(arch, Graph):
                genotype = convert_naslib_to_genotype(arch)
            elif hasattr(arch, 'normal') and hasattr(arch, 'reduce'):
                genotype = arch
            elif hasattr(arch, 'get_genotype'):
                genotype = arch.get_genotype()
            else:
                raise TypeError(f"Unknown architecture object type: {type(arch)}")
        except Exception as e:
            print(f"[Stringifier Error] Could not convert {type(arch)} to Genotype.")
            raise e

        # 1. Stringify Cells
        normal_cell_def = self._stringify_cell_class(genotype.normal, "NormalCell", reduction=False)
        reduce_cell_def = self._stringify_cell_class(genotype.reduce, "ReductionCell", reduction=True)

        full_code = ["import torch", "import torch.nn as nn", ""]

        # 2. Optionally include Primitives
        if self.include_primitives:
            full_code.append(self._get_primitives())
            full_code.append("")

        # 3. Add Cell Definitions (Backbone removed)
        full_code.append(normal_cell_def)
        full_code.append("")
        full_code.append(reduce_cell_def)

        return "\n".join(full_code)

    def _stringify_cell_class(self, cell_ops, class_name, reduction):
        edge_ops = {}
        for edge_idx, (op_name, src_idx) in enumerate(cell_ops):
            dst_node = 2 + (edge_idx // 2)
            edge_ops[(src_idx, dst_node)] = op_name

        lines = [f"class {class_name}(nn.Module):"]
        lines.append("    def __init__(self, in_channels, out_channels):")
        lines.append("        super().__init__()")
        
        # Operations for intermediate nodes
        # We need to track which edges are simple identities to skip defining a layer for them
        identity_edges = [] 

        for dst in range(2, 6):
            for src in range(dst):
                if (src, dst) in edge_ops:
                    op = edge_ops[(src, dst)]
                    if op == 'none': continue
                    
                    stride = 2 if reduction and src < 2 else 1
                    
                    # SMART IDENTITY CHECK:
                    # If it's a skip connect with stride 1, it's a true Identity.
                    # We don't define a layer for it.
                    if op == 'skip_connect' and stride == 1:
                        identity_edges.append((src, dst))
                        continue

                    # Otherwise, generate code
                    if self.include_primitives:
                        op_code = self._get_short_op_code(op, stride)
                    else:
                        op_code = self._get_descriptive_op_call(op, stride)
                        
                    lines.append(f"        self.op_{src}_{dst} = {op_code}")

        lines.append("")
        lines.append("    def forward(self, s0, s1):")
        
        concat_list = []
        for dst in range(2, 6):
            inputs = []
            for src in range(dst):
                if (src, dst) in edge_ops:
                    op = edge_ops[(src, dst)]
                    if op == 'none': continue
                    
                    # SMART IDENTITY USE:
                    if (src, dst) in identity_edges:
                        # Just pass the tensor directly!
                        inputs.append('s0' if src==0 else 's1' if src==1 else f's{src}')
                    else:
                        # Call the defined layer
                        inputs.append(f"self.op_{src}_{dst}({'s0' if src==0 else 's1' if src==1 else f's{src}'})")
            
            if not inputs:
                lines.append(f"        s{dst} = torch.zeros_like(s0)") 
            else:
                lines.append(f"        s{dst} = {' + '.join(inputs)}")
            concat_list.append(f"s{dst}")

        lines.append(f"        return torch.cat([{', '.join(concat_list)}], dim=1)")
        return "\n".join(lines)

    def _get_short_op_code(self, op_name, stride):
        """Used when include_primitives=True"""
        if op_name == 'skip_connect':
            # This branch only hit if stride != 1 (FactorizedReduce)
            return "FactorizedReduce(C_in, C_out)" 
        elif op_name == 'sep_conv_3x3': return f"SepConv(C_in, C_out, 3, {stride}, 1)"
        elif op_name == 'sep_conv_5x5': return f"SepConv(C_in, C_out, 5, {stride}, 2)"
        elif op_name == 'dil_conv_3x3': return f"DilConv(C_in, C_out, 3, {stride}, 2, 2)"
        elif op_name == 'dil_conv_5x5': return f"DilConv(C_in, C_out, 5, {stride}, 4, 2)"
        elif op_name == 'max_pool_3x3': return f"nn.MaxPool2d(3, stride={stride}, padding=1)"
        elif op_name == 'avg_pool_3x3': return f"nn.AvgPool2d(3, stride={stride}, padding=1, count_include_pad=False)"
        return "Identity()"

    def _get_descriptive_op_call(self, op_name, stride):
        """Used when include_primitives=False"""
        if op_name == 'skip_connect':
            # Only hit if stride != 1
            return "FactorizedReduce(in_channels=C, out_channels=C, stride=2)"
        elif op_name == 'sep_conv_3x3': 
            return f"SeparableConv2d_BN_ReLU(in_channels=C, out_channels=C, kernel_size=3, stride={stride}, padding=1)"
        elif op_name == 'sep_conv_5x5': 
            return f"SeparableConv2d_BN_ReLU(in_channels=C, out_channels=C, kernel_size=5, stride={stride}, padding=2)"
        elif op_name == 'dil_conv_3x3': 
            return f"DilatedConv2d_BN_ReLU(in_channels=C, out_channels=C, kernel_size=3, stride={stride}, padding=2, dilation=2)"
        elif op_name == 'dil_conv_5x5': 
            return f"DilatedConv2d_BN_ReLU(in_channels=C, out_channels=C, kernel_size=5, stride={stride}, padding=4, dilation=2)"
        elif op_name == 'max_pool_3x3': 
            return f"MaxPool2d(kernel_size=3, stride={stride}, padding=1)"
        elif op_name == 'avg_pool_3x3': 
            return f"AvgPool2d(kernel_size=3, stride={stride}, padding=1)"
        return "Zero()"

    def _get_primitives(self):
        return """
class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    def forward(self, x): return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    def forward(self, x): return self.op(x)

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    def forward(self, x): return self.op(x)

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        return self.bn(out)
"""

# ==========================================
# 3. THE LLM PREDICTOR WRAPPER
# ==========================================
class LLM_NB301_Predictor:
    def __init__(self, base_predictor_cls, use_pca=True, pca_components=128, include_primitives=False, **kwargs):
        """
        Args:
            include_primitives (bool): 
                If True: Includes Helper Class defs (SepConv, etc) and uses concise op calls.
                If False: Excludes Class defs and uses verbose, descriptive op calls.
        """
        debug_log(f"LLM_NB301_Predictor init")
        self.llm = CodeLlamaEmbedder()
        
        # PASS TOGGLE TO STRINGIFIER
        self.stringifier = NB301Stringifier(include_primitives=include_primitives)
        
        self.predictor = base_predictor_cls(encoding_type=None, **kwargs)
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None
        self._debug_printed = False

    def _get_hash_key(self, arch):
        """Helper to ensure consistent hashing for Read and Write"""
        if isinstance(arch, Graph):
            try:
                g_obj = convert_naslib_to_genotype(arch)
                return str(g_obj)
            except Exception:
                return str(arch)
        elif hasattr(arch, 'get_genotype'):
            return str(arch.get_genotype())
        else:
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
        return getattr(self.predictor, name)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 