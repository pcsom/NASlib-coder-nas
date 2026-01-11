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
# GLOBAL DEBUG FLAG - Set to True to enable verbose logging
# ==========================================
DEBUG_LOGGING = False

def debug_log(msg):
    """Helper function for conditional debug logging"""
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
        debug_log("CodeLlamaEmbedder.__new__() called")
        if cls._instance is None:
            debug_log("Creating new singleton instance")
            cls._instance = super(CodeLlamaEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        else:
            debug_log("Returning existing singleton instance")
        return cls._instance

    def __init__(self):
        debug_log("CodeLlamaEmbedder.__init__() called")
        if self._initialized:
            debug_log("Already initialized, skipping initialization")
            return
        
        debug_log("Starting initialization process")
        self.model_name = "codellama/CodeLlama-7b-Python-hf"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 8
        self._debug_printed = False
        
        print(f"[LLM Singleton] Initializing {self.model_name} on {self.device}...")
        debug_log(f"Model name: {self.model_name}, Device: {self.device}, Batch size: {self.batch_size}")
        
        try:
            debug_log("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            debug_log("Tokenizer loaded successfully")
            
            if self.tokenizer.pad_token is None:
                debug_log("Setting pad_token to eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            debug_log("Loading model... (this may take a while)")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                device_map="auto",
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            debug_log("Model loaded, setting to eval mode")
            self.model.eval()
            self._initialized = True
            print("[LLM Singleton] Model loaded successfully.")
            debug_log("Initialization complete")
            
        except Exception as e:
            print(f"[LLM Singleton] CRITICAL ERROR loading model: {e}")
            debug_log(f"Exception during initialization: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise e

    @torch.no_grad()
    def get_embeddings(self, code_strings: List[str]) -> np.ndarray:
        debug_log(f"get_embeddings() called with {len(code_strings)} code strings")
        all_embeddings = []
        num_batches = (len(code_strings) + self.batch_size - 1) // self.batch_size
        debug_log(f"Will process {num_batches} batches with batch_size={self.batch_size}")
        
        for i in range(0, len(code_strings), self.batch_size):
            batch_num = i // self.batch_size + 1
            debug_log(f"Processing batch {batch_num}/{num_batches}")
            
            if not self._debug_printed:
                print("\n" + "="*40)
                print(">>> DEBUG: PREVIEW OF FIRST ARCHITECTURE STRINGIFICATION <<<")
                print("="*40)
                print(code_strings[i])
                print("="*40 + "\n")
                self._debug_printed = True

            batch_texts = code_strings[i : i + self.batch_size]
            debug_log(f"Batch contains {len(batch_texts)} texts")
            
            debug_log("Tokenizing batch...")
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
            ).to(self.model.device)
            debug_log(f"Tokenization complete. Input shape: {inputs['input_ids'].shape}")

            debug_log("Running model inference...")
            outputs = self.model(**inputs)
            debug_log("Model inference complete")
            
            last_hidden = outputs.last_hidden_state
            debug_log(f"Last hidden state shape: {last_hidden.shape}")
            
            # Mask padding
            debug_log("Applying attention mask and mean pooling...")
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            debug_log(f"Mean pooled shape: {mean_pooled.shape}")
            
            debug_log("Moving embeddings to CPU...")
            all_embeddings.append(mean_pooled.cpu().numpy())
            debug_log(f"Batch {batch_num}/{num_batches} complete")

        debug_log("Stacking all embeddings...")
        result = np.vstack(all_embeddings)
        debug_log(f"get_embeddings() returning array of shape {result.shape}")
        return result

# ==========================================
# 2. NB301 ARCHITECTURE STRINGIFIER (FIXED)
# ==========================================
class NB301Stringifier:
    def stringify(self, arch):
        # --- FIX: ROBUST CONVERSION LOGIC ---
        try:
            # Case A: It's a NASLib Graph object (e.g. from Bananas optimizer)
            if isinstance(arch, Graph):
                genotype = convert_naslib_to_genotype(arch)
            
            # Case B: It's already a Genotype named tuple
            elif hasattr(arch, 'normal') and hasattr(arch, 'reduce'):
                genotype = arch
                
            # Case C: It has a getter method
            elif hasattr(arch, 'get_genotype'):
                genotype = arch.get_genotype()
                
            else:
                raise TypeError(f"Unknown architecture object type: {type(arch)}")
                
        except Exception as e:
            print(f"[Stringifier Error] Could not convert {type(arch)} to Genotype.")
            print(f"Error details: {e}")
            raise e
        # ------------------------------------

        primitives = self._get_primitives()
        normal_cell_def = self._stringify_cell_class(genotype.normal, "NormalCell", reduction=False)
        reduce_cell_def = self._stringify_cell_class(genotype.reduce, "ReductionCell", reduction=True)
        network_def = self._get_network_backbone()

        full_code = [
            "import torch",
            "import torch.nn as nn",
            "",
            primitives,
            "",
            normal_cell_def,
            "",
            reduce_cell_def,
            "",
            network_def,
            "",
            "def get_model():",
            "    return Network(C=36, layers=20, classes=10)"
        ]
        return "\n".join(full_code)

    def _stringify_cell_class(self, cell_ops, class_name, reduction):
        # cell_ops is a list of tuples: [('op_name', src_index), ...]
        edge_ops = {}
        for edge_idx, (op_name, src_idx) in enumerate(cell_ops):
            # In Genotype format, every 2 edges belong to one node (2, 3, 4, 5)
            dst_node = 2 + (edge_idx // 2)
            edge_ops[(src_idx, dst_node)] = op_name

        lines = [f"class {class_name}(nn.Module):"]
        lines.append("    def __init__(self, C_prev_prev, C_prev, C, reduction, reduction_prev):")
        lines.append("        super().__init__()")
        
        # DARTS Logic: FactorizedReduce if reduction_prev is True
        lines.append("        if reduction_prev:")
        lines.append("            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)")
        lines.append("        else:")
        lines.append("            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)")
        
        lines.append("        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)")
        
        # Operations for intermediate nodes 2, 3, 4, 5
        for dst in range(2, 6):
            for src in range(dst):
                if (src, dst) in edge_ops:
                    op = edge_ops[(src, dst)]
                    if op == 'none': continue
                    
                    # If current cell is Reduction, edges from input nodes (0, 1) have stride 2
                    stride = 2 if reduction and src < 2 else 1
                    op_code = self._get_op_code(op, stride)
                    lines.append(f"        self.op_{src}_{dst} = {op_code}")

        lines.append("")
        lines.append("    def forward(self, s0, s1):")
        lines.append("        s0 = self.preprocess0(s0)")
        lines.append("        s1 = self.preprocess1(s1)")
        
        concat_list = []
        for dst in range(2, 6):
            inputs = []
            for src in range(dst):
                if (src, dst) in edge_ops:
                    op = edge_ops[(src, dst)]
                    if op == 'none': continue
                    inputs.append(f"self.op_{src}_{dst}({'s0' if src==0 else 's1' if src==1 else f's{src}'})")
            
            if not inputs:
                lines.append(f"        s{dst} = torch.zeros_like(s0)") 
            else:
                lines.append(f"        s{dst} = {' + '.join(inputs)}")
            concat_list.append(f"s{dst}")

        lines.append(f"        return torch.cat([{', '.join(concat_list)}], dim=1)")
        return "\n".join(lines)

    def _get_op_code(self, op_name, stride):
        if op_name == 'skip_connect':
            return "nn.Identity()" if stride == 1 else "FactorizedReduce(C, C, affine=False)"
        elif op_name == 'sep_conv_3x3': return f"SepConv(C, C, 3, {stride}, 1, affine=False)"
        elif op_name == 'sep_conv_5x5': return f"SepConv(C, C, 5, {stride}, 2, affine=False)"
        elif op_name == 'dil_conv_3x3': return f"DilConv(C, C, 3, {stride}, 2, 2, affine=False)"
        elif op_name == 'dil_conv_5x5': return f"DilConv(C, C, 5, {stride}, 4, 2, affine=False)"
        elif op_name == 'max_pool_3x3': return f"nn.MaxPool2d(3, stride={stride}, padding=1)"
        elif op_name == 'avg_pool_3x3': return f"nn.AvgPool2d(3, stride={stride}, padding=1, count_include_pad=False)"
        return "nn.Identity()"

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

    def _get_network_backbone(self):
        return """
class Network(nn.Module):
    def __init__(self, C, layers, classes):
        super().__init__()
        self._layers = layers
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
                
            if reduction:
                cell = ReductionCell(C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            else:
                cell = NormalCell(C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
                
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev = C_prev
            C_prev = C_curr * 4

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, classes)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
"""

# ==========================================
# 3. THE LLM PREDICTOR WRAPPER (FIXED)
# ==========================================
class LLM_NB301_Predictor:
    def __init__(self, base_predictor_cls, use_pca=True, pca_components=128, **kwargs):
        """
        Args:
            use_pca (bool): Whether to apply PCA dimensionality reduction.
            pca_components (int or float): Number of components (int) or variance ratio (float).
        """
        debug_log(f"LLM_NB301_Predictor.__init__() called")
        debug_log(f"base_predictor_cls={base_predictor_cls}, use_pca={use_pca}, pca_components={pca_components}")
        debug_log(f"Additional kwargs: {kwargs}")
        
        debug_log("Initializing CodeLlamaEmbedder...")
        self.llm = CodeLlamaEmbedder()
        debug_log("CodeLlamaEmbedder initialized")
        
        debug_log("Initializing NB301Stringifier...")
        self.stringifier = NB301Stringifier()
        debug_log("NB301Stringifier initialized")
        
        # Pass encoding_type=None so the base predictor doesn't try standard conversions
        debug_log(f"Initializing base predictor: {base_predictor_cls.__name__}")
        self.predictor = base_predictor_cls(encoding_type=None, **kwargs)
        debug_log("Base predictor initialized")
        
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None
        
        self._debug_printed = False
        debug_log("LLM_NB301_Predictor initialization complete")

    def _get_hash_key(self, arch):
        """Helper to ensure consistent hashing for Read and Write"""
        # 1. Handle NASLib Graph Objects (Canonicalize)
        if isinstance(arch, Graph):
            try:
                g_obj = convert_naslib_to_genotype(arch)
                return str(g_obj)
            except Exception:
                # Fallback only if conversion fails
                return str(arch)
        
        # 2. Handle Objects with explicit genotype getter
        elif hasattr(arch, 'get_genotype'):
            return str(arch.get_genotype())
            
        # 3. Fallback (Strings, primitive types)
        else:
            return str(arch)
        
    def _get_llm_embeddings_online(self, architectures):
        debug_log(f"_get_llm_embeddings_online() called with {len(architectures)} architectures")
        embeddings = []
        indices_to_compute = []
        strings_to_compute = []

        # Use shared cache from Singleton
        shared_cache = self.llm.cache
        debug_log(f"Cache currently contains {len(shared_cache)} entries")
        
        debug_log("Checking cache for each architecture...")
        for i, arch in enumerate(architectures):
            debug_log(f"Processing architecture {i+1}/{len(architectures)}")
            arch_hash = self._get_hash_key(arch)
            debug_log(f"Hash generated (length={len(arch_hash)})")
            
            # 2. Check Cache
            if arch_hash in shared_cache:
                debug_log(f"Cache HIT for arch {i+1}")
                embeddings.append(shared_cache[arch_hash])
            else:
                debug_log(f"Cache MISS for arch {i+1}, will compute embedding")
                embeddings.append(None)
                indices_to_compute.append(i)
                debug_log(f"Stringifying arch {i+1}...")
                code_str = self.stringifier.stringify(arch)
                debug_log(f"Stringification complete (length={len(code_str)})")
                strings_to_compute.append(code_str)

        if strings_to_compute:
            print(f"[LLM Predictor] Encoding batch of {len(strings_to_compute)} new architectures...")
            debug_log(f"Computing embeddings for {len(strings_to_compute)} architectures...")
            new_embs = self.llm.get_embeddings(strings_to_compute)
            debug_log(f"Embeddings computed, shape: {new_embs.shape}")
            
            debug_log("Updating cache with new embeddings...")
            for idx_in_batch, original_idx in enumerate(indices_to_compute):
                emb_vector = new_embs[idx_in_batch]
                embeddings[original_idx] = emb_vector
                
                # Update Cache
                arch = architectures[original_idx]
                arch_hash = self._get_hash_key(arch)
                shared_cache[arch_hash] = emb_vector
                debug_log(f"Cached embedding for arch {original_idx+1}")
            debug_log(f"Cache now contains {len(shared_cache)} entries")

        debug_log("Stacking embeddings...")
        result = np.vstack(embeddings)
        debug_log(f"_get_llm_embeddings_online() returning array of shape {result.shape}")
        return result

    # --- FIXED PROXY METHODS ---
    # We EXPLICITLY transform data here before calling the base predictor.

    def fit(self, xtrain, ytrain, train_info=None, **kwargs):
        debug_log(f"fit() called with {len(xtrain)} training samples")
        debug_log(f"ytrain shape: {np.array(ytrain).shape if hasattr(ytrain, 'shape') else len(ytrain)}")
        debug_log(f"train_info: {train_info}")
        debug_log(f"Additional kwargs: {kwargs}")
        
        # 1. Convert Archs -> Embeddings
        debug_log("Step 1: Converting architectures to embeddings...")
        xtrain_emb = self._get_llm_embeddings_online(xtrain)
        debug_log(f"Embeddings shape: {xtrain_emb.shape}")

        # 2. Fit & Apply PCA
        if self.use_pca:
            debug_log("Step 2: Applying PCA...")
            # We fit PCA every time the predictor is updated (online learning)
            n_samples = len(xtrain_emb)
            # Cap components at min(n_samples, desired_dims)
            current_dims = min(self.pca_components, n_samples)
            debug_log(f"PCA components: {current_dims} (requested: {self.pca_components}, samples: {n_samples})")

            if current_dims < self.pca_components and not self._debug_printed:
                print(f"[LLM Predictor] WARNING: Reducing PCA components to {current_dims} due to limited samples ({n_samples}).")

            debug_log("Fitting PCA...")
            self.pca = PCA(n_components=current_dims)
            xtrain_emb = self.pca.fit_transform(xtrain_emb)
            debug_log(f"PCA applied, new shape: {xtrain_emb.shape}")
        else:
            debug_log("Step 2: Skipping PCA (use_pca=False)")
        
        # 3. Pass Embeddings (Features) to Base Predictor
        debug_log("Step 3: Fitting base predictor...")
        result = self.predictor.fit(xtrain_emb, ytrain, train_info, **kwargs)
        debug_log("fit() complete")
        return result

    def query(self, xtest, info=None, *args, **kwargs):
        debug_log(f"query() called with {len(xtest)} test samples")
        debug_log(f"info: {info}")
        debug_log(f"Additional args: {args}")
        debug_log(f"Additional kwargs: {kwargs}")
        
        # 1. Convert Archs -> Embeddings
        debug_log("Step 1: Converting architectures to embeddings...")
        xtest_emb = self._get_llm_embeddings_online(xtest)
        debug_log(f"Embeddings shape: {xtest_emb.shape}")

        # 2. Apply PCA (using the fitted transform from training)
        if self.use_pca and self.pca is not None:
            debug_log("Step 2: Applying PCA transform...")
            xtest_emb = self.pca.transform(xtest_emb)
            debug_log(f"PCA applied, new shape: {xtest_emb.shape}")
        else:
            debug_log("Step 2: Skipping PCA (not fitted or use_pca=False)")
        
        # 3. Query Base Predictor with Features
        debug_log("Step 3: Querying base predictor...")
        result = self.predictor.query(xtest_emb, info, *args, **kwargs)
        debug_log(f"query() complete, result shape/type: {np.array(result).shape if hasattr(result, 'shape') else type(result)}")
        return result
    
    def get_model(self, **kwargs):
        debug_log(f"get_model() called with kwargs: {kwargs}")
        result = self.predictor.get_model(**kwargs)
        debug_log(f"get_model() returning: {type(result)}")
        return result
        
    def __getattr__(self, name):
        debug_log(f"__getattr__() called for attribute: {name}")
        result = getattr(self.predictor, name)
        debug_log(f"__getattr__() returning: {type(result)}")
        return result