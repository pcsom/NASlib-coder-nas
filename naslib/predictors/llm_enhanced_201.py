import numpy as np
import pandas as pd
import json
import traceback
from sklearn.decomposition import PCA

# NASLib Utilities for NASBench201
from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_str
from naslib.search_spaces.nasbench201.conversions import convert_op_indices_to_str

# ==========================================
# 1. PRE-COMPUTED EMBEDDING CACHE LOADER (SINGLETON)
# ==========================================
class EmbeddingCacheLoader:
    """
    Singleton loader for pre-computed embeddings from CSV/Pickle file for NASBench201.
    Replaces online LLM inference with O(1) cache lookups.
    Implemented as singleton to avoid loading the corpus multiple times.
    """
    _instance = None
    _cache = {}
    
    def __new__(cls, corpus_path=None, embedding_col='codellama_python_7b_pytorch_code_embedding'):
        if cls._instance is None:
            cls._instance = super(EmbeddingCacheLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, corpus_path=None, embedding_col='codellama_python_7b_pytorch_code_embedding'):
        if self._initialized:
            return
        
        if corpus_path is None:
            raise ValueError("corpus_path must be provided on first initialization")
        
        self.corpus_path = corpus_path
        self.embedding_col = embedding_col
        self._load_corpus()
        self._initialized = True
    
    def _load_corpus(self):
        """Load embeddings from CSV/Pickle and build arch_string -> embedding cache"""
        print(f"[Cache Loader] Loading embeddings from {self.corpus_path}...")
        
        try:
            # Handle Pickle vs CSV
            if self.corpus_path.endswith('.pkl'):
                df = pd.read_pickle(self.corpus_path)
            else:
                df = pd.read_csv(self.corpus_path)
            
            # Parse embeddings - handle both JSON strings and numpy arrays
            def parse_embedding(val):
                if isinstance(val, (np.ndarray, list)):
                    return np.array(val, dtype=np.float32)
                
                val_str = str(val).strip()
                
                # Try JSON format (comma-separated)
                if ',' in val_str:
                    try:
                        return np.array(json.loads(val_str), dtype=np.float32)
                    except:
                        pass
                
                # Try numpy string format (space-separated)
                cleaned = val_str.replace('[', '').replace(']', '').replace('\n', ' ').strip()
                try:
                    return np.fromstring(cleaned, sep=' ', dtype=np.float32)
                except Exception:
                    # Final fallback: literal_eval
                    from ast import literal_eval
                    return np.array(literal_eval(val_str), dtype=np.float32)
            
            # Build cache: arch_string -> embedding (use class-level cache)
            EmbeddingCacheLoader._cache = {
                k: parse_embedding(v)
                for k, v in zip(df['arch_string'], df[self.embedding_col])
            }
            
            print(f"[Cache Loader] Successfully loaded {len(EmbeddingCacheLoader._cache)} embeddings.")
            
        except Exception as e:
            print(f"[Cache Loader] ERROR loading corpus: {e}")
            traceback.print_exc()
            EmbeddingCacheLoader._cache = {}
    
    def get_embedding(self, arch_string):
        """Retrieve embedding for a single architecture string"""
        if arch_string not in EmbeddingCacheLoader._cache:
            raise ValueError(f"Architecture not found in cache: {arch_string}")
        return EmbeddingCacheLoader._cache[arch_string]
    
    def get_embeddings(self, arch_strings):
        """Retrieve embeddings for multiple architecture strings"""
        embeddings = []
        for arch_str in arch_strings:
            embeddings.append(self.get_embedding(arch_str))
        return np.vstack(embeddings)


# ==========================================
# 2. LLM-ENHANCED PREDICTOR FOR NASBENCH201
# ==========================================
class LLM_NB201_Predictor:
    """
    Wrapper for any NASLib predictor that uses pre-computed LLM embeddings
    from NASBench201 architectures instead of standard encodings.
    
    Based on the working architecture from LLM_NB301_Predictor but adapted
    for NASBench201 with cache-based embedding retrieval.
    """
    
    def __init__(self, base_predictor_cls, corpus_path, 
                 embedding_col='codellama_python_7b_pytorch_code_embedding',
                 use_pca=True, pca_components=128, **kwargs):
        """
        Args:
            base_predictor_cls: The predictor class to wrap (e.g., VarSparseGPPredictor)
            corpus_path: Path to CSV/PKL file with pre-computed embeddings
            embedding_col: Column name containing embeddings
            use_pca (bool): Whether to apply PCA dimensionality reduction
            pca_components (int): Target number of PCA components
            **kwargs: Additional arguments passed to base_predictor_cls
        """
        # print(f"[LLM NB201 Predictor] Initializing with {base_predictor_cls.__name__}...")
        
        # Initialize cache loader
        self.cache_loader = EmbeddingCacheLoader(corpus_path, embedding_col)
        
        # Initialize base predictor with encoding_type=None (we handle encoding)
        self.predictor = base_predictor_cls(encoding_type=None, **kwargs)
        
        # PCA configuration
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None
        
        self._pca_warning_printed = False
        # print(f"[LLM NB201 Predictor] Initialization complete.")
    
    def _get_embeddings_from_cache(self, architectures):
        """
        Convert NASLib architecture objects to embeddings via cache lookup.
        Uses convert_naslib_to_str for NASBench201 architecture strings.
        """
        # Convert architectures to strings
        # arch_strings = [convert_naslib_to_str(arch) for arch in architectures]
        arch_strings = []
        for arch in architectures:
            # Use the fast, graph-free conversion
            if hasattr(arch, 'op_indices') and arch.op_indices is not None:
                s = convert_op_indices_to_str(arch.op_indices)
            else:
                # Fallback (should not be reached with hollow patch)
                s = convert_naslib_to_str(arch)
            arch_strings.append(s)
        
        # Retrieve from cache
        embeddings = self.cache_loader.get_embeddings(arch_strings)
        
        return embeddings
    
    def fit(self, xtrain, ytrain, train_info=None, **kwargs):
        """
        Fit the predictor on training data.
        
        Workflow:
        1. Convert architectures to embeddings via cache lookup
        2. Apply PCA (fit and transform)
        3. Pass transformed embeddings to base predictor
        """
        # Step 1: Get embeddings from cache
        xtrain_emb = self._get_embeddings_from_cache(xtrain)
        
        # Step 2: Apply PCA (fit and transform)
        if self.use_pca:
            n_samples = len(xtrain_emb)
            current_dims = min(self.pca_components, n_samples)
            
            if current_dims < self.pca_components and not self._pca_warning_printed:
                print(f"[LLM NB201 Predictor] WARNING: Reducing PCA to {current_dims} components (requested {self.pca_components}, have {n_samples} samples)")
                self._pca_warning_printed = True
            
            self.pca = PCA(n_components=current_dims)
            xtrain_emb = self.pca.fit_transform(xtrain_emb)
        
        # Step 3: Train base predictor
        return self.predictor.fit(xtrain_emb, ytrain, train_info, **kwargs)
    
    def query(self, xtest, info=None, *args, **kwargs):
        """
        Query the predictor on test data.
        
        Workflow:
        1. Convert architectures to embeddings via cache lookup
        2. Apply PCA transform (using fitted PCA from training)
        3. Query base predictor
        """
        # Step 1: Get embeddings from cache
        xtest_emb = self._get_embeddings_from_cache(xtest)
        
        # Step 2: Apply PCA transform (using fitted transform)
        if self.use_pca and self.pca is not None:
            xtest_emb = self.pca.transform(xtest_emb)
        
        # Step 3: Query base predictor
        return self.predictor.query(xtest_emb, info, *args, **kwargs)
    
    def get_model(self, **kwargs):
        """Delegate to base predictor"""
        return self.predictor.get_model(**kwargs)
    
    def __getattr__(self, name):
        """Forward any other attribute access to the base predictor"""
        return getattr(self.predictor, name)
