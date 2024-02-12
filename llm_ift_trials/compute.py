from joblib import Memory


COMPUTE_CACHE_DIR = "__compute_cache__"

memory = Memory(COMPUTE_CACHE_DIR, verbose=0)
