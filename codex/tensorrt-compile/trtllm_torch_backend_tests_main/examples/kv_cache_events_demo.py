import importlib
import time

class ShimKVCacheEventManager:
    def on_created(self, key): print(f"[KVEvent] CREATED  key={key}")
    def on_updated(self, key): print(f"[KVEvent] UPDATED  key={key}")
    def on_removed(self, key): print(f"[KVEvent] REMOVED  key={key}")
    def on_stored(self,  key): print(f"[KVEvent] STORED   key={key}")

def get_kv_event_manager():
    candidates = [
        "tensorrt_llm._torch.kv_cache.events",
        "tensorrt_llm.kv_cache.events",
        "tensorrt_llm._torch.scheduler.kv_events",
    ]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            for attr in ("KVCacheEventManager", "EventManager", "manager"):
                if hasattr(mod, attr):
                    return getattr(mod, attr)()
        except Exception:
            continue
    return ShimKVCacheEventManager()

class SimpleKVCache:
    def __init__(self, manager):
        self.mgr = manager
        self.store = {}

    def create(self, key, value):
        self.store[key] = value
        self.mgr.on_created(key)

    def update(self, key, value):
        self.store[key] = value
        self.mgr.on_updated(key)

    def remove(self, key):
        self.store.pop(key, None)
        self.mgr.on_removed(key)

    def persist(self, key):
        time.sleep(0.01)
        self.mgr.on_stored(key)

def main():
    mgr = get_kv_event_manager()
    kv = SimpleKVCache(mgr)
    kv.create("req#1_layer0", b"\x00" * 64)
    kv.update("req#1_layer0", b"\x01" * 64)
    kv.persist("req#1_layer0")
    kv.remove("req#1_layer0")
    print("KV-cache lifecycle demo complete (note: these are NOT CUDA events).")

if __name__ == "__main__":
    main()
