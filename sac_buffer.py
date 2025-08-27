from dataclasses import dataclass
import numpy as np
import os
from typing import Optional, Tuple

@dataclass
class TransitionBatch:
    s_img: np.ndarray; s_vec: np.ndarray; a: np.ndarray; r: np.ndarray
    s2_img: np.ndarray; s2_vec: np.ndarray; done: np.ndarray
    limits: np.ndarray; limits2: np.ndarray

class ReplayBuffer:
    """
    Replay buffer with optional SSD-backed storage (numpy.memmap) for the large image tensors.

    Parameters
    ----------
    capacity : int
        Max number of transitions.
    img_shape : Tuple[int, int]
        (H, W) of the ESDF/local patch. Channel is fixed to 1 (CxHxW with C=1).
    vec_dim : int
        Dimension of the low-dimensional state vector.
    act_dim : int
        Action dimension.
    disk_dir : Optional[str]
        If provided, `s_img` and `s2_img` will be stored as memory-mapped files
        under this directory to save RAM. Other arrays remain in RAM.
    img_dtype : str
        Storage dtype for images when using memmap or RAM:
        - "uint8": store 0..255; will be de/quantized to [0,1] float32 on sample()
        - "float16": store float16; will be upcast to float32 on sample()
        - "float32": store float32 (highest RAM/SSD usage)
        Default "float16" if disk_dir is None; if disk_dir is set and not specified, default "uint8".
    mmap_mode : str
        File mode for memmap, e.g., "w+" (create/truncate), "r+" (open existing for read/write).
        Default: "w+" so each run starts with a clean buffer.

    Notes
    -----
    - API is backward compatible with the previous version: add(), size(), sample().
    - sample(B) ALWAYS returns float32 arrays for images/vectors, ready for torch tensors.
    - Images are stored in shape (capacity, 1, H, W).
    """
    def __init__(self, capacity: int, img_shape: Tuple[int,int], vec_dim: int, act_dim: int,
                 disk_dir: Optional[str] = None, img_dtype: Optional[str] = None,
                 mmap_mode: str = "w+"):
        self.cap = int(capacity)
        self.ptr = 0
        self.full = False
        H, W = img_shape
        self._img_shape = (1, H, W)

        # Decide image storage dtype
        if img_dtype is None:
            img_dtype = "uint8" if disk_dir else "float16"
        img_dtype = img_dtype.lower()
        if img_dtype not in ("uint8", "float16", "float32"):
            raise ValueError(f"img_dtype must be one of 'uint8'|'float16'|'float32', got {img_dtype}")
        self._img_is_u8 = (img_dtype == "uint8")
        _np_img_dtype = np.uint8 if self._img_is_u8 else (np.float16 if img_dtype=="float16" else np.float32)

        # Create arrays
        if disk_dir:
            os.makedirs(disk_dir, exist_ok=True)
            # Two large image arrays go to disk
            self.s_img  = np.memmap(os.path.join(disk_dir, "s_img.dat"),
                                    dtype=_np_img_dtype, mode=mmap_mode, shape=(self.cap,)+self._img_shape)
            self.s2_img = np.memmap(os.path.join(disk_dir, "s2_img.dat"),
                                    dtype=_np_img_dtype, mode=mmap_mode, shape=(self.cap,)+self._img_shape)
            self._on_disk = True
            self._disk_dir = disk_dir
        else:
            # In-RAM storage
            self.s_img  = np.zeros((self.cap,)+self._img_shape, _np_img_dtype)
            self.s2_img = np.zeros((self.cap,)+self._img_shape, _np_img_dtype)
            self._on_disk = False
            self._disk_dir = None

        # The rest are small enough to keep in RAM (float32)
        self.s_vec   = np.zeros((self.cap, vec_dim), np.float32)
        self.a       = np.zeros((self.cap, act_dim), np.float32)
        self.r       = np.zeros((self.cap, 1), np.float32)
        self.s2_vec  = np.zeros((self.cap, vec_dim), np.float32)
        self.done    = np.zeros((self.cap, 1), np.float32)
        self.limits  = np.zeros((self.cap, act_dim), np.float32)
        self.limits2 = np.zeros((self.cap, act_dim), np.float32)

    # ----------------------------- helpers -----------------------------
    def _coerce_img(self, x: np.ndarray) -> np.ndarray:
        # Accept (H,W) or (1,H,W) or (H,W,1) and return (1,H,W)
        if x.ndim == 2:
            x = x[None, ...]
        elif x.ndim == 3 and x.shape[-1] == 1 and x.shape[0] != 1:
            # (H,W,1) -> (1,H,W)
            x = np.transpose(x, (2,0,1))
        if x.shape != self._img_shape:
            raise ValueError(f"Expected image shape {self._img_shape}, got {x.shape}")
        return x

    def _store_img(self, arr: np.ndarray, i: int, x: np.ndarray):
        """Store one image at index i with proper dtype/quantization."""
        x = self._coerce_img(x)
        if self._img_is_u8:
            # assume input already in [0,1]; quantize to 0..255
            arr[i] = np.clip(x, 0.0, 1.0).astype(np.float32)  # ensure float for mul
            arr[i] = (arr[i] * 255.0).astype(np.uint8, copy=False)
        else:
            arr[i] = x.astype(arr.dtype, copy=False)

    # ------------------------------ API --------------------------------
    def add(self, s_img, s_vec, a, r, s2_img, s2_vec, done, limits, limits2):
        i = self.ptr
        self._store_img(self.s_img,  i, s_img)
        self._store_img(self.s2_img, i, s2_img)

        self.s_vec[i]  = np.asarray(s_vec,  dtype=np.float32)
        self.a[i]      = np.asarray(a,      dtype=np.float32)
        self.r[i]      = np.asarray(r,      dtype=np.float32)
        self.s2_vec[i] = np.asarray(s2_vec, dtype=np.float32)
        self.done[i]   = np.asarray(done,   dtype=np.float32)
        self.limits[i] = np.asarray(limits, dtype=np.float32)
        self.limits2[i]= np.asarray(limits2,dtype=np.float32)

        self.ptr += 1
        if self.ptr >= self.cap:
            self.ptr = 0
            self.full = True

    def size(self) -> int:
        return self.cap if self.full else self.ptr

    def _load_imgs_for_indices(self, arr, idx) -> np.ndarray:
        x = arr[idx]
        # Dequantize / upcast to float32 for PyTorch
        if self._img_is_u8:
            x = x.astype(np.float32) / 255.0
        else:
            # float16/float32 -> float32
            if x.dtype != np.float32:
                x = x.astype(np.float32, copy=False)
        return x

    def sample(self, B: int) -> TransitionBatch:
        if self.size() <= 0:
            raise RuntimeError("ReplayBuffer is empty")
        idx = np.random.randint(0, self.size(), size=int(B))

        s_img  = self._load_imgs_for_indices(self.s_img,  idx)
        s2_img = self._load_imgs_for_indices(self.s2_img, idx)

        # Ensure the rest are float32 (already allocated as float32)
        return TransitionBatch(
            s_img = s_img,
            s_vec = self.s_vec[idx],
            a     = self.a[idx],
            r     = self.r[idx],
            s2_img= s2_img,
            s2_vec= self.s2_vec[idx],
            done  = self.done[idx],
            limits= self.limits[idx],
            limits2= self.limits2[idx],
        )

    # --------------------------- maintenance ---------------------------
    def flush(self):
        """If using memmap, flush pending writes to disk."""
        if self._on_disk:
            try:
                self.s_img.flush()
                self.s2_img.flush()
            except Exception:
                pass

    def __del__(self):
        # Best-effort flush
        try:
            self.flush()
        except Exception:
            pass
