"""
ROCA_MEDIA_BRAIN_FIXED.py - Fixed version with working capsule system
"""

import os
import sys
import json
import time
import uuid
import hashlib
import shutil
import threading
import multiprocessing as mp
import fnmatch
import zlib
import struct
import binascii
import io
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from enum import Enum, auto
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from numpy.linalg import norm

# PyQt6 for production UI
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# Try to import media processing libraries with fallbacks
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available - thumbnails will be limited")

try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    print("⚠️ imagehash not available - style detection will be limited")

# Files and directories to ignore when scanning
IGNORE_PATTERNS = [
    '__pycache__', '.git', '.svn', 'node_modules',
    'backup', 'wip', 'old', 'render_output', 'cache',
    '*.blend1', '*.blend2', '*.autosave', '*.blend@'
]

# Icon color map for tiny embedded icons (RGB tuples)
ICON_COLORS = {
    'character': (200, 70, 70),
    'environment': (70, 200, 120),
    'texture': (220, 200, 70),
    'model': (70, 120, 200),
    'video': (200, 120, 200),
    'document': (150, 150, 150),
    'default': (120, 120, 120)
}

def _make_solid_png_bytes(rgb: Tuple[int, int, int], size: Tuple[int, int] = (64, 64)) -> bytes:
    """Create a minimal solid-color PNG byte sequence without PIL.

    Produces a valid PNG using zlib compression of raw scanlines.
    """
    width, height = size
    r, g, b = rgb

    # PNG file signature
    png_sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack('!I', len(data))
        crc = struct.pack('!I', binascii.crc32(chunk_type + data) & 0xffffffff)
        return length + chunk_type + data + crc

    # IHDR chunk
    ihdr_data = struct.pack('!IIBBBBB', width, height, 8, 2, 0, 0, 0)  # color type 2 = truecolor RGB
    ihdr = _chunk(b'IHDR', ihdr_data)

    # Build raw image data: each scanline starts with filter byte 0
    raw = bytearray()
    for _ in range(height):
        raw.append(0)  # no filter
        for _ in range(width):
            raw.extend(bytes((r, g, b)))

    idat = _chunk(b'IDAT', zlib.compress(bytes(raw), level=9))
    iend = _chunk(b'IEND', b'')

    return png_sig + ihdr + idat + iend


def _embed_text(text: str, dim: int = 128) -> np.ndarray:
    """Deterministic lightweight text embedding fallback.

    Produces a stable pseudo-embedding by hashing the text with incremental salts.
    Not a substitute for a neural encoder but works offline and deterministically.
    """
    vec = np.zeros(dim, dtype=float)
    if not text:
        return vec

    # Normalize text
    t = text.strip().lower()

    for i in range(dim):
        h = hashlib.sha256(f"{t}::{i}".encode('utf-8')).digest()
        # take first 8 bytes as unsigned integer to form a float
        val = int.from_bytes(h[:8], 'big') / (2**64 - 1)
        vec[i] = val

    # zero-mean, unit-norm
    vec = vec - vec.mean()
    n = norm(vec)
    if n > 0:
        vec = vec / n
    return vec


def _embed_image(path: str, dim: int = 128) -> np.ndarray:
    """Create a simple image embedding by downsampling pixels and computing a compact vector.

    This is a cheap fallback when no neural image encoder is available.
    """
    try:
        with Image.open(path) as im:
            im = im.convert('RGB')
            im = im.resize((32, 32))
            arr = np.asarray(im).astype(float) / 255.0
            flat = arr.reshape(-1)
            # reduce to `dim` by averaging blocks
            if flat.size < dim:
                # pad
                vec = np.zeros(dim, dtype=float)
                vec[:flat.size] = flat
            else:
                vec = np.array([flat[i::dim].mean() for i in range(dim)], dtype=float)

            # zero mean and normalize
            vec = vec - vec.mean()
            n = norm(vec)
            if n > 0:
                vec = vec / n
            return vec
    except Exception:
        return np.zeros(dim, dtype=float)

# ============================================================================
# FIXED CORE CAPSULE SYSTEM
# ============================================================================

class MediaType(Enum):
    """All media types a 3D animator works with"""
    IMAGE = auto()           # PNG, JPG, HDR, EXR
    MODEL_3D = auto()        # FBX, OBJ, GLTF, BLEND, MA, MB
    ANIMATION = auto()       # FBX (anim), BVH, TRC, CHO
    VIDEO = auto()           # MP4, MOV, AVI
    TEXTURE = auto()         # PBR textures: albedo, normal, roughness, etc.
    SHADER = auto()          # Unity shaders, MaterialX, OSL
    RIG = auto()             # Maya rigs, character rigs
    MOCAP = auto()           # BVH, TRC, C3D
    SCRIPT = auto()          # Python, MEL, MaxScript
    BRUSH = auto()           # ZBrush brushes, Substance brushes
    SCENE = auto()           # C4D, 3DS, Blender scenes
    AUDIO = auto()           # WAV, MP3
    DOCUMENT = auto()        # PDF, TXT, MD
    PROJECT = auto()         # Unity project, Unreal project
    UNKNOWN = auto()

@dataclass
class MediaCapsule:
    """
    FIXED Media Capsule with all required attributes
    """
    
    # Required attributes with default values
    source_path: str
    media_type: MediaType = MediaType.UNKNOWN
    activity_vector: np.ndarray = field(default_factory=lambda: np.zeros(128))
    style_hash: str = ""
    content_hash: str = ""
    complexity: float = 0.5  # FIXED: Added this missing attribute
    poly_count: int = 0
    texture_count: int = 0
    animation_length: float = 0.0
    rig_bones: int = 0
    material_types: List[str] = field(default_factory=list)
    style_tags: List[str] = field(default_factory=list)
    usage_context: List[str] = field(default_factory=list)
    emotional_tone: str = ""
    parent_projects: List[str] = field(default_factory=list)
    related_capsules: List[str] = field(default_factory=list)
    used_with: List[str] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    file_size: int = 0
    capsule_path: str = ""
    thumbnail_path: str = ""
    
    def __post_init__(self):
        """Initialize after dataclass creation"""
        if self.id is None:
            self.id = str(uuid.uuid5(uuid.NAMESPACE_DNS,
                                    f"{self.source_path}_{time.time()}"))
        if not self.filename:
            self.filename = os.path.basename(self.source_path)
        if not self.extension:
            self.extension = os.path.splitext(self.source_path)[1].lower()
        if self.media_type == MediaType.UNKNOWN:
            self.media_type = self._detect_media_type()
        if not self.content_hash and os.path.exists(self.source_path):
            self.content_hash = self._compute_content_hash()
        if not self.file_size and os.path.exists(self.source_path):
            self.file_size = os.path.getsize(self.source_path)
    
    # These are NOT dataclass fields (they're instance attributes)
    id: Optional[str] = None
    filename: str = ""
    extension: str = ""
    _thumbnail_cache: Optional[Any] = None
    _metadata_cache: Optional[Dict] = None
    _style_hash_obj: Optional[Any] = None
    _image_dimensions: Optional[Tuple[int, int]] = None
    
    def _detect_media_type(self) -> MediaType:
        """Detect media type from file extension"""
        ext = self.extension.lower()
        
        # Images/Textures
        if ext in ['.png', '.jpg', '.jpeg', '.tga', '.tif', '.tiff', '.exr', '.hdr', '.bmp', '.webp']:
            return MediaType.TEXTURE if self._is_likely_texture() else MediaType.IMAGE
        
        # 3D Models
        elif ext in ['.fbx', '.obj', '.gltf', '.glb', '.blend', '.ma', '.mb', '.max', '.c4d', '.3ds', '.dae']:
            return MediaType.MODEL_3D
        
        # Animation/Mocap
        elif ext in ['.bvh', '.trc', '.c3d', '.cho']:
            return MediaType.MOCAP
        
        # Video
        elif ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv']:
            return MediaType.VIDEO
        
        # Audio
        elif ext in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
            return MediaType.AUDIO
        
        # Documents
        elif ext in ['.pdf', '.txt', '.md', '.doc', '.docx']:
            return MediaType.DOCUMENT
        
        return MediaType.UNKNOWN
    
    def _is_likely_texture(self) -> bool:
        """Check if image is likely a texture (PBR naming conventions)"""
        filename_lower = self.filename.lower()
        texture_indicators = [
            'albedo', 'diffuse', 'normal', 'roughness', 'metallic',
            'ao', 'ambient', 'occlusion', 'height', 'displacement',
            'bump', 'specular', 'gloss', 'opacity', 'alpha', 'emission',
            '_d', '_n', '_r', '_m', '_h', '_b'  # Common texture suffixes
        ]
        return any(indicator in filename_lower for indicator in texture_indicators)
    
    def _compute_content_hash(self) -> str:
        """Compute SHA256 hash of file content"""
        try:
            with open(self.source_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return ""
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze media file"""
        if self._metadata_cache:
            return self._metadata_cache
        
        metadata = {
            'basic': self._analyze_basic(),
            'technical': self._analyze_technical(),
            'creative': self._analyze_creative(),
            'relationships': self._analyze_relationships()
        }
        
        self._metadata_cache = metadata
        # Ensure activity vector / semantic embedding is computed
        try:
            self.compute_activity_vector()
        except Exception:
            pass
        return metadata

    def compute_activity_vector(self, dim: int = 128):
        """Compute a semantic embedding for this capsule combining text and image features.

        Tries to use image data (if PIL available) and filename/description text.
        Falls back to deterministic text hashing when richer models aren't available.
        """
        # Text source: filename + style tags + usage + description if present
        text_parts = [self.filename or '', ' '.join(self.style_tags or []), ' '.join(self.usage_context or [])]
        if getattr(self, 'description', None):
            text_parts.append(self.description)
        text_src = ' '.join([p for p in text_parts if p])

        txt_vec = _embed_text(text_src, dim=dim)

        img_vec = None
        if PIL_AVAILABLE and self.media_type in [MediaType.IMAGE, MediaType.TEXTURE]:
            try:
                img_vec = _embed_image(self.source_path, dim=dim)
            except Exception:
                img_vec = None

        if img_vec is not None:
            # combine vectors (weighted)
            vec = 0.6 * txt_vec + 0.4 * img_vec
        else:
            vec = txt_vec

        # normalize
        n = norm(vec)
        if n > 0:
            vec = vec / n

        self.activity_vector = vec
    
    def _analyze_basic(self) -> Dict[str, Any]:
        """Basic file analysis"""
        return {
            'filename': self.filename,
            'path': self.source_path,
            'size_bytes': self.file_size,
            'extension': self.extension,
            'media_type': self.media_type.name,
            'content_hash': self.content_hash
        }
    
    def _analyze_technical(self) -> Dict[str, Any]:
        """Technical analysis"""
        tech_data = {}
        
        try:
            if self.media_type in [MediaType.IMAGE, MediaType.TEXTURE] and PIL_AVAILABLE:
                with Image.open(self.source_path) as img:
                    tech_data.update({
                        'dimensions': img.size,
                        'mode': img.mode,
                        'format': img.format,
                        'has_alpha': 'A' in img.mode
                    })
                
                # Cache some useful attributes
                self._image_dimensions = img.size

                # Compute perceptual style hash when available
                if IMAGEHASH_AVAILABLE:
                    try:
                        ph = imagehash.phash(img)
                        self._style_hash_obj = ph
                        self.style_hash = str(ph)
                    except Exception:
                        pass

                # Estimate complexity based on image properties
                self.complexity = self._estimate_image_complexity(img)
            
            elif self.media_type == MediaType.DOCUMENT and self.extension == '.pdf':
                tech_data['document_type'] = 'PDF'
                # Simple complexity estimate for documents
                self.complexity = min(1.0, self.file_size / 1000000)  # 1MB = max complexity
            
            else:
                # Generic complexity estimate based on file size
                self.complexity = min(1.0, self.file_size / 10000000)  # 10MB = max complexity
            
            tech_data['complexity'] = self.complexity
            
        except Exception as e:
            tech_data['error'] = str(e)
        
        return tech_data
    
    def _estimate_image_complexity(self, img: Image.Image) -> float:
        """Estimate visual complexity of an image"""
        try:
            # Convert to grayscale for analysis
            gray = img.convert('L')
            
            # Calculate edge density (simplified)
            from PIL import ImageFilter
            edges = gray.filter(ImageFilter.FIND_EDGES())
            
            # Count edge pixels
            edge_pixels = sum(1 for pixel in edges.getdata() if pixel > 50)
            total_pixels = gray.width * gray.height
            
            edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0
            
            # Consider color variation
            if img.mode in ['RGB', 'RGBA']:
                colors = img.getcolors(maxcolors=10000)
                if colors:
                    unique_colors = len(colors)
                    color_variation = min(1.0, unique_colors / 1000)
                else:
                    color_variation = 0.5
            else:
                color_variation = 0.3
            
            # Combine factors
            complexity = (edge_density * 0.6) + (color_variation * 0.4)
            
            return min(1.0, complexity)
        except:
            return 0.5
    
    def _analyze_creative(self) -> Dict[str, Any]:
        """Creative/style analysis"""
        creative_data = {
            'style_tags': [],
            'usage_suggestions': [],
            'complexity_score': self.complexity,
            'style_consistency': 1.0
        }
        
        # Auto-tag based on filename
        filename = self.filename.lower()
        
        # Style detection from filename
        styles = []
        if any(word in filename for word in ['realistic', 'photoreal', 'pbr']):
            styles.append('realistic')
        if any(word in filename for word in ['stylized', 'cartoon', 'toon']):
            styles.append('stylized')
        if any(word in filename for word in ['anime', 'manga', 'cel']):
            styles.append('anime')
        if any(word in filename for word in ['cyberpunk', 'scifi', 'futuristic']):
            styles.append('cyberpunk')
        if any(word in filename for word in ['fantasy', 'medieval', 'magic']):
            styles.append('fantasy')
        if any(word in filename for word in ['lowpoly', 'low_poly']):
            styles.append('lowpoly')
        
        creative_data['style_tags'] = styles
        self.style_tags = styles
        
        # Usage suggestions
        usages = []
        if any(word in filename for word in ['character', 'char', 'hero', 'villain']):
            usages.append('character')
        if any(word in filename for word in ['environment', 'env', 'level', 'terrain']):
            usages.append('environment')
        if any(word in filename for word in ['prop', 'weapon', 'vehicle', 'furniture']):
            usages.append('prop')
        if any(word in filename for word in ['ui', 'interface', 'hud', 'menu']):
            usages.append('ui')
        if 'texture' in filename or any(ext in filename for ext in ['.png', '.jpg', '.tga']):
            usages.append('texture')
        
        creative_data['usage_suggestions'] = usages
        self.usage_context = usages
        
        return creative_data
    
    def _analyze_relationships(self) -> Dict[str, Any]:
        """Analyze relationships with other files"""
        return {
            'likely_project': self._guess_project(),
            'common_formats': self._get_common_formats(),
            'associated_files': []
        }
    
    def _guess_project(self) -> str:
        """Guess which project this belongs to"""
        path = self.source_path.lower()
        
        if 'unity' in path:
            return 'Unity'
        elif 'unreal' in path or 'ue' in path:
            return 'Unreal Engine'
        elif 'blender' in path:
            return 'Blender'
        elif 'maya' in path:
            return 'Maya'
        elif '3ds' in path or 'max' in path:
            return '3DS Max'
        elif 'c4d' in path:
            return 'Cinema 4D'
        elif 'zbrush' in path:
            return 'ZBrush'
        elif 'substance' in path:
            return 'Substance'
        
        return 'Unknown'
    
    def _get_common_formats(self) -> List[str]:
        """Get common formats for this media type"""
        format_map = {
            MediaType.MODEL_3D: ['FBX', 'OBJ', 'GLTF', 'BLEND'],
            MediaType.TEXTURE: ['PNG', 'JPEG', 'EXR', 'HDR', 'TGA'],
            MediaType.MOCAP: ['BVH', 'FBX', 'TRC'],
            MediaType.VIDEO: ['MP4', 'MOV', 'AVI'],
            MediaType.DOCUMENT: ['PDF', 'TXT', 'MD']
        }
        return format_map.get(self.media_type, [])
    
    def generate_thumbnail(self, size: Tuple[int, int] = (256, 256)) -> Optional[Image.Image]:
        """Generate thumbnail preview"""
        if self._thumbnail_cache:
            return self._thumbnail_cache
        try:
            if self.media_type in [MediaType.IMAGE, MediaType.TEXTURE] and PIL_AVAILABLE:
                try:
                    img = Image.open(self.source_path)
                    img.thumbnail(size, Image.Resampling.LANCZOS)
                    self._thumbnail_cache = img
                    return img
                except Exception as e:
                    # Fall through to icon generation
                    print(f"Thumbnail generation failed for {self.filename}: {e}")
        except Exception:
            # Pillow import state may be flaky; fall back
            pass

        # Determine an appropriate icon color based on media type and filename
        try:
            key = 'default'
            fname = (self.filename or '').lower()
            if 'character' in fname or 'char' in fname:
                key = 'character'
            elif 'env' in fname or 'environment' in fname:
                key = 'environment'
            elif 'texture' in fname or self.media_type == MediaType.TEXTURE:
                key = 'texture'
            elif self.media_type == MediaType.MODEL_3D:
                key = 'model'
            elif self.media_type == MediaType.VIDEO:
                key = 'video'
            elif self.media_type == MediaType.DOCUMENT:
                key = 'document'
            color = ICON_COLORS.get(key, ICON_COLORS['default'])

            png_bytes = _make_solid_png_bytes(color, size=size)
        except Exception:
            png_bytes = _make_solid_png_bytes(ICON_COLORS['default'], size=size)

        # If PIL is available, return an Image object
        if PIL_AVAILABLE:
            try:
                return Image.open(io.BytesIO(png_bytes))
            except Exception:
                return None

        return None
        
        # Add media type indicator
        color_map = {
            MediaType.IMAGE: (100, 200, 100),
            MediaType.TEXTURE: (200, 100, 100),
            MediaType.MODEL_3D: (100, 100, 200),
            MediaType.DOCUMENT: (200, 200, 100)
        }
        
        color = color_map.get(self.media_type, (100, 100, 100))
        draw.rectangle([10, 10, size[0]-10, size[1]-10], outline=color, width=3)
        
        # Add text
        from PIL import ImageFont
        try:
            font = ImageFont.load_default()
            text = f"{self.media_type.name}\n{self.extension}"
            text_width = draw.textlength(text[:15], font=font)
            draw.text(((size[0]-text_width)//2, size[1]//2-20), 
                     text[:15], fill=color, font=font)
        except:
            pass
        
        self._thumbnail_cache = img
        return img
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capsule to dictionary for serialization"""
        return {
            'id': self.id,
            'source_path': self.source_path,
            'filename': self.filename,
            'media_type': self.media_type.name,
            'activity_vector': self.activity_vector.tolist(),
            'style_hash': self.style_hash,
            'content_hash': self.content_hash,
            'complexity': self.complexity,
            'style_tags': self.style_tags,
            'usage_context': self.usage_context,
            'created': self.created,
            'last_modified': self.last_modified,
            'file_size': self.file_size,
            'metadata': self.analyze()
        }
    
    def save_capsule(self, capsule_dir: str) -> str:
        """Save capsule to disk"""
        # Create directories
        capsule_path = os.path.join(capsule_dir, f"{self.id}.capsule")
        thumb_dir = os.path.join(capsule_dir, "thumbnails")
        os.makedirs(thumb_dir, exist_ok=True)
        
        thumb_path = os.path.join(thumb_dir, f"{self.id}.png")
        try:
            thumb = None
            if PIL_AVAILABLE:
                try:
                    thumb = self.generate_thumbnail()
                except Exception:
                    thumb = None

            if thumb is not None:
                try:
                    thumb.save(thumb_path)
                    self.thumbnail_path = thumb_path
                except Exception:
                    # Fallback: write raw png bytes
                    png_bytes = _make_solid_png_bytes(ICON_COLORS['default'], size=(256, 256))
                    with open(thumb_path, 'wb') as tb:
                        tb.write(png_bytes)
                    self.thumbnail_path = thumb_path
            else:
                # PIL not available or thumbnail couldn't be created: write icon bytes directly
                # Choose color as in generate_thumbnail
                try:
                    key = 'default'
                    fname = (self.filename or '').lower()
                    if 'character' in fname or 'char' in fname:
                        key = 'character'
                    elif 'env' in fname or 'environment' in fname:
                        key = 'environment'
                    elif 'texture' in fname or self.media_type == MediaType.TEXTURE:
                        key = 'texture'
                    elif self.media_type == MediaType.MODEL_3D:
                        key = 'model'
                    elif self.media_type == MediaType.VIDEO:
                        key = 'video'
                    elif self.media_type == MediaType.DOCUMENT:
                        key = 'document'
                    color = ICON_COLORS.get(key, ICON_COLORS['default'])
                except Exception:
                    color = ICON_COLORS['default']

                png_bytes = _make_solid_png_bytes(color, size=(256, 256))
                with open(thumb_path, 'wb') as tb:
                    tb.write(png_bytes)
                self.thumbnail_path = thumb_path
        except Exception as e:
            print(f"Failed to save thumbnail: {e}")
        
        # Save capsule data
        capsule_data = self.to_dict()
        with open(capsule_path, 'w') as f:
            json.dump(capsule_data, f, indent=2)
        
        self.capsule_path = capsule_path
        return capsule_path
    
    @classmethod
    def load_capsule(cls, capsule_path: str) -> 'MediaCapsule':
        """Load capsule from disk"""
        with open(capsule_path, 'r') as f:
            data = json.load(f)
        
        # Create capsule with required fields
        capsule = cls(
            source_path=data['source_path'],
            media_type=MediaType[data['media_type']],
            activity_vector=np.array(data['activity_vector']),
            style_hash=data['style_hash'],
            content_hash=data['content_hash'],
            complexity=data['complexity'],
            style_tags=data['style_tags'],
            usage_context=data['usage_context'],
            created=data['created'],
            last_modified=data['last_modified'],
            file_size=data['file_size']
        )
        
        # Set additional attributes
        capsule.id = data['id']
        capsule.filename = data['filename']
        capsule._metadata_cache = data.get('metadata', {})
        
        return capsule

# ============================================================================
# SIMPLIFIED ROCA MEDIA BRAIN
# ============================================================================

class SimpleROCA:
    """
    Simplified ROCA Media Brain without complex dependencies
    """
    
    def __init__(self, library_path: str):
        self.library_path = Path(library_path)
        self.capsule_dir = self.library_path / "capsules"
        self.thumb_dir = self.library_path / "thumbnails"
        
        # Create directories
        for dir_path in [self.library_path, self.capsule_dir, self.thumb_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Core data
        self.capsules: Dict[str, MediaCapsule] = {}
        self.capsule_index: Dict[str, List[str]] = {}
        
        # Load existing
        self._load_existing_capsules()
        
        print(f"🧠 Simple ROCA initialized at: {library_path}")
        print(f"📦 Found {len(self.capsules)} existing capsules")
    
    def _load_existing_capsules(self):
        """Load existing capsules"""
        capsule_files = list(self.capsule_dir.glob("*.capsule"))
        
        for cap_file in capsule_files:
            try:
                capsule = MediaCapsule.load_capsule(str(cap_file))
                self.capsules[capsule.id] = capsule
                
                # Update index
                media_type = capsule.media_type.name
                if media_type not in self.capsule_index:
                    self.capsule_index[media_type] = []
                self.capsule_index[media_type].append(capsule.id)
                
            except Exception as e:
                print(f"Failed to load capsule {cap_file}: {e}")

    def _is_ignored(self, path: str) -> bool:
        """Return True if path matches any ignore pattern."""
        try:
            # Check basename and full path against fnmatch patterns
            name = os.path.basename(path)
            for pat in IGNORE_PATTERNS:
                if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(path, pat):
                    return True
        except Exception:
            return False

        return False
    
    def import_media(self, source_path: Union[str, List[str]]) -> List[MediaCapsule]:
        """Import media files"""
        if isinstance(source_path, str):
            source_path = [source_path]
        
        all_files = []
        for path in source_path:
            path_obj = Path(path)
            if path_obj.is_file():
                all_files.append(str(path_obj))
            elif path_obj.is_dir():
                # Simple recursive scan
                for root, _, files in os.walk(path):
                    # Prune ignored directories in-place
                    # (os.walk will use the updated list of dirs if we modify it)
                    # Build a fresh list of directories to keep
                    dirs = []
                    for d in os.listdir(root):
                        full_d = os.path.join(root, d)
                        if os.path.isdir(full_d) and not self._is_ignored(full_d):
                            dirs.append(d)

                    for file in files:
                        file_path = os.path.join(root, file)
                        if self._is_ignored(file_path):
                            continue
                        if self._is_media_file(file_path):
                            all_files.append(file_path)
        
        print(f"📥 Found {len(all_files)} media files to import")
        
        imported = []
        for file_path in all_files:
            try:
                capsule = MediaCapsule(source_path=file_path)
                capsule.analyze()  # Generate metadata

                # Check for duplicates (exact + near)
                duplicates = self._find_duplicates(capsule)
                if duplicates['exact']:
                    existing = duplicates['exact'][0][0]
                    print(f"✓ Already exists (exact): {existing.filename}")
                    imported.append(existing)
                    continue
                elif duplicates['near']:
                    # Report near-duplicates but still import
                    msg_items = [f"{d.filename} ({reason})" for d, reason in duplicates['near']]
                    print(f"⚠️ Near-duplicates found for {capsule.filename}: {', '.join(msg_items)}")
                
                # Save capsule
                capsule.save_capsule(str(self.capsule_dir))
                
                # Add to memory
                self.capsules[capsule.id] = capsule
                
                # Update index
                media_type = capsule.media_type.name
                if media_type not in self.capsule_index:
                    self.capsule_index[media_type] = []
                self.capsule_index[media_type].append(capsule.id)
                
                imported.append(capsule)
                print(f"✓ Imported: {capsule.filename}")
                
            except Exception as e:
                print(f"✗ Failed to import {file_path}: {e}")
        
        print(f"✅ Import complete: {len(imported)} new capsules")
        return imported
    
    def _is_media_file(self, file_path: str) -> bool:
        """Check if file is a supported media type"""
        ext = os.path.splitext(file_path)[1].lower()
        
        media_extensions = [
            # Images
            '.png', '.jpg', '.jpeg', '.tga', '.tif', '.tiff', '.exr', '.hdr',
            '.bmp', '.webp',
            # 3D Models
            '.fbx', '.obj', '.gltf', '.glb', '.blend', '.ma', '.mb', '.max',
            '.c4d', '.3ds',
            # Documents
            '.pdf', '.txt', '.md',
            # Other
            '.mp4', '.mov', '.avi', '.wav', '.mp3'
        ]
        
        return ext in media_extensions
    
    def _find_by_content_hash(self, content_hash: str) -> Optional[MediaCapsule]:
        """Find capsule by content hash"""
        for capsule in self.capsules.values():
            if capsule.content_hash and capsule.content_hash == content_hash:
                return capsule
        return None

    def _find_duplicates(self, capsule: MediaCapsule, size_tolerance: float = 0.1,
                         phash_threshold: int = 6) -> Dict[str, List[Tuple[MediaCapsule, str]]]:
        """Find exact and near-duplicates for a capsule.

        Returns a dict with keys 'exact' and 'near', each a list of (capsule, reason).
        """
        exact = []
        near = []

        for other in self.capsules.values():
            # Skip comparing to self when saving/loading
            if other is capsule:
                continue

            reasons = []

            # Exact content hash
            if capsule.content_hash and other.content_hash and capsule.content_hash == other.content_hash:
                exact.append((other, 'content_hash'))
                continue

            # Filename + similar size (±size_tolerance)
            try:
                if capsule.filename and other.filename and capsule.filename.lower() == other.filename.lower():
                    if capsule.file_size and other.file_size:
                        a = capsule.file_size
                        b = other.file_size
                        if b * (1 - size_tolerance) <= a <= b * (1 + size_tolerance):
                            reasons.append('filename+size')
            except Exception:
                pass

            # Perceptual image hash similarity
            if IMAGEHASH_AVAILABLE and getattr(capsule, '_style_hash_obj', None) is not None and getattr(other, '_style_hash_obj', None) is not None:
                try:
                    dist = capsule._style_hash_obj - other._style_hash_obj
                    if dist <= phash_threshold:
                        reasons.append(f'perceptual_hash(dist={dist})')
                except Exception:
                    pass

            # Same poly count
            try:
                if capsule.poly_count and other.poly_count and capsule.poly_count == other.poly_count:
                    reasons.append('poly_count')
            except Exception:
                pass

            # Same animation length / duration
            try:
                if capsule.animation_length and other.animation_length and abs(capsule.animation_length - other.animation_length) < 1e-3:
                    reasons.append('duration')
            except Exception:
                pass

            # Same image dimensions when available
            try:
                a_dims = getattr(capsule, '_image_dimensions', None)
                b_dims = getattr(other, '_image_dimensions', None)
                if a_dims and b_dims and a_dims == b_dims:
                    reasons.append('dimensions')
            except Exception:
                pass

            if reasons:
                near.append((other, ','.join(reasons)))

        return {'exact': exact, 'near': near}

    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[MediaCapsule, float]]:
        """Perform a simple semantic search over capsules using cosine similarity.

        Uses `_embed_text` to embed the query and compares against `activity_vector` on each capsule.
        """
        qvec = _embed_text(query, dim=len(next(iter(self.capsules.values())).activity_vector)) if self.capsules else _embed_text(query)
        results: List[Tuple[MediaCapsule, float]] = []

        for capsule in self.capsules.values():
            try:
                vec = capsule.activity_vector
                if vec is None:
                    continue
                # ensure numpy
                vec = np.asarray(vec, dtype=float)
                denom = (norm(vec) * norm(qvec))
                if denom == 0:
                    score = 0.0
                else:
                    score = float(np.dot(vec, qvec) / denom)
                results.append((capsule, score))
            except Exception:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search(self, query: str) -> List[MediaCapsule]:
        """Simple search"""
        results = []
        query_lower = query.lower()
        
        for capsule in self.capsules.values():
            # Search in filename
            if query_lower in capsule.filename.lower():
                results.append(capsule)
                continue
            
            # Search in style tags
            if any(query_lower in tag.lower() for tag in capsule.style_tags):
                results.append(capsule)
                continue
            
            # Search in usage context
            if any(query_lower in usage.lower() for usage in capsule.usage_context):
                results.append(capsule)
        
        # Additional filters: support '4k' queries to prioritize high-resolution media
        try:
            if "4k" in query_lower or "4k " in query_lower:
                filtered = []
                for c in results:
                    fname = (c.filename or '').lower()
                    tech = c.analyze().get('technical', {})
                    dims = tech.get('dimensions', (0, 0))
                    width = dims[0] if isinstance(dims, (list, tuple)) and len(dims) > 0 else 0
                    if "4k" in fname or width >= 3840:
                        filtered.append(c)
                results = filtered
        except Exception:
            pass

        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        total_size = sum(c.file_size for c in self.capsules.values())
        
        return {
            'capsules': len(self.capsules),
            'total_size_gb': total_size / 1e9,
            'media_types': {k: len(v) for k, v in self.capsule_index.items()},
            'library_path': str(self.library_path)
        }

# ============================================================================
# SIMPLIFIED UI
# ============================================================================

class SimpleROCAUI(QMainWindow):
    """Simplified UI for ROCA"""
    
    def __init__(self, brain: SimpleROCA):
        super().__init__()
        self.brain = brain
        
        # Window setup
        self.setWindowTitle("🧠 ROCA Media Brain")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # Dashboard tab
        self._create_dashboard_tab()
        
        # Import tab
        self._create_import_tab()
        
        # Browse tab
        self._create_browse_tab()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(1000)
        
        # Initial update
        self.update_status()
        
        print("🎨 ROCA Interface initialized")
    
    def _create_dashboard_tab(self):
        """Create dashboard tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("📊 ROCA Dashboard")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Stats
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        layout.addWidget(self.stats_text)
        
        # Recent imports
        recent_label = QLabel("Recent Imports")
        recent_label.setStyleSheet("font-size: 18px; margin-top: 20px;")
        layout.addWidget(recent_label)
        
        self.recent_list = QListWidget()
        layout.addWidget(self.recent_list)
        
        self.tabs.addTab(tab, "📊 Dashboard")
    
    def _create_import_tab(self):
        """Create import tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("📥 Import Media")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select file or folder...")
        file_layout.addWidget(self.file_path)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_files)
        file_layout.addWidget(browse_btn)
        
        layout.addLayout(file_layout)
        
        # Import button
        import_btn = QPushButton("🚀 Import Selected")
        import_btn.clicked.connect(self.start_import)
        import_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        layout.addWidget(import_btn)
        
        # Progress
        self.progress_label = QLabel("Ready to import")
        layout.addWidget(self.progress_label)
        
        # Log
        self.import_log = QTextEdit()
        self.import_log.setReadOnly(True)
        layout.addWidget(self.import_log)
        
        self.tabs.addTab(tab, "📥 Import")
    
    def _create_browse_tab(self):
        """Create browse tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Search
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search media...")
        self.search_input.returnPressed.connect(self.perform_search)
        search_layout.addWidget(self.search_input)
        
        search_btn = QPushButton("🔍 Search")
        search_btn.clicked.connect(self.perform_search)
        search_layout.addWidget(search_btn)
        
        layout.addLayout(search_layout)
        
        # Results grid
        self.results_scroll = QScrollArea()
        self.results_widget = QWidget()
        self.results_grid = QGridLayout(self.results_widget)
        self.results_scroll.setWidget(self.results_widget)
        self.results_scroll.setWidgetResizable(True)
        
        layout.addWidget(self.results_scroll)
        
        self.tabs.addTab(tab, "📁 Browse")
    
    def browse_files(self):
        """Browse for files or folders"""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        
        if dialog.exec():
            selected = dialog.selectedFiles()
            if selected:
                self.file_path.setText(selected[0])
    
    def start_import(self):
        """Start import process"""
        path = self.file_path.text()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Warning", "Please select a valid file or folder")
            return
        
        # Clear log
        self.import_log.clear()
        self.progress_label.setText("Importing...")
        
        # Import in background
        self.import_thread = threading.Thread(
            target=self._import_background,
            args=(path,)
        )
        self.import_thread.start()
    
    def _import_background(self, path: str):
        """Background import"""
        try:
            # Import media
            capsules = self.brain.import_media(path)
            
            # Update UI
            self.import_log.append(f"✅ Imported {len(capsules)} capsules")
            
            for capsule in capsules[:10]:  # Show first 10
                self.import_log.append(f"  ✓ {capsule.filename}")
            
            if len(capsules) > 10:
                self.import_log.append(f"  ... and {len(capsules) - 10} more")
            
            self.progress_label.setText("Import complete!")
            
            # Refresh UI
            QTimer.singleShot(100, self.update_status)
            
        except Exception as e:
            self.import_log.append(f"❌ Import failed: {str(e)}")
            self.progress_label.setText("Import failed")
    
    def update_status(self):
        """Update status bar and dashboard"""
        stats = self.brain.get_stats()
        
        # Update dashboard
        stats_text = f"📦 Total Capsules: {stats['capsules']}\n"
        stats_text += f"💾 Total Size: {stats['total_size_gb']:.2f} GB\n"
        stats_text += f"📁 Library: {stats['library_path']}\n\n"
        
        stats_text += "Media Types:\n"
        for media_type, count in stats['media_types'].items():
            stats_text += f"  • {media_type}: {count}\n"
        
        self.stats_text.setText(stats_text)
        
        # Update status bar
        status_text = f"Capsules: {stats['capsules']} | Size: {stats['total_size_gb']:.2f} GB"
        self.status_bar.showMessage(status_text)
    
    def perform_search(self):
        """Perform search"""
        query = self.search_input.text()
        if not query:
            return
        
        # Clear grid
        for i in reversed(range(self.results_grid.count())):
            self.results_grid.itemAt(i).widget().setParent(None)
        
        # Search
        results = self.brain.search(query)
        
        # Display results
        row, col = 0, 0
        max_cols = 4
        
        for capsule in results[:32]:  # Limit display
            widget = self._create_capsule_widget(capsule)
            self.results_grid.addWidget(widget, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Update widget size
        self.results_widget.adjustSize()
    
    def _create_capsule_widget(self, capsule: MediaCapsule) -> QWidget:
        """Create widget for capsule"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Thumbnail placeholder
        thumb_label = QLabel()
        thumb_label.setFixedSize(150, 150)

        # Create colored border based on media type
        color_map = {
            MediaType.IMAGE.name: "#64c864",
            MediaType.TEXTURE.name: "#c86464", 
            MediaType.MODEL_3D.name: "#6464c8",
            MediaType.DOCUMENT.name: "#c8c864"
        }

        color = color_map.get(capsule.media_type.name, "#808080")
        thumb_label.setStyleSheet(f"""
            border: 3px solid {color};
            border-radius: 8px;
            background-color: #2a2a2a;
        """)

        # Show thumbnail image if available, otherwise draw a colored rectangle with text
        try:
            if getattr(capsule, 'thumbnail_path', None) and os.path.exists(capsule.thumbnail_path):
                pix = QPixmap(capsule.thumbnail_path).scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                thumb_label.setPixmap(pix)
            else:
                pix = QPixmap(150, 150)
                pix.fill(QColor('#2a2a2a'))

                painter = QPainter(pix)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)

                # Draw inner rounded rect with media color
                inner = pix.rect().adjusted(8, 8, -8, -8)
                painter.setBrush(QColor(color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(inner, 8, 8)

                # Draw centered text (media type)
                painter.setPen(QColor('white'))
                font = QFont()
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, capsule.media_type.name)
                painter.end()

                thumb_label.setPixmap(pix)
        except Exception:
            # Final fallback: plain text
            thumb_label.setText(f"{capsule.media_type.name}\n{capsule.extension}")
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(thumb_label)
        
        # Filename
        name_label = QLabel(capsule.filename[:20] + ("..." if len(capsule.filename) > 20 else ""))
        name_label.setToolTip(capsule.filename)
        name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(name_label)
        
        # Tags
        if capsule.style_tags:
            tags_label = QLabel(", ".join(capsule.style_tags[:2]))
            tags_label.setStyleSheet("color: #4a9cff; font-size: 11px;")
            layout.addWidget(tags_label)
        
        widget.setToolTip(f"Click for details\nPath: {capsule.source_path}")
        
        return widget

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main function"""
    print("""
    🧠 ROCA MEDIA BRAIN - Simplified Version
    ======================================
    Professional Media Organization System
    """)
    
    # Create library in user directory
    home = Path.home()
    library_path = home / "ROCA_Media_Library"
    library_path.mkdir(exist_ok=True)
    
    # Initialize brain
    brain = SimpleROCA(str(library_path))
    
    # Start UI
    app = QApplication(sys.argv)
    window = SimpleROCAUI(brain)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()