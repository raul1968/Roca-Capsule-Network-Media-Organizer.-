# production/config.py
"""
ROCA Media Registry - Production Configuration
Enterprise-grade media registration and exchange platform
"""

import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Set
import hashlib
import json
from datetime import datetime
import numpy as np

class RegistryMode(Enum):
    """Registry operation modes"""
    STANDALONE = "standalone"  # Single user, local storage
    TEAM = "team"              # Shared registry with permissions
    ENTERPRISE = "enterprise"  # Multi-tenant, cloud-backed
    CLOUD = "cloud"            # Full cloud registry

@dataclass
class RegistryConfig:
    """Production registry configuration"""
    # Core settings
    mode: RegistryMode = RegistryMode.TEAM
    registry_path: Path = Path.home() / ".roca_registry"
    auto_backup: bool = True
    backup_interval_hours: int = 24
    
    # Performance settings
    max_workers: int = 32  # Threadripper optimized
    batch_size: int = 100
    cache_size_mb: int = 1024  # 1GB cache
    
    # Security settings
    enable_encryption: bool = True
    encryption_key_path: Optional[Path] = None
    require_authentication: bool = False
    
    # Network settings
    enable_p2p: bool = True
    p2p_port: int = 8765
    discovery_broadcast: bool = True
    
    # Storage settings
    use_compression: bool = True
    compression_level: int = 6
    deduplicate_files: bool = True
    thumbnail_size: tuple = (256, 256)
    
    # Export/Import settings
    default_export_format: str = "rocapkg"  # rocapkg, zip, directory
    include_metadata: bool = True
    include_previews: bool = True
    include_thumbnails: bool = True