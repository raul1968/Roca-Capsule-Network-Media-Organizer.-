# production/exchange.py
"""
Universal Exchange Format - ROCAPKG (.rocapkg)
Standardized package format for media exchange
"""

import zipfile
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import msgpack
import base64
from datetime import datetime
import hashlib

class ROCAPackage:
    """ROCA Package format for universal media exchange"""
    
    VERSION = "1.0"
    MANIFEST_FILENAME = "manifest.roca"
    METADATA_FILENAME = "metadata.msgpack"
    THUMBNAILS_DIR = "thumbnails/"
    PREVIEWS_DIR = "previews/"
    
    def __init__(self, package_path: Path):
        self.package_path = package_path
        self.manifest = None
        self.metadata = None
    
    @classmethod
    def create(cls, media_items: List[Dict], output_path: Path,
              include_previews: bool = True,
              include_thumbnails: bool = True,
              compress: bool = True) -> 'ROCAPackage':
        """Create a new ROCA package"""
        
        # Prepare manifest
        manifest = {
            'version': cls.VERSION,
            'created_at': datetime.now().isoformat(),
            'creator': os.environ.get('USER', 'unknown'),
            'media_count': len(media_items),
            'total_size': sum(item.get('file_size', 0) for item in media_items),
            'includes': {
                'previews': include_previews,
                'thumbnails': include_thumbnails
            }
        }
        
        # Create package
        compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        
        with zipfile.ZipFile(output_path, 'w', compression) as zipf:
            # Add manifest
            manifest_json = json.dumps(manifest, indent=2)
            zipf.writestr(cls.MANIFEST_FILENAME, manifest_json)
            
            # Add metadata
            metadata = {
                'media_items': media_items,
                'registry_info': {
                    'source_registry': 'roca_media_registry',
                    'export_time': datetime.now().isoformat()
                }
            }
            
            metadata_bytes = msgpack.packb(metadata, use_bin_type=True)
            zipf.writestr(cls.METADATA_FILENAME, metadata_bytes)
            
            # Add media files
            for item in media_items:
                source_path = Path(item['original_path'])
                if source_path.exists():
                    # Store in structured path
                    arcname = f"media/{item['media_id']}/{source_path.name}"
                    zipf.write(source_path, arcname)
                    
                    # Add thumbnail if exists
                    if include_thumbnails and item.get('thumbnail_path'):
                        thumb_path = Path(item['thumbnail_path'])
                        if thumb_path.exists():
                            thumb_arcname = f"{cls.THUMBNAILS_DIR}{item['media_id']}.jpg"
                            zipf.write(thumb_path, thumb_arcname)
                    
                    # Add preview if exists
                    if include_previews and item.get('preview_path'):
                        preview_path = Path(item['preview_path'])
                        if preview_path.exists():
                            preview_arcname = f"{cls.PREVIEWS_DIR}{item['media_id']}.mp4"
                            zipf.write(preview_path, preview_arcname)
        
        package = cls(output_path)
        package.manifest = manifest
        package.metadata = metadata
        
        print(f"📦 Created ROCA package: {output_path} ({len(media_items)} items)")
        return package
    
    def extract(self, target_dir: Path, verify: bool = True) -> Dict[str, Any]:
        """Extract ROCA package"""
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Read manifest first
        with zipfile.ZipFile(self.package_path, 'r') as zipf:
            manifest_data = zipf.read(self.MANIFEST_FILENAME)
            self.manifest = json.loads(manifest_data)
            
            metadata_data = zipf.read(self.METADATA_FILENAME)
            self.metadata = msgpack.unpackb(metadata_data, raw=False)
            
            # Extract all files
            zipf.extractall(target_dir)
            
            # Verify extraction
            if verify:
                extracted_files = list(target_dir.rglob("*"))
                print(f"📦 Extracted {len(extracted_files)} files")
        
        return {
            'manifest': self.manifest,
            'extracted_to': str(target_dir),
            'media_count': len(self.metadata['media_items'])
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get package information without extracting"""
        with zipfile.ZipFile(self.package_path, 'r') as zipf:
            manifest_data = zipf.read(self.MANIFEST_FILENAME)
            manifest = json.loads(manifest_data)
            
            metadata_data = zipf.read(self.METADATA_FILENAME)
            metadata = msgpack.unpackb(metadata_data, raw=False)
            
            # Get file list
            file_list = zipf.namelist()
            
            return {
                'manifest': manifest,
                'file_count': len(file_list),
                'media_items': metadata['media_items'][:10],  # First 10
                'file_list': file_list[:20]  # First 20 files
            }
    
    def verify_integrity(self) -> bool:
        """Verify package integrity"""
        try:
            with zipfile.ZipFile(self.package_path, 'r') as zipf:
                # Check required files
                required = [self.MANIFEST_FILENAME, self.METADATA_FILENAME]
                for req in required:
                    if req not in zipf.namelist():
                        return False
                
                # Verify zip integrity
                return zipf.testzip() is None
        except:
            return False
    
    @classmethod
    def create_from_registry(cls, registry: MediaRegistry, 
                           media_ids: List[str],
                           output_path: Path) -> 'ROCAPackage':
        """Create package from registry media IDs"""
        media_items = []
        
        for media_id in media_ids:
            media_info = registry.get_by_id(media_id)
            if media_info:
                media_items.append(media_info)
        
        return cls.create(media_items, output_path)

# Quick share function for email/chat
def create_quick_share(files: List[Path], output_path: Path) -> ROCAPackage:
    """Create quick share package for email/chat"""
    media_items = []
    
    for file_path in files:
        # Quick metadata extraction
        media_type = guess_media_type(file_path)
        file_size = file_path.stat().st_size
        
        media_items.append({
            'original_path': str(file_path),
            'media_type': media_type,
            'file_size': file_size,
            'filename': file_path.name,
            'quick_share': True
        })
    
    return ROCAPackage.create(media_items, output_path, 
                            include_previews=False, 
                            include_thumbnails=False)