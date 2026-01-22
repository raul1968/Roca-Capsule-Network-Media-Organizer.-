# production/registry.py
"""
Universal Media Registry - Production Grade
Maintains central registry of all media with unique IDs and metadata
"""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Generator, Optional, List, Dict, Any
from dataclasses import asdict
import pickle
import zlib
import msgpack

class MediaRegistry:
    """Production media registry with ACID compliance"""
    
    def __init__(self, config: RegistryConfig):
        self.config = config
        self.db_path = config.registry_path / "registry.db"
        self._lock = threading.RLock()
        self._init_database()
        self._cache = {}
        
        # Initialize sub-systems
        self.thumbnail_manager = ThumbnailManager(config)
        self.metadata_extractor = MetadataExtractor()
        self.duplicate_detector = DuplicateDetector()
        self.export_manager = ExportManager(config)
        
        print(f"📁 Media Registry initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with proper indices"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            
            # Main registry table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS media_registry (
                    media_id TEXT PRIMARY KEY,
                    original_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    metadata BLOB,
                    thumbnail_path TEXT,
                    preview_path TEXT,
                    tags TEXT,
                    projects TEXT,
                    status TEXT DEFAULT 'registered',
                    registered_by TEXT,
                    permissions TEXT,
                    INDEX idx_content_hash (content_hash),
                    INDEX idx_media_type (media_type),
                    INDEX idx_status (status),
                    INDEX idx_tags (tags),
                    INDEX idx_projects (projects)
                )
            """)
            
            # File references table (for deduplication)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_references (
                    reference_id TEXT PRIMARY KEY,
                    media_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    is_primary BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (media_id) REFERENCES media_registry(media_id)
                )
            """)
            
            # User activity table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_activity (
                    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    action TEXT,
                    media_id TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
    
    def register_media(self, file_path: Path, user_id: str = "system") -> Dict[str, Any]:
        """Register media file with comprehensive processing"""
        with self._lock:
            # Generate unique ID
            media_id = self._generate_media_id(file_path)
            
            # Check if already registered
            existing = self.get_by_path(file_path)
            if existing:
                print(f"⏭️  Already registered: {file_path}")
                return existing
            
            # Extract metadata
            metadata = self.metadata_extractor.extract(file_path)
            
            # Generate content hash
            content_hash = self._compute_content_hash(file_path)
            
            # Check for duplicates by content hash
            duplicate = self.get_by_hash(content_hash)
            if duplicate:
                print(f"🔍 Duplicate found: {file_path} matches {duplicate['original_path']}")
                # Add as reference to existing media
                self._add_file_reference(media_id, duplicate['media_id'], file_path)
                return duplicate
            
            # Create thumbnail
            thumbnail_path = self.thumbnail_manager.create_thumbnail(file_path, media_id)
            
            # Create preview (if applicable)
            preview_path = self._create_preview(file_path, media_id)
            
            # Store in registry
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO media_registry 
                (media_id, original_path, content_hash, media_type, file_size, 
                 metadata, thumbnail_path, preview_path, tags, projects, registered_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                media_id,
                str(file_path.absolute()),
                content_hash,
                metadata.get('type', 'unknown'),
                metadata.get('file_size', 0),
                pickle.dumps(metadata),  # Serialize metadata
                str(thumbnail_path) if thumbnail_path else None,
                str(preview_path) if preview_path else None,
                ','.join(metadata.get('tags', [])),
                ','.join(metadata.get('projects', [])),
                user_id
            ))
            
            # Add primary file reference
            cursor.execute("""
                INSERT INTO file_references (reference_id, media_id, file_path, is_primary)
                VALUES (?, ?, ?, 1)
            """, (f"REF_{media_id}", media_id, str(file_path.absolute())))
            
            # Log activity
            cursor.execute("""
                INSERT INTO user_activity (user_id, action, media_id, details)
                VALUES (?, ?, ?, ?)
            """, (user_id, 'register', media_id, f'Registered {file_path.name}'))
            
            conn.commit()
            conn.close()
            
            # Update cache
            self._cache[media_id] = {
                'media_id': media_id,
                'original_path': str(file_path.absolute()),
                'content_hash': content_hash,
                'metadata': metadata
            }
            
            print(f"✅ Registered: {file_path.name} -> {media_id}")
            
            return self._cache[media_id]
    
    def bulk_register(self, directory: Path, user_id: str = "system") -> Dict[str, Any]:
        """Register all media in directory with progress tracking"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import tqdm
        
        media_files = []
        supported_extensions = {
            '.png', '.jpg', '.jpeg', '.tga', '.tif', '.tiff', '.exr', '.hdr', 
            '.bmp', '.webp', '.fbx', '.obj', '.gltf', '.glb', '.blend', 
            '.ma', '.mb', '.max', '.c4d', '.3ds', '.dae', '.bvh', '.trc', 
            '.c3d', '.cho', '.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv',
            '.wav', '.mp3', '.ogg', '.flac', '.m4a', '.pdf', '.txt', '.md',
            '.doc', '.docx', '.psd', '.ai', '.afdesign', '.afphoto', '.zip',
            '.rar', '.7z', '.py', '.json', '.yaml', '.yml', '.xml'
        }
        
        # Find all media files
        for ext in supported_extensions:
            media_files.extend(directory.rglob(f"*{ext}"))
        
        print(f"📁 Found {len(media_files)} media files")
        
        results = {
            'registered': [],
            'duplicates': [],
            'errors': [],
            'skipped': []
        }
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_file = {
                executor.submit(self.register_media, file, user_id): file
                for file in media_files
            }
            
            for future in tqdm.tqdm(as_completed(future_to_file), total=len(media_files)):
                file = future_to_file[future]
                try:
                    result = future.result()
                    if result.get('status') == 'duplicate':
                        results['duplicates'].append(result)
                    else:
                        results['registered'].append(result)
                except Exception as e:
                    results['errors'].append({
                        'file': str(file),
                        'error': str(e)
                    })
        
        # Generate summary
        summary = {
            'total_files': len(media_files),
            'registered': len(results['registered']),
            'duplicates': len(results['duplicates']),
            'errors': len(results['errors']),
            'registry_size': self.get_registry_size(),
            'storage_saved': self.calculate_storage_saved(results['duplicates'])
        }
        
        return {
            'summary': summary,
            'details': results
        }
    
    def get_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get media by content hash (for duplicate detection)"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM media_registry 
                WHERE content_hash = ? AND status = 'registered'
                LIMIT 1
            """, (content_hash,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return self._row_to_dict(row)
            return None
    
    def get_by_path(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Get media by file path"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT mr.* FROM media_registry mr
                JOIN file_references fr ON mr.media_id = fr.media_id
                WHERE fr.file_path = ? AND mr.status = 'registered'
                LIMIT 1
            """, (str(file_path.absolute()),))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return self._row_to_dict(row)
            return None
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Advanced search with filters"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            base_query = """
                SELECT * FROM media_registry 
                WHERE status = 'registered'
            """
            params = []
            
            # Text search
            if query:
                base_query += """
                    AND (original_path LIKE ? OR tags LIKE ? OR projects LIKE ?)
                """
                search_term = f"%{query}%"
                params.extend([search_term, search_term, search_term])
            
            # Apply filters
            if filters:
                if 'media_type' in filters:
                    base_query += " AND media_type = ?"
                    params.append(filters['media_type'])
                
                if 'min_size' in filters:
                    base_query += " AND file_size >= ?"
                    params.append(filters['min_size'])
                
                if 'max_size' in filters:
                    base_query += " AND file_size <= ?"
                    params.append(filters['max_size'])
                
                if 'tags' in filters:
                    tags = filters['tags']
                    if isinstance(tags, list):
                        tags = ','.join(tags)
                    base_query += " AND tags LIKE ?"
                    params.append(f"%{tags}%")
                
                if 'date_from' in filters:
                    base_query += " AND created_at >= ?"
                    params.append(filters['date_from'])
                
                if 'date_to' in filters:
                    base_query += " AND created_at <= ?"
                    params.append(filters['date_to'])
            
            # Order by most recently accessed
            base_query += " ORDER BY last_accessed DESC"
            
            cursor.execute(base_query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_dict(row) for row in rows]
    
    def export_package(self, media_ids: List[str], output_path: Path, 
                      include_references: bool = True) -> Dict[str, Any]:
        """Export media as ROCA package"""
        return self.export_manager.create_package(
            media_ids=media_ids,
            output_path=output_path,
            include_references=include_references
        )
    
    def import_package(self, package_path: Path, target_dir: Path, 
                      user_id: str = "system") -> Dict[str, Any]:
        """Import ROCA package"""
        return self.export_manager.import_package(
            package_path=package_path,
            target_dir=target_dir,
            user_id=user_id
        )
    
    def _generate_media_id(self, file_path: Path) -> str:
        """Generate unique media ID"""
        stat = file_path.stat()
        unique_string = f"{file_path.absolute()}:{stat.st_size}:{stat.st_mtime}"
        return f"MED_{hashlib.sha256(unique_string.encode()).hexdigest()[:16]}"
    
    def _compute_content_hash(self, file_path: Path) -> str:
        """Compute content hash with progress for large files"""
        import hashlib
        
        sha256 = hashlib.sha256()
        buffer_size = 65536  # 64KB chunks
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(buffer_size):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary"""
        columns = [
            'media_id', 'original_path', 'content_hash', 'media_type', 
            'file_size', 'created_at', 'last_accessed', 'access_count',
            'metadata', 'thumbnail_path', 'preview_path', 'tags', 
            'projects', 'status', 'registered_by', 'permissions'
        ]
        
        result = dict(zip(columns, row))
        
        # Deserialize metadata
        if result['metadata']:
            result['metadata'] = pickle.loads(result['metadata'])
        
        # Parse tags and projects
        if result['tags']:
            result['tags'] = result['tags'].split(',')
        else:
            result['tags'] = []
        
        if result['projects']:
            result['projects'] = result['projects'].split(',')
        else:
            result['projects'] = []
        
        return result
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            stats = {}
            
            # Total media count
            cursor.execute("SELECT COUNT(*) FROM media_registry WHERE status = 'registered'")
            stats['total_media'] = cursor.fetchone()[0]
            
            # By type
            cursor.execute("""
                SELECT media_type, COUNT(*) 
                FROM media_registry 
                WHERE status = 'registered'
                GROUP BY media_type
            """)
            stats['by_type'] = dict(cursor.fetchall())
            
            # Storage used
            cursor.execute("SELECT SUM(file_size) FROM media_registry WHERE status = 'registered'")
            stats['total_size'] = cursor.fetchone()[0] or 0
            
            # Duplicate savings
            cursor.execute("""
                SELECT COUNT(DISTINCT content_hash) as unique_files,
                       COUNT(*) as total_files
                FROM media_registry 
                WHERE status = 'registered'
            """)
            unique, total = cursor.fetchone()
            stats['duplicate_savings'] = total - unique
            
            # Recent activity
            cursor.execute("""
                SELECT action, COUNT(*) 
                FROM user_activity 
                WHERE timestamp > datetime('now', '-7 days')
                GROUP BY action
            """)
            stats['weekly_activity'] = dict(cursor.fetchall())
            
            conn.close()
            
            return stats