# production/web_portal.py
"""
Web Portal for ROCA Media Sharing
Enable sharing via web browser, QR codes, and links
"""

from flask import Flask, render_template, send_file, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Any
import secrets

class ROCAWebPortal:
    """Web portal for sharing media packages"""
    
    def __init__(self, registry: MediaRegistry, host: str = "0.0.0.0", port: int = 5000):
        self.registry = registry
        self.host = host
        self.port = port
        self.active_shares = {}  # share_id -> package_info
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        CORS(self.app)
        
        self._setup_routes()
        
        # Cleanup thread for expired shares
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/shares', methods=['GET'])
        def list_shares():
            """List active shares"""
            return jsonify({
                'shares': [
                    {
                        'id': share_id,
                        'name': info['name'],
                        'expires': info['expires'],
                        'downloads': info['downloads']
                    }
                    for share_id, info in self.active_shares.items()
                ]
            })
        
        @self.app.route('/api/shares', methods=['POST'])
        def create_share():
            """Create new share"""
            data = request.json
            media_ids = data.get('media_ids', [])
            share_name = data.get('name', 'Untitled Share')
            expires_hours = data.get('expires_hours', 24)
            
            # Create package
            share_id = f"SHARE_{secrets.token_urlsafe(8)}"
            package_path = Path(f"/tmp/{share_id}.rocapkg")
            
            package = ROCAPackage.create_from_registry(
                self.registry, media_ids, package_path
            )
            
            # Store share info
            self.active_shares[share_id] = {
                'id': share_id,
                'name': share_name,
                'package_path': str(package_path),
                'created': datetime.now().isoformat(),
                'expires': datetime.now().timestamp() + (expires_hours * 3600),
                'downloads': 0,
                'max_downloads': data.get('max_downloads', 100),
                'password': data.get('password'),  # Optional password
                'media_count': len(media_ids)
            }
            
            # Generate share link and QR code
            share_url = f"http://{self.host}:{self.port}/share/{share_id}"
            qr_code = self._generate_qr_code(share_url)
            
            return jsonify({
                'share_id': share_id,
                'share_url': share_url,
                'qr_code': qr_code,
                'expires': self.active_shares[share_id]['expires']
            })
        
        @self.app.route('/share/<share_id>')
        def share_page(share_id):
            """Share page for downloading"""
            if share_id not in self.active_shares:
                return "Share not found or expired", 404
            
            share_info = self.active_shares[share_id]
            
            # Check if password protected
            if share_info.get('password'):
                if not request.args.get('password') == share_info['password']:
                    return render_template('password_prompt.html', share_id=share_id)
            
            return render_template('share.html', 
                                 share_name=share_info['name'],
                                 media_count=share_info['media_count'])
        
        @self.app.route('/download/<share_id>')
        def download_share(share_id):
            """Download share package"""
            if share_id not in self.active_shares:
                return "Share not found or expired", 404
            
            share_info = self.active_shares[share_id]
            
            # Check limits
            if share_info['downloads'] >= share_info['max_downloads']:
                return "Download limit reached", 403
            
            # Increment download count
            share_info['downloads'] += 1
            
            # Send file
            package_path = Path(share_info['package_path'])
            return send_file(
                package_path,
                as_attachment=True,
                download_name=f"{share_info['name']}.rocapkg"
            )
        
        @self.app.route('/api/scan_qr', methods=['POST'])
        def scan_qr():
            """Scan QR code and import package"""
            data = request.json
            qr_data = data.get('qr_data')
            
            # Extract share ID from QR
            share_id = self._extract_share_id(qr_data)
            
            if share_id and share_id in self.active_shares:
                # Download and import package
                share_info = self.active_shares[share_id]
                package_path = Path(share_info['package_path'])
                
                # Import to registry
                result = self.registry.import_package(
                    package_path,
                    target_dir=Path.home() / "Downloads" / "ROCA_Imports"
                )
                
                return jsonify({
                    'success': True,
                    'imported': result.get('imported_count', 0),
                    'message': 'Package imported successfully'
                })
            
            return jsonify({'success': False, 'error': 'Invalid QR code'})
        
        # WebSocket for real-time updates
        @self.socketio.on('connect')
        def handle_connect():
            emit('connected', {'message': 'Connected to ROCA Portal'})
        
        @self.socketio.on('request_share_update')
        def handle_share_update(data):
            share_id = data.get('share_id')
            if share_id in self.active_shares:
                emit('share_update', self.active_shares[share_id])
    
    def _cleanup_expired(self):
        """Cleanup expired shares"""
        import time
        
        while True:
            time.sleep(3600)  # Check every hour
            
            now = time.time()
            expired = []
            
            for share_id, info in self.active_shares.items():
                if info['expires'] < now:
                    expired.append(share_id)
                    # Delete package file
                    package_path = Path(info['package_path'])
                    if package_path.exists():
                        package_path.unlink()
            
            for share_id in expired:
                del self.active_shares[share_id]
            
            if expired:
                print(f"🧹 Cleaned up {len(expired)} expired shares")
    
    def _generate_qr_code(self, data: str) -> str:
        """Generate QR code as base64"""
        import qrcode
        import io
        import base64
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def start(self):
        """Start web portal"""
        print(f"🌐 ROCA Web Portal starting on http://{self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=False)