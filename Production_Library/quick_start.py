# production/quick_start.py
"""
Quick start for production deployment
"""

def quick_start():
    """Quick start ROCA Media Registry"""
    
    print("""
    🚀 ROCA Media Registry - Production Edition
    ==========================================
    
    Features:
    ✅ Universal media registration
    ✅ Smart deduplication
    ✅ ROCAPKG exchange format
    ✅ Web sharing portal
    ✅ Team collaboration
    ✅ Enterprise security
    
    Quick Commands:
    1. Start registry: python -m roca.registry
    2. Start web portal: python -m roca.web_portal
    3. Import media: python -m roca.cli import /path/to/media
    4. Export package: python -m roca.cli export --output package.rocapkg
    5. Create share: python -m roca.cli share --name "My Assets"
    
    Configuration file: ~/.roca_registry/config.yaml
    Database: ~/.roca_registry/registry.db
    Logs: ~/.roca_registry/logs/
    
    For enterprise deployment, see docker-compose.yml
    """)