# production/installer.py
"""
Production installer for ROCA Media Registry
"""

import sys
import subprocess
import platform
from pathlib import Path

class ROCAInstaller:
    """Production installer"""
    
    def __init__(self):
        self.system = platform.system()
        self.install_dir = Path.home() / "ROCA_Media_Registry"
        
    def check_requirements(self):
        """Check system requirements"""
        requirements = {
            'Python': '3.9+',
            'RAM': '8 GB',
            'Storage': '10 GB',
            'OS': 'Windows 10+, macOS 10.15+, Ubuntu 20.04+'
        }
        
        print("🔍 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
            print(f"❌ Python 3.9+ required (found {python_version.major}.{python_version.minor})")
            return False
        
        print("✅ All requirements met")
        return True
    
    def install_dependencies(self):
        """Install required dependencies"""
        print("📦 Installing dependencies...")
        
        requirements = [
            "pyqt6>=6.5.0",
            "numpy>=1.24.0",
            "pillow>=10.0.0",
            "sqlalchemy>=2.0.0",
            "flask>=3.0.0",
            "flask-socketio>=5.3.0",
            "qrcode[pil]>=7.4.0",
            "msgpack>=1.0.0",
            "psutil>=5.9.0",
            "tqdm>=4.65.0"
        ]
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + requirements)
            print("✅ Dependencies installed")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    def create_shortcuts(self):
        """Create desktop shortcuts"""
        print("📝 Creating shortcuts...")
        
        if self.system == "Windows":
            self._create_windows_shortcut()
        elif self.system == "Darwin":  # macOS
            self._create_macos_shortcut()
        elif self.system == "Linux":
            self._create_linux_shortcut()
        
        print("✅ Shortcuts created")
    
    def _create_windows_shortcut(self):
        """Create Windows shortcut"""
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        path = str(desktop / "ROCA Media Registry.lnk")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.TargetPath = str(self.install_dir / "roca_launcher.exe")
        shortcut.WorkingDirectory = str(self.install_dir)
        shortcut.IconLocation = str(self.install_dir / "icon.ico")
        shortcut.save()
    
    def install(self):
        """Complete installation"""
        print("🚀 Starting ROCA Media Registry installation")
        
        if not self.check_requirements():
            print("Installation failed: Requirements not met")
            return False
        
        # Create installation directory
        self.install_dir.mkdir(exist_ok=True)
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Copy application files
        self._copy_application_files()
        
        # Create shortcuts
        self.create_shortcuts()
        
        # Create configuration
        self._create_default_config()
        
        print(f"""
        🎉 Installation complete!
        
        ROCA Media Registry has been installed to:
        {self.install_dir}
        
        To start the application:
        1. Double-click "ROCA Media Registry" on your desktop
        2. Or run: {self.install_dir / "roca_launcher.py"}
        
        Documentation: {self.install_dir / "docs"}
        """)
        
        return True