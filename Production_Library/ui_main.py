# production/ui_main.py
"""
Production-ready PyQt6 UI for ROCA Media Registry
"""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QTabWidget, QSplitter,
                            QTreeWidget, QTreeWidgetItem, QListWidget,
                            QListWidgetItem, QMenuBar, QStatusBar, QToolBar,
                            QFileDialog, QMessageBox, QProgressDialog,
                            QDockWidget, QTextEdit, QGroupBox, QGridLayout,
                            QLineEdit, QComboBox, QCheckBox, QSlider)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon, QAction, QPixmap, QFont
import sys
from pathlib import Path

class RegistryMainWindow(QMainWindow):
    """Main window for ROCA Media Registry"""
    
    def __init__(self, config: RegistryConfig):
        super().__init__()
        self.config = config
        self.registry = MediaRegistry(config)
        self.current_directory = Path.home()
        
        self.setWindowTitle("ROCA Media Registry - Professional Edition")
        self.setGeometry(100, 100, 1600, 900)
        
        # Set application icon
        self.setWindowIcon(self._create_icon())
        
        # Initialize UI
        self._init_ui()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_status)
        self.update_timer.start(5000)
        
        # Load recent session
        self._load_session()
    
    def _init_ui(self):
        """Initialize user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar
        self._create_toolbar()
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Directory tree and registry info
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Center panel: Media browser
        center_panel = self._create_center_panel()
        main_splitter.addWidget(center_panel)
        
        # Right panel: Preview and metadata
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter sizes
        main_splitter.setSizes([300, 800, 300])
        
        main_layout.addWidget(main_splitter)
        
        # Create menu bar
        self._create_menu_bar()
    
    def _create_toolbar(self):
        """Create main toolbar"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Register actions
        register_action = QAction("📁 Register Media", self)
        register_action.triggered.connect(self._register_media)
        toolbar.addAction(register_action)
        
        register_folder_action = QAction("📂 Register Folder", self)
        register_folder_action.triggered.connect(self._register_folder)
        toolbar.addAction(register_folder_action)
        
        toolbar.addSeparator()
        
        # Search bar
        search_label = QLabel("Search:")
        toolbar.addWidget(search_label)
        
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search media...")
        self.search_bar.setMaximumWidth(200)
        self.search_bar.returnPressed.connect(self._search_media)
        toolbar.addWidget(self.search_bar)
        
        toolbar.addSeparator()
        
        # Export/Import
        export_action = QAction("📤 Export", self)
        export_action.triggered.connect(self._export_selected)
        toolbar.addAction(export_action)
        
        import_action = QAction("📥 Import", self)
        import_action.triggered.connect(self._import_package)
        toolbar.addAction(import_action)
        
        toolbar.addSeparator()
        
        # Share button
        share_action = QAction("🔗 Share", self)
        share_action.triggered.connect(self._share_selected)
        toolbar.addAction(share_action)
        
        # Stats button
        stats_action = QAction("📊 Stats", self)
        stats_action.triggered.connect(self._show_stats)
        toolbar.addAction(stats_action)
    
    def _create_left_panel(self):
        """Create left panel with directory tree and registry info"""
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Directory tree
        dir_group = QGroupBox("Directory")
        dir_layout = QVBoxLayout(dir_group)
        
        self.dir_tree = QTreeWidget()
        self.dir_tree.setHeaderLabel("Folders")
        self.dir_tree.itemDoubleClicked.connect(self._on_dir_selected)
        dir_layout.addWidget(self.dir_tree)
        
        layout.addWidget(dir_group)
        
        # Registry stats
        stats_group = QGroupBox("Registry Stats")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("""
            QLabel {
                font-family: monospace;
                font-size: 9pt;
                color: #ccc;
            }
        """)
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        
        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        # Scan for duplicates button
        dup_btn = QPushButton("🔍 Find Duplicates")
        dup_btn.clicked.connect(self._find_duplicates)
        actions_layout.addWidget(dup_btn)
        
        # Clean registry button
        clean_btn = QPushButton("🧹 Clean Registry")
        clean_btn.clicked.connect(self._clean_registry)
        actions_layout.addWidget(clean_btn)
        
        # Backup button
        backup_btn = QPushButton("💾 Backup")
        backup_btn.clicked.connect(self._backup_registry)
        actions_layout.addWidget(backup_btn)
        
        actions_layout.addStretch()
        layout.addWidget(actions_group)
        
        return left_widget
    
    def _create_center_panel(self):
        """Create center panel with media browser"""
        center_widget = QWidget()
        layout = QVBoxLayout(center_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Media browser
        browser_group = QGroupBox("Media Browser")
        browser_layout = QVBoxLayout(browser_group)
        
        # Filter controls
        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        
        # Media type filter
        type_filter = QComboBox()
        type_filter.addItems(["All Types", "Images", "3D Models", "Videos", "Audio", "Documents"])
        type_filter.currentTextChanged.connect(self._filter_by_type)
        filter_layout.addWidget(QLabel("Type:"))
        filter_layout.addWidget(type_filter)
        
        # Sort options
        sort_filter = QComboBox()
        sort_filter.addItems(["Recently Added", "Name", "Size", "Type"])
        sort_filter.currentTextChanged.connect(self._sort_media)
        filter_layout.addWidget(QLabel("Sort:"))
        filter_layout.addWidget(sort_filter)
        
        filter_layout.addStretch()
        browser_layout.addWidget(filter_widget)
        
        # Thumbnail grid
        self.media_grid = QListWidget()
        self.media_grid.setViewMode(QListWidget.ViewMode.IconMode)
        self.media_grid.setIconSize(QPixmap(150, 150).size())
        self.media_grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.media_grid.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.media_grid.itemDoubleClicked.connect(self._on_media_selected)
        browser_layout.addWidget(self.media_grid)
        
        layout.addWidget(browser_group)
        
        return center_widget
    
    def _create_right_panel(self):
        """Create right panel with preview and metadata"""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Preview panel
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        preview_layout.addWidget(self.preview_label)
        
        layout.addWidget(preview_group)
        
        # Metadata panel
        meta_group = QGroupBox("Metadata")
        meta_layout = QVBoxLayout(meta_group)
        
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setMaximumHeight(300)
        meta_layout.addWidget(self.metadata_text)
        
        layout.addWidget(meta_group)
        
        # Quick export panel
        export_group = QGroupBox("Quick Export")
        export_layout = QVBoxLayout(export_group)
        
        # Export format
        format_combo = QComboBox()
        format_combo.addItems(["ROCAPKG (.rocapkg)", "ZIP Archive", "Folder Copy"])
        export_layout.addWidget(QLabel("Format:"))
        export_layout.addWidget(format_combo)
        
        # Include options
        include_thumb = QCheckBox("Include Thumbnails")
        include_thumb.setChecked(True)
        export_layout.addWidget(include_thumb)
        
        include_meta = QCheckBox("Include Metadata")
        include_meta.setChecked(True)
        export_layout.addWidget(include_meta)
        
        # Export button
        export_btn = QPushButton("📤 Export Selected")
        export_btn.clicked.connect(self._quick_export)
        export_layout.addWidget(export_btn)
        
        layout.addWidget(export_group)
        
        return right_widget
    
    def _create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_registry_action = QAction("New Registry", self)
        file_menu.addAction(new_registry_action)
        
        open_registry_action = QAction("Open Registry...", self)
        open_registry_action.triggered.connect(self._open_registry)
        file_menu.addAction(open_registry_action)
        
        file_menu.addSeparator()
        
        import_menu = file_menu.addMenu("Import")
        import_package_action = QAction("ROCA Package", self)
        import_package_action.triggered.connect(self._import_package)
        import_menu.addAction(import_package_action)
        
        export_menu = file_menu.addMenu("Export")
        export_selected_action = QAction("Selected Media", self)
        export_selected_action.triggered.connect(self._export_selected)
        export_menu.addAction(export_selected_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        batch_process_action = QAction("Batch Processing", self)
        tools_menu.addAction(batch_process_action)
        
        deduplicate_action = QAction("Deduplicate Files", self)
        deduplicate_action.triggered.connect(self._deduplicate_files)
        tools_menu.addAction(deduplicate_action)
        
        # Share menu
        share_menu = menubar.addMenu("Share")
        
        create_share_action = QAction("Create Share", self)
        create_share_action.triggered.connect(self._create_share)
        share_menu.addAction(create_share_action)
        
        web_portal_action = QAction("Web Portal", self)
        web_portal_action.triggered.connect(self._start_web_portal)
        share_menu.addAction(web_portal_action)
    
    def _register_media(self):
        """Register selected media files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Media Files",
            str(self.current_directory),
            "Media Files (*.png *.jpg *.jpeg *.tga *.tif *.tiff *.exr *.hdr *.bmp *.webp *.fbx *.obj *.gltf *.glb *.blend *.ma *.mb *.max *.c4d *.3ds *.dae *.bvh *.trc *.c3d *.cho *.mp4 *.mov *.avi *.mkv *.webm *.wmv *.wav *.mp3 *.ogg *.flac *.m4a *.pdf *.txt *.md *.doc *.docx *.psd *.ai *.afdesign *.afphoto);;All Files (*)"
        )
        
        if files:
            progress = QProgressDialog("Registering media...", "Cancel", 0, len(files), self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            
            for i, file_path in enumerate(files):
                if progress.wasCanceled():
                    break
                
                self.registry.register_media(Path(file_path))
                progress.setValue(i + 1)
            
            progress.close()
            self._refresh_media_browser()
    
    def _register_folder(self):
        """Register all media in selected folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            str(self.current_directory)
        )
        
        if folder:
            # Run in background thread
            self.worker = RegistryWorker(self.registry, Path(folder))
            self.worker.progress.connect(self._on_registry_progress)
            self.worker.finished.connect(self._on_registry_finished)
            self.worker.start()
    
    def _refresh_media_browser(self):
        """Refresh media browser with current filters"""
        # Clear current items
        self.media_grid.clear()
        
        # Get filtered media
        media_items = self.registry.search(self.search_bar.text())
        
        # Add to grid
        for item in media_items:
            list_item = QListWidgetItem()
            list_item.setText(item['filename'])
            list_item.setData(Qt.ItemDataRole.UserRole, item)
            
            # Set thumbnail if available
            if item.get('thumbnail_path'):
                pixmap = QPixmap(item['thumbnail_path'])
                if not pixmap.isNull():
                    list_item.setIcon(QIcon(pixmap.scaled(150, 150, 
                                                         Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation)))
            
            self.media_grid.addItem(list_item)
        
        self.status_bar.showMessage(f"Showing {self.media_grid.count()} media items")
    
    def _on_media_selected(self, item):
        """Handle media selection"""
        media_data = item.data(Qt.ItemDataRole.UserRole)
        
        # Update preview
        if media_data.get('preview_path'):
            pixmap = QPixmap(media_data['preview_path'])
            if not pixmap.isNull():
                self.preview_label.setPixmap(
                    pixmap.scaled(400, 400, 
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
                )
        
        # Update metadata
        metadata_text = f"""
        <b>File:</b> {media_data['filename']}<br>
        <b>Type:</b> {media_data['media_type']}<br>
        <b>Size:</b> {self._format_size(media_data['file_size'])}<br>
        <b>Registered:</b> {media_data['created_at']}<br>
        <b>Path:</b> {media_data['original_path']}<br>
        <b>Content Hash:</b> {media_data['content_hash'][:16]}...<br>
        """
        
        if media_data.get('tags'):
            metadata_text += f"<b>Tags:</b> {', '.join(media_data['tags'])}<br>"
        
        if media_data.get('projects'):
            metadata_text += f"<b>Projects:</b> {', '.join(media_data['projects'])}<br>"
        
        self.metadata_text.setHtml(metadata_text)
    
    def _export_selected(self):
        """Export selected media"""
        selected_items = self.media_grid.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select media to export")
            return
        
        # Get output path
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Package",
            str(self.current_directory / "export.rocapkg"),
            "ROCA Packages (*.rocapkg);;ZIP Archives (*.zip)"
        )
        
        if output_path:
            media_ids = []
            for item in selected_items:
                media_data = item.data(Qt.ItemDataRole.UserRole)
                media_ids.append(media_data['media_id'])
            
            # Create package
            result = self.registry.export_package(media_ids, Path(output_path))
            
            if result.get('success'):
                QMessageBox.information(self, "Export Complete", 
                                      f"Exported {len(media_ids)} items to {output_path}")
    
    def _import_package(self):
        """Import ROCA package"""
        package_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Package",
            str(self.current_directory),
            "ROCA Packages (*.rocapkg);;ZIP Archives (*.zip)"
        )
        
        if package_path:
            target_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Import Directory",
                str(self.current_directory)
            )
            
            if target_dir:
                result = self.registry.import_package(Path(package_path), Path(target_dir))
                
                if result.get('success'):
                    QMessageBox.information(self, "Import Complete",
                                          f"Imported {result.get('imported_count', 0)} items")
                    self._refresh_media_browser()
    
    def _share_selected(self):
        """Share selected media"""
        selected_items = self.media_grid.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select media to share")
            return
        
        # Create share dialog
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QSpinBox, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Share")
        layout = QVBoxLayout(dialog)
        
        form = QFormLayout()
        
        share_name = QLineEdit("Media Share")
        form.addRow("Share Name:", share_name)
        
        expires = QSpinBox()
        expires.setRange(1, 168)  # 1 hour to 1 week
        expires.setValue(24)
        expires.setSuffix(" hours")
        form.addRow("Expires After:", expires)
        
        max_downloads = QSpinBox()
        max_downloads.setRange(1, 1000)
        max_downloads.setValue(100)
        form.addRow("Max Downloads:", max_downloads)
        
        password = QLineEdit()
        password.setPlaceholderText("Optional password")
        form.addRow("Password:", password)
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                  QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get media IDs
            media_ids = []
            for item in selected_items:
                media_data = item.data(Qt.ItemDataRole.UserRole)
                media_ids.append(media_data['media_id'])
            
            # Create share
            share_info = {
                'media_ids': media_ids,
                'name': share_name.text(),
                'expires_hours': expires.value(),
                'max_downloads': max_downloads.value(),
                'password': password.text() if password.text() else None
            }
            
            # Start web portal if not running
            if not hasattr(self, 'web_portal'):
                self.web_portal = ROCAWebPortal(self.registry, port=8765)
                threading.Thread(target=self.web_portal.start, daemon=True).start()
            
            # Create share
            import requests
            response = requests.post(f"http://localhost:8765/api/shares", json=share_info)
            
            if response.status_code == 200:
                share_data = response.json()
                self._show_share_info(share_data)
    
    def _show_share_info(self, share_data):
        """Show share information with QR code"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextBrowser
        from PyQt6.QtGui import QPixmap
        import base64
        from io import BytesIO
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Share Created")
        layout = QVBoxLayout(dialog)
        
        # QR code
        qr_data = share_data.get('qr_code', '').split(',')[1]
        qr_bytes = base64.b64decode(qr_data)
        pixmap = QPixmap()
        pixmap.loadFromData(qr_bytes)
        
        qr_label = QLabel()
        qr_label.setPixmap(pixmap.scaled(200, 200))
        layout.addWidget(qr_label)
        
        # Share URL
        url_label = QLabel(f"<b>Share URL:</b> {share_data['share_url']}")
        url_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(url_label)
        
        # Share ID
        id_label = QLabel(f"<b>Share ID:</b> {share_data['share_id']}")
        layout.addWidget(id_label)
        
        # Expiry
        expiry_label = QLabel(f"<b>Expires:</b> {share_data['expires']}")
        layout.addWidget(expiry_label)
        
        dialog.exec()
    
    def _update_status(self):
        """Update status bar with registry info"""
        stats = self.registry.get_registry_stats()
        
        status_text = f"""
        Registry: {stats['total_media']} media | 
        Storage: {self._format_size(stats['total_size'])} | 
        Unique: {stats.get('unique_files', 0)} | 
        Saved: {self._format_size(stats.get('duplicate_savings', 0) * 1024 * 1024)}
        """
        
        self.status_bar.showMessage(status_text)
        self._update_stats_display(stats)
    
    def _update_stats_display(self, stats):
        """Update stats display in left panel"""
        stats_text = f"""
        <b>Registry Statistics</b>
        <hr>
        <b>Total Media:</b> {stats['total_media']}
        <b>Total Size:</b> {self._format_size(stats['total_size'])}
        <b>Unique Files:</b> {stats.get('unique_files', 0)}
        <b>Storage Saved:</b> {self._format_size(stats.get('duplicate_savings', 0) * 1024 * 1024)}
        
        <b>By Type:</b>
        """
        
        for media_type, count in stats.get('by_type', {}).items():
            stats_text += f"  {media_type}: {count}\n"
        
        self.stats_label.setText(stats_text)
    
    def _format_size(self, size_bytes):
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

class RegistryWorker(QThread):
    """Worker thread for registry operations"""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(dict)      # results
    
    def __init__(self, registry, directory):
        super().__init__()
        self.registry = registry
        self.directory = directory
    
    def run(self):
        results = self.registry.bulk_register(self.directory)
        self.finished.emit(results)