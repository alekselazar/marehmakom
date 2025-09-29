# tools/postinstall.py
import os
from setuptools.command.install import install

class CustomInstall(install):
    """Run a best-effort asset download after wheel install."""
    def run(self):
        super().run()
        try:
            from marehmakom.download import ensure_assets
            print("[marehmakom] Post-install: ensuring model assets...")
            ensure_assets()
        except Exception as e:
            print(f"[marehmakom] Post-install skipped/failed: {e}")
