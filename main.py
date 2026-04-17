"""
AI Trading Bot - NiceGUI Desktop Application
Entry point for the application
"""

import asyncio
import os
import signal
import sys
from nicegui import ui, app

# Global reference to bot_service for cleanup
bot_service_ref = None
_cleanup_done = False


def signal_handler(signum, frame):
    """Handle shutdown signals by letting NiceGUI's own shutdown proceed."""
    print("\n[INFO] Shutting down gracefully...")
    # Restore default handler so a second Ctrl+C force-kills immediately.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    # Ask the running event loop to shut the app down instead of
    # calling sys.exit(), which collides with uvloop teardown.
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(app.shutdown)
    except RuntimeError:
        # No running loop — nothing left to shut down.
        pass

if __name__ in {"__main__", "__mp_main__"}:
    kernel_version = ""
    try:
        kernel_version = open("/proc/version", encoding="utf-8").read().lower()
    except OSError:
        pass
    is_wsl = "microsoft" in kernel_version

    # Prefer the GTK/Wayland stack on Linux/WSL when available.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("PYWEBVIEW_GUI", "gtk")
        if os.getenv("WAYLAND_DISPLAY"):
            os.environ.setdefault("GDK_BACKEND", "wayland")

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Import app module to register the @ui.page('/') handler
    from src.gui.app import bot_service

    # Save reference to bot_service for cleanup
    bot_service_ref = bot_service

    # Register shutdown handler with NiceGUI app
    async def on_app_shutdown():
        """Called when NiceGUI app is shutting down"""
        global _cleanup_done
        if _cleanup_done:
            return
        _cleanup_done = True
        print("[INFO] NiceGUI app shutdown event triggered")
        if bot_service_ref and bot_service_ref.is_running():
            print("[INFO] Stopping bot engine...")
            try:
                await bot_service_ref.stop()
                print("[INFO] Bot stopped successfully")
            except Exception as e:
                print(f"[WARN] Error stopping bot: {e}")

    app.on_shutdown(on_app_shutdown)

    # WSL can expose DISPLAY/WAYLAND but still hang when pywebview initializes GTK.
    native_mode = not is_wsl
    if is_wsl:
        print("[INFO] WSL detected; launching NiceGUI in browser mode instead of native desktop mode.")

    ui.run(
        native=native_mode,
        window_size=(1400, 900),  # Window dimensions
        fullscreen=False,
        title="AI Trading Bot",
        favicon="🤖",
        dark=True,                # Dark theme
        reload=False,             # Disable hot reload in production
        show=True,                # Show window immediately
        port=8080,                # Default port
        binding_refresh_interval=0.1  # Faster UI updates
    )
