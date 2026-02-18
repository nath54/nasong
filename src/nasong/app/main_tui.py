# Copyright (C) 2026 Nathan Cerisara <https://github.com/nath54/nasong>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
The NaSong Algo-Rave TUI application.

This module provides a multi-tab terminal interface for live-coding music,
including a documentation browser, application logs, and session controls (BPM, Volume).
"""

#
### Import Modules. ###
#
import os
import sys

#
from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    Label,
    TabbedContent,
    TabPane,
    TextArea,
    Button,
    Tree,
    Input,
    RichLog,
)
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.binding import Binding
from textual.reactive import reactive

#
from nasong.app.live_session import LiveSession
from nasong.app.docs_utils import get_module_docs


class Editor(TextArea):
    """
    A specialized code editor widget for the Algo-Rave environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.language = "python"
        self.theme = "dracula"
        self.show_line_numbers = True


class DocBrowser(Container):
    """
    A documentation browser widget that introspects the NaSong API.

    Uses `get_module_docs` to build a navigable tree of classes and functions
    available in the NaSong DSL and Theory modules.
    """

    def compose(self) -> ComposeResult:
        yield Label("Documentation", classes="header")
        yield Input(placeholder="Search...", id="doc-search", classes="search-box")
        yield Tree("NaSong API", id="doc-tree")

    def on_mount(self):
        self.populate_tree()

    def populate_tree(self):
        tree = self.query_one("#doc-tree", Tree)
        tree.root.expand()

        # DSL Docs
        dsl_docs = get_module_docs("nasong.dsl")
        self.add_docs_to_tree(tree.root, dsl_docs, "DSL")

        # Theory Docs
        theory_docs = get_module_docs("nasong.theory")
        self.add_docs_to_tree(tree.root, theory_docs, "Theory")

    def add_docs_to_tree(self, root_node, docs, label):
        node = root_node.add(label, expand=False)

        for name, _doc in docs.get("classes", {}).items():
            node.add(f"Class: {name}", allow_expand=False)

        for name, doc in docs.get("functions", {}).items():
            node.add(f"Func: {name}", allow_expand=False)

        for name, sub_docs in docs.get("submodules", {}).items():
            self.add_docs_to_tree(node, sub_docs, name)


class LogScreen(Screen):
    """
    A dedicated screen for viewing application-wide logs and errors.
    """

    BINDINGS = [("escape", "app.pop_screen", "Close Logs")]

    def compose(self) -> ComposeResult:
        yield Label("Application Logs (Press ESC to close)", classes="header")
        yield RichLog(highlight=True, markup=True, id="log-view")

    def on_mount(self):
        self.query_one("#log-view", RichLog).write("Log Console Started...")


class AlgoRaveApp(App):
    """
    The main Algo-Rave application class.

    Provides a rich TUI for live music performance, managing multiple editor tabs,
    real-time session controls, and background log capturing.

    Bindings:
        ctrl+n: Open a new tab.
        ctrl+w: Close the current tab.
        f5: Reload the current script.
        ctrl+s: Save the current file.
        ctrl+l: Toggle the log screen.
        q: Quit the application.
    """

    CSS = """
    Screen {
    }

    #main-content {
        width: 3fr;
        height: 100%;
    }

    #sidebar {
        width: 1fr;
        height: 100%;
        border-left: solid green;
    }

    .box {
        height: 50%;
        padding: 1;
        border-bottom: solid white;
    }

    .header {
        text-align: center;
        background: $accent;
        color: $text;
        width: 100%;
    }

    .search-box {
        dock: top;
        margin-bottom: 1;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text;
        text-align: right;
        padding: 0 1;
    }

    #status-bar.reloading {
        color: yellow;
        text-style: bold;
    }

    #status-bar.success {
        color: green;
        text-style: bold;
    }

    #status-bar.error {
        color: red;
        text-style: bold;
    }

    .sidebar-row {
        height: 3;
        margin-bottom: 1;
    }

    .sidebar-btn {
        width: 1fr;
        height: 3;
        margin: 0 1;
    }

    .sidebar-label {
        width: 1fr;
        text-align: center;
        content-align: center middle;
    }


    """

    BINDINGS = [
        Binding("ctrl+n", "new_tab", "New Tab"),
        Binding("ctrl+w", "close_tab", "Close Tab"),
        Binding("f5", "reload_code", "Reload Code"),
        Binding("ctrl+s", "save_file", "Save File"),
        Binding("ctrl+l", "toggle_log", "Log"),
        ("q", "quit", "Quit"),
    ]

    SCREENS = {"log": LogScreen}

    current_file = reactive("")
    is_playing = reactive(False)
    bpm = reactive(120.0)
    volume = reactive(0.8)

    def __init__(
        self,
        device=None,
        sample_rate=44100,
        volume=0.8,
        initial_file=None,
        block_size=2048,
    ):
        super().__init__()
        self.session = LiveSession(
            device=device, sample_rate=sample_rate, block_size=block_size
        )
        self.session.set_volume(volume)
        self.volume = volume
        self.initial_file = initial_file
        self.session.set_error_callback(self.on_session_error)
        self.session.set_log_callback(self.log_message)

    def log_message(self, msg: str):
        # Forward specific messages to Status Bar if desirable, or always simple status
        # And always write to LogScreen
        if self.is_mounted:
            # We can't access screen widgets directly if not active,
            # but we can access installed screens?
            # Or just push to log when screen is active?
            # Better: Keep a buffer or try to access the screen instance.
            try:
                # This assumes 'log' screen is instantiated.
                # Textual lazy loads screens usually, but we installed it in SCREENS.
                # Only writes if screen is instantiated.
                self.install_screen(
                    LogScreen(), "log"
                ) if "log" not in self._installed_screens else None
                self.get_screen("log").query_one(RichLog).write(msg)
            except Exception:  # pylint: disable=broad-except
                pass

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Container(id="main-content"):
                with TabbedContent(id="tabs"):
                    with TabPane("Editor", id="tab-demo"):
                        yield Editor(id="editor-demo")

            with Vertical(id="sidebar"):
                with Container(classes="box"):
                    yield Label("Live Settings", classes="header")
                    with Horizontal(classes="sidebar-row"):
                        yield Button(
                            "Start Audio",
                            id="btn-audio",
                            variant="success",
                            classes="sidebar-btn",
                        )
                        yield Button(
                            "Reload (F5)",
                            id="btn-reload",
                            variant="primary",
                            classes="sidebar-btn",
                        )

                    with Horizontal(classes="sidebar-row"):
                        # BPM Row
                        with Vertical(classes="sidebar-label"):
                            yield Label(f"BPM: {self.bpm}", id="lbl-bpm")
                            with Horizontal():
                                yield Button("-", id="btn-bpm-dec")
                                yield Button("+", id="btn-bpm-inc")

                    with Horizontal(classes="sidebar-row"):
                        # Volume Row
                        with Vertical(classes="sidebar-label"):
                            yield Label(
                                f"Volume: {int(self.volume * 100)}%", id="lbl-vol"
                            )
                            with Horizontal():
                                yield Button("-", id="btn-vol-dec")
                                yield Button("+", id="btn-vol-inc")

                with Container(classes="box"):
                    yield DocBrowser()
        yield Label("Ready", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "NaSong Algo-Rave"
        target = self.initial_file or "demo_theory.py"
        try:
            with open(target, "r", encoding="utf-8") as f:
                content = f.read()
                editor = self.query_one("#editor-demo", Editor)
                editor.text = content
                self.current_file = os.path.abspath(target)

                # Dynamic tab name
                tab_pane = self.query_one("#tab-demo", TabPane)
                tab_pane.label = os.path.basename(target)

                # Auto-load into session if it's there
                success = self.session.load_script(self.current_file)
                status = self.query_one("#status-bar", Label)
                if success:
                    status.update("● ONLINE")
                    status.classes = "success"
                else:
                    status.update("● ERROR")
                    status.classes = "error"
        except Exception as e:  # pylint: disable=broad-except
            self.log_message(f"Could not load {target}: {e}")

    def save_current_file(self) -> None:
        """Helper to save the current file content."""
        editor = self.query_one("#editor-demo", Editor)
        content = editor.text
        if self.current_file:
            with open(self.current_file, "w", encoding="utf-8") as f:
                f.write(content)
            self.notify(f"Saved {self.current_file}")

    def action_save_file(self) -> None:
        """Saves file and optionally reloads if playing."""
        self.save_current_file()
        if self.is_playing:
            self.action_reload_code()

    def action_reload_code(self) -> None:
        """Reloads the code. Saves first."""
        if self.current_file:
            # We ONLY save here. We do NOT call action_save_file()
            # because that might trigger action_reload_code again!
            self.save_current_file()

            self.notify(f"Reloading {self.current_file}...")

            # Update Status Bar
            status = self.query_one("#status-bar", Label)
            status.update("Reloading...")
            status.classes = "reloading"

            # Pass globals? No, load_script just re-imports.
            # We need to consider how BPM interacts.
            # If the script uses `render(..., bpm=120)`, it is hardcoded.
            # Ideally the script should read BPM from somewhere or accept it.
            # For this prototype, we just reload.
            success = self.session.load_script(self.current_file)
            if success:
                self.notify(
                    "Reloaded Successfully!", title="Success", severity="information"
                )
                status.update("● ONLINE")
                status.classes = "success"
            else:
                self.notify("Reload Failed!", title="Error", severity="error")
                status.update("● ERROR")
                status.classes = "error"

    def action_toggle_log(self) -> None:
        self.push_screen("log")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-audio":
            if not self.is_playing:
                self.session.start()
                self.is_playing = True
                event.button.label = "Stop Audio"
                event.button.variant = "error"
                if self.current_file:
                    self.session.load_script(self.current_file)
            else:
                self.session.stop()
                self.is_playing = False
                event.button.label = "Start Audio"
                event.button.variant = "success"

        elif event.button.id == "btn-bpm-inc":
            self.bpm += 5
        elif event.button.id == "btn-bpm-dec":
            self.bpm -= 5
        elif event.button.id == "btn-vol-inc":
            self.volume = min(10.0, self.volume + 0.1)  # Boost allowed
        elif event.button.id == "btn-vol-dec":
            self.volume = max(0.0, self.volume - 0.1)
        elif event.button.id == "btn-reload":
            self.action_reload_code()

    def watch_bpm(self, val):
        try:
            self.query_one("#lbl-bpm", Label).update(f"BPM: {val}")
        except Exception as _e:
            pass

    def watch_volume(self, val):
        try:
            self.query_one("#lbl-vol", Label).update(f"Volume: {int(val * 100)}%")
            self.session.set_volume(val)
        except Exception as _e:
            pass

    def on_session_error(self, err_msg: str):
        if self.is_mounted:
            self.notify(err_msg, title="Audio/Script Error", severity="error")
        else:
            # Print to console as fallback
            sys.__stdout__.write(f"SESSION ERROR: {err_msg}\n")

    def on_unmount(self) -> None:
        self.session.stop()


def main():
    """
    Entry point for the Algo-Rave TUI.

    Parses command-line arguments and launches the application.
    """
    import argparse

    parser = argparse.ArgumentParser(description="NaSong Algo-Rave TUI")
    parser.add_argument(
        "script",
        nargs="?",
        default="nasong_examples/demo_loop.py",
        help="Initial script",
    )
    parser.add_argument(
        "--rate", type=int, default=48000, help="Sample rate (try 44100 if silent)"
    )
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--volume", type=float, default=0.8, help="Initial volume")
    parser.add_argument(
        "--block-size",
        type=int,
        default=2048,
        help="Audio chunk size in samples (default: 2048). "
        "Smaller = lower latency, larger = more stable.",
    )
    args = parser.parse_args()

    device = args.device
    # Simple check for exists in args (nargs=? script might be missing if only flags)
    # But usually it's handled by default=
    initial_script = args.script

    app = AlgoRaveApp(
        device=device,
        sample_rate=args.rate,
        volume=args.volume,
        initial_file=initial_script,
        block_size=args.block_size,
    )
    app.run()


if __name__ == "__main__":
    main()
