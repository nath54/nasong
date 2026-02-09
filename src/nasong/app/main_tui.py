from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    Static,
    Label,
    TabbedContent,
    TabPane,
    TextArea,
    Button,
    DirectoryTree,
    Tree,
)
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.binding import Binding
from textual.reactive import reactive
from pathlib import Path
import os

from nasong.app.live_session import LiveSession


class Editor(TextArea):
    """
    Code editor widget.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.language = "python"
        self.show_line_numbers = True


class DocBrowser(Container):
    """
    Documentation browser.
    """

    def compose(self) -> ComposeResult:
        yield Label("Documentation", classes="header")
        yield TextArea("Search...", id="doc-search", classes="search-box")
        # Placeholder tree
        tree = Tree("NaSong API")
        tree.root.expand()
        dsl = tree.root.add("DSL", expand=True)
        dsl.add("units (BPM, Hz)")
        dsl.add("chain (>>)")
        theory = tree.root.add("Theory", expand=True)
        theory.add("Scale")
        theory.add("Chord")
        yield tree


class AlgoRaveApp(App):
    """
    The main TUI application for live coding music.
    """

    CSS = """
    Screen {
        layout: horizontal;
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
        height: 3;
        dock: top;
    }
    
    #status-bar {
        dock: bottom;
        height: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+n", "new_tab", "New Tab"),
        Binding("ctrl+w", "close_tab", "Close Tab"),
        Binding("f5", "reload_code", "Reload Code"),
        Binding("ctrl+s", "save_file", "Save File"),
        ("q", "quit", "Quit"),
    ]

    current_file = reactive("")
    is_playing = reactive(False)

    def __init__(self):
        super().__init__()
        self.session = LiveSession()
        self.session.set_error_callback(self.on_session_error)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Container(id="main-content"):
                with TabbedContent(id="tabs"):
                    with TabPane("demo_theory.py", id="tab-demo"):
                        yield Editor(id="editor-demo")

            with Vertical(id="sidebar"):
                with Container(classes="box"):
                    yield Label("Live Settings", classes="header")
                    yield Button("Start Audio", id="btn-audio", variant="success")
                    yield Label("BPM: 120", id="lbl-bpm")

                with Container(classes="box"):
                    yield DocBrowser()
        yield Footer()

    def on_mount(self) -> None:
        self.title = "NaSong Algo-Rave"
        # Load demo content
        try:
            with open("demo_theory.py", "r") as f:
                content = f.read()
                self.query_one("#editor-demo", Editor).text = content
                self.current_file = os.path.abspath("demo_theory.py")
        except FileNotFoundError:
            pass

    def action_save_file(self) -> None:
        editor = self.query_one("#editor-demo", Editor)
        content = editor.text
        if self.current_file:
            with open(self.current_file, "w") as f:
                f.write(content)
            self.notify(f"Saved {self.current_file}")

            # Auto-reload if playing?
            if self.is_playing:
                self.action_reload_code()

    def action_reload_code(self) -> None:
        if self.current_file:
            self.notify(f"Reloading {self.current_file}...")
            # Save first to be sure
            self.action_save_file()
            success = self.session.load_script(self.current_file)
            if success:
                self.notify(
                    "Reloaded Successfully!", title="Success", severity="information"
                )
            else:
                self.notify("Reload Failed!", title="Error", severity="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-audio":
            if not self.is_playing:
                self.session.start()
                self.is_playing = True
                event.button.label = "Stop Audio"
                event.button.variant = "error"
                # Load initial script
                if self.current_file:
                    self.session.load_script(self.current_file)
            else:
                self.session.stop()
                self.is_playing = False
                event.button.label = "Start Audio"
                event.button.variant = "success"

    def on_session_error(self, err_msg: str):
        self.notify(err_msg, title="Audio/Script Error", severity="error")

    def on_unmount(self) -> None:
        self.session.stop()


if __name__ == "__main__":
    app = AlgoRaveApp()
    app.run()
