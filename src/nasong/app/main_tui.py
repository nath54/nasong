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
    Input,
)
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.binding import Binding
from textual.reactive import reactive
from textual.message import Message
from pathlib import Path
import os
import sounddevice as sd

from nasong.app.live_session import LiveSession
from nasong.app.docs_utils import get_module_docs, flatten_docs


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

        for name, doc in docs.get("classes", {}).items():
            node.add(f"Class: {name}", allow_expand=False)

        for name, doc in docs.get("functions", {}).items():
            node.add(f"Func: {name}", allow_expand=False)

        for name, sub_docs in docs.get("submodules", {}).items():
            self.add_docs_to_tree(node, sub_docs, name)


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
        dock: top;
        margin-bottom: 1;
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
    bpm = reactive(120.0)
    volume = reactive(0.8)

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

                    yield Label(f"BPM: {self.bpm}", id="lbl-bpm")
                    with Horizontal():
                        yield Button("-", id="btn-bpm-dec")
                        yield Button("+", id="btn-bpm-inc")

                    yield Label(f"Volume: {int(self.volume * 100)}%", id="lbl-vol")
                    # Volume controls could be buttons or slider if available
                    with Horizontal():
                        yield Button("-", id="btn-vol-dec")
                        yield Button("+", id="btn-vol-inc")

                with Container(classes="box"):
                    yield DocBrowser()
        yield Footer()

    def on_mount(self) -> None:
        self.title = "NaSong Algo-Rave"
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
            if self.is_playing:
                self.action_reload_code()

    def action_reload_code(self) -> None:
        if self.current_file:
            self.notify(f"Reloading {self.current_file}...")
            self.action_save_file()
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
            else:
                self.notify("Reload Failed!", title="Error", severity="error")

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
            self.volume = min(1.0, self.volume + 0.1)
        elif event.button.id == "btn-vol-dec":
            self.volume = max(0.0, self.volume - 0.1)

    def watch_bpm(self, val):
        try:
            self.query_one("#lbl-bpm", Label).update(f"BPM: {val}")
        except:
            pass

    def watch_volume(self, val):
        try:
            self.query_one("#lbl-vol", Label).update(f"Volume: {int(val * 100)}%")
            # Update volume in session?
            # session doesn't have set_volume yet.
        except:
            pass

    def on_session_error(self, err_msg: str):
        self.notify(err_msg, title="Audio/Script Error", severity="error")

    def on_unmount(self) -> None:
        self.session.stop()


def main():
    app = AlgoRaveApp()
    app.run()


if __name__ == "__main__":
    main()
