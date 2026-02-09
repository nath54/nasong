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
)
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.binding import Binding


class Editor(Static):
    """
    A wrapper around TextArea for editing code.
    """

    def compose(self) -> ComposeResult:
        yield TextArea.code_editor("", language="python", id="code-editor")


class Sidebar(Container):
    """
    Right sidebar with live settings and documentation.
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            with Container(id="live-settings", classes="box"):
                yield Label("Live Settings")
                yield Button("BPM: 120", id="btn-bpm")
                yield Button("Volume: 80%", id="btn-vol")
                yield Button("Swing: 0%", id="btn-swing")

            with Container(id="docs-browser", classes="box"):
                yield Label("Documentation")
                yield TextArea.code_editor(
                    "Search instruments...", language=None, id="doc-search"
                )
                # Tree view will go here


class AlgoRaveApp(App):
    """
    The main TUI application.
    """

    CSS = """
    Screen {
        layout: horizontal;
    }
    
    #main-content {
        width: 3fr;
        height: 100%;
        border: solid green;
    }
    
    #sidebar {
        width: 1fr;
        height: 100%;
        border: solid blue;
    }
    
    .box {
        height: 1fr;
        border: solid white;
        margin: 1;
        padding: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+n", "new_tab", "New Tab"),
        Binding("ctrl+w", "close_tab", "Close Tab"),
        Binding("f5", "reload_code", "Reload Code"),
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Container(id="main-content"):
                with TabbedContent(initial="tab-1"):
                    with TabPane("mainsong.py", id="tab-1"):
                        yield Editor()
                    with TabPane("scratchpad.py", id="tab-2"):
                        yield Editor()

            yield Sidebar(id="sidebar")
        yield Footer()

    def action_new_tab(self) -> None:
        # Logic to add a new tab
        pass

    def action_reload_code(self) -> None:
        self.notify("Hot-reloading code...")
        # Logic to trigger live session reload


if __name__ == "__main__":
    app = AlgoRaveApp()
    app.run()
