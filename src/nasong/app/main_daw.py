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
The NaSong DAW (Digital Audio Workstation) TUI application.

This module provides a Textual-based terminal interface for live-coding music,
featuring a real-time waveform and spectrogram visualization, a file browser,
and an integrated code editor.
"""

#
### Import Modules. ###
#
import os
import sys
import threading
import subprocess
import numpy as np

#
from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    Label,
    TextArea,
    DirectoryTree,
    Static,
    Button,
)
from textual.containers import Container, Horizontal, Vertical
from textual.binding import Binding
from textual.reactive import reactive
from textual import events
from rich.text import Text

#
from nasong.app.live_session import LiveSession


class TimelineWidget(Static):
    """
    A widget that visualizes the audio timeline including waveform and spectrogram.

    This widget renders a character-based visualization of the audio state,
    indicating which parts of the timeline are cached, stale, or currently being played.
    It also handles mouse interactions for seeking.

    Attributes:
        cursor_time: The current playback position in seconds.
        duration: The total duration visible in the timeline window.
    """

    cursor_time = reactive(0.0)
    duration = reactive(60.0)  # Default view duration in seconds

    def __init__(self, session: LiveSession, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.render_engine = session.render_engine

    def on_mount(self):
        self.set_interval(0.1, self.refresh)

    def render(self):
        width = self.content_size.width
        if width == 0:
            return Text("")

        # Prepare multi-line output
        height = self.content_size.height
        if height < 2:
            height = 10  # Fallback

        # We will build a list of strings (lines)
        grid = [[" " for _ in range(width)] for _ in range(height)]

        blocks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

        # Calculate samples per character
        seconds_per_char = self.duration / width
        samples_per_char = int(seconds_per_char * self.session.sample_rate)

        cursor_x = int((self.cursor_time / self.duration) * width)

        # Pre-calculation loop
        for x in range(width):
            t = (x / width) * self.duration
            sample_idx = int(t * self.session.sample_rate)

            # Get chunk
            chunk, version = self.render_engine.get_audio_chunk(sample_idx)

            # Spectrogram / Waveform Logic
            if chunk is not None:
                chunk_size = self.render_engine.chunk_size
                chunk_start = (sample_idx // chunk_size) * chunk_size
                offset = sample_idx - chunk_start

                # Window for analysis
                if offset + samples_per_char < len(chunk):
                    window = chunk[offset : offset + samples_per_char]
                    if len(window) > 0:
                        # 1. Waveform (Top line, index 0)
                        amp = np.max(np.abs(window))
                        idx = int(amp * 7)
                        idx = max(0, min(7, idx))
                        grid[0][x] = blocks[idx]

                        # 2. Spectrogram (Remaining lines)
                        # perform FFT
                        fft_res = np.abs(np.fft.rfft(window))

                        num_bins_y = height - 1
                        if num_bins_y > 0 and len(fft_res) > 0:
                            # Log scale frequency? Linear for now.
                            bin_size = len(fft_res) / num_bins_y

                            for y in range(num_bins_y):
                                bin_start = int(y * bin_size)
                                bin_end = int((y + 1) * bin_size)
                                if bin_end <= bin_start:
                                    bin_end = bin_start + 1

                                band_energy = np.mean(fft_res[bin_start:bin_end])
                                # Log scale magnitude
                                mag = np.log1p(band_energy) * 5  # Scale factor

                                mag_idx = int(mag)
                                mag_idx = max(0, min(7, mag_idx))

                                # Draw from bottom up
                                row_idx = height - 1 - y
                                if 0 < row_idx < height:
                                    grid[row_idx][x] = blocks[mag_idx]

        # Construct Text
        final_text = Text()
        for y in range(height):
            line = Text()
            for x in range(width):
                char = grid[y][x]

                # Re-calculate style per cell
                t = (x / width) * self.duration
                sample_idx = int(t * self.session.sample_rate)

                _, version = self.render_engine.get_audio_chunk(sample_idx)
                current_version = self.render_engine.current_version_id

                # Default style (Grey)
                bg_style = "on #1a1a1a" if x % 2 == 0 else "on #2a2a2a"

                if version != -1:  # Found
                    if version == current_version:
                        # Green
                        bg_style = "on #004400" if x % 2 == 0 else "on #005500"
                    else:
                        # Blue (Stale/Edited)
                        bg_style = "on #000044" if x % 2 == 0 else "on #000055"

                # Cursor style
                style = bg_style
                if x == cursor_x:
                    style = "reverse"
                    if y == 0:
                        char = "|"  # Draw cursor line

                line.append(char, style=style)

            final_text.append(line)
            if y < height - 1:
                final_text.append("\n")

        return final_text

    def on_click(self, event: events.Click):
        width = self.content_size.width
        if width > 0:
            # Calculate time (relative to widget content box)
            new_time = (event.x / width) * self.duration
            self.session.seek(new_time)
            self.cursor_time = new_time
            # Force refresh to show immediate feedback
            self.refresh()


class Editor(TextArea):
    """
    A specialized code editor widget for Python music scripts.

    Provides syntax highlighting (Dracula theme) and line numbers optimized
    for the NaSong live-coding workflow.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.language = "python"
        self.theme = "dracula"
        self.show_line_numbers = True


class NasongDAWApp(App):
    """
    The main application class for the NaSong DAW.

    Coordinates the TUI layout, audio session integration, file management,
    and compilation of scripts to WAV files.

    Bindings:
        space: Toggle Play/Pause.
        f5: Reload Code.
        ctrl+s: Save current file.
        q: Quit the application.
    """

    CSS = """
    #sidebar {
        width: 30;
        dock: left;
        border-right: solid green;
    }
    
    #controls {
        height: 3;
        dock: top;
        padding: 0 1;
        background: $surface;
        color: $text;
    }
    
    #controls Button {
        margin-right: 1;
    }
    
    #timeline-container {
        height: 10;
        border-bottom: solid blue;
        margin: 1;
    }
    
    #editor-container {
        height: 1fr;
    }
    
    TimelineWidget {
        height: 100%;
        background: $surface-darken-1;
        color: $text;
    }
    """

    BINDINGS = [
        Binding("space", "toggle_play", "Play/Pause"),
        Binding("f5", "reload_code", "Reload Code"),
        Binding("ctrl+s", "save_file", "Save File"),
        Binding("q", "quit", "Quit"),
    ]

    current_file = reactive("")
    is_playing = reactive(False)

    def __init__(self, initial_file=None, sample_rate=48000):
        super().__init__()
        self.initial_file = initial_file
        # Initialize Audio Session
        self.session = LiveSession(sample_rate=sample_rate)
        self.session.set_error_callback(self.on_session_error)

    def compose(self) -> ComposeResult:
        yield Header()

        with Container():
            with Horizontal():
                with Vertical(id="sidebar"):
                    yield Label("Files", classes="header")
                    # Start at current working directory or nasong examples
                    yield DirectoryTree("./nasong_examples", id="file-tree")

                with Vertical():
                    # Controls bar
                    with Horizontal(id="controls"):
                        yield Button("Play", id="btn-play-stop", variant="success")
                        yield Button(
                            "Compile (WAV)", id="btn-compile", variant="primary"
                        )

                    with Container(id="timeline-container"):
                        yield Label("Timeline (Click to seek)")
                        yield TimelineWidget(self.session, id="timeline")

                    with Container(id="editor-container"):
                        yield Editor(id="code-editor")

        yield Footer()

    def on_mount(self):
        self.title = "NaSong DAW"

        # Load initial file if provided
        if self.initial_file:
            self.load_file(self.initial_file)

        # Start timer to sync cursor from session
        self.set_interval(0.1, self.sync_cursor)

    def sync_cursor(self):
        # Update timeline cursor from session
        # LiveSession.cursor is in samples
        current_time = self.session.cursor / self.session.sample_rate
        self.query_one("#timeline", TimelineWidget).cursor_time = current_time

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected):
        self.load_file(event.path)

    def load_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                self.query_one("#code-editor", Editor).text = content
                self.current_file = str(path)

                # Auto-load into session
                if self.session.load_script(self.current_file):
                    self.notify(f"Loaded {path}")
                else:
                    self.notify("Failed to load script logic", severity="warning")

        except Exception as e:  # pylint: disable=broad-except
            self.notify(f"Error loading file: {e}", severity="error")

    def action_save_file(self):
        if self.current_file:
            editor = self.query_one("#code-editor", Editor)
            content = editor.text
            try:
                with open(self.current_file, "w", encoding="utf-8") as f:
                    f.write(content)
                self.notify(f"Saved {self.current_file}")
            except Exception as e:  # pylint: disable=broad-except
                self.notify(f"Error saving: {e}", severity="error")
        else:
            self.notify("No file loaded to save", severity="warning")

    def action_reload_code(self):
        self.action_save_file()
        if self.current_file:
            if self.session.load_script(self.current_file):
                self.notify("Reloaded Script & Audio Engine")
            else:
                self.notify("Reload Failed - Check Logs/Console", severity="error")

    def action_toggle_play(self):
        # Toggle session
        if self.is_playing:
            self.session.stop()
            self.is_playing = False
            self.notify("Stopped")

            # Update button
            try:
                btn = self.query_one("#btn-play-stop", Button)
                btn.label = "Play"
                btn.variant = "success"
            except Exception:  # pylint: disable=broad-except
                pass
        else:
            self.session.start()
            self.is_playing = True
            self.notify("Playing")

            # Update button
            try:
                btn = self.query_one("#btn-play-stop", Button)
                btn.label = "Stop"
                btn.variant = "error"
            except Exception:  # pylint: disable=broad-except
                pass

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn-play-stop":
            self.action_toggle_play()
        elif event.button.id == "btn-compile":
            self.action_compile()

    def action_compile(self):
        if not self.current_file:
            self.notify("No file loaded to compile", severity="warning")
            return

        self.notify("Compiling logic to WAV...", title="Compiling")

        # Output directory
        output_dir = "generated_songs_wav"
        os.makedirs(output_dir, exist_ok=True)

        # Derive output filename
        basename = os.path.basename(self.current_file)
        name, _ = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{name}.wav")

        # Construct command
        # Use simple python -m nasong because we assume running in venv
        cmd = [
            sys.executable,
            "-m",
            "nasong",
            "-i",
            self.current_file,
            "-o",
            output_path,
        ]

        # Run in background (non-blocking would be better but simple subprocess for now)
        # To avoid blocking UI, we should probably run in thread, but for now blocking nicely notifies.
        # Actually blocking will freeze UI. Let's use threading.

        def compile_thread():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if result.returncode == 0:
                    self.app.call_from_thread(
                        self.notify,
                        f"Compiled to {output_path}",
                        title="Compile Success",
                        severity="information",
                    )
                else:
                    self.app.call_from_thread(
                        self.notify,
                        f"Compile Error:\n{result.stderr}",
                        title="Compile Failed",
                        severity="error",
                    )
            except Exception as e:  # pylint: disable=broad-except
                self.app.call_from_thread(self.notify, f"Error: {e}", severity="error")

        threading.Thread(target=compile_thread, daemon=True).start()

    def on_session_error(self, err):
        self.notify(str(err), severity="error")

    def on_unmount(self):
        self.session.stop()


def main():
    """
    Entry point for the nasong-daw application.

    Parses command-line arguments and launches the Textual TUI.
    """
    import argparse

    parser = argparse.ArgumentParser(description="NaSong DAW")
    parser.add_argument("file", nargs="?", help="Initial file to open")
    parser.add_argument(
        "--rate", type=int, default=48000, help="Audio sample rate (default: 48000)"
    )
    args = parser.parse_args()

    app = NasongDAWApp(initial_file=args.file, sample_rate=args.rate)
    app.run()


if __name__ == "__main__":
    main()
