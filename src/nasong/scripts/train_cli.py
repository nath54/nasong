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


"""Training execution CLI wrapper.

This script acts as a convenience entry point for launching NaSong instrument
training sessions.
"""

#
### Import Modules. ###
#
import nasong.trainable.train as train_module


def main() -> None:
    """Main entry point for the training CLI."""
    train_module.main()


if __name__ == "__main__":
    main()
