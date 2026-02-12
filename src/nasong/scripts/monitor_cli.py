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


"""Experiment monitoring CLI tool.

This script provides a command-line interface for listing, inspecting, and
deleting NaSong training experiments using the `ExperimentManager`.
"""

#
### Import Modules. ###
#
import argparse
from datetime import datetime

#
from nasong.scripts.experiment_manager import ExperimentManager


def main() -> None:
    """Main entry point for the experiment monitoring CLI."""
    parser = argparse.ArgumentParser(description="Monitor Nasong experiments")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # List
    _list_parser = subparsers.add_parser("list", help="List all experiments")

    # Show
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("experiment_id", help="Experiment ID")

    # Delete
    del_parser = subparsers.add_parser("delete", help="Delete experiment")
    del_parser.add_argument("experiment_id", help="Experiment ID")

    args = parser.parse_args()
    manager = ExperimentManager()

    if args.command == "list":
        experiments = manager.list_experiments()
        print(
            f"{'ID':<10} | {'Date':<20} | {'Instrument':<15} | {'Loss':<10} | {'Status':<10}"
        )
        print("-" * 75)
        for exp in experiments:
            date_str = datetime.fromtimestamp(exp.timestamp).strftime("%Y-%m-%d %H:%M")
            instr = exp.params.get("instrument", "N/A")
            loss = exp.metrics.get("best_loss", "N/A")
            if isinstance(loss, float):
                loss = f"{loss:.4f}"
            print(
                f"{exp.id:<10} | {date_str:<20} | {instr[:15]:<15} | {loss:<10} | {exp.status:<10}"
            )

    elif args.command == "show":
        exp = manager.get_experiment(args.experiment_id)
        if not exp:
            print("Experiment not found.")
            return

        print(f"Experiment: {exp.name} ({exp.id})")
        print(f"Path: {exp.path}")
        print(f"Status: {exp.status}")
        print("-" * 20)
        print("Parameters:")
        for k, v in exp.params.items():
            print(f"  {k}: {v}")
        print("-" * 20)
        print("Metrics:")
        for k, v in exp.metrics.items():
            print(f"  {k}: {v}")

    elif args.command == "delete":
        if manager.delete_experiment(args.experiment_id):
            print(f"Deleted experiment {args.experiment_id}")
        else:
            print("Experiment not found.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
