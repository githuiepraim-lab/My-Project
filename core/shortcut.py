"""Create a desktop shortcut that launches the assistant with its pet.

    python -m core.shortcut            # shortcut that starts the app + pet
    python -m core.shortcut --pet-only # pet on the desktop, main window minimised
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "Ephraim AI"


def _command(flag: str) -> tuple[str, list[str]]:
    """(program, args) that launch the app. Uses pythonw on Windows so no console window."""
    if getattr(sys, "frozen", False):
        return sys.executable, [flag]
    exe = sys.executable
    if os.name == "nt" and exe.lower().endswith("python.exe"):
        exe = exe[:-len("python.exe")] + "pythonw.exe"
    return exe, [str(ROOT / "main.py"), flag]


def desktop_dir() -> Path:
    if os.name == "nt":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "[Environment]::GetFolderPath('Desktop')"],
                                 capture_output=True, text=True, timeout=15).stdout.strip()
            if out:
                return Path(out)
        except Exception:
            pass
    d = Path.home() / "Desktop"
    return d if d.exists() else Path.home()


def windows_script(lnk: Path, exe: str, args: list[str], icon: Path | None) -> str:
    q = lambda s: str(s).replace("'", "''")            # PowerShell single-quote escaping
    arg = " ".join(f'"{a}"' for a in args)
    ico = f"$s.IconLocation = '{q(icon)}'; " if icon and icon.exists() else ""
    return (f"$w = New-Object -ComObject WScript.Shell; $s = $w.CreateShortcut('{q(lnk)}'); "
            f"$s.TargetPath = '{q(exe)}'; $s.Arguments = '{q(arg)}'; "
            f"$s.WorkingDirectory = '{q(ROOT)}'; {ico}$s.Description = 'Ephraim AI with desktop pet'; $s.Save()")


def create_shortcut(pet_only: bool = False, folder: Path | None = None) -> Path:
    flag = "--pet-only" if pet_only else "--pet"
    exe, args = _command(flag)
    folder = Path(folder) if folder else desktop_dir()
    folder.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        lnk = folder / f"{NAME}.lnk"
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        windows_script(lnk, exe, args, ROOT / "config" / "jarvis.ico")],
                       check=True, capture_output=True, timeout=30)
        return lnk
    cmd = " ".join(f'"{x}"' for x in [exe, *args])
    if sys.platform == "darwin":
        f = folder / f"{NAME}.command"
        f.write_text(f'#!/bin/bash\ncd "{ROOT}"\nexec {cmd}\n', encoding="utf-8")
    else:
        f = folder / f"{NAME}.desktop"
        f.write_text("[Desktop Entry]\nType=Application\n"
                     f"Name={NAME}\nComment=Ephraim AI with desktop pet\nPath={ROOT}\nExec={cmd}\nTerminal=false\n",
                     encoding="utf-8")
    f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return f


if __name__ == "__main__":
    print("Created:", create_shortcut(pet_only="--pet-only" in sys.argv))
