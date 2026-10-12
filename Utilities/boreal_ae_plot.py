"""Plot BorealHDR exposure at the moment SLAM opens each prepared left frame.

Run directly with a prepared AE sequence directory to attach to an active demo,
or let ``borealhdr demo --ae --ae-plot`` launch it before SLAM starts.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import struct
import sys
import tkinter as tk
from pathlib import Path


IN_OPEN = 0x20
IN_Q_OVERFLOW = 0x4000
EVENT_HEADER = struct.Struct('iIII')


def load_exposure_trace(sequence: Path):
    with (sequence / 'exposure.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'No exposure frames in {sequence / "exposure.csv"}')
    frames = [int(row['frame_num']) for row in rows]
    if frames != list(range(len(rows))):
        raise ValueError('Exposure frames must be consecutive and zero-based')
    exposures = [float(row['exposure_ms']) for row in rows]
    by_filename = {f"{row['timestamp_ns']}.png": i for i, row in enumerate(rows)}
    if len(by_filename) != len(rows):
        raise ValueError('Exposure timestamps must be unique')
    return exposures, by_filename


class FrameOpenMonitor:
    """Linux inotify events for images opened in a prepared rgb_0 directory."""

    def __init__(self, directory: Path):
        if not sys.platform.startswith('linux'):
            raise RuntimeError('Frame-open monitoring requires Linux inotify')
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.inotify_init1.argtypes = [ctypes.c_int]
        self._libc.inotify_init1.restype = ctypes.c_int
        self._libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self._libc.inotify_add_watch.restype = ctypes.c_int
        self.fd = self._libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_init1 failed')
        watch = self._libc.inotify_add_watch(self.fd, os.fsencode(directory), IN_OPEN)
        if watch < 0:
            error = ctypes.get_errno()
            os.close(self.fd)
            raise OSError(error, f'Cannot watch {directory}')

    def read(self):
        """Return opened filenames and whether the event queue overflowed."""
        names = []
        overflow = False
        while True:
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                break
            if not data:
                break
            offset = 0
            while offset + EVENT_HEADER.size <= len(data):
                _, mask, _, size = EVENT_HEADER.unpack_from(data, offset)
                offset += EVENT_HEADER.size
                name = data[offset:offset + size].split(b'\0', 1)[0]
                offset += size
                if mask & IN_Q_OVERFLOW:
                    overflow = True
                if mask & IN_OPEN and name:
                    names.append(os.fsdecode(name))
        return names, overflow

    def close(self):
        os.close(self.fd)


def show_plot(sequence: Path):
    exposures, by_filename = load_exposure_trace(sequence)
    monitor = FrameOpenMonitor(sequence / 'rgb_0')
    try:
        root = tk.Tk()
    except Exception:
        monitor.close()
        raise
    root.title(f'BorealHDR AE exposure — {sequence.name}')
    root.geometry('850x380')
    canvas = tk.Canvas(root, background='#ffffff', highlightthickness=0)
    canvas.pack(fill='both', expand=True)
    current = {'frame': None, 'overflow': False}

    def draw(_event=None):
        canvas.delete('all')
        width = max(canvas.winfo_width(), 400)
        height = max(canvas.winfo_height(), 200)
        x0, x1, y0, y1 = 62, width - 22, 48, height - 48
        upper = max(max(exposures) * 1.1, 1.)
        x = lambda i: x0 + (x1 - x0) * i / max(len(exposures) - 1, 1)
        y = lambda value: y1 - (y1 - y0) * value / upper
        canvas.create_line(x0, y1, x1, y1, fill='#777777')
        canvas.create_line(x0, y0, x0, y1, fill='#777777')
        for fraction in (0, .25, .5, .75, 1):
            value = upper * fraction
            yy = y(value)
            canvas.create_line(x0, yy, x1, yy, fill='#eeeeee')
            canvas.create_text(x0 - 7, yy, text=f'{value:g}', anchor='e', fill='#555555')
        canvas.create_text(x0, y1 + 20, text='0', anchor='w', fill='#555555')
        canvas.create_text(x1, y1 + 20, text=str(len(exposures) - 1), anchor='e', fill='#555555')
        canvas.create_text((x0 + x1) / 2, y1 + 20, text='Frame', fill='#555555')
        if len(exposures) > 1:
            points = [coordinate for i, value in enumerate(exposures) for coordinate in (x(i), y(value))]
            canvas.create_line(*points, fill='#b7c7d7', width=2)
        frame = current['frame']
        if frame is None:
            label = 'Waiting for SLAM to open a left frame'
        else:
            canvas.create_line(x(frame), y0, x(frame), y1, fill='#e4572e', dash=(4, 3))
            canvas.create_oval(x(frame)-5, y(exposures[frame])-5,
                               x(frame)+5, y(exposures[frame])+5,
                               fill='#e4572e', outline='')
            label = f'Frame {frame}/{len(exposures)-1}     Exposure {exposures[frame]:.6g} ms'
        if current['overflow']:
            label += '     WARNING: frame-open events were lost'
        canvas.create_text(x0, 20, text=label, anchor='w', fill='#172b4d', font=('TkDefaultFont', 12, 'bold'))

    def poll():
        names, overflow = monitor.read()
        if overflow:
            current['overflow'] = True
        for name in names:
            if name in by_filename:
                current['frame'] = by_filename[name]
        if names or overflow:
            draw()
        root.after(50, poll)

    def close():
        monitor.close()
        root.destroy()

    canvas.bind('<Configure>', draw)
    root.protocol('WM_DELETE_WINDOW', close)
    root.update_idletasks()
    draw()
    print('READY', flush=True)
    root.after(50, poll)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', type=Path, help='Prepared AE sequence directory')
    args = parser.parse_args()
    show_plot(args.sequence.expanduser().resolve())


if __name__ == '__main__':
    main()
