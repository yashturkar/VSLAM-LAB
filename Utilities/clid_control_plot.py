"""Live light/exposure plot for a prepared CLID stereo demo.

The marker tracks the latest prepared left PNG opened by SLAM. It is a file-read
indicator, not a guarantee that the native SLAM system processed that frame.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tkinter as tk
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Utilities.boreal_ae_plot import FrameOpenMonitor


def load_control_trace(sequence: Path) -> tuple[list[dict[str, float]], dict[str, int]]:
    path = sequence / 'relight.csv'
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        required = {'frame_num', 'timestamp_ns', 'light_commanded', 'exposure_ev',
                    'left_effective_power', 'right_effective_power'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CLID control trace is missing columns: {path}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'No CLID controls in {path}')
    if [int(row['frame_num']) for row in rows] != list(range(len(rows))):
        raise ValueError('CLID control frames must be consecutive and zero-based')
    values = []
    by_filename = {}
    for index, row in enumerate(rows):
        stamp = int(row['timestamp_ns'])
        filename = f'{stamp}.png'
        if filename in by_filename:
            raise ValueError('CLID control timestamps must be unique')
        item = {key: float(row[key]) for key in ('light_commanded', 'exposure_ev',
                                                 'left_effective_power', 'right_effective_power')}
        if not all(math.isfinite(value) for value in item.values()):
            raise ValueError(f'Non-finite CLID control at frame {index}')
        if not 0 <= item['light_commanded'] <= 1 or not -2 <= item['exposure_ev'] <= 2:
            raise ValueError(f'Out-of-range CLID control at frame {index}')
        if not all(0 <= item[key] <= 1 for key in ('left_effective_power', 'right_effective_power')):
            raise ValueError(f'Out-of-range effective power at frame {index}')
        values.append(item)
        by_filename[filename] = index
    return values, by_filename


def show_plot(sequence: Path) -> None:
    values, by_filename = load_control_trace(sequence)
    monitor = FrameOpenMonitor(sequence / 'rgb_0')
    try:
        root = tk.Tk()
    except Exception:
        monitor.close()
        raise
    root.title(f'CLID stereo controls — {sequence.name}')
    root.geometry('960x560')
    canvas = tk.Canvas(root, background='#f8fafc', highlightthickness=0)
    canvas.pack(fill='both', expand=True)
    current = {'frame': None, 'overflow': False}

    def draw(_event=None) -> None:
        canvas.delete('all')
        width, height = max(canvas.winfo_width(), 520), max(canvas.winfo_height(), 360)
        left, right = 85, width - 28
        top1, bottom1 = 75, (height - 65) / 2
        top2, bottom2 = bottom1 + 55, height - 60
        x = lambda index: left + (right - left) * index / max(len(values) - 1, 1)

        def panel(top: float, bottom: float, key: str, low: float, high: float,
                  ticks: tuple[float, ...], color: str, title: str) -> None:
            y = lambda value: bottom - (bottom - top) * (value - low) / (high - low)
            for tick in ticks:
                yy = y(tick)
                canvas.create_line(left, yy, right, yy, fill='#dbe4ec')
                canvas.create_text(left - 8, yy, text=f'{tick:g}', anchor='e', fill='#475569')
            canvas.create_line(left, top, left, bottom, fill='#64748b')
            canvas.create_line(left, bottom, right, bottom, fill='#64748b')
            canvas.create_text(left, top - 18, text=title, anchor='w', fill='#0f172a',
                               font=('TkDefaultFont', 11, 'bold'))
            if len(values) > 1:
                points = [coordinate for index, item in enumerate(values)
                          for coordinate in (x(index), y(item[key]))]
                canvas.create_line(*points, fill=color, width=2, smooth=False)
            frame = current['frame']
            if frame is not None:
                xx, yy = x(frame), y(values[frame][key])
                canvas.create_line(xx, top, xx, bottom, fill='#334155', dash=(4, 4))
                canvas.create_oval(xx - 5, yy - 5, xx + 5, yy + 5, fill=color, outline='#ffffff', width=2)

        panel(top1, bottom1, 'light_commanded', 0, 1, (0, .25, .5, .75, 1),
              '#087e8b', 'Commanded onboard light · fraction [0, 1]')
        panel(top2, bottom2, 'exposure_ev', -2, 2, (-2, -1, 0, 1, 2),
              '#e16428', 'Exposure · EV (stops)')
        canvas.create_text(left, bottom2 + 20, text='0', anchor='w', fill='#475569')
        canvas.create_text(right, bottom2 + 20, text=str(len(values) - 1), anchor='e', fill='#475569')
        canvas.create_text((left + right) / 2, bottom2 + 20, text='Paired frame', fill='#475569')
        frame = current['frame']
        if frame is None:
            label = 'Waiting for SLAM to open a prepared left frame'
        else:
            item = values[frame]
            label = (f'Frame {frame}/{len(values) - 1}   light {item["light_commanded"]:.3f} commanded'
                     f'   exposure {item["exposure_ev"]:+.3f} EV'
                     f'   effective L/R {item["left_effective_power"]:.3f}/{item["right_effective_power"]:.3f}')
        if current['overflow']:
            label += '   WARNING: frame-open events lost'
        canvas.create_text(left, 28, text=label, anchor='w', fill='#0f172a',
                           font=('TkDefaultFont', 11, 'bold'))
        canvas.create_text(left, height - 13,
                           text='Marker follows left image file opens; native SLAM processing may lag or skip frames.',
                           anchor='w', fill='#64748b')

    def poll() -> None:
        names, overflow = monitor.read()
        if overflow:
            current['overflow'] = True
        for name in names:
            if name in by_filename:
                current['frame'] = by_filename[name]
        if names or overflow:
            draw()
        root.after(50, poll)

    def close() -> None:
        monitor.close()
        root.destroy()

    canvas.bind('<Configure>', draw)
    root.protocol('WM_DELETE_WINDOW', close)
    root.update_idletasks()
    draw()
    print('READY', flush=True)
    root.after(50, poll)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', type=Path, help='Prepared CLID stereo sequence')
    args = parser.parse_args()
    show_plot(args.sequence.expanduser().resolve())


if __name__ == '__main__':
    main()
