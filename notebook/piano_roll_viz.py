import pretty_midi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np
import os

# 한글 폰트 설정
_korean_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_korean_font):
    plt.rcParams["font.family"] = fm.FontProperties(fname=_korean_font).get_name()
    fm.fontManager.addfont(_korean_font)

MIDI_PATH = "/home/sangheon/Desktop/Pansori_Data/rosvot_midi/01-김소희-춘향가_어사또_방자_만나는_대목_vocal_000_우조.mid"

midi = pretty_midi.PrettyMIDI(MIDI_PATH)

filename = os.path.basename(MIDI_PATH).replace(".mid", "")
total_time = midi.get_end_time()

print(f"파일: {filename}")
print(f"총 길이: {total_time:.2f}초")
print(f"트랙 수: {len(midi.instruments)}")
for i, inst in enumerate(midi.instruments):
    print(f"  트랙 {i}: {inst.name!r}, 노트 수={len(inst.notes)}, program={inst.program}")

fig, ax = plt.subplots(figsize=(18, 6))

colors = plt.cm.tab10.colors

for inst_idx, inst in enumerate(midi.instruments):
    color = colors[inst_idx % len(colors)]
    for note in inst.notes:
        start = note.start
        duration = note.end - note.start
        pitch = note.pitch
        rect = mpatches.FancyBboxPatch(
            (start, pitch - 0.4),
            duration,
            0.8,
            boxstyle="round,pad=0.02",
            linewidth=0.3,
            edgecolor="black",
            facecolor=color,
            alpha=0.85,
        )
        ax.add_patch(rect)

# pitch 범위 계산
all_pitches = [n.pitch for inst in midi.instruments for n in inst.notes]
if all_pitches:
    p_min, p_max = min(all_pitches) - 2, max(all_pitches) + 2
else:
    p_min, p_max = 48, 84

ax.set_xlim(0, total_time)
ax.set_ylim(p_min, p_max)

# Y축: MIDI pitch → 음이름
note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
yticks = list(range(int(p_min), int(p_max) + 1))
ylabels = [f"{note_names[p % 12]}{p // 12 - 1}" for p in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=7)

# 반음 경계선 (흰건반/검은건반 구분용 수평선)
for p in yticks:
    if note_names[p % 12] in ["C", "F"]:  # 옥타브/4도 경계
        ax.axhline(p - 0.5, color="gray", linewidth=0.4, alpha=0.5)

ax.set_xlabel("Time (seconds)", fontsize=12)
ax.set_ylabel("MIDI Pitch", fontsize=12)
ax.set_title(f"Piano Roll — {filename}", fontsize=11, pad=10)
ax.grid(axis="x", linestyle="--", alpha=0.3)

if len(midi.instruments) > 1:
    handles = [
        mpatches.Patch(color=colors[i % len(colors)], label=inst.name or f"Track {i}")
        for i, inst in enumerate(midi.instruments)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

plt.tight_layout()
out_path = "/home/sangheon/Desktop/piano_roll.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n저장 완료: {out_path}")
plt.show()
