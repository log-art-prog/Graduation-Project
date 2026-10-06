"""真实卫星 IQ 录音 → 包络序列提取管线（路径 B：SatNOGS 数据利用）。

功能：
1. 分块读取大体积 raw IQ 文件（memmap，避免 537MB 全量入内存）
2. 计算包络 |x|，低通滤波 + 抽取到目标包络采样率
3. 若提供 TLE 与地面站坐标，用 skyfield 计算仰角轨迹
4. 按仰角区间切片为定长序列，输出 npz（与 generative 管线格式对齐）

用法示例：
    python scripts/extract_iq_envelope.py \
        --input datasets/real/by70_2/BY02_2020-07-12T113950_436152kHz_192ksps.raw \
        --fs 192000 --start-utc "2020-07-12T11:39:50Z" \
        --lat 40.595865 --lon -3.699069 \
        --tle-file datasets/real/by70_2/by70_2_tle.txt \
        --out datasets/real/by70_2/envelope.npz

注意：BY70-2 已于 2024-08-19 再入，CelesTrak 无在轨 TLE，
历史 TLE 需从 Space-Track.org（免费注册）按 NORAD 45857 + 历元 2020-07 下载。
无 TLE 时脚本仍可提取包络并定长切片，但仰角列填 NaN。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal


def detect_dtype(path: Path, fs: float) -> np.dtype:
    """启发式判定 IQ 存储格式：complex64（SatNOGS/GNU Radio 默认）或 int16 交错。"""
    probe = np.fromfile(path, dtype=np.float32, count=200_000)
    # float32 IQ 典型量级 |x| < 10；int16 误判为 float32 时会出现大量极端值
    if np.isfinite(probe).all() and np.percentile(np.abs(probe), 99.9) < 10:
        return np.complex64
    return np.dtype(np.int16)


def load_iq(path: Path, dtype: np.dtype):
    """memmap 加载 IQ，统一返回 complex64 视图。"""
    if dtype == np.complex64:
        return np.memmap(path, dtype=np.complex64, mode="r")
    raw = np.memmap(path, dtype=np.int16, mode="r")
    iq = raw.reshape(-1, 2).astype(np.float32) / 32768.0
    return iq[:, 0] + 1j * iq[:, 1]


def extract_envelope(iq, fs: float, out_rate: float, chunk: int = 1 << 22):
    """分块取模 → 低通 → 抽取，返回 (envelope, t_axis)。"""
    dec = int(round(fs / out_rate))
    assert dec >= 1 and abs(fs / dec - out_rate) < 1e-6, "fs 必须能被 out_rate 整除"
    # 抗混叠低通：截止 0.45 * out_rate
    taps = signal.firwin(257, 0.45 * out_rate, fs=fs)
    n = len(iq)
    # 滤波需跨块保持状态，简单起见先分块取模再整体滤波（包络已为非负实数，内存可承受）
    mag = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        mag[s:e] = np.abs(iq[s:e])
    filt = signal.filtfilt(taps, 1.0, mag)
    env = filt[::dec][: n // dec]
    t = np.arange(len(env)) / out_rate
    return env, t


def elevation_track(tle_lines, lat, lon, alt_m, start_utc, duration_s, step_s=1.0):
    """用 skyfield 计算仰角轨迹，返回 (t_rel, elev_deg)。"""
    from skyfield.api import EarthSatellite, load, wgs84

    ts = load.timescale()
    lines = [l.rstrip("\n") for l in tle_lines if l.strip()]
    if len(lines) == 3:
        sat = EarthSatellite(lines[1], lines[2], lines[0], ts)
    else:
        sat = EarthSatellite(lines[0], lines[1], None, ts)
    gs = wgs84.latlon(lat, lon, elevation_m=alt_m)
    t_rel = np.arange(0, duration_s, step_s)
    base = np.datetime64(start_utc.replace("Z", ""), "s")
    elev = np.empty(len(t_rel))
    for i, dt in enumerate(t_rel):
        t64 = (base + np.timedelta64(int(round(dt)), "s")).astype(object)
        tt = ts.utc(t64.year, t64.month, t64.day, t64.hour, t64.minute, t64.second)
        alt, _, _ = (sat - gs).at(tt).altaz()
        elev[i] = alt.degrees
    return t_rel, elev


def segment_by_elevation(env, out_rate, seq_len, t_rel, elev, bins):
    """按仰角区间切片。"""
    n_seq = len(env) // seq_len
    env = env[: n_seq * seq_len].reshape(n_seq, seq_len)
    seg_t0 = np.arange(n_seq) * seq_len / out_rate
    if elev is not None:
        seg_elev = np.interp(seg_t0, t_rel, elev)
    else:
        seg_elev = np.full(n_seq, np.nan)
    keep = np.zeros(n_seq, dtype=bool)
    for lo, hi in zip(bins[:-1], bins[1:]):
        keep |= (seg_elev >= lo) & (seg_elev < hi)
    return env[keep], seg_elev[keep]


def main():
    ap = argparse.ArgumentParser(description="SatNOGS IQ 录音 → 包络序列提取")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=192000)
    ap.add_argument("--out-rate", type=float, default=1000, help="包络采样率 Hz")
    ap.add_argument("--seq-len", type=int, default=1000, help="切片长度（与模型输入对齐）")
    ap.add_argument("--dtype", choices=["auto", "complex64", "int16"], default="auto")
    ap.add_argument("--start-utc", default=None, help="录音起始时刻 ISO，如 2020-07-12T11:39:50Z")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--alt-m", type=float, default=0.0)
    ap.add_argument("--tle-file", type=Path, default=None)
    ap.add_argument("--elev-bins", type=float, nargs="+",
                    default=[20, 30, 40, 50, 60, 70, 80, 90])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    dtype = detect_dtype(args.input, args.fs) if args.dtype == "auto" else (
        np.complex64 if args.dtype == "complex64" else np.dtype(np.int16))
    print(f"[1/4] IQ 格式: {dtype}, 文件 {args.input.stat().st_size/1e6:.1f} MB")
    iq = load_iq(args.input, dtype)
    dur = len(iq) / args.fs
    print(f"      采样点 {len(iq):,}，时长 {dur:.1f} s @ {args.fs/1e3:.0f} ksps")

    print(f"[2/4] 包络提取 → {args.out_rate} Hz ...")
    env, t = extract_envelope(iq, args.fs, args.out_rate)
    print(f"      包络点 {len(env):,}，|x| 中位数 {np.median(env):.4f}")

    elev = None
    if args.tle_file and args.start_utc and args.lat is not None:
        print("[3/4] 计算仰角轨迹 ...")
        t_rel, elev = elevation_track(
            args.tle_file.read_text().splitlines(),
            args.lat, args.lon, args.alt_m, args.start_utc, t[-1])
        print(f"      仰角范围 {np.nanmin(elev):.1f}° ~ {np.nanmax(elev):.1f}°")
    else:
        print("[3/4] 未提供 TLE/坐标，跳过仰角（输出列填 NaN）")

    print("[4/4] 切片 ...")
    segs, seg_elev = segment_by_elevation(
        env, args.out_rate, args.seq_len,
        np.arange(0, t[-1], 1.0), elev, args.elev_bins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        envelope=segs.astype(np.float32),
        elevation_deg=seg_elev.astype(np.float32),
        meta=json.dumps({
            "source": args.input.name, "fs_iq": args.fs,
            "out_rate_hz": args.out_rate, "seq_len": args.seq_len,
            "start_utc": args.start_utc, "lat": args.lat, "lon": args.lon,
        }, ensure_ascii=False))
    print(f"      输出 {len(segs)} 条序列 → {args.out}")
    if elev is not None:
        for lo, hi in zip(args.elev_bins[:-1], args.elev_bins[1:]):
            n = int(((seg_elev >= lo) & (seg_elev < hi)).sum())
            print(f"      {lo:.0f}°~{hi:.0f}°: {n} 条")


if __name__ == "__main__":
    main()
