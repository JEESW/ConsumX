import os
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import whisper
from pyannote.audio import Pipeline


# ===== pyannote diarization 모델 (HF에서 약관 동의 필요) =====
DIAR_MODEL = "pyannote/speaker-diarization-3.1"

# diarization 구간 후처리(합치기) 파라미터
MIN_SEGMENT_SEC = 1.5   # 너무 짧은 구간은 버림
MERGE_GAP_SEC = 1.2     # 같은 화자 구간 사이의 간격이 이보다 작으면 합침
MAX_SEGMENT_SEC = 30.0   # 너무 길면 쪼개기

@dataclass
class Segment:
    start: float
    end: float
    speaker: str

def run_ffmpeg_cut(src: Path, out_wav: Path, start: float, end: float) -> None:
    """
    src에서 [start, end] 구간을 잘라 out_wav로 저장 (16kHz mono wav)
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg cut failed:\nCMD: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}"
        )

def ensure_wav_16k_mono(src: Path, out_wav: Path) -> Path:
    """
    pyannote/torchaudio가 잘 읽도록 src를 16kHz mono wav로 변환.
    이미 wav면 그대로 반환.
    """
    if src.suffix.lower() == ".wav":
        return src

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(out_wav),
    ]
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg convert failed:\nCMD: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}"
        )
        
    return out_wav

def diarize_audio(
    audio_path: Path,
    hf_token: str,
    device: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Segment]:
    """
    pyannote로 화자 분리 후 Segment 리스트 반환
    """
    pipeline = Pipeline.from_pretrained(DIAR_MODEL, use_auth_token=hf_token)

    # 버전에 따라 pipeline.to() 지원 여부가 달라서 try 처리
    try:
        pipeline.to(torch.device(device))
    except Exception:
        pass

    # pyannote는 보통 {"audio": "..."} 형태 입력 권장
    file_dict = {"audio": str(audio_path)}

    # 힌트 kwargs 구성 (없으면 빈 dict -> 자동 추정)
    kwargs = {}

    if num_speakers is not None:
        if num_speakers <= 0:
            raise ValueError("--num-speakers는 1 이상의 정수여야 합니다.")
        kwargs["num_speakers"] = num_speakers
    else:
        # num_speakers가 없을 때만 min/max 적용
        if min_speakers is not None:
            if min_speakers <= 0:
                raise ValueError("--min-speakers는 1 이상의 정수여야 합니다.")
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            if max_speakers <= 0:
                raise ValueError("--max-speakers는 1 이상의 정수여야 합니다.")
            kwargs["max_speakers"] = max_speakers

        if (min_speakers is not None) and (max_speakers is not None) and (min_speakers > max_speakers):
            raise ValueError("--min-speakers는 --max-speakers보다 클 수 없습니다.")

    if kwargs:
        print(f"[INFO] diarization speaker hint: {kwargs}")
    else:
        print("[INFO] diarization speaker hint: (auto)")

    diar = pipeline(file_dict, **kwargs)

    segs: List[Segment] = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        if end - start >= MIN_SEGMENT_SEC:
            segs.append(Segment(start=start, end=end, speaker=speaker))

    return segs


def merge_segments(segments: List[Segment], merge_gap: float = MERGE_GAP_SEC) -> List[Segment]:
    """
    diarization 결과가 잘게 쪼개지므로:
    - 같은 speaker가 연속으로 등장하고
    - gap이 merge_gap 이하이면
    => 하나로 합친다.
    """
    if not segments:
        return []

    segments = sorted(segments, key=lambda s: (s.start, s.end))
    merged: List[Segment] = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]
        if seg.speaker == prev.speaker and seg.start - prev.end <= merge_gap:
            prev.end = max(prev.end, seg.end)
        else:
            merged.append(seg)

    # 합친 뒤에도 너무 짧으면 제거
    merged = [s for s in merged if (s.end - s.start) >= MIN_SEGMENT_SEC]
    return merged


def split_long_segments(segs, max_len=30.0):
    """
    max_len을 기준으로 너무 긴 구간은 자름
    """
    out=[]
    for s in segs:
        cur=s.start
        while s.end-cur > max_len:
            out.append(Segment(cur, cur+max_len, s.speaker))
            cur += max_len
        out.append(Segment(cur, s.end, s.speaker))
    return out


def stt_by_speaker(
    audio_path: Path,
    segments: List[Segment],
    whisper_model: whisper.Whisper,
    device: str,
    language: str,
    out_dir: Path
) -> List[Tuple[str, float, float, str]]:
    """
    (speaker, start, end, text) 형태로 결과 반환
    """
    tmp_dir = out_dir / "_tmp_segments"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = (device == "cuda")
    results: List[Tuple[str, float, float, str]] = []

    for idx, seg in enumerate(segments, 1):
        start, end, speaker = seg.start, seg.end, seg.speaker

        seg_wav = tmp_dir / f"seg_{idx:05d}_{speaker}.wav"
        run_ffmpeg_cut(audio_path, seg_wav, start, end)

        r = whisper_model.transcribe(
            str(seg_wav),
            language=language,
            fp16=use_fp16,
            verbose=False,
            # 더 안정적으로 만들고 싶으면:
            # condition_on_previous_text=False,
            # temperature=0.0,
        )
        text = (r.get("text") or "").strip()
        if text:
            results.append((speaker, start, end, text))

    return results


def write_outputs(out_dir: Path, stt_rows: List[Tuple[str, float, float, str]]) -> None:
    """
    1) 시간순 화자별 대화 로그
    2) 화자별로 모아보기
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 시간순 로그
    timeline_path = out_dir / "transcript_by_speaker_timeline.txt"
    lines = [
        f"{speaker} ({start:.1f}-{end:.1f}) : {text}"
        for speaker, start, end, text in stt_rows
    ]
    timeline_path.write_text("\n".join(lines), encoding="utf-8")

    # 2) 화자별 모아보기
    by_speaker = {}
    for speaker, start, end, text in stt_rows:
        by_speaker.setdefault(speaker, []).append((start, end, text))

    grouped_path = out_dir / "transcript_by_speaker_grouped.txt"
    grouped_lines = []
    for speaker in sorted(by_speaker.keys()):
        grouped_lines.append(f"===== {speaker} =====")
        for start, end, text in by_speaker[speaker]:
            grouped_lines.append(f"({start:.1f}-{end:.1f}) {text}")
        grouped_lines.append("")  # blank line

    grouped_path.write_text("\n".join(grouped_lines).rstrip() + "\n", encoding="utf-8")

    print(f"[DONE] saved:\n- {timeline_path}\n- {grouped_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="pyannote diarization + whisper STT -> speaker-labeled transcript"
    )
    p.add_argument(
        "--audio",
        required=True,
        help="입력 오디오 파일 경로 (예: meeting.wav)"
    )
    p.add_argument(
        "--model",
        default="small",
        help="Whisper 모델명 (tiny/base/small/medium/large)"
    )
    p.add_argument(
        "--lang",
        default="ko",
        help="언어 코드 (예: ko, en)"
    )
    p.add_argument(
        "--out",
        default="outputs",
        help="출력 폴더명 (기본: outputs)"
    )
    p.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="화자 수를 정확히 고정 (예: 3)"
    )
    p.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="최소 화자 수 (예: 2)"
    )
    p.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="최대 화자 수 (예: 5)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # HF 토큰은 환경변수에서만 읽음
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN 환경변수가 없습니다.\n"
            "PowerShell:  $env:HF_TOKEN=\"<토큰>\"\n"
            "CMD:        set HF_TOKEN=<토큰>\n"
        )

    base_dir = Path(__file__).resolve().parent

    audio_path = Path(args.audio)
    if not audio_path.is_absolute():
        audio_path = (base_dir / audio_path).resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {audio_path}")

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()
    out_dir.mkdir(exist_ok=True)

    # GPU 체크
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")
    if device == "cuda":
        print(f"[INFO] gpu = {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] GPU를 못 잡았습니다. CPU로 실행됩니다(느릴 수 있음).")
        
    # pyannote가 m4a 같은 형식을 못 읽는 경우가 있어 wav로 변환해서 사용
    wav_for_diar = out_dir / "input_for_diarization.wav"
    audio_for_diar = ensure_wav_16k_mono(audio_path, wav_for_diar)

    # 1) diarization
    print("[INFO] diarization(pyannote) running...")
    segs = diarize_audio(audio_for_diar, hf_token, device, args.num_speakers, args.min_speakers, args.max_speakers)
    print(f"[INFO] diarization raw segments = {len(segs)}")

    segs = merge_segments(segs)
    print(f"[INFO] diarization merged segments = {len(segs)}")
    
    segs = split_long_segments(segs)
    print(f"[INFO] diarization splited segments = {len(segs)}")

    if not segs:
        raise RuntimeError("diarization 결과 세그먼트가 없습니다. (오디오/모델/환경 확인)")

    # 2) whisper load
    print(f"[INFO] whisper model load: {args.model}")
    w = whisper.load_model(args.model, device=device)

    # 3) STT per speaker segment
    print("[INFO] STT per speaker segments...")
    rows = stt_by_speaker(
        audio_path=audio_path,
        segments=segs,
        whisper_model=w,
        device=device,
        language=args.lang,
        out_dir=out_dir
    )

    # 4) save
    write_outputs(out_dir, rows)

    # preview
    timeline_file = out_dir / "transcript_by_speaker_timeline.txt"
    print("\n=== PREVIEW (first 20 lines) ===")
    for line in timeline_file.read_text(encoding="utf-8").splitlines()[:20]:
        print(line)


if __name__ == "__main__":
    main()