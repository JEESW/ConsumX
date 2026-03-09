import os
import gc
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline


def seconds_to_hhmmss(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3_600_000
    m = (total_ms % 3_600_000) // 60_000
    s = (total_ms % 60_000) / 1000.0
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def normalize_path(base_dir: Path, path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_speaker_args(
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> None:
    if num_speakers is not None and num_speakers <= 0:
        raise ValueError("--num-speakers는 1 이상의 정수여야 합니다.")
    if min_speakers is not None and min_speakers <= 0:
        raise ValueError("--min-speakers는 1 이상의 정수여야 합니다.")
    if max_speakers is not None and max_speakers <= 0:
        raise ValueError("--max-speakers는 1 이상의 정수여야 합니다.")
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise ValueError("--min-speakers는 --max-speakers보다 클 수 없습니다.")
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        print("[WARN] --num-speakers가 지정되면 --min-speakers/--max-speakers는 무시됩니다.")


def pick_compute_type(device: str, user_compute_type: Optional[str]) -> str:
    if user_compute_type:
        return user_compute_type

    if device == "cuda":
        return "float16"
    return "int8"


def extract_timeline_rows(result: Dict[str, Any]) -> List[Tuple[str, float, float, str]]:
    rows: List[Tuple[str, float, float, str]] = []

    for seg in result.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker") or "SPEAKER_UNKNOWN"

        if text:
            rows.append((speaker, start, end, text))

    rows.sort(key=lambda x: (x[1], x[2]))
    return rows


def write_outputs(
    out_dir: Path,
    result: Dict[str, Any],
    rows: List[Tuple[str, float, float, str]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 시간순 화자별 로그
    timeline_path = out_dir / "transcript_by_speaker_timeline.txt"
    timeline_lines = [
        f"{speaker} ({start:.1f}-{end:.1f}) : {text}"
        for speaker, start, end, text in rows
    ]
    timeline_path.write_text("\n".join(timeline_lines), encoding="utf-8")

    # 2) 화자별로 모아보기
    grouped_path = out_dir / "transcript_by_speaker_grouped.txt"
    by_speaker: Dict[str, List[Tuple[float, float, str]]] = {}
    for speaker, start, end, text in rows:
        by_speaker.setdefault(speaker, []).append((start, end, text))

    grouped_lines: List[str] = []
    for speaker in sorted(by_speaker.keys()):
        grouped_lines.append(f"===== {speaker} =====")
        for start, end, text in by_speaker[speaker]:
            grouped_lines.append(f"({start:.1f}-{end:.1f}) {text}")
        grouped_lines.append("")

    grouped_path.write_text("\n".join(grouped_lines).rstrip() + "\n", encoding="utf-8")

    # 3) WhisperX raw JSON 저장
    json_path = out_dir / "whisperx_result.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 4) 좀 더 보기 쉬운 상세 로그
    detailed_path = out_dir / "transcript_detailed.txt"
    detailed_lines: List[str] = []
    for idx, seg in enumerate(result.get("segments", []), 1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        speaker = seg.get("speaker") or "SPEAKER_UNKNOWN"
        text = (seg.get("text") or "").strip()

        detailed_lines.append(
            f"[{idx:04d}] {speaker} | {seconds_to_hhmmss(start)} ~ {seconds_to_hhmmss(end)}"
        )
        detailed_lines.append(text)

        words = seg.get("words") or []
        if words:
            detailed_lines.append("  - words:")
            for w in words:
                w_start = w.get("start")
                w_end = w.get("end")
                w_word = (w.get("word") or "").strip()
                w_speaker = w.get("speaker")

                span = ""
                if w_start is not None and w_end is not None:
                    span = f"{float(w_start):.2f}-{float(w_end):.2f}"

                if w_speaker:
                    detailed_lines.append(f"    * [{span}] ({w_speaker}) {w_word}")
                else:
                    detailed_lines.append(f"    * [{span}] {w_word}")

        detailed_lines.append("")

    detailed_path.write_text("\n".join(detailed_lines).rstrip() + "\n", encoding="utf-8")

    print("[DONE] saved:")
    print(f"- {timeline_path}")
    print(f"- {grouped_path}")
    print(f"- {json_path}")
    print(f"- {detailed_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WhisperX diarization + alignment + speaker-labeled transcript"
    )
    p.add_argument(
        "--audio",
        required=True,
        help="입력 오디오 파일 경로 (예: meeting.wav, meeting.m4a)"
    )
    p.add_argument(
        "--model",
        default="small",
        help="WhisperX ASR 모델명 (예: tiny, base, small, medium, large-v2, large-v3)"
    )
    p.add_argument(
        "--lang",
        default=None,
        help="언어 코드 (예: ko, en). 지정하지 않으면 자동 감지"
    )
    p.add_argument(
        "--out",
        default="outputs",
        help="출력 폴더명 (기본: outputs)"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="배치 크기 (GPU 메모리 부족 시 줄이기)"
    )
    p.add_argument(
        "--compute-type",
        default=None,
        help="연산 타입 (예: float16, int8, int8_float16). 미지정 시 device 기준 자동 선택"
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
    p.add_argument(
        "--no-diarize",
        action="store_true",
        help="화자 분리를 끄고 순수 STT + alignment만 수행"
    )
    p.add_argument(
        "--device",
        default=None,
        help="실행 디바이스 강제 지정 (cuda / cpu)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    validate_speaker_args(args.num_speakers, args.min_speakers, args.max_speakers)

    base_dir = Path(__file__).resolve().parent
    audio_path = normalize_path(base_dir, args.audio)
    out_dir = normalize_path(base_dir, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {audio_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = pick_compute_type(device, args.compute_type)

    print(f"[INFO] device = {device}")
    if device == "cuda":
        print(f"[INFO] gpu = {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] GPU를 못 잡았습니다. CPU로 실행됩니다(느릴 수 있음).")
    print(f"[INFO] compute_type = {compute_type}")

    hf_token = os.environ.get("HF_TOKEN")
    if not args.no_diarize and not hf_token:
        raise RuntimeError(
            "화자 분리를 사용하려면 HF_TOKEN 환경변수가 필요합니다.\n"
            "PowerShell:  $env:HF_TOKEN=\"<토큰>\"\n"
            "CMD:         set HF_TOKEN=<토큰>\n"
            "또한 Hugging Face에서 pyannote diarization 모델 약관 동의가 필요합니다."
        )

    # 1) 오디오 로드
    print("[INFO] loading audio...")
    audio = whisperx.load_audio(str(audio_path))

    # 2) ASR
    print(f"[INFO] loading whisperx model: {args.model}")
    asr_model = whisperx.load_model(
        args.model,
        device=device,
        compute_type=compute_type,
        language=args.lang,
    )

    print("[INFO] transcribing...")
    result = asr_model.transcribe(
        audio,
        batch_size=args.batch_size,
    )

    print(f"[INFO] detected language = {result.get('language')}")
    print(f"[INFO] raw segments = {len(result.get('segments', []))}")

    # ASR 모델 메모리 해제
    del asr_model
    cleanup_cuda()

    # 3) Alignment
    language_code = args.lang or result.get("language")
    if not language_code:
        raise RuntimeError("언어를 판별하지 못했습니다. --lang 옵션을 직접 지정해보세요.")

    print(f"[INFO] loading align model for language = {language_code}")
    align_model, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=device,
    )

    print("[INFO] aligning...")
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    print(f"[INFO] aligned segments = {len(result.get('segments', []))}")

    # align 모델 메모리 해제
    del align_model
    cleanup_cuda()

    # 4) Diarization
    if not args.no_diarize:
        print("[INFO] diarization running...")
        diarize_model = DiarizationPipeline(
            token=hf_token,
            device=device,
        )

        diar_kwargs = {}
        if args.num_speakers is not None:
            diar_kwargs["num_speakers"] = args.num_speakers
        else:
            if args.min_speakers is not None:
                diar_kwargs["min_speakers"] = args.min_speakers
            if args.max_speakers is not None:
                diar_kwargs["max_speakers"] = args.max_speakers

        if diar_kwargs:
            print(f"[INFO] diarization speaker hint = {diar_kwargs}")
        else:
            print("[INFO] diarization speaker hint = (auto)")

        diarize_segments = diarize_model(audio, **diar_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # diarization 모델 메모리 해제
        del diarize_model
        cleanup_cuda()
    else:
        print("[INFO] diarization skipped (--no-diarize)")

    # 5) 저장
    rows = extract_timeline_rows(result)
    if not rows:
        raise RuntimeError("최종 결과 세그먼트가 없습니다. 오디오/모델/언어 설정을 확인하세요.")

    write_outputs(out_dir, result, rows)

    # 6) preview
    timeline_file = out_dir / "transcript_by_speaker_timeline.txt"
    print("\n=== PREVIEW (first 20 lines) ===")
    for line in timeline_file.read_text(encoding="utf-8").splitlines()[:20]:
        print(line)


if __name__ == "__main__":
    main()