#!/usr/bin/env python3
"""Recursively OCR images into a CSV, omitting files with no recognized text.

CPU: python ocr_images.py ./images -o text.csv --workers auto
GPU: python ocr_images.py ./images -o text.csv --engine easyocr --device auto

Requires Python 3.10+ and Pillow. CPU mode also needs the Tesseract executable.
GPU mode needs EasyOCR and a compatible PyTorch installation. See README.md.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator


EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".tif", ".tiff", ".bmp",
    ".dib", ".webp", ".gif", ".avif", ".heic", ".heif", ".jp2", ".j2k",
    ".ppm", ".pgm", ".pbm", ".pnm", ".ico", ".pcx", ".tga",
})
CSV_FIELDS = ["path", "file_name", "extracted_text"]


def cpu_count() -> int:
    """Respect CPU affinity on platforms that expose it."""
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return number


def worker_count(value: str) -> int:
    return cpu_count() if value.lower() == "auto" else positive_int(value)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folder", type=Path, help="folder to scan recursively")
    parser.add_argument("-o", "--output", type=Path, default=Path("ocr_results.csv"))
    parser.add_argument("--engine", choices=["tesseract", "easyocr"], default="tesseract")
    parser.add_argument("--workers", type=worker_count, default=None,
                        metavar="N|auto", help="parallel Tesseract processes; default: all available CPUs")
    parser.add_argument("--languages", default=None,
                        help="Tesseract: tur+eng; EasyOCR: en,tr; default: English + Turkish")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true", help="replace an existing output CSV")
    output_mode.add_argument("--resume", action="store_true",
                             help="append to an existing CSV, skipping paths already recorded")
    parser.add_argument("--first-frame", action="store_true",
                        help="read only the first page/frame of TIFFs, GIFs, etc.")
    parser.add_argument("--max-side", type=positive_int, default=None,
                        help="optionally shrink large images to this longest edge in pixels")
    parser.add_argument("--quiet", action="store_true", help="suppress progress; still print errors")
    cpu = parser.add_argument_group("Tesseract CPU settings")
    cpu.add_argument("--tesseract", default="tesseract", help="executable name or full path")
    cpu.add_argument("--psm", type=int, choices=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], default=11,
                     help="page segmentation mode: 11=sparse text, 3=documents, 6=single text block")
    cpu.add_argument("--timeout", type=positive_float, default=120.0,
                     help="maximum Tesseract runtime per page/frame, in seconds")
    gpu = parser.add_argument_group("EasyOCR settings")
    gpu.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                     help="auto selects CUDA, then Apple MPS, then CPU")
    gpu.add_argument("--batch-size", type=positive_int, default=4,
                     help="number of detected text regions per recognition batch")
    gpu.add_argument("--gpu-restart-every", type=positive_int, default=100,
                     help="refresh the GPU process after this many image files")
    gpu.add_argument("--gpu-memory-limit-mb", type=positive_int, default=4096,
                     help="refresh the GPU process if idle memory exceeds this soft limit (MiB)")
    gpu.add_argument("--cpu-threads", type=positive_int, default=min(4, cpu_count()),
                     help="PyTorch CPU threads used by EasyOCR")
    gpu.add_argument("--model-dir", type=Path, default=Path(".ocr-models"),
                     help="EasyOCR model cache, downloaded on first use")
    gpu.add_argument("--offline", action="store_true",
                     help="disable EasyOCR model downloads; use an existing cache")
    args = parser.parse_args(argv)
    if args.engine == "easyocr" and args.workers is not None:
        parser.error("--workers is for Tesseract; EasyOCR uses --batch-size and --cpu-threads")
    return args


def initialize_images() -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is missing. Install it with: python -m pip install Pillow") from exc
    # Load decoders once before starting threads.
    Image.init()
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        pass  # HEIC/HEIF files will be reported as unreadable unless this is installed.
    else:
        register_heif_opener()


def image_paths(folder: Path, on_error: Callable[[OSError], None]) -> Iterator[Path]:
    """Discover lazily; skip symlinks to avoid loops and escaping the input tree."""
    for directory, subdirs, names in os.walk(folder, onerror=on_error, followlinks=False):
        subdirs[:] = [name for name in subdirs if not Path(directory, name).is_symlink()]
        for name in names:
            path = Path(directory, name)
            if path.suffix.lower() not in EXTENSIONS or path.is_symlink():
                continue
            try:
                if path.is_file():
                    yield path
            except OSError as exc:
                on_error(exc)


def prepared_frames(path: Path, args: argparse.Namespace) -> Iterator:
    """Orient and flatten one frame at a time; never load a whole image collection."""
    from PIL import Image, ImageOps, ImageSequence

    with Image.open(path) as source:
        for frame in ImageSequence.Iterator(source):
            prepared = ImageOps.exif_transpose(frame)
            try:
                if "A" in prepared.getbands() or "transparency" in prepared.info:
                    rgba = prepared.convert("RGBA")
                    try:
                        rgb = Image.new("RGB", rgba.size, "white")
                        rgb.paste(rgba, mask=rgba.getchannel("A"))
                    finally:
                        rgba.close()
                else:
                    rgb = prepared.convert("RGB")
                try:
                    if args.max_side:
                        rgb.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
                    yield rgb
                finally:
                    rgb.close()
            finally:
                prepared.close()
            if args.first_frame:
                break


class Tesseract:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.processes: set[subprocess.Popen] = set()
        self.executable = shutil.which(args.tesseract)
        if not self.executable:
            raise RuntimeError("Tesseract executable not found. Install Tesseract or supply --tesseract PATH.")
        self.environment = dict(os.environ, OMP_THREAD_LIMIT="1", OMP_NUM_THREADS="1")
        codes = (args.languages or "tur+eng").replace(",", "+").split("+")
        self.languages = "+".join(code.strip() for code in codes)
        available = subprocess.run(
            [self.executable, "--list-langs"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=self.environment, timeout=30, check=True,
        ).stdout.splitlines()
        missing = set(self.languages.split("+")) - set(line.strip() for line in available)
        if missing:
            raise RuntimeError("Tesseract language data not installed: " + ", ".join(sorted(missing)))
        self.description = f"Tesseract; {args.workers or cpu_count()} parallel processes; languages={self.languages}"

    def read(self, frame) -> str:
        # PPM avoids spending CPU on PNG compression just to hand an image to OCR.
        data = io.BytesIO()
        frame.save(data, format="PPM")
        command = [self.executable, "stdin", "stdout", "-l", self.languages,
                   "--psm", str(self.args.psm)]
        with self.lock:
            if self.stopped.is_set():
                raise InterruptedError("scan cancelled")
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, env=self.environment)
            self.processes.add(process)
        try:
            try:
                stdout, stderr = process.communicate(data.getvalue(), timeout=self.args.timeout)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.communicate()
                raise TimeoutError(f"Tesseract exceeded {self.args.timeout:g}s for this frame") from exc
            if process.returncode:
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(detail or f"Tesseract exited with code {process.returncode}")
            return stdout.decode("utf-8", errors="replace")
        finally:
            with self.lock:
                self.processes.discard(process)

    def cancel(self) -> None:
        with self.lock:
            self.stopped.set()
            for process in self.processes:
                try:
                    process.kill()
                except OSError:
                    pass  # It may have exited between completion and cancellation.


class GPUMemoryError(RuntimeError):
    """Request process-level recovery instead of treating OOM as an ordinary file error."""


class EasyOCR:
    def __init__(self, args: argparse.Namespace):
        try:
            import torch
            import easyocr
        except ImportError as exc:
            raise RuntimeError("EasyOCR is missing. Install GPU requirements; see README.md.") from exc
        self.torch = torch
        self.args = args
        self.stopped = threading.Event()
        torch.set_num_threads(args.cpu_threads)
        cuda = torch.cuda.is_available()
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        device = args.device
        if device == "auto":
            device = "cuda" if cuda else "mps" if mps else "cpu"
        if (device == "cuda" and not cuda) or (device == "mps" and not mps):
            raise RuntimeError(f"Requested {device} device is unavailable in this PyTorch installation.")
        if args.device == "auto" and device == "cpu":
            print("No supported GPU found; EasyOCR is using CPU. Tesseract supports parallel CPU jobs.", file=sys.stderr)
        self.device = device
        self.frames_read = 0
        model_dir = args.model_dir.expanduser().resolve()
        languages = [code.strip() for code in (args.languages or "en,tr").replace("+", ",").split(",")]
        self.reader = easyocr.Reader(
            languages, gpu=False if device == "cpu" else device,
            model_storage_directory=str(model_dir),
            user_network_directory=str(model_dir / "user_network"),
            download_enabled=not args.offline, quantize=(device == "cpu"), verbose=False,
        )
        self.description = f"EasyOCR; device={device}; recognition batch={args.batch_size}; languages={','.join(languages)}"

    def _recognize(self, frame) -> str:
        import numpy as np

        # EasyOCR expects BGR for color arrays. Grayscale avoids a channel-order ambiguity.
        with frame.convert("L") as gray:
            pixels = np.array(gray)
        with self.torch.inference_mode():
            pieces = self.reader.readtext(pixels, detail=0, paragraph=False,
                                          batch_size=self.args.batch_size, workers=0)
        return "\n".join(pieces)

    def read(self, frame) -> str:
        try:
            # Return from inference before collecting: its tensor locals must be out of scope.
            text = self._recognize(frame)
            self.frames_read += 1
            if self.frames_read % 25 == 0:
                gc.collect()
            if self.device == "mps":
                self.torch.mps.synchronize()
                self.torch.mps.empty_cache()
            elif self.device == "cuda" and self.frames_read % 25 == 0:
                self.torch.cuda.synchronize()
                self.torch.cuda.empty_cache()
            return text
        except RuntimeError as exc:
            if self.device != "cpu" and "out of memory" in str(exc).lower():
                raise GPUMemoryError(str(exc)) from None
            raise

    def memory_bytes(self) -> int:
        if self.device == "mps":
            return self.torch.mps.driver_allocated_memory()
        if self.device == "cuda":
            return self.torch.cuda.memory_reserved()
        return 0

    def cancel(self) -> None:
        self.stopped.set()


@dataclass
class Result:
    path: Path
    text: str
    errors: list[str]


def extract(path: Path, engine, args: argparse.Namespace) -> Result:
    texts: list[str] = []
    seen: set[str] = set()
    errors: list[str] = []
    try:
        for index, frame in enumerate(prepared_frames(path, args), start=1):
            if engine.stopped.is_set():
                break
            try:
                text = engine.read(frame).replace("\x00", "").strip()
                if text and text not in seen:
                    texts.append(text)
                    seen.add(text)
            except GPUMemoryError:
                raise
            except Exception as exc:
                errors.append(f"frame {index}: {type(exc).__name__}: {exc}")
    except GPUMemoryError:
        raise
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return Result(path, "\n\n".join(texts), errors)


def easyocr_worker(connection, args: argparse.Namespace) -> None:
    """Own all GPU state in a disposable process. Send only paths and text over IPC."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # The parent handles Ctrl+C and terminates us.
    try:
        initialize_images()
        engine = EasyOCR(args)
        connection.send({"kind": "ready", "device": engine.device, "description": engine.description})
        while True:
            path = connection.recv()
            if path is None:
                return
            try:
                result = extract(path, engine, args)
                memory = engine.memory_bytes()
            except GPUMemoryError as exc:
                connection.send({"kind": "oom", "error": str(exc)})
                return  # Exiting also releases Metal graph/driver allocations, not just tensor caches.
            connection.send({"kind": "result", "result": result, "memory_bytes": memory})
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({"kind": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class EasyOCRProcess:
    """Sequential file dispatch with bounded GPU lifetime and bounded OOM retries."""
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.process = None
        self.connection = None
        self.files_in_worker = 0
        self.last_memory_bytes = 0
        self.peak_idle_memory_bytes = 0
        self.restart_count = 0
        self.batch_size = args.batch_size
        self.device = args.device
        self.refresh_reason = ""
        self._start(self.device)

    def _start(self, device: str) -> None:
        worker_args = argparse.Namespace(**vars(self.args))
        worker_args.device = device
        worker_args.batch_size = self.batch_size
        context = multiprocessing.get_context("spawn")  # Never fork an initialized GPU runtime.
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(target=easyocr_worker, args=(child_connection, worker_args), daemon=True)
        try:
            self.process.start()
            child_connection.close()
            ready = self._receive()
            if ready["kind"] != "ready":
                raise RuntimeError(ready.get("error", "OCR worker did not initialize"))
            self.device = ready["device"]
            self.description = ready["description"]
            self.files_in_worker = 0
            self.refresh_reason = ""
        except BaseException:
            child_connection.close()
            self.cancel()
            raise

    def _receive(self) -> dict:
        try:
            while not self.connection.poll(0.25):
                if not self.process.is_alive():
                    raise RuntimeError(f"OCR worker exited unexpectedly (code {self.process.exitcode})")
            message = self.connection.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            raise RuntimeError("OCR worker disconnected; completed CSV rows have been kept") from exc
        if message["kind"] == "fatal":
            raise RuntimeError(message["error"])
        return message

    def cancel(self) -> None:
        if self.process is not None:
            if self.process.pid is not None:
                if self.process.is_alive():
                    self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join()
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _restart(self, device: str) -> None:
        self.cancel()  # Wait for the old process to die before allocating the next model.
        self.restart_count += 1
        self._start(device)

    def _request(self, path: Path) -> Result:
        self.connection.send(path)
        reply = self._receive()
        if reply["kind"] == "oom":
            raise GPUMemoryError(reply["error"])
        if reply["kind"] != "result":
            raise RuntimeError("Unexpected response from OCR worker")
        self.files_in_worker += 1
        self.last_memory_bytes = reply["memory_bytes"]
        self.peak_idle_memory_bytes = max(self.peak_idle_memory_bytes, self.last_memory_bytes)
        if self.device != "cpu":
            if self.last_memory_bytes >= self.args.gpu_memory_limit_mb * 1024**2:
                self.refresh_reason = f"idle GPU memory reached {self.last_memory_bytes / 1024**2:.0f} MiB"
            elif self.files_in_worker >= self.args.gpu_restart_every:
                self.refresh_reason = f"{self.files_in_worker} images processed"
        return reply["result"]

    def extract(self, path: Path) -> Result:
        if self.process is None:
            self._start(self.device)
        elif self.refresh_reason:
            if not self.args.quiet:
                print(f"Refreshing GPU worker: {self.refresh_reason}.", file=sys.stderr, flush=True)
            self._restart(self.device)
        gpu_device = self.device
        try:
            return self._request(path)
        except GPUMemoryError:
            # Leave the exception scope before retrying so traceback references can be freed.
            pass
        print(f"GPU memory recovery: {path}; retrying in a fresh {gpu_device} process with batch size 1.",
              file=sys.stderr, flush=True)
        self.batch_size = 1  # Retain the lower batch size for the remaining scan.
        self._restart(gpu_device)
        try:
            return self._request(path)
        except GPUMemoryError:
            pass
        print(f"GPU memory recovery: {path}; retrying this image with EasyOCR on CPU.",
              file=sys.stderr, flush=True)
        self._restart("cpu")
        try:
            return self._request(path)
        finally:
            self.cancel()
            self.device = gpu_device  # The following image gets a fresh GPU process.


def results(paths: Iterable[Path], engine, args: argparse.Namespace) -> Iterator[Result]:
    if args.engine == "easyocr":
        try:
            for path in paths:
                yield engine.extract(path)
        finally:
            engine.cancel()
        return

    workers = args.workers or cpu_count()
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = set()
    paths = iter(paths)
    exhausted = False
    try:
        while pending or not exhausted:
            # Bound pending work instead of queuing the entire folder in memory.
            while not exhausted and len(pending) < workers * 2:
                try:
                    path = next(paths)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(pool.submit(extract, path, engine, args))
            if pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
    finally:
        engine.cancel()
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def read_existing_csv(output: Path) -> tuple[set[str], int]:
    """Validate before appending, including records whose OCR text exceeds csv's default limit."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    paths: set[str] = set()
    count = 0
    with output.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(f"Cannot resume: CSV header must be {CSV_FIELDS}")
        for row in reader:
            if (None in row or any(row.get(key) is None for key in CSV_FIELDS)
                    or not Path(row["path"]).is_absolute() or not row["file_name"]
                    or not row["extracted_text"].strip()):
                raise ValueError(f"Cannot resume: incomplete or invalid CSV record near line {reader.line_num}")
            paths.add(os.path.normcase(row["path"]))
            count += 1
    return paths, count


def run(args: argparse.Namespace) -> int:
    folder = args.folder.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Input is not a directory: {folder}")
    if output.suffix.lower() != ".csv":
        raise ValueError("Output must end in .csv")
    if output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"Output already exists: {output}. Use another path, --resume, or --overwrite.")
    recorded, saved = read_existing_csv(output) if args.resume else (set(), 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    initialize_images()
    if not args.quiet:
        print(f"Preparing {args.engine}...", file=sys.stderr, flush=True)
    engine = Tesseract(args) if args.engine == "tesseract" else EasyOCRProcess(args)
    count = empty = failed = scan_errors = skipped = 0
    started = last_report = time.monotonic()

    def scan_error(exc: OSError) -> None:
        nonlocal scan_errors
        scan_errors += 1
        print(f"SCAN ERROR: {exc}", file=sys.stderr, flush=True)

    def progress() -> None:
        elapsed = max(time.monotonic() - started, 0.001)
        memory = (f" | GPU idle={engine.last_memory_bytes / 1024**2:.0f} MiB"
                  if isinstance(engine, EasyOCRProcess) and engine.device != "cpu" else "")
        print(f"Processed={count:,} | CSV rows={saved:,} | no text={empty:,} | "
              f"files with errors={failed:,} | scan errors={scan_errors:,} | "
              f"already saved={skipped:,} | {count / elapsed:.2f} images/s | {elapsed:.1f}s{memory}",
              file=sys.stderr, flush=True)

    def pending_paths() -> Iterator[Path]:
        nonlocal skipped
        for path in image_paths(folder, scan_error):
            if os.path.normcase(str(path)) in recorded:
                skipped += 1
            else:
                yield path

    if not args.quiet:
        print(f"{engine.description}\nScanning: {folder}\nCSV: {output}", file=sys.stderr, flush=True)
    stream = results(pending_paths(), engine, args)
    interrupted = False
    # utf-8-sig helps spreadsheet applications detect Unicode. csv handles embedded newlines/quotes.
    mode = "a" if args.resume else "w" if args.overwrite else "x"
    try:
        needs_newline = False
        if args.resume:
            with output.open("rb") as original:
                original.seek(-1, os.SEEK_END)
                needs_newline = original.read(1) not in (b"\n", b"\r")
        with output.open(mode, newline="", encoding="utf-8" if args.resume else "utf-8-sig") as handle:
            writer = csv.writer(handle)
            if not args.resume:
                writer.writerow(CSV_FIELDS)
            elif needs_newline:
                handle.write("\n")
            handle.flush()
            try:
                for result in stream:
                    count += 1
                    if result.errors:
                        failed += 1
                        for error in result.errors:
                            print(f"OCR ERROR: {result.path}: {error}", file=sys.stderr, flush=True)
                    if result.text:
                        writer.writerow([str(result.path), result.path.name, result.text])
                        handle.flush()
                        saved += 1
                    elif not result.errors:
                        empty += 1
                    if not args.quiet and time.monotonic() - last_report >= 5:
                        progress()
                        last_report = time.monotonic()
            except KeyboardInterrupt:
                interrupted = True
                print("Interrupted; keeping CSV rows already written.", file=sys.stderr, flush=True)
            finally:
                stream.close()
    finally:
        engine.cancel()
    if not args.quiet:
        progress()
    if interrupted:
        return 130
    return 1 if failed or scan_errors else 0


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
