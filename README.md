# Recursive image OCR to CSV

`ocr_images.py` scans a folder and its subfolders, extracts text locally, and writes **one CSV record per image with nonempty recognized text**. English and Turkish are enabled by default.

The CSV contains:

| Column | Content |
| --- | --- |
| `path` | Absolute path to the image, including its filename |
| `file_name` | Filename with extension |
| `extracted_text` | Recognized text, retaining line breaks |

Images with empty or whitespace-only OCR output do not produce records. The file uses UTF-8 with a BOM for spreadsheet compatibility. Commas, quotes, and multiline text are escaped by Python's CSV library. One CSV record can span multiple physical lines when its text contains line breaks.

## Quick start: CPU

Requires Python 3.10+, Pillow, and Tesseract with English and Turkish language data. Run these commands from the directory containing the downloaded script and requirements files.

Create an environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
```

On Windows, use `py -3 -m venv .venv`, then `.venv\Scripts\Activate.ps1` in PowerShell and the same pip command.

Install Tesseract and its language data:

```bash
# macOS with Homebrew; tesseract-lang includes Turkish
brew install tesseract tesseract-lang

# Ubuntu / Debian
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-tur
```

For Windows, follow the [Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html), include Turkish language data, and add Tesseract to PATH. Alternatively, supply its full executable path with `--tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"`.

Confirm that `tesseract --list-langs` lists both `eng` and `tur`, then run:

```bash
# Use all logical CPUs available to the process (also the default)
python ocr_images.py "/path/to/images" -o results.csv --workers auto

# Limit OCR to eight concurrent Tesseract processes
python ocr_images.py "/path/to/images" -o results-8.csv --workers 8
```

CPU mode runs separate Tesseract processes concurrently, coordinated by Python threads. Each Tesseract process is limited to one OpenMP thread to reduce CPU oversubscription. This follows the [Tesseract guidance for processing many images](https://tesseract-ocr.github.io/tessdoc/FAQ.html).

The default language order is `tur+eng`: prioritizing Turkish preserved accented letters better in the included development checks while still recognizing English. Change it with `--languages eng`, `--languages tur`, or another combination of installed Tesseract languages.

## Optional GPU mode

GPU mode uses EasyOCR and PyTorch. It supports NVIDIA CUDA and Apple Silicon MPS when available in the installed PyTorch build. Python 3.12 was used to validate this mode.

For NVIDIA, first install matching `torch` and `torchvision` packages using the command from the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) for your OS and CUDA setup. For Apple Silicon, the standard PyTorch packages can be installed with `python -m pip install torch torchvision`. Then install the OCR dependencies:

```bash
python -m pip install -r requirements-gpu.txt

# Automatically select CUDA, then MPS, then CPU
python ocr_images.py "/path/to/images" -o results-gpu.csv --engine easyocr --device auto

# Explicitly use the Apple Silicon GPU
python ocr_images.py "/path/to/images" -o results-mps.csv --engine easyocr --device mps --batch-size 16

# Explicitly use NVIDIA CUDA
python ocr_images.py "/path/to/images" -o results-cuda.csv --engine easyocr --device cuda --batch-size 32
```

English and Turkish are also the defaults here (`en,tr`). EasyOCR uses different language codes from Tesseract; for example, `--languages en,tr,de` enables English, Turkish, and German. Additional combinations must be compatible with EasyOCR's recognition models. See its [language documentation](https://www.jaided.ai/easyocr/).

GPU cores are scheduled by the GPU runtime, so the script does not expose a GPU-core count. Instead, `--batch-size` controls how many **detected text regions within an image** are recognized together. It does not batch entire image files. One reader stays loaded for the scan; the script does not launch a model process per GPU core. EasyOCR's [API documentation](https://jaided.ai/easyocr/documentation/) explains the batch size and memory tradeoff.

`--workers` is for Tesseract only. In EasyOCR mode, use `--cpu-threads N` to control PyTorch CPU threads, and `--batch-size N` for recognition batching. `--device cpu` is available for comparison, using one EasyOCR reader.

The first EasyOCR run downloads its models into `.ocr-models` in your current directory. Choose a different cache with `--model-dir /path/to/models`. Once the models are cached, use `--offline` to disable model downloads. Images are processed locally and are not uploaded to a service. Explicit `--device cuda` or `--device mps` fails clearly if that device is unavailable; `--device auto` reports when it falls back to CPU.

## Formats and behavior

- Scans common raster image extensions, including PNG, JPEG, TIFF, BMP, WebP, GIF, AVIF, HEIC/HEIF, JPEG 2000, PNM, ICO, PCX, and TGA. Reading depends on the codecs available in Pillow. PDF, SVG, and camera RAW formats are not included.
- For HEIC/HEIF, install the optional decoder with `python -m pip install pillow-heif`.
- Corrects EXIF orientation and composites transparent pixels onto white before OCR.
- Reads all pages/frames of multipage TIFFs and animated images by default. Text from the frames is combined into one record; identical complete frame results are included once. Use `--first-frame` to scan only the first page/frame.
- Discovers files while processing and keeps at most twice the worker count queued. Memory still depends on image dimensions and concurrent workers; it does not load the whole image collection into memory.
- Skips symbolic links, including linked directories. This prevents loops and duplicate scans through aliases.
- Writes and flushes each nonempty result as it finishes. CPU CSV row order follows completion order and can vary between runs.
- Reports unreadable files and OCR failures to stderr, continues scanning, and returns a nonzero exit code. If some frames fail but others produce text, the file still gets one row and the frame failures are reported.
- Refuses to replace an existing CSV unless `--overwrite` is passed. It does not resume or append to earlier results.
- On Ctrl+C, keeps the CSV rows already written and cancels pending CPU work. There is no hard runtime timeout for EasyOCR; GPU cancellation can wait for an in-flight device operation.

Progress reports show processed images, CSV records, no-text images, errors, throughput, and elapsed time. The throughput counter begins after engine initialization/model loading. There is no initial full-folder counting pass.

## Tuning and troubleshooting

| Option | Purpose |
| --- | --- |
| `--workers N` | Concurrent CPU OCR processes; default is all available logical CPUs |
| `--psm 11` | Default Tesseract layout for scattered text in general images |
| `--psm 3` | Alternative Tesseract layout for scanned documents |
| `--psm 6` | Alternative Tesseract layout for a single text block |
| `--timeout 120` | Tesseract time limit in seconds for each frame; excludes image decoding |
| `--max-side 2400` | Resize large images before OCR; reduces memory but may lose small text |
| `--first-frame` | Reduce work for animated/multipage files |
| `--batch-size 16` | EasyOCR recognition batch size; lower it if GPU memory is exhausted |
| `--cpu-threads 4` | PyTorch CPU threads for EasyOCR |
| `--languages ...` | Override the default English/Turkish language set |
| `--quiet` | Suppress script progress; errors and library warnings remain visible |
| `--overwrite` | Replace an existing output CSV |

More workers or a larger GPU batch is not always faster. Compare a representative sample with different worker counts; disk throughput, RAM, image size, and OCR layout all affect speed. GPU startup overhead can outweigh any benefit on small jobs. If memory pressure rises, reduce `--workers` or `--batch-size` before enabling resizing.

OCR can misread characters or detect noise as text. Empty output means the chosen engine recognized no text, not a guarantee that the original image contains none. The script preserves nonempty OCR output without applying a confidence threshold. Pillow's default oversized-image protections remain enabled; unusually large files may be reported as errors.

To retain diagnostics for a large run:

```bash
python ocr_images.py "/path/to/images" -o results.csv --workers 8 2>scan.log
```

Exit codes: `0` = completed without reported errors; `1` = scan completed with one or more file/directory errors; `2` = setup, argument, or fatal output error; `130` = interrupted. An empty input folder succeeds with a header-only CSV.

Run `python ocr_images.py --help` for all options.

## Validation

Validated with real Tesseract OCR on macOS, including serial/parallel agreement, English/Turkish text, blank-image filtering, nested and Unicode paths, CSV quoting, transparent images, EXIF rotation, multipage TIFFs, animated GIFs, a corrupt image, timeouts, output protection, and interruption. Queue bounds and the worker concurrency limit were also checked.

EasyOCR was exercised on an Apple Silicon MPS GPU with the same English/Turkish image set. CUDA support follows the EasyOCR/PyTorch interface; no NVIDIA hardware was available for a CUDA execution test.
