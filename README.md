# Indian Electoral Roll OCR Pipeline

A Python OCR pipeline for extracting structured voter data from Hindi Indian electoral roll PDFs into CSV files.

The script is designed for electoral roll PDFs where voter entries are laid out in fixed rectangular boxes across multiple columns. It starts extraction from page 3, detects voter boxes dynamically, runs PaddleOCR, extracts fields, and writes one CSV per PDF.

## What it extracts

Each output CSV contains the following columns:

| Column | Description |
|---|---|
| `Serial_Number` | Voter serial number in the roll |
| `EPIC_Number` | EPIC / voter ID number, kept as string |
| `Voter_Name` | Voter name in original Hindi text |
| `Relative_Name` | Father / husband / mother name, where detected |
| `Relation_Type` | `पिता`, `पति`, or `माता` |
| `House_Number` | House number |
| `Age` | Age |
| `Gender` | Gender |

This version intentionally does **not** extract page-level metadata.

## Features

- Batch processes a folder of PDFs.
- Skips the first two pages and starts voter extraction from page 3.
- Dynamically detects three-column voter-entry layouts.
- Dynamically detects voter boxes on each page.
- Uses full-width crops for EPIC and serial extraction.
- Uses body crops and fallback crops for voter field extraction.
- Ignores photo-area text such as `फोटो उपलब्ध है`.
- Removes common OCR field bleed, for example names swallowing `मकान संख्या`, `आयु`, or `लिंग`.
- Preserves Hindi text without translation.
- Writes one CSV per PDF.
- Supports debug mode with per-box crops and OCR text files.

## Repository structure

Recommended structure:

```text
.
├── electoral_roll_pipeline_final.py
├── requirements.txt
├── README.md
├── pdfs/                # input PDFs, not committed
└── output/              # generated CSV/debug output, not committed
```

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it.

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install PaddlePaddle

PaddleOCR 3.x needs PaddlePaddle 3.0 or above. Install either CPU or GPU PaddlePaddle depending on your machine.

For CPU:

```bash
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

For NVIDIA GPU with CUDA 11.8:

```bash
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

For any other CUDA version or OS, use the official PaddlePaddle installation selector.

### 4. Install project requirements

```bash
pip install -r requirements.txt
```

## Usage

Basic GPU run:

```bash
python electoral_roll_pipeline_final.py ./pdfs ./output --device gpu
```

With debug output:

```bash
python electoral_roll_pipeline_final.py ./pdfs ./output --device gpu --debug
```

Custom DPI:

```bash
python electoral_roll_pipeline_final.py ./pdfs ./output --device gpu --dpi 300 --debug
```

Run:

```bash
pip install -r requirements.txt
```

### No PDFs found

Make sure your input folder contains `.pdf` files:

```bash
python electoral_roll_pipeline_final.py ./pdfs ./output --device gpu
```