---
layout: "post"
show-avatar: false
title: "DCM-Parser: A effecient tool to extract dicom meta data from files"
date: "2023-03-11"
tags: [DICOM, tools]
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/dcm_parser.png
---

![GitHub Liscence](https://img.shields.io/github/license/PL97/DICOM_Parser)
![PyPi Download](https://img.shields.io/pypi/dm/DCM-parser)
![Pypi version](https://img.shields.io/pypi/v/DCM-parser)

Source: [GitHub](https://github.com/PL97/DICOM_Parser)

## DCM-Parser
A effecient tool for extract dicom meta data from files.

This tool offers:
- convert dicom to png (if pixel array is 2D) / mp4 (3D)
- extract all meta data (dicom header) to a single csv file
- examize corrupted files and files with missing values
- multi-thread speed up exact HUNDREDS files in seconds

## Installation
```bash
pip install -r requirements.txt
pip install dcm-parser
```


## Usage

**ONE line for everything!**

```python
from dcm_parser import batch_extraction

batch_extraction(source_dir=[YOUR_SOURCE_DIR], dest_dir=[YOUR_TARGET_DIR], save_img=True)

```

## How to cite this work

If you find this git repo useful, please consider citing it using the snippet below:

```bibtex
@misc{DCM-parser,
    author={Le Peng},
    title={DCM-parser: A Lightning DICOM Parser},
    howpublished={\url{https://github.com/PL97/DICOM_Parser}},
    year={2022}
}
```
