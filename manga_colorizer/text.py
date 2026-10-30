# SPDX-License-Identifier: GPL-3.0-only
"""Locate text, then copy original pixels; OCR never rewrites the wording."""
import csv
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import cv2
import numpy as np


def reviewed_regions(path):
    document = json.loads(Path(path).read_text())
    result = {}
    for entry in document['images'].values():
        if entry['sha256'] in result and result[entry['sha256']] != entry['regions']:
            raise ValueError('Conflicting reviewed regions for the same source checksum')
        result[entry['sha256']] = entry['regions']
    return result


def check_ocr(language):
    executable = shutil.which('tesseract')
    if not executable:
        raise RuntimeError('New pages need Tesseract OCR. On macOS run: brew install tesseract')
    result = subprocess.run([executable, '--list-langs'], capture_output=True, text=True, check=True, timeout=15)
    installed = set(result.stdout.splitlines()[1:])
    if not set(language.split('+')).issubset(installed):
        raise RuntimeError(f'Tesseract language {language!r} is not installed')
    return executable


def detect_regions(original, raw, language='eng'):
    executable = check_ocr(language)
    buffer = io.BytesIO();original.save(buffer, format='PNG')
    gray = np.asarray(original.convert('L'));rgb = np.asarray(raw)
    height, width = gray.shape;regions = [];seen = set()
    for mode in [3,11]:
        result = subprocess.run([executable,'stdin','stdout','-l',language,'--psm',str(mode),'tsv'],
                                input=buffer.getvalue(), capture_output=True, check=True, timeout=120)
        grouped = {}
        for word in csv.DictReader(io.StringIO(result.stdout.decode('utf-8','replace')), delimiter='\t'):
            token = (word.get('text') or '').strip()
            if word['level'] != '5' or float(word['conf']) < 45 or not re.search('[A-Za-z0-9]',token):
                continue
            key = tuple(word[k] for k in ['block_num','par_num','line_num'])
            grouped.setdefault(key,[]).append(word)
        for words in grouped.values():
            token = ' '.join(word['text'] for word in words)
            letters = len(re.findall('[A-Za-z0-9]',token))
            confidence = float(np.mean([float(word['conf']) for word in words]))
            if (letters < 2 and token not in ('I','A')) or (letters < 4 and confidence < 75):
                continue
            x = min(int(word['left']) for word in words);y = min(int(word['top']) for word in words)
            right = max(int(word['left'])+int(word['width']) for word in words)
            bottom = max(int(word['top'])+int(word['height']) for word in words)
            line_height = bottom-y
            if line_height < 6 or line_height > height*.06:continue
            pad = max(6,round(line_height*.5))
            box = (max(0,x-pad),max(0,y-pad),min(width,right+pad),min(height,bottom+pad))
            left,top,right,bottom = box
            white = gray[top:bottom,left:right] >= 230
            if white.size == 0 or white.mean() < .5 or white.sum() < 30:continue
            colors = rgb[top:bottom,left:right][white].astype(np.int16)
            if np.median(colors.mean(1)) < 210 or np.median(colors.max(1)-colors.min(1)) > 26:continue
            if box not in seen:
                seen.add(box);regions.append({'box':list(box),'confidence':confidence})
    return regions


def region_mask(original, regions):
    """Restore white halos/letter holes; keep gray artwork between floating words."""
    gray = np.asarray(original.convert('L'));height,width = gray.shape
    mask = np.zeros_like(gray,np.uint8)
    for region in regions:
        left,top,right,bottom = map(int,region['box'])
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError('Text region lies outside the source image')
        white = (gray[top:bottom,left:right] >= 230).astype(np.uint8)
        contours,_ = cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(white);cv2.drawContours(filled,contours,-1,255,cv2.FILLED)
        filled = cv2.dilate(filled,np.ones((3,3),np.uint8),iterations=1)
        mask[top:bottom,left:right] |= filled
    return mask


def restore_text(original, ink, raw, regions=None, language='eng'):
    started = time.perf_counter()
    reviewed = regions is not None
    if regions is None:regions = detect_regions(original,raw,language)
    mask = region_mask(original,regions)
    chosen = mask > 0
    source = np.asarray(original);base = np.asarray(ink)
    result = base.copy();result[chosen] = source[chosen]
    if not np.array_equal(result[~chosen],base[~chosen]) or not np.array_equal(result[chosen],source[chosen]):
        raise ValueError('Text restoration changed pixels outside its mask')
    return result, mask, {'method':'reviewed' if reviewed else 'ocr', 'regions':regions,
        'restored_pixels':int(chosen.sum()), 'restored_fraction':float(chosen.mean()),
        'seconds':time.perf_counter()-started,
        'scope':'Reviewed English dialogue/captions and selected small effects; logos, Japanese and oversized effects excluded.' if reviewed else 'Automatic English text detection; review masks because some text can be missed.'}
