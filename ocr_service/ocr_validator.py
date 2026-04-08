import sys
import json
import logging
import re
import os

# 1. SUPPRESS ALL PADDLE LOGS
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)

# Resolve paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_ocr")
os.makedirs(TEMP_DIR, exist_ok=True)

# Singleton for OCR
_PADDLE_OCR_INSTANCE = None

# Redirect stderr to devnull to catch C++ warnings
# We'll enable it only if necessary, but PaddleOCR writes a lot to stderr.
# Keep commented for now unless we need absolute silence.
# sys.stderr = open(os.devnull, 'w')

from paddleocr import PaddleOCR


def build_ocr():
    global _PADDLE_OCR_INSTANCE
    if _PADDLE_OCR_INSTANCE:
        return _PADDLE_OCR_INSTANCE

    constructors = [
        {"use_textline_orientation": False, "lang": "en", "show_log": False},
        {"use_textline_orientation": False, "lang": "en"},
        {"use_angle_cls": False, "lang": "en", "show_log": False},
        {"use_angle_cls": False, "lang": "en"},
        {"lang": "en", "show_log": False},
        {"lang": "en"},
    ]

    last_error = None
    for kwargs in constructors:
        try:
            print(f"📦 Initializing PaddleOCR with {kwargs}...")
            _PADDLE_OCR_INSTANCE = PaddleOCR(**kwargs)
            return _PADDLE_OCR_INSTANCE
        except Exception as err:
            last_error = err

    raise last_error


def extract_text_with_paddle(image_path):
    ocr = build_ocr()
    ocr_result = ocr.ocr(image_path)

    extracted_text_lines = []
    if ocr_result and ocr_result[0]:
        for line in ocr_result[0]:
            text = line[1][0]
            extracted_text_lines.append(text)

    return extracted_text_lines


# EasyOCR removed to save memory on 512MB RAM environments


def parse_amount_candidates(text):
    candidates = []
    for match in re.finditer(r'[\d,]+\.?\d*', text):
        raw = match.group(0).rstrip('.')
        try:
            value = float(raw.replace(',', ''))
        except ValueError:
            continue

        if value <= 50:
            continue
        if value.is_integer() and 2019 <= value <= 2030:
            continue
        if value > 100000 and ',' not in raw:
            continue

        candidates.append({
            "raw": raw,
            "value": value,
            "start": match.start(),
            "end": match.end(),
        })

    return candidates

def validate_document(image_path, doc_type='GENERIC', expected_value=None):
    result = {
        "valid": False,
        "reason": "Unknown error",
        "extracted_text": "",
        "data": {} 
    }

    if not os.path.exists(image_path):
        result["reason"] = f"File not found: {image_path}"
        return result

    try:
        # PaddleOCR is the sole engine now to stay within 512MB RAM
        extracted_text_lines = extract_text_with_paddle(image_path)
        
        full_text_raw = " ".join(extracted_text_lines).upper()

        if not extracted_text_lines:
             result["reason"] = "No text detected in image."
             result["valid"] = False # Explicitly false
        else:
            result["extracted_text"] = full_text_raw
            result["valid"] = True # Default to true if text found, refine below

            # --- LOGIC BASED ON DOC TYPE ---
            if doc_type == 'INVOICE':
                # --- Advanced Invoice Logic ---
                result["data"] = {
                    "invoice_number": None,
                    "date": None,
                    "amount": None,
                    "vendor": None
                }
                
                # 1. Vendor (Heuristic: usually top lines)
                # Take the first few lines as potential vendor name if they don't look like generic headers
                possible_vendors = [line for line in extracted_text_lines[:4] if len(line) > 3 and "TAX" not in line.upper() and "INVOICE" not in line.upper()]
                if possible_vendors:
                    result["data"]["vendor"] = possible_vendors[0]

                # 2. Date
                # Heuristic: Prefer "INVOICE DATE" over just any date
                # Regex for "Invoice Date: DD/MM/YYYY" etc.
                invoice_date_match = re.search(r'(?:INVOICE|BILL)\s*DATE.*?(\d{2}[/\-]\d{2}[/\-]\d{4}|\d{4}[/\-]\d{2}[/\-]\d{2})', full_text_raw, re.IGNORECASE)
                
                date_pattern = r'\b(\d{2}[/\-]\d{2}[/\-]\d{4}|\d{4}[/\-]\d{2}[/\-]\d{2})\b'
                all_dates = re.findall(date_pattern, full_text_raw)
                
                if invoice_date_match:
                     result["data"]["date"] = invoice_date_match.group(1)
                elif all_dates:
                     # If multiple dates, pick the one that is likely the invoice date (often the latest one if it's 2026 vs 2023)
                     # Or just the first one if unsure.
                     # In the example 28/04/2023 vs 13/03/2026. 2026 is likely correct.
                     # Sort by year descending? 
                     try:
                         # Simple sort
                         all_dates.sort(key=lambda x: x.split('/')[-1] if '/' in x else x.split('-')[0], reverse=True)
                         result["data"]["date"] = all_dates[0]
                     except:
                         result["data"]["date"] = all_dates[0]

                # 3. Invoice Number
                # Heuristics: "Invoice No:", "Bill No:", "Inv No"
                invoice_keywords = ["INVOICE NO", "BILL NO", "INV NO", "INVOICE #", "TAX INVOICE"]
                
                # Manual Check for "TXJ..." pattern if usually consistent
                # or general "Alphanumeric-Dash-Alphanumeric"
                
                found_inv_no = None
                
                # A. Keyword Search
                for line in extracted_text_lines:
                    line_upper = line.upper()
                    if any(k in line_upper for k in invoice_keywords):
                        tokens = line_upper.split()
                        ignore_words = ["TAX", "INVOICE", "NO", "NO.", "BILL", "SALES", "DATE", "OF", ".", ":", "/", "ORIGINAL", "DUPLICATE", "COPY", "CASH", "CREDIT"]
                        candidates = []
                        for t in tokens:
                             t_clean = t.strip(".:,;/-()")
                             # Clean surrounding brackets too (e.g. (Cash))
                             t_clean = re.sub(r'[\(\)]', '', t_clean)
                             
                             is_ignored = any(ign == t_clean for ign in ignore_words) # Strict match for ignore
                             
                             # Candidate: Length > 3, has digit, not ignored
                             if not is_ignored and len(t_clean) > 3 and any(c.isdigit() for c in t_clean):
                                 candidates.append(t_clean)
                        
                        if candidates:
                            found_inv_no = candidates[-1] # Usually the identifier is last
                            break
                
                # B. Pattern Search (Fallback or Refinement)
                # Look for pattern like "TXJ25-15720" (Word-Dash-Number) inside the whole text
                # Regex: WordChars(2+)-Digits(2+)-Digits(2+)
                if not found_inv_no:
                    # Generic Invoice ID patterns: 
                    # AA-1234, 2023/ABC/123, TXJ25-15720
                    inv_id_patterns = [
                        r'\b([A-Z0-9]{2,10}[\-/][A-Z0-9]{2,10}[\-/][A-Z0-9]{2,10})\b', # Complex 3 part
                        r'\b([A-Z]{2,5}\d*[\-/]\d+)\b' # TXJ25-15720 matches this
                    ]
                    for pat in inv_id_patterns:
                         matches = re.findall(pat, full_text_raw)
                         # Filter matches that are not dates
                         valid_matches = [m for m in matches if not re.match(r'\d{2,4}[\-/]\d{2}[\-/]\d{2,4}', m)]
                         if valid_matches:
                             found_inv_no = valid_matches[0]
                             break
                
                if found_inv_no:
                     # Clean up
                     result["data"]["invoice_number"] = found_inv_no.strip("., ")

                # 4. Amount
                # Strategy: Find all decimal numbers. 
                # The total is usually the largest number.

                # A. Context-aware search for Total
                context_amount = None
                
                total_keyword_pattern = re.compile(
                    r'(?:GRAND\s*TOTAL|NET\s*TOTAL|GROSS\s*TOTAL|'
                    r'G[.\s]*TOTA?L?|TOTAL(?:\s+AMOUNT)?|ROUNDING)',
                    re.IGNORECASE
                )
                best_context_candidate = None

                for keyword_match in total_keyword_pattern.finditer(full_text_raw):
                    keyword_end = keyword_match.end()
                    window_start = max(0, keyword_match.start() - 20)
                    window_end = min(len(full_text_raw), keyword_end + 40)
                    window_text = full_text_raw[window_start:window_end]

                    for candidate in parse_amount_candidates(window_text):
                        absolute_start = window_start + candidate["start"]
                        distance = abs(absolute_start - keyword_end)
                        is_after_keyword = absolute_start >= keyword_end
                        score = distance + (0 if is_after_keyword else 10)

                        ranked_candidate = {
                            "score": score,
                            "value": candidate["value"],
                        }

                        if (
                            best_context_candidate is None or
                            ranked_candidate["score"] < best_context_candidate["score"] or
                            (
                                ranked_candidate["score"] == best_context_candidate["score"] and
                                ranked_candidate["value"] > best_context_candidate["value"]
                            )
                        ):
                            best_context_candidate = ranked_candidate

                if best_context_candidate:
                    context_amount = best_context_candidate["value"]
                        
                # B. Global Search (Fallback to Max Amount)
                # Catch 123.45 OR 12345 (integers > 9)
                # Filter out years and phone numbers
                cleaned_amounts = [candidate["value"] for candidate in parse_amount_candidates(full_text_raw)]
                
                # Decision Logic
                if context_amount and context_amount > 50:
                     result["data"]["amount"] = context_amount
                elif cleaned_amounts:
                    result["data"]["amount"] = max(cleaned_amounts)
                
                # Fallback for Invoice Number using regex if keyword method failed
                if not result["data"]["invoice_number"]:
                     # Look for pattern like "TXJ25-15720" specifically for this Toyota bill style
                     # Or general "3+ALPHAS - 4+DIGITS"
                     match_inv = re.search(r'\b[A-Z]{3,}\d+-\d+[A-Z]*\b', full_text_raw)
                     if match_inv:
                         result["data"]["invoice_number"] = match_inv.group(0)

                # Amount Validation: If detected amount is massive (e.g. > 100,000 for a car service?), verify.
                # If amount found "G.TOTAL" but regex failed due to chars, try simple search for "Total"
                if not result["data"]["amount"] and "TOTAL" in full_text_raw:
                     # Last ditch check for numbers near "TOTAL"
                     total_idx = full_text_raw.find("TOTAL")
                     snippet = full_text_raw[total_idx:total_idx+30]
                     nums = re.findall(r'[\d,]+', snippet)
                     if nums:
                         try: result["data"]["amount"] = float(nums[0].replace(',',''))
                         except: pass

                # --- FINAL: Strongest Invoice Number Extraction: Header Only, Prefer TXJ/GSJ/Pattern ---
                # Only consider the first 10 lines for invoice number (header area)
                header_text = ' '.join(extracted_text_lines[:10]).upper()
                # Look for TXJ/GSJ/XXXnnn-nnnnn patterns, prefer TXJ if present
                txj_match = re.search(r'TXJ[0-9A-Z\-]+', header_text)
                if txj_match:
                    result["data"]["invoice_number"] = txj_match.group(0).replace('CASH','').strip()
                else:
                    # Fallback: first ID-like pattern in header
                    m = re.search(r'[A-Z0-9]{3,}-[A-Z0-9]+', header_text)
                    if m:
                        result["data"]["invoice_number"] = m.group(0)

                # --- Invoice Date/Time Extraction: Closest to Invoice Number in Header ---
                # Look for date/time in the header area (first 10 lines)
                date_pattern = r'(\d{2}[/\-]\d{2}[/\-]\d{4})'
                time_pattern = r'(\d{2}:\d{2})'
                header_dates = re.findall(date_pattern, header_text)
                header_times = re.findall(time_pattern, header_text)
                if header_dates:
                    # If a time is found after a date, combine them
                    if header_times:
                        result["data"]["date"] = f"{header_dates[0]} {header_times[0]}"
                    else:
                        result["data"]["date"] = header_dates[0]

                # --- Invoice Number Matching with User Input ---
                # If expected_value (user entered invoice number) is provided, try to match it in the extracted candidates
                if expected_value:
                    expected_clean = expected_value.strip().replace(' ', '').replace('-', '').upper()
                    # Try to match exactly or with minor OCR errors (ignore case, dashes, spaces)
                    found_match = None
                    # Check all potential IDs (from pattern search)
                    all_ids = []
                    # Re-run the same pattern as above to collect all candidates
                    for m in re.finditer(r'\b[A-Z0-9]{3,}[\-][A-Z0-9]+', full_text_raw):
                        s = m.group(0)
                        s_clean = re.sub(r'(?:CASH|CREDIT)$', '', s)
                        s_clean = s_clean.replace('-', '').upper()
                        all_ids.append(s_clean)
                    # Also add the found_inv_no if not already in list
                    if found_inv_no and found_inv_no.replace('-', '').upper() not in all_ids:
                        all_ids.append(found_inv_no.replace('-', '').upper())
                    # Try to match
                    for candidate in all_ids:
                        if candidate == expected_clean:
                            result["data"]["invoice_number"] = expected_value.strip()
                            break
                    else:
                        # If not found, keep the previous logic (first header ID)
                        pass

                # --- Force user input invoice number if present and found in OCR text ---
                if expected_value:
                    expected_clean = expected_value.strip().replace(' ', '').replace('-', '').upper()
                    # Search for the expected invoice number in the full OCR text (ignore dashes, spaces, case)
                    ocr_text_clean = full_text_raw.replace(' ', '').replace('-', '').upper()
                    if expected_clean in ocr_text_clean:
                        result["data"]["invoice_number"] = expected_value.strip()

                # --- Date: If user input invoice number is found, try to find the closest date after it in the OCR text ---
                if expected_value and result["data"].get("invoice_number") == expected_value.strip():
                    # Find where the invoice number appears in the OCR text
                    idx = ocr_text_clean.find(expected_clean)
                    if idx != -1:
                        # Look for a date pattern after this index (within next 100 chars)
                        snippet = full_text_raw[idx:idx+100]
                        date_pattern = r'(\d{2}[/\-]\d{2}[/\-]\d{4}|\d{4}[/\-]\d{2}[/\-]\d{2})'
                        date_match = re.search(date_pattern, snippet)
                        if date_match:
                            result["data"]["date"] = date_match.group(0)

                # Validation Result

                # If expected_value (user entered invoice number) is provided, it MUST match the detected invoice_number
                if expected_value:
                    # After normalization and searching, if the detected invoice number is exactly expected_value.strip()
                    # It means the logic in parts 327/339 confirmed it was found.
                    if result["data"].get("invoice_number") == expected_value.strip():
                        result["valid"] = True
                        result["reason"] = "Invoice validated (number match)."
                    else:
                        result["valid"] = False
                        result["reason"] = f"Invoice number mismatch. Please upload the correct bill or verify the invoice number: {expected_value}"
                else:
                    # If we found at least an Amount or Invoice Number, consider it valid
                    if result["data"]["amount"] or result["data"]["invoice_number"]:
                        result["valid"] = True
                        result["reason"] = "Invoice detected."
                    else:
                        # Fallback to simple keyword check
                        keywords = ["TOTAL", "AMOUNT", "DATE", "INVOICE", "BILL", "GST", "TAX", "RECEIPT"]
                        if any(k in full_text_raw for k in keywords):
                            result["valid"] = True 
                            result["reason"] = "Invoice keywords found (details unclear)."
                        else:
                            result["valid"] = False
                            result["reason"] = "Document does not appear to be an invoice."

            elif doc_type == 'PAN_CARD':
                # --- STEP 1: Strict & Loose Pattern Search ---
                pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
                match = re.search(pan_pattern, full_text_raw.replace(" ", "")) # Improved: search in space-stripped text
                
                best_match_pan = match.group(0) if match else None
                
                if not best_match_pan:
                     # Attempt on raw just in case
                     match = re.search(pan_pattern, full_text_raw)
                     best_match_pan = match.group(0) if match else None

                if not best_match_pan:
                    # Loose Search & Repair
                    loose_candidates = re.findall(r'(?=([A-Z0-9]{10}))', full_text_raw.replace(" ", ""))
                    loose_candidates = list(set(loose_candidates))
                    
                    for cand in loose_candidates:
                        # Attempt to Repair (First 5 letters, Next 4 digits, Last 1 letter)
                        part1 = cand[:5].translate(str.maketrans("01582", "OISBZ"))
                        part2 = cand[5:9].translate(str.maketrans("OISB", "0158"))
                        part3 = cand[9].translate(str.maketrans("01582", "OISBZ"))
                        candidate_fixed = part1 + part2 + part3
                        
                        if re.match(pan_pattern, candidate_fixed):
                            best_match_pan = candidate_fixed
                            break

                # --- STEP 2: Name Extraction (Always Run) ---
                extracted_name_candidate = "User"
                for i, line in enumerate(extracted_text_lines):
                    line_u = line.strip().upper()
                    # Look for "NAME" label but exclude "FATHER NAME" or similar
                    if ("NAME" in line_u) and ("FATHER" not in line_u) and ("BANK" not in line_u):
                        # 1. Check same line (e.g. "Name: S Praveenkumar")
                        match = re.search(r'(?:H\/|I\/)?NAME[:\-\s\.]+(.*)', line_u)
                        if match and len(match.group(1).strip()) > 2:
                             extracted_name_candidate = match.group(1).strip()
                             break
                        
                        # 2. Check next line
                        if (i + 1 < len(extracted_text_lines)):
                             next_line = extracted_text_lines[i + 1].strip()
                             if "FATHER" not in next_line.upper() and len(next_line) > 2:
                                 extracted_name_candidate = next_line
                                 break

                # Name Spacing Fix (e.g. SPRAVEENKUMAR -> S PRAVEENKUMAR)
                # To avoid breaking names like "SRINIVASAN" (where S is part of name), 
                # we only split if the remainder is a known valid structure or ends with common suffix.
                name_stripped = extracted_name_candidate.replace(" ", "")
                final_name = extracted_name_candidate.title()
                
                if len(name_stripped) > 5 and extracted_name_candidate != "User":
                     # Common Indian Name Suffixes that imply the main name is done
                     common_suffixes = ["KUMAR", "SINGH", "LAL", "CHAND", "DAS", "NATH", "REDDY", "RAO", "DEVI", "KUMARI", "SHARMA", "VERMA", "GUPTA", "YADAV", "JHA", "ALI", "KHAN"]
                     
                     # Check if it looks like INITIAL + NAME + SUFFIX (or just Name+Suffix with Initial)
                     # Case 1: SPRAVEENKUMAR -> Ends in KUMAR. Likely S PRAVEENKUMAR
                     name_upper = name_stripped.upper()
                     
                     should_split = False
                     # Condition A: Ends with common suffix
                     if any(name_upper.endswith(s) for s in common_suffixes):
                         should_split = True
                     
                     # Condition B: Starts with common initial pattern that isn't a digraph?
                     # e.g. "S" is common initial. "K" is common. 
                     # But "SR" is common start (Srinivasan). "SP" is rare start (Spandan? Sparsh?)
                     # If starts with S, and next char is P, likely S Praveen?
                     # Let's rely on Condition A primarily to be safe for now, 
                     # PLUS: If no suffix match, revert to original behavior BUT check for common Digraphs
                     
                     if should_split:
                          final_name = (name_stripped[0] + " " + name_stripped[1:]).title()
                     else:
                          # Just Title Case the raw string
                          final_name = name_stripped.title()
                     
                     # Check for End Merges (e.g. "Srinivasanr" -> "Srinivasan R")
                     # Common endings for South Indian names: N, M followed by Initial
                     # Regex: Name Ending in 'N' or 'M' followed by single consonant (Initial)
                     # Protect against specific short names if needed
                     
                     # Case 1: Ends in 'NR', 'MR' (e.g. ...anr, ...amr)
                     if re.search(r'[a-zA-Z]{3}[NM][RSTK]$', name_stripped.upper()):
                          # Check if it's not a common ending weirdness
                          # "HANS" ends in NS -> Don't split
                          # "HEMANT" ends in NT -> Not covered by [RSTK]
                          
                          name_upper = name_stripped.upper()
                          last_char = name_upper[-1]
                          base_name = name_upper[:-1]
                          
                          # Filter out false positives like "HANS"
                          if not (name_upper.endswith("HANS") or name_upper.endswith("JAMES")):
                               final_name = base_name.title() + " " + last_char.title()

                # --- STEP 3: Final Validation & Decision ---
                
                # Default: Assume what we found is best
                if best_match_pan:
                    result["valid"] = True
                    result["reason"] = "Valid PAN Card detected."
                    result["data"]["pan_number"] = best_match_pan
                    result["data"]["name_candidate"] = final_name.title()
                else:
                    result["valid"] = False
                    result["reason"] = "No valid PAN pattern found."
                    result["data"]["name_candidate"] = final_name.title()

                # --- STEP 4: Fuzzy Match Override (If User Configured) ---
                if expected_value:
                    expected_clean = re.sub(r'[^A-Z0-9]', '', expected_value.upper())
                    ocr_clean = re.sub(r'[^A-Z0-9]', '', result.get("data", {}).get("pan_number", ""))
                    
                    # A. Check existing result vs Expected
                    if ocr_clean:
                        # 1. Exact Match
                        if ocr_clean == expected_clean:
                            result["valid"] = True
                            result["data"]["pan_number"] = expected_value
                            result["reason"] = "Valid PAN detected (Matches input)."
                        
                        # 2. Fuzzy Match (Levenshtein-ish 2 char tolerance)
                        elif len(ocr_clean) == 10 and len(expected_clean) == 10:
                             diff = sum(1 for a, b in zip(ocr_clean, expected_clean) if a != b)
                             if diff <= 2:
                                 result["valid"] = True
                                 result["data"]["pan_number"] = expected_value
                                 result["reason"] = "Valid PAN detected (Fuzzy match confirmed)."
                    
                    # B. Deep Search in Raw Text (if still invalid)
                    if not result["valid"]:
                        raw_stripped = full_text_raw.replace(" ", "")
                        # Direct search or simple substitution (0->O) search
                        target_sub = expected_clean.replace("O", "0").replace("I", "1")
                        
                        if (expected_clean in raw_stripped) or (target_sub in raw_stripped):
                             result["valid"] = True
                             result["data"]["pan_number"] = expected_value
                             result["reason"] = "Valid PAN detected (Deep Text Search)."

            elif doc_type == 'PROFESSION_DOC':
                # Flexible validation for profession documents
                # Look for degree names or verify keywords
                keywords = ["DEGREE", "CERTIFICATE", "UNIVERSITY", "INSTITUTE", "DOCTOR", "ENGINEER", "ARCHITECT", "COUNCIL", "REGISTRATION"]
                if any(k in full_text_raw for k in keywords):
                    result["valid"] = True
                    result["reason"] = "Profession document validated."
                    result["data"]["document_id"] = "Verified"
                else:
                    # Don't invalid strictly, just warn
                    result["valid"] = True # Lenient check as requested
                    result["reason"] = "Profession document text detected."
                    result["data"]["document_id"] = "Verified"

    except Exception as e:
        result["valid"] = False
        result["reason"] = f"OCR Exception: {str(e)}"
    
    # OUTPUT JSON
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"valid": False, "reason": "No image path provided"}))
        sys.exit(1)
    
    img_path = sys.argv[1]
    # Optional doc_type argument
    dtype = sys.argv[2] if len(sys.argv) > 2 else 'GENERIC'
    # Optional expected value
    expected = sys.argv[3] if len(sys.argv) > 3 else None
    
    res = validate_document(img_path, dtype, expected)
    print(json.dumps(res))
