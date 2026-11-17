"""Dataset importer and validator for RegGraph gold-standard evaluation facts.

Supports importing from:
- Seed Excel workbook (.xlsx) with zero external dependencies (std library zipfile + XML)
- Canonical JSONL (.jsonl)
- Structured CSV (.csv)
- Raw JSON (.json)
"""
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import xml.etree.ElementTree as ET
import zipfile

from app.core.logging import logger
from evaluation.models import GoldFact, VerificationStatus


GOLD_FACT_CSV_FIELDS = [
    "gold_fact_id",
    "document_title",
    "document_ref",
    "document_id",
    "document_last_updated",
    "section_ref",
    "clause_reference",
    "source_chunk_id",
    "pdf_page",
    "source_clause_text",
    "subject",
    "predicate",
    "object",
    "conditions",
    "exceptions",
    "threshold",
    "time",
    "rule_reference",
    "evidence_text",
    "fact_category",
    "annotation_status",
    "is_verified",
    "review_notes",
]


def _parse_list_field(val: Any) -> List[str]:
    """Parse comma/semicolon/newline-separated strings or existing lists into clean string lists."""
    if not val:
        return []
    if isinstance(val, list):
        return [str(item).strip() for item in val if str(item).strip()]
    val_str = str(val).strip()
    if val_str.lower() in ["none", "none stated", "null", "n/a", ""]:
        return []
    # Split on semicolon or newline
    items = [part.strip() for part in val_str.replace("\n", ";").split(";") if part.strip()]
    return items if items else [val_str]


def _clean_str(val: Any) -> Optional[str]:
    """Clean empty, whitespace, or none-like string values."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ["none", "none stated", "null", "n/a"]:
        return None
    return s


def _determine_verification(status_str: Optional[str], explicit_verified: Optional[Any] = None) -> bool:
    """Determine strict human verification status."""
    if explicit_verified is not None:
        if isinstance(explicit_verified, bool):
            return explicit_verified
        if str(explicit_verified).strip().lower() in ["true", "1", "yes", "verified"]:
            return True

    if not status_str:
        return False

    status_upper = status_str.strip().upper()
    if "SEED" in status_upper or "REVIEW REQUIRED" in status_upper or "UNVERIFIED" in status_upper:
        return False
    if status_upper in ["VERIFIED", "HUMAN_VERIFIED", "APPROVED", "CONFIRMED"]:
        return True

    return False


class DatasetImporter:
    """Imports, validates, and serializes gold facts."""

    @staticmethod
    def read_xlsx(xlsx_path: Union[str, Path], sheet_name: Optional[str] = None) -> List[Dict[str, str]]:
        """Parse an .xlsx file using stdlib zipfile and XML parsing without openpyxl."""
        path = Path(xlsx_path)
        if not path.exists():
            raise FileNotFoundError(f"Workbook not found at {path}")

        with zipfile.ZipFile(path, "r") as z:
            # 1. Read shared strings table if present
            shared_strings: List[str] = []
            if "xl/sharedStrings.xml" in z.namelist():
                ss_tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
                ns_main = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
                for si in ss_tree.findall(f"{ns_main}si"):
                    texts = [t.text for t in si.iter(f"{ns_main}t") if t.text]
                    shared_strings.append("".join(texts))

            # 2. Determine target sheet XML path
            wb_tree = ET.fromstring(z.read("xl/workbook.xml"))
            ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            sheet_map: Dict[str, str] = {}
            for idx, s in enumerate(wb_tree.findall(".//ns:sheet", ns), start=1):
                sheet_map[s.attrib.get("name", f"sheet{idx}")] = f"xl/worksheets/sheet{idx}.xml"

            if sheet_name and sheet_name in sheet_map:
                target_xml = sheet_map[sheet_name]
            elif "Gold_Facts" in sheet_map:
                target_xml = sheet_map["Gold_Facts"]
            elif "sheet2" in [Path(p).stem for p in sheet_map.values()]:
                target_xml = "xl/worksheets/sheet2.xml"
            else:
                target_xml = list(sheet_map.values())[0]

            # 3. Parse row cells
            sheet_root = ET.fromstring(z.read(target_xml))
            rows_data: List[List[str]] = []

            for r in sheet_root.findall(".//ns:row", ns):
                # Map cell column letter to 0-based column index
                row_cells: Dict[int, str] = {}
                max_col = 0
                for c in r.findall("ns:c", ns):
                    coord = c.attrib.get("r", "")
                    col_str = "".join(ch for ch in coord if ch.isalpha())
                    if not col_str:
                        continue
                    # Convert column letter (A, B, AA) to 0-based index
                    col_idx = 0
                    for ch in col_str:
                        col_idx = col_idx * 26 + (ord(ch.upper()) - ord("A") + 1)
                    col_idx -= 1
                    max_col = max(max_col, col_idx)

                    cell_type = c.attrib.get("t")
                    val = ""
                    if cell_type == "s":
                        v_el = c.find("ns:v", ns)
                        if v_el is not None and v_el.text:
                            idx_str = v_el.text.strip()
                            if idx_str.isdigit() and int(idx_str) < len(shared_strings):
                                val = shared_strings[int(idx_str)]
                    elif cell_type == "inlineStr":
                        is_el = c.find("ns:is", ns)
                        if is_el is not None:
                            val = "".join(is_el.itertext())
                    else:
                        v_el = c.find("ns:v", ns)
                        if v_el is not None and v_el.text:
                            val = v_el.text

                    row_cells[col_idx] = val

                row_list = [row_cells.get(i, "") for i in range(max_col + 1)]
                if any(cell.strip() for cell in row_list):
                    rows_data.append(row_list)

            if not rows_data:
                return []

            # 4. Map header row to dictionaries
            headers = [h.strip() for h in rows_data[0]]
            records: List[Dict[str, str]] = []
            for r in rows_data[1:]:
                rec: Dict[str, str] = {}
                for h_idx, h in enumerate(headers):
                    rec[h] = r[h_idx] if h_idx < len(r) else ""
                records.append(rec)

            return records

    @classmethod
    def dict_to_gold_fact(cls, raw: Dict[str, Any]) -> GoldFact:
        """Convert raw dict from Excel/CSV/JSON to normalized GoldFact model."""
        # Field name normalization
        gold_id = str(raw.get("gold_fact_id") or raw.get("sample_id") or "").strip()
        doc_title = str(raw.get("document_title") or "Unknown Document").strip()
        doc_ref = _clean_str(raw.get("document_ref") or raw.get("circular_ref"))
        doc_id = _clean_str(raw.get("document_id"))
        doc_date = _clean_str(raw.get("document_last_updated") or raw.get("document_date"))

        sec_ref = _clean_str(raw.get("section_ref"))
        clause_ref = _clean_str(raw.get("clause_reference") or raw.get("clause_ref") or raw.get("section_ref"))
        chunk_id = _clean_str(raw.get("source_chunk_id") or raw.get("chunk_id"))

        pdf_page_val = raw.get("pdf_page") or raw.get("page_number")
        pdf_page = int(pdf_page_val) if pdf_page_val and str(pdf_page_val).isdigit() else None

        source_clause = _clean_str(raw.get("source_clause_text") or raw.get("clause_text"))

        subject = str(raw.get("subject") or "").strip()
        predicate = str(raw.get("predicate") or "").strip().upper()
        obj = str(raw.get("object") or raw.get("object_or_action") or "").strip()

        conditions = _parse_list_field(raw.get("conditions") or raw.get("conditions_scope"))
        exceptions = _parse_list_field(raw.get("exceptions") or raw.get("exceptions_exclusions"))
        threshold = _clean_str(raw.get("threshold") or raw.get("thresholds_amounts"))
        time = _clean_str(raw.get("time") or raw.get("deadline_timing"))
        rule_ref = _clean_str(raw.get("rule_reference"))
        evidence = _clean_str(raw.get("evidence_text") or raw.get("evidence_span") or raw.get("evidence"))
        category = _clean_str(raw.get("fact_category"))
        status_str = str(raw.get("annotation_status") or "Seed annotation — human review required").strip()

        is_verified = _determine_verification(status_str, raw.get("is_verified"))
        review_notes = _clean_str(raw.get("review_notes"))

        return GoldFact(
            gold_fact_id=gold_id,
            document_title=doc_title,
            document_ref=doc_ref,
            document_id=doc_id,
            document_last_updated=doc_date,
            section_ref=sec_ref,
            clause_reference=clause_ref,
            source_chunk_id=chunk_id,
            pdf_page=pdf_page,
            source_clause_text=source_clause,
            subject=subject,
            predicate=predicate,
            object=obj,
            conditions=conditions,
            exceptions=exceptions,
            threshold=threshold,
            time=time,
            rule_reference=rule_ref,
            evidence_text=evidence,
            fact_category=category,
            annotation_status=status_str,
            is_verified=is_verified,
            review_notes=review_notes,
        )

    @classmethod
    def load_dataset(cls, file_path: Union[str, Path]) -> List[GoldFact]:
        """Load a gold-standard dataset from XLSX, CSV, JSONL, or JSON."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {path}")

        ext = path.suffix.lower()
        facts: List[GoldFact] = []

        if ext == ".xlsx":
            records = cls.read_xlsx(path)
            for r in records:
                if r.get("sample_id") or r.get("gold_fact_id"):
                    facts.append(cls.dict_to_gold_fact(r))
        elif ext == ".csv":
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    facts.append(cls.dict_to_gold_fact(r))
        elif ext == ".jsonl":
            with open(path, mode="r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        facts.append(cls.dict_to_gold_fact(json.loads(line)))
        elif ext == ".json":
            with open(path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        facts.append(cls.dict_to_gold_fact(item))
                elif isinstance(data, dict) and "facts" in data:
                    for item in data["facts"]:
                        facts.append(cls.dict_to_gold_fact(item))
        else:
            raise ValueError(f"Unsupported dataset format: {ext}. Use .xlsx, .csv, or .jsonl")

        logger.info(
            "Loaded %d gold facts from %s (%d verified, %d unverified)",
            len(facts),
            path,
            sum(1 for f in facts if f.is_verified),
            sum(1 for f in facts if not f.is_verified),
        )
        return facts

    @staticmethod
    def save_dataset_jsonl(facts: List[GoldFact], output_path: Union[str, Path]) -> Path:
        """Save list of GoldFact objects to a JSONL file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for fact in facts:
                f.write(fact.model_dump_json() + "\n")
        logger.info("Saved %d gold facts to %s", len(facts), out)
        return out

    @staticmethod
    def save_dataset_csv(facts: List[GoldFact], output_path: Union[str, Path]) -> Path:
        """Save list of GoldFact objects to a CSV file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=GOLD_FACT_CSV_FIELDS)
            writer.writeheader()
            for fact in facts:
                d = fact.model_dump()
                d["conditions"] = "; ".join(fact.conditions) if fact.conditions else ""
                d["exceptions"] = "; ".join(fact.exceptions) if fact.exceptions else ""
                writer.writerow({k: d.get(k) for k in GOLD_FACT_CSV_FIELDS})
        logger.info("Saved %d gold facts to %s", len(facts), out)
        return out

    @staticmethod
    def generate_template_csv(output_path: Union[str, Path]) -> Path:
        """Generate a documented empty CSV template for human annotation."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sample_rows = [
            {
                "gold_fact_id": "SEBI-SURV-EXAMPLE-001",
                "document_title": "Master Circular on Surveillance of Securities Market",
                "document_ref": "HO/43/15/12(3)2025-ISD-POD2/I/11734/2026",
                "document_id": "",
                "document_last_updated": "2026-05-15",
                "section_ref": "1.1",
                "clause_reference": "1.1.1",
                "source_chunk_id": "",
                "pdf_page": 7,
                "source_clause_text": "Trading shall take place in the Trade for Trade (TFT) segment for the first 10 trading days in specified cases.",
                "subject": "Stock exchanges",
                "predicate": "REQUIRES",
                "object": "Place affected securities in the TFT segment",
                "conditions": "Applies to cases listed in 1.1.1–1.1.3",
                "exceptions": "Original scrips on which derivatives are available",
                "threshold": "First 10 trading days",
                "time": "First 10 trading days",
                "rule_reference": "Section 1",
                "evidence_text": "Trading shall take place in Trade for Trade (TFT) segment for first 10 trading days",
                "fact_category": "OBLIGATION",
                "annotation_status": "VERIFIED",
                "is_verified": True,
                "review_notes": "Verified against official PDF Clause 1.1",
            }
        ]
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=GOLD_FACT_CSV_FIELDS)
            writer.writeheader()
            for r in sample_rows:
                writer.writerow(r)
        logger.info("Generated gold dataset template at %s", out)
        return out
