"""Script to generate realistic synthetic regulatory circular PDFs using PyMuPDF."""
from pathlib import Path
import fitz  # PyMuPDF


def generate_2022_circular(output_path: Path) -> Path:
    """Generate initial 2022 circular with 12 illustrative activities."""
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page(width=595, height=842)  # A4
    text_page1 = """NATIONAL STOCK EXCHANGE OF INDIA LIMITED
Department : COMPLIANCE
Download Ref No: NSE/COMP/50957
Circular Ref. No: 02/2022
Date : January 07, 2022

To All Members,

Sub: Clarification to Rule 8(1)(f) of the Securities Contracts (Regulation) Rules, 1957 - Illustrative Activities

Dear Members,

Attention of members is drawn to Rule 8(1)(f) of the Securities Contracts (Regulation) Rules, 1957 regarding permissible and non-permissible activities for Trading Members of the Exchange.

The Exchange has compiled an illustrative list of prohibited and restricted business activities in Annexure A.

Annexure A
Illustrative List of Activities

Point No. 1: Financing securities transactions of clients directly or indirectly without regulatory clearance.
Point No. 2: Sharing client brokerage or commission with unregistered entities or unauthorized intermediaries.
Point No. 3: Engaging in proprietary commodity arbitrage without maintaining separate earmarked capital adequacy.
Point No. 4: Operating offshore branch offices without prior written intimation to the Exchange.
Point No. 5: Providing margin loans at discretionary undisclosed interest rates.
Point No. 6: Facilitating unapproved fund routing through non-banking financial entities.
"""
    page1.insert_text((50, 60), text_page1, fontsize=10, fontname="helv")

    # Page 2
    page2 = doc.new_page(width=595, height=842)
    text_page2 = """NATIONAL STOCK EXCHANGE OF INDIA LIMITED
Download Ref No: NSE/COMP/50957
Circular Ref. No: 02/2022

Point No. 7: Conducting client portfolio management schemes without obtaining SEBI PMS registration.
Point No. 8: Entering into bilateral derivative swaps with unhedged retail counterparties.
Point No. 9: Accepting cash deposits or bearer instruments towards trading margins.
Point No. 10: Issuing guaranteed return advisory services to equity investors.
Point No. 11: Delegating core KYC and anti-money laundering verification to third-party marketing agents.
Point No. 12: Providing unapproved algorithmic trading execution without exchange system audits.

All members are advised to take note of the above and ensure strict compliance.

For and on behalf of
National Stock Exchange of India Limited

Compliance Officer
"""
    page2.insert_text((50, 60), text_page2, fontsize=10, fontname="helv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()
    return output_path


def generate_2026_circular(output_path: Path) -> Path:
    """Generate 2026 amending circular modifying Points 1 & 2, deleting Points 3-12, and adding FAQs."""
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page(width=595, height=842)
    text_page1 = """NATIONAL STOCK EXCHANGE OF INDIA LIMITED
Department : COMPLIANCE
Download Ref No: NSE/COMP/62114
Circular Ref. No: 15/2026
Date : June 23, 2026

To All Members,

Sub: Amendment to Circular NSE/COMP/50957 dated January 07, 2022 regarding Illustrative Activities under Rule 8(1)(f)

Dear Members,

This circular is issued with reference to our earlier circular NSE/COMP/50957 dated January 07, 2022 wherein the Exchange had specified illustrative prohibited activities under Rule 8(1)(f).

Based on representations received from Trading Members and subsequent regulatory deliberations, the provisions of Circular NSE/COMP/50957 are hereby amended as follows:

1. Point No. 1 is amended to permit structured securities lending with prior exchange authorization.
2. Point No. 2 is modified to require registered affiliate verification prior to brokerage sharing.
3. Point No. 3 to 12 stand deleted with immediate effect.

The revised provisions shall come into force with effect from June 23, 2026.
"""
    page1.insert_text((50, 60), text_page1, fontsize=10, fontname="helv")

    # Page 2
    page2 = doc.new_page(width=595, height=842)
    text_page2 = """NATIONAL STOCK EXCHANGE OF INDIA LIMITED
Download Ref No: NSE/COMP/62114
Circular Ref. No: 15/2026

Annexure B
Frequently Asked Questions (FAQs)

FAQ on Rule 8 Clarifications

Question No. 1: Does Rule 8 permit member affiliates to share co-location server racks?
Answer: Yes, member affiliates are permitted to co-locate provided physical and logical network separation is maintained.

Question No. 2: What is the effective timeline for deleting Points 3 to 12?
Answer: Points 3 to 12 stand deleted immediately from June 23, 2026 and members need not submit quarterly returns against those clauses.

For and on behalf of
National Stock Exchange of India Limited

Chief Compliance Officer
"""
    page2.insert_text((50, 60), text_page2, fontsize=10, fontname="helv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()
    return output_path


if __name__ == "__main__":
    base_dir = Path(__file__).parent / "sample_data"
    doc1 = generate_2022_circular(base_dir / "circular_2022_50957.pdf")
    doc2 = generate_2026_circular(base_dir / "circular_2026_amendment.pdf")
    print(f"Generated sample PDFs:\n1. {doc1}\n2. {doc2}")
