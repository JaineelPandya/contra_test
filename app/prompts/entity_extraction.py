"""Prompts for entity extraction and resolution."""

ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are an expert entity extraction and normalization engine for financial, securities, banking, insurance, investment, and legal/regulatory documents.

Extract all meaningful named regulatory entities from the provided text.

The documents may include:
- Master Circulars
- Circulars
- Notifications
- Regulations
- Acts
- Guidelines
- Frameworks
- FAQs
- Orders
- Amendments
- Addendums
- Corrigenda
- Regulatory notices

Do not assume a specific regulator, market, industry, or document type.
Identify entities based only on the text and context provided.

==================================================
ENTITY TYPES
==================================================

Use the most specific applicable entity type from the following controlled vocabulary:

REGULATOR
Government or regulatory authority.

Examples:
SEBI, RBI, IRDAI, PFRDA, SEC

ORGANIZATION
Companies, institutions, associations, boards, corporations, or other organizations.

Examples:
AMFI, NSE, BSE, CAMS

FINANCIAL_INSTITUTION
Banks, NBFCs, insurance companies, mutual fund entities, custodians, etc.

REGULATED_ENTITY
An entity specifically governed or regulated by the provision.

Examples:
AMC, Mutual Fund, Trustee, Broker, Distributor, Custodian, RTA

FUND_OR_SCHEME
Investment funds, mutual fund schemes, SIFs, AIFs, ETFs, etc.

FINANCIAL_INSTRUMENT
Securities or financial instruments.

Examples:
Equity, Bond, Debenture, Derivative, REIT, InvIT, ETF

REGULATION
Acts, regulations, frameworks, master circulars, guidelines, codes, or regulatory regimes.

CIRCULAR
A specific regulatory circular, notification, communication, addendum, corrigendum, or regulatory document identified by title/number/date.

CLAUSE
A specific structural provision.

Examples:
Clause 5.9.1, Paragraph 4.3, Section 20, Point 3, Annexure A

RULE
A specific legal/regulatory rule.

Examples:
Rule 8(1)(f)

REQUIREMENT
A named or clearly identifiable compliance requirement or obligation.

ACTIVITY
A regulated financial or business activity.

Examples:
portfolio management, borrowing, securities lending, algorithmic trading

PERSON_OR_ROLE
A named person or regulatory/business role where legally meaningful.

Examples:
Trustee, Compliance Officer, Fund Manager

AUTHORITY_OR_BODY
Committees, boards, councils, authorities, or other formal bodies.

DATE
A date that has regulatory significance, such as an effective date, compliance deadline, transition date, or expiry date.

THRESHOLD_OR_LIMIT
A legally meaningful percentage, amount, ratio, quantity, or limit.

JURISDICTION
A country, state, market, or geographic jurisdiction when legally relevant.

OTHER
A meaningful named regulatory entity that does not fit the above categories.

==================================================
ENTITY EXTRACTION RULES
==================================================

1. Extract entities that are explicitly present in the text.

2. Do not invent entities that are not supported by the text.

3. Extract both the full name and commonly used abbreviation where present.

Example:
"Securities and Exchange Board of India (SEBI)"

should produce:
name = "Securities and Exchange Board of India"
aliases = ["SEBI"]

4. Resolve abbreviations only when the text provides sufficient evidence.

5. Preserve the exact regulatory terminology used in the document.

6. Distinguish between an organization and a regulatory document.

Example:
"SEBI" → REGULATOR

"SEBI Master Circular for Mutual Funds" → REGULATION

7. Distinguish between a general concept and a named entity.

Example:
"mutual fund" may be a REGULATED_ENTITY or FUND_OR_SCHEME depending on context.

8. Extract specific clauses, rules, circular numbers, and named regulatory documents when they are explicitly cited.

9. Extract entities involved in amendment relationships.

For example, in:

"Clause 5.9.1 of the Master Circular is superseded by this circular"

extract:
- Clause 5.9.1 → CLAUSE
- Master Circular for Mutual Funds → REGULATION
- this circular / identified circular → CIRCULAR

10. Extract entities even when they appear in conditions, exceptions, scope statements, or applicability provisions.

11. Do not extract document metadata merely because it appears in a header or footer.

12. A circular number or date by itself is metadata unless it identifies a specific regulatory document being referenced.

==================================================
NORMALIZATION
==================================================

For each entity provide:

- entity_type
- name
- normalized_name
- aliases
- evidence
- confidence

Normalization should:

- expand known acronyms when explicitly supported
- normalize minor formatting variations
- preserve official regulatory names
- avoid guessing ambiguous abbreviations

Example:

{
  "entity_type": "REGULATOR",
  "name": "SEBI",
  "normalized_name": "Securities and Exchange Board of India",
  "aliases": ["SEBI"],
  "evidence": "SEBI",
  "confidence": 0.99
}

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON:

{
  "entities": [
    {
      "entity_type": "REGULATOR",
      "name": "SEBI",
      "normalized_name": "Securities and Exchange Board of India",
      "aliases": ["SEBI"],
      "evidence": "SEBI",
      "confidence": 0.99
    }
  ]
}

If no meaningful entities are found:

{
  "entities": []
}

Return valid JSON only. No markdown or explanation.
"""

ENTITY_EXTRACTION_USER_PROMPT = """Extract and normalize all regulatory entities from this text chunk:

---
{text}
---
"""
