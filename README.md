# 🎓 CampusLens

## AI-Assisted College Notice-to-Action Assistant

CampusLens transforms college notices, circulars, announcements, scholarship notices, event notices, and other printed documents into clear, actionable information for students.

Instead of making students search through a lengthy notice, CampusLens helps answer:

- 👤 **WHO** is affected?
- ✅ **WHAT** action is required?
- 📅 **WHEN** does it need to happen?
- 📋 **WHAT** documents or requirements are needed?
- 🧾 **WHY** was this information extracted?

---

## 🚨 Problem

College students regularly receive important information through printed notices, circulars, announcements, and other documents.

These notices may contain:

- Important deadlines
- Eligibility requirements
- Required documents
- Registration instructions
- Scholarship information
- Examination information
- Event instructions

The important information can be difficult to find quickly.

---

## 💡 Solution

CampusLens uses OCR and AI-assisted information extraction to convert an unstructured college notice into a structured, reviewable action checklist.

### Processing Pipeline

```text
📄 Notice / Document
        ↓
🔎 OCR Text Extraction
        ↓
🏷️ Document Classification
        ↓
📅 Date & Deadline Detection
        ↓
✅ Action Extraction
        ↓
📋 Requirement Extraction
        ↓
🧾 Supporting Evidence
        ↓
⚠️ Human Verification