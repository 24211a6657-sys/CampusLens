# ============================================================
# CampusLens — College Notice-to-Action Assistant
# SIH26188
# ============================================================

import json
import re
from datetime import datetime

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image


# ============================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CampusLens | Notice-to-Action Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        background: linear-gradient(
            135deg,
            #172554 0%,
            #1e3a8a 50%,
            #2563eb 100%
        );
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .hero h1 {
        margin-bottom: 0.3rem;
        font-size: 2.3rem;
    }

    .hero p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.92;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
        min-height: 110px;
    }

    .metric-title {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #172554;
    }

    .section-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }

    .small-note {
        color: #64748b;
        font-size: 0.85rem;
    }

    .review-box {
        padding: 1rem;
        border-radius: 12px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
    }

    .success-box {
        padding: 1rem;
        border-radius: 12px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
    }

    .warning-box {
        padding: 1rem;
        border-radius: 12px;
        background: #fffbeb;
        border: 1px solid #fde68a;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. TESSERACT CONFIGURATION
# ============================================================

WINDOWS_TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT_PATH


# ============================================================
# 4. TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Clean OCR output while preserving useful information.
    """

    if not text:
        return ""

    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def get_lines(text: str):
    """
    Return meaningful non-empty lines.
    """

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


# ============================================================
# 5. OCR PREPROCESSING
# ============================================================

def preprocess_image(image: Image.Image):
    """
    Prepare image for OCR.
    """

    img = np.array(image.convert("RGB"))

    # RGB -> grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Upscale small documents
    height, width = gray.shape

    if width < 1200:
        scale = 1200 / width
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    # Mild denoising
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return processed


# ============================================================
# 6. OCR EXECUTION
# ============================================================

def perform_ocr(image: Image.Image):
    """
    Perform OCR using Tesseract.

    Two OCR attempts are made:
    1. Original image
    2. Preprocessed image

    The longer meaningful result is returned.
    """

    try:
        # OCR on original image
        original_text = pytesseract.image_to_string(
            image,
            config="--psm 6",
        )

        # OCR on processed image
        processed = preprocess_image(image)

        processed_text = pytesseract.image_to_string(
            processed,
            config="--psm 6",
        )

        original_text = normalize_text(original_text)
        processed_text = normalize_text(processed_text)

        # Choose the result containing more usable text
        if len(processed_text) > len(original_text):
            final_text = processed_text
        else:
            final_text = original_text

        return final_text

    except Exception as exc:
        return f"__OCR_ERROR__:{exc}"


# ============================================================
# 7. DOCUMENT CLASSIFICATION
# ============================================================

def classify_document(text: str) -> str:
    """
    Rule-based document classification.
    """

    lower = text.lower()

    categories = {
        "Scholarship / Financial Aid": [
            "scholarship",
            "financial aid",
            "fee reimbursement",
            "tuition fee",
            "freeship",
        ],
        "Examination / Academic": [
            "examination",
            "exam",
            "internal",
            "semester",
            "mid",
            "assessment",
            "hall ticket",
        ],
        "Event / Workshop": [
            "workshop",
            "seminar",
            "event",
            "conference",
            "hackathon",
            "webinar",
            "registration",
        ],
        "Placement / Career": [
            "placement",
            "campus recruitment",
            "job",
            "internship",
            "career",
            "recruitment",
        ],
        "Holiday / Schedule": [
            "holiday",
            "working day",
            "timetable",
            "schedule",
            "academic calendar",
        ],
        "General College Notice": [
            "notice",
            "circular",
            "announcement",
            "students are informed",
            "all students",
        ],
    }

    scores = {}

    for category, keywords in categories.items():
        score = sum(
            1 for keyword in keywords
            if keyword in lower
        )
        scores[category] = score

    best_category = max(
        scores,
        key=scores.get,
    )

    if scores[best_category] == 0:
        return "Unknown document"

    return best_category


# ============================================================
# 8. AUDIENCE EXTRACTION
# ============================================================

def extract_audience(text: str) -> str:
    lower = text.lower()

    audience_patterns = [
        (r"\b(all students)\b", "All students"),
        (r"\b(b\.?tech students)\b", "B.Tech students"),
        (r"\b(m\.?tech students)\b", "M.Tech students"),
        (r"\b(ug students)\b", "Undergraduate students"),
        (r"\b(pg students)\b", "Postgraduate students"),
        (r"\b(final year students)\b", "Final-year students"),
        (r"\b(second year students)\b", "Second-year students"),
        (r"\b(first year students)\b", "First-year students"),
        (r"\b(third year students)\b", "Third-year students"),
        (r"\b(faculty)\b", "Faculty"),
        (r"\b(staff)\b", "Staff"),
    ]

    for pattern, label in audience_patterns:
        if re.search(pattern, lower):
            return label

    return "Not clearly identified"


# ============================================================
# 9. DATE EXTRACTION
# ============================================================

def extract_dates(text: str):
    """
    Detect common Indian/academic notice date formats.
    """

    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{1,2}\s+(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\s+\d{2,4}\b",
        r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},?\s+\d{2,4}\b",
    ]

    found = []

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        for match in matches:
            if match not in found:
                found.append(match)

    return found


# ============================================================
# 10. DEADLINE DETECTION
# ============================================================

def extract_deadlines(text: str):
    """
    Find lines that appear to contain deadlines.
    """

    lines = get_lines(text)

    deadline_keywords = [
        "last date",
        "last date for",
        "deadline",
        "submit by",
        "submission date",
        "before",
        "on or before",
        "due date",
        "apply before",
        "registration closes",
    ]

    results = []

    for line in lines:
        lower = line.lower()

        if any(
            keyword in lower
            for keyword in deadline_keywords
        ):
            results.append(line)

    return results


# ============================================================
# 11. ACTION EXTRACTION
# ============================================================

def extract_actions(text: str):
    """
    Detect likely actions students/users need to perform.
    """

    lines = get_lines(text)

    action_keywords = [
        "submit",
        "apply",
        "register",
        "attend",
        "report",
        "upload",
        "download",
        "fill",
        "complete",
        "contact",
        "visit",
        "bring",
        "participate",
        "appear",
        "pay",
        "verify",
        "enroll",
    ]

    actions = []

    for line in lines:
        lower = line.lower()

        if any(
            keyword in lower
            for keyword in action_keywords
        ):
            if line not in actions:
                actions.append(line)

    return actions[:10]


# ============================================================
# 12. REQUIREMENT EXTRACTION
# ============================================================

def extract_requirements(text: str):
    """
    Detect lines describing documents or conditions.
    """

    lines = get_lines(text)

    requirement_keywords = [
        "required",
        "requirements",
        "documents",
        "certificate",
        "id card",
        "proof",
        "marks memo",
        "marksheet",
        "resume",
        "photo",
        "signature",
        "aadhaar",
        "application form",
        "eligibility",
        "criteria",
    ]

    requirements = []

    for line in lines:
        lower = line.lower()

        if any(
            keyword in lower
            for keyword in requirement_keywords
        ):
            if line not in requirements:
                requirements.append(line)

    return requirements[:10]


# ============================================================
# 13. EVIDENCE EXTRACTION
# ============================================================

def extract_evidence(
    text: str,
    actions,
    requirements,
    deadlines,
):
    """
    Return the original OCR lines supporting
    extracted information.
    """

    lines = get_lines(text)

    evidence = []

    important_terms = [
        "deadline",
        "last date",
        "submit",
        "apply",
        "register",
        "required",
        "eligibility",
        "documents",
        "students",
        "date",
        "venue",
        "contact",
    ]

    for line in lines:

        lower = line.lower()

        if any(
            term in lower
            for term in important_terms
        ):
            if line not in evidence:
                evidence.append(line)

    # Add extracted lines if they are not already included
    for collection in (
        actions,
        requirements,
        deadlines,
    ):
        for item in collection:
            if item not in evidence:
                evidence.append(item)

    return evidence[:15]


# ============================================================
# 14. HUMAN REVIEW GENERATION
# ============================================================

def generate_review_notes(
    text,
    document_type,
    dates,
    deadlines,
    actions,
    requirements,
):
    notes = []

    if not text.strip():
        notes.append(
            "OCR did not produce reliable readable text."
        )

    if document_type == "Unknown document":
        notes.append(
            "Document type could not be classified reliably."
        )

    if not dates:
        notes.append(
            "No reliable date was detected."
        )

    if not actions:
        notes.append(
            "No clear user action was automatically extracted."
        )

    if not requirements:
        notes.append(
            "No reliable requirements were extracted."
        )

    notes.append(
        "Verify all extracted information against the original notice before taking action."
    )

    return notes


# ============================================================
# 15. CHECKLIST JSON
# ============================================================

def create_checklist(
    filename,
    document_type,
    audience,
    dates,
    deadlines,
    actions,
    requirements,
    evidence,
    review_notes,
):
    return {
        "application": "CampusLens",
        "project": "SIH26188",
        "analysis_timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "source_file": filename,
        "document_type": document_type,
        "target_audience": audience,
        "dates": dates,
        "possible_deadlines": deadlines,
        "possible_actions": actions,
        "requirements": requirements,
        "supporting_evidence": evidence,
        "human_review_required": True,
        "human_review_notes": review_notes,
    }


# ============================================================
# 16. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 CampusLens")

    st.markdown(
        """
        **Notice-to-Action Assistant**

        Upload a college circular, announcement,
        event notice, scholarship notice, timetable,
        or other document containing printed text.
        """
    )

    st.divider()

    st.markdown("### 🔍 What CampusLens does")

    st.markdown(
        """
        1. 📄 Reads the uploaded document
        2. 🔎 Extracts text using OCR
        3. 🏷️ Classifies the document
        4. 📅 Detects dates and deadlines
        5. ✅ Identifies possible actions
        6. 📋 Extracts requirements
        7. 🧾 Shows supporting evidence
        8. 👤 Flags information for human verification
        """
    )

    st.divider()

    st.warning(
        "Prototype: Always verify extracted information "
        "against the original document."
    )


# ============================================================
# 17. HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🎓 CampusLens</h1>
        <p>
            AI-assisted College Notice-to-Action Assistant
        </p>
        <p>
            Convert college notices into actionable,
            reviewable checklists.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 18. FILE UPLOADER
# ============================================================

st.markdown("## 📤 Upload College Notice")

uploaded_file = st.file_uploader(
    "Upload a clear notice, circular, announcement, poster, or college document",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp",
        "bmp",
        "tiff",
    ],
    help="For best OCR results, upload a clear image with readable printed text.",
)


if uploaded_file is None:

    st.info(
        "👆 Upload a college notice to begin analysis."
    )

    st.markdown(
        """
        ### Recommended test documents

        - 📢 College circular
        - 📝 Examination notice
        - 🎓 Scholarship announcement
        - 💼 Placement notice
        - 🏆 Hackathon/event announcement
        - 📅 Timetable
        - 📚 Workshop/seminar notice

        **Avoid testing first with signature-only images or blank documents.**
        """
    )

    st.stop()


# ============================================================
# 19. LOAD IMAGE
# ============================================================

try:

    image = Image.open(uploaded_file)

except Exception as exc:

    st.error(
        f"Unable to read the uploaded image: {exc}"
    )

    st.stop()


# ============================================================
# 20. DOCUMENT PREVIEW
# ============================================================

left, right = st.columns([1, 1])

with left:

    st.markdown("### 🖼️ Uploaded Document")

    st.image(
        image,
        width="stretch",
    )

with right:

    st.markdown("### 📄 File Information")

    st.write(
        f"**Filename:** {uploaded_file.name}"
    )

    st.write(
        f"**Format:** {image.format or 'Unknown'}"
    )

    st.write(
        f"**Image size:** {image.width} × {image.height}px"
    )

    st.write(
        f"**File size:** {uploaded_file.size / 1024:.1f} KB"
    )


# ============================================================
# 21. OCR
# ============================================================

with st.spinner("🔎 Reading document with OCR..."):

    ocr_text = perform_ocr(image)


if ocr_text.startswith("__OCR_ERROR__"):

    error_message = ocr_text.replace(
        "__OCR_ERROR__:",
        "",
        1,
    )

    st.error(
        "OCR could not be completed."
    )

    st.code(error_message)

    st.info(
        "Check that Tesseract OCR is installed at "
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    st.stop()


# ============================================================
# 22. EMPTY OCR HANDLING
# ============================================================

if not ocr_text.strip():

    st.error(
        "❌ No readable printed text was detected."
    )

    st.warning(
        """
        Try uploading a clearer college notice.

        For better OCR:
        - Use a high-resolution image
        - Keep the document straight
        - Avoid glare/shadows
        - Make sure printed text is visible
        - Avoid signature-only images
        """
    )

    st.markdown("### 🧑‍💻 Human Review")

    st.markdown(
        """
        <div class="review-box">
        <strong>Manual verification required.</strong><br><br>
        CampusLens could not obtain reliable OCR text from
        this document.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# 23. ANALYSIS
# ============================================================

document_type = classify_document(
    ocr_text
)

audience = extract_audience(
    ocr_text
)

dates = extract_dates(
    ocr_text
)

deadlines = extract_deadlines(
    ocr_text
)

actions = extract_actions(
    ocr_text
)

requirements = extract_requirements(
    ocr_text
)

evidence = extract_evidence(
    ocr_text,
    actions,
    requirements,
    deadlines,
)

review_notes = generate_review_notes(
    ocr_text,
    document_type,
    dates,
    deadlines,
    actions,
    requirements,
)


checklist = create_checklist(
    uploaded_file.name,
    document_type,
    audience,
    dates,
    deadlines,
    actions,
    requirements,
    evidence,
    review_notes,
)


# ============================================================
# 24. SUMMARY CARDS
# ============================================================

st.markdown("## 📊 Analysis Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Document Type</div>
            <div class="metric-value">
                {document_type}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Target Audience</div>
            <div class="metric-value">
                {audience}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Dates Found</div>
            <div class="metric-value">
                {len(dates)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Possible Actions</div>
            <div class="metric-value">
                {len(actions)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 25. CLASSIFICATION
# ============================================================

st.markdown("## 🏷️ Document Classification")

st.success(
    f"**Detected document type:** {document_type}"
)

if audience != "Not clearly identified":

    st.info(
        f"🎯 **Possible audience:** {audience}"
    )

else:

    st.warning(
        "🎯 Primary audience could not be confidently identified."
    )


# ============================================================
# 26. OCR TEXT
# ============================================================

st.markdown("## 🔎 Extracted OCR Text")

with st.expander(
    "View OCR text",
    expanded=False,
):

    st.text_area(
        "OCR output",
        value=ocr_text,
        height=300,
    )


# ============================================================
# 27. DATES AND DEADLINES
# ============================================================

st.markdown("## 📅 Dates & Deadlines")

date_col, deadline_col = st.columns(2)

with date_col:

    st.markdown("### Detected Dates")

    if dates:

        for date in dates:
            st.write(f"📅 {date}")

    else:

        st.info(
            "No date detected."
        )


with deadline_col:

    st.markdown("### Possible Deadlines")

    if deadlines:

        for deadline in deadlines:
            st.write(f"⏰ {deadline}")

    else:

        st.info(
            "No clear deadline detected."
        )


# ============================================================
# 28. POSSIBLE ACTIONS
# ============================================================

st.markdown("## ✅ Possible Actions")

if actions:

    for index, action in enumerate(
        actions,
        start=1,
    ):

        st.write(
            f"**{index}.** {action}"
        )

else:

    st.warning(
        "No reliable action was automatically extracted."
    )


# ============================================================
# 29. REQUIREMENTS
# ============================================================

st.markdown("## 📋 Requirements")

if requirements:

    for requirement in requirements:

        st.write(
            f"• {requirement}"
        )

else:

    st.info(
        "No reliable requirement was extracted."
    )


# ============================================================
# 30. SUPPORTING EVIDENCE
# ============================================================

st.markdown("## 🧾 Supporting Evidence")

if evidence:

    with st.expander(
        "View evidence lines",
        expanded=True,
    ):

        for item in evidence:

            st.markdown(
                f"> {item}"
            )

else:

    st.info(
        "No supporting evidence lines were identified."
    )


# ============================================================
# 31. HUMAN REVIEW
# ============================================================

st.markdown("## 👤 Human Review")

st.markdown(
    """
    <div class="review-box">
    <strong>Human verification is required.</strong><br><br>
    CampusLens provides AI-assisted extraction only.
    The original college notice should always be checked
    before a student acts on the extracted information.
    </div>
    """,
    unsafe_allow_html=True,
)

if review_notes:

    for note in review_notes:

        st.write(
            f"⚠️ {note}"
        )


# ============================================================
# 32. CHECKLIST JSON
# ============================================================

st.markdown("## 🧩 Action Checklist")

json_text = json.dumps(
    checklist,
    indent=4,
    ensure_ascii=False,
)

with st.expander(
    "View generated JSON checklist",
    expanded=False,
):

    st.code(
        json_text,
        language="json",
    )


# ============================================================
# 33. DOWNLOAD JSON
# ============================================================

st.download_button(
    label="⬇️ Download JSON Checklist",
    data=json_text,
    file_name="campuslens_checklist.json",
    mime="application/json",
    width="stretch",
)


# ============================================================
# 34. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b;">
        <strong>CampusLens</strong> · SIH26188<br>
        AI-assisted College Notice-to-Action Assistant<br>
        <small>
        Prototype output must be verified against the original document.
        </small>
    </div>
    """,
    unsafe_allow_html=True,
)