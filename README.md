# Intelligent Certificate Extractor

Upload certificates (ex-PDF/image), teach template-specific examples, and automatically extract fields from similar certificates.

## Output JSON Fields
- `student_name`
- `course_name`
- `issue_date`
- `certificate_id`
- `issuer`
- `institute_details`
- `student_details`
- `course_details`
- `result_summary`
- `trimester_wise_performance`
- `final_summary`
- `result_declaration`
- `confidence_score`

## Setup
1. Create/activate your virtualenv.
2. Install Python dependencies:
   `pip install -r requirements.txt`
3. Install system dependencies:
   - Poppler (`pdftoppm`) required by `pdf2image`
   - Tesseract OCR binary (`tesseract`) required by `pytesseract`

## Run Frontend
`streamlit run app.py`

Then open the URL shown by Streamlit (usually `http://localhost:8501`).

## Teach Template (Learning Mode)
1. Open the **Teach Template** tab.
2. Upload a certificate that represents a template you want to learn.
3. Create/select a template and fill (or auto-fill then correct) these fields:
   - `student_name`
   - `course_name`
   - `issue_date`
   - `certificate_id`
   - `issuer`
4. Optional: enable **Train with full labeled JSON (all relevant fields)** and provide the complete corrected JSON for nested/extended fields.
5. Click **Save Training Example**.

The app stores template learning data in `storage/template_learning/templates.json` and improves as you add more examples.

## Extract (Inference Mode)
1. Open the **Extract** tab.
2. Upload a new certificate and run extraction.
3. Choose fields to extract and use **Add custom fields (comma separated)** for any new keys or dotted paths.
4. Save reusable college-level field sets from **Sidebar -> College Library**.
5. The pipeline now:
   - tries template matching/layout extraction first
   - merges template result with OCR + rule-based extraction
   - prefers learned template fields when template confidence is high
6. If any field is wrong, use **Improve Accuracy With Corrections** under the result:
   - edit corrected values
   - optional: enable **Train with full labeled JSON (all relevant fields)** to train nested sections too
   - select/create a template
   - click **Save Corrections As Training Example**

This creates a continuous learning loop where corrected predictions immediately become new training data.


## Run CLI
`python main.py /path/to/certificate.pdf --pretty`

## Measure Accuracy
Create ground-truth JSON files in `evaluation/ground_truth/` named as `<input_stem>.json`.

Evaluate one file:
`python scripts/evaluate_accuracy.py --input-file storage/uploads/RBLDCBH759-M.pdf`

Evaluate a folder and save report:
`python scripts/evaluate_accuracy.py --input-dir storage/uploads --report-file evaluation/report.json`

Notes:
- By default `confidence_score` is excluded from accuracy scoring.
- Use `--include-confidence` if you want to include it.
