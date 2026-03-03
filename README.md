# Certificate Extractor

Upload a certificate (PDF/image) and send it directly to LLaMA Vision 3.2 to get structured JSON.

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
4. Start Ollama with a vision model available, for example:
   - `ollama pull llama3.2-vision`
   - `ollama serve`

## Run Frontend (Streamlit)
`streamlit run app.py`

Then open the URL shown by Streamlit (usually `http://localhost:8501`), upload a certificate, and click **Extract Data**.

## Run CLI
`python main.py /path/to/certificate.pdf --pretty`
