import streamlit as st
import json

from backend.pipeline import run_pipeline
from backend.template_learning import LEARNABLE_FIELDS, TemplateLearningEngine
from frontend.components.confidence_display import show_confidence
from frontend.components.file_uploader import upload_file
from frontend.components.json_viewer import show_json
from frontend.components.mode_selector import select_mode
from frontend.components.preview_panel import show_preview
from frontend.utils.file_handler import save_uploaded_file


FIELD_LABELS: dict[str, str] = {
    "name": "Name",
    "examination": "Examination",
    "held_in": "Held In",
    "seat_number": "Seat Number",
    "specialisation": "Specialisation",
    "aicte_number": "AICTE Number",
    "course_code": "Course Code",
    "course_title": "Course Title",
    "maximum_marks": "Maximum Marks",
    "minimum_marks": "Minimum Marks",
    "marks_obtained": "Marks Obtained",
    "course_credits": "Course Credits",
    "grade": "Grade",
    "credits_earned_c": "Credits Earned (C)",
    "grade_points_g": "Grade Points (G)",
    "cxg": "CxG",
    "remark": "Remark",
    "percentage": "Percentage",
    "gpa": "GPA",
    "overall_grade": "Overall Grade",
    "range": "Range",
    "trimester_i": "Trimester I",
    "trimester_ii": "Trimester II",
    "trimester_iii": "Trimester III",
    "trimester_iv": "Trimester IV",
    "trimester_v": "Trimester V",
    "trimester_vi": "Trimester VI",
    "final_cgpa": "FINAL CGPA",
    "total_credits": "Total Credits",
    "total_grade_points": "Total Grade Points",
    "total_marks_obtained": "Total Marks Obtained",
    "result_declared_on": "Result Declared On",
    "institute_address": "Institute Address",
}

FIELD_PATHS: dict[str, list[str]] = {
    "name": ["student_details.name", "student_name"],
    "examination": ["student_details.examination", "course_name"],
    "held_in": ["student_details.held_in", "issue_date"],
    "seat_number": ["student_details.seat_number", "certificate_id"],
    "specialisation": ["student_details.specialization"],
    "aicte_number": ["student_details.aicte_number"],
    "course_code": ["course_details.0.course_code"],
    "course_title": ["course_details.0.course_title"],
    "maximum_marks": ["course_details.0.maximum_marks", "result_summary.total_maximum_marks"],
    "minimum_marks": ["course_details.0.minimum_marks"],
    "marks_obtained": ["course_details.0.marks_obtained", "result_summary.total_marks_obtained"],
    "course_credits": ["course_details.0.course_credits"],
    "grade": ["course_details.0.grade"],
    "credits_earned_c": ["course_details.0.credits_earned"],
    "grade_points_g": ["course_details.0.grade_points"],
    "cxg": ["course_details.0.cxg"],
    "remark": ["course_details.0.remark", "result_summary.result"],
    "percentage": ["result_summary.percentage"],
    "gpa": ["result_summary.gpa"],
    "overall_grade": ["result_summary.overall_grade"],
    "range": ["result_summary.grade_range"],
    "trimester_i": ["trimester_wise_performance.0"],
    "trimester_ii": ["trimester_wise_performance.1"],
    "trimester_iii": ["trimester_wise_performance.2"],
    "trimester_iv": ["trimester_wise_performance.3"],
    "trimester_v": ["trimester_wise_performance.4"],
    "trimester_vi": ["trimester_wise_performance.5"],
    "final_cgpa": ["final_summary.final_cgpa"],
    "total_credits": ["final_summary.total_credits"],
    "total_grade_points": ["final_summary.total_grade_points"],
    "total_marks_obtained": ["final_summary.total_marks_obtained", "result_summary.total_marks_obtained"],
    "result_declared_on": ["result_declaration.result_declared_on", "issue_date"],
    "institute_address": ["institute_details.address"],
}


def _field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " ").title())


def _get_by_dotted_path(payload: dict[str, object], dotted_path: str) -> object | None:
    current: object = payload
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, list) and part.isdigit():
            idx = int(part)
            if 0 <= idx < len(current):
                current = current[idx]
                continue
        return None
    return current


def _trimester_to_text(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    ordered_keys = ["credits_earned", "marks", "percentage", "gpa"]
    parts = [f"{key}: {value.get(key)}" for key in ordered_keys if value.get(key)]
    if parts:
        return ", ".join(parts)
    return json.dumps(value, ensure_ascii=False)


def _resolve_field_value(extracted_result: dict[str, object], field: str) -> str:
    raw = extracted_result.get(field)
    if raw is not None and str(raw).strip():
        return str(raw).strip()

    for dotted_path in FIELD_PATHS.get(field, []):
        value = _get_by_dotted_path(extracted_result, dotted_path)
        if value is None:
            continue
        if field.startswith("trimester_"):
            text = _trimester_to_text(value)
            if text:
                return text
            continue
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        text = str(value).strip()
        if text:
            return text
    return ""


def _show_template_learning_status(engine: TemplateLearningEngine) -> None:
    templates = engine.list_templates()
    if not templates:
        st.caption("Template learner is empty. Add examples in the Teach Template tab.")
        return

    total_examples = sum(int(row.get("examples_count", 0)) for row in templates)
    st.caption(f"Templates: {len(templates)} | Training examples: {total_examples}")


def _save_correction_feedback(
    engine: TemplateLearningEngine,
    file_path: str,
    annotations: dict[str, str],
    template_name: str,
    template_id: str | None,
    include_all_fields: bool,
    full_payload: dict[str, str] | None,
) -> None:
    if not template_name.strip():
        st.error("Enter a template name or choose an existing template.")
        return
    if not any(value.strip() for value in annotations.values()):
        st.error("Provide at least one corrected value before saving.")
        return

    with st.spinner("Saving correction feedback..."):
        try:
            summary = engine.add_training_example(
                file_path=file_path,
                annotations=annotations,
                template_name=template_name,
                template_id=template_id,
                include_all_fields=include_all_fields,
                full_payload=full_payload,
            )
        except Exception as exc:
            st.error(f"Failed to save correction feedback: {exc}")
            return

    unresolved = ", ".join(summary.get("unresolved_fields", [])) or "None"
    st.success(
        f"Feedback saved to '{summary['template_name']}'. "
        f"Learned {summary['learned_fields']}/{summary['total_annotated_fields']} corrected fields."
    )
    st.caption(f"Fields not localized in OCR (still stored as values): {unresolved}")


def _render_feedback_form(engine: TemplateLearningEngine, file_path: str, extracted_result: dict[str, str]) -> None:
    st.subheader("Improve Accuracy With Corrections")
    st.caption("Correct any fields below and save. Each save becomes a new template-learning example.")

    templates = engine.list_templates()
    template_options = ["Create New Template"] + [
        f"{row['template_name']} ({row['examples_count']} examples)" for row in templates
    ]
    selected_option = st.selectbox(
        "Save feedback to template",
        options=template_options,
        index=0,
        key="extract_feedback_template_select",
    )

    selected_template_id = None
    if selected_option == "Create New Template":
        default_name = st.session_state.get("extract_feedback_suggested_template_name", "")
        template_name = st.text_input("New template name", value=default_name, key="extract_feedback_template_name")
    else:
        selected_index = template_options.index(selected_option) - 1
        selected_template = templates[selected_index]
        selected_template_id = selected_template["template_id"]
        template_name = selected_template["template_name"]
        st.caption(f"Selected template: {template_name}")

    annotations: dict[str, str] = {}
    for field in LEARNABLE_FIELDS:
        field_key = f"extract_feedback_field_{field}"
        if field_key not in st.session_state:
            st.session_state[field_key] = _resolve_field_value(extracted_result, field)
        annotations[field] = st.text_input(_field_label(field), key=field_key)

    include_all_fields = st.checkbox(
        "Train with full labeled JSON (all relevant fields)",
        value=False,
        key="extract_feedback_train_full_json",
    )
    full_payload: dict[str, str] | None = None
    if include_all_fields:
        if "extract_feedback_full_json" not in st.session_state:
            st.session_state["extract_feedback_full_json"] = json.dumps(extracted_result, indent=2, ensure_ascii=False)
        full_json_text = st.text_area(
            "Labeled JSON",
            key="extract_feedback_full_json",
            height=260,
        )
        try:
            parsed_json = json.loads(full_json_text)
            if isinstance(parsed_json, dict):
                full_payload = parsed_json
            else:
                st.error("Labeled JSON must be a JSON object.")
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc.msg}")

    if st.button("Save Corrections As Training Example", type="primary", key="extract_feedback_save"):
        if include_all_fields and full_payload is None:
            st.error("Fix labeled JSON before saving.")
            return
        _save_correction_feedback(
            engine=engine,
            file_path=file_path,
            annotations=annotations,
            template_name=template_name,
            template_id=selected_template_id,
            include_all_fields=include_all_fields,
            full_payload=full_payload,
        )


def _extract_tab(
    engine: TemplateLearningEngine,
    timeout: int,
    use_custom_timeout: bool,
    show_debug: bool,
) -> None:
    st.subheader("Automatic Extraction")
    uploaded_file = upload_file(key="extract_upload")
    if not uploaded_file:
        st.info("Upload a certificate file to begin.")
        return

    file_path = save_uploaded_file(uploaded_file)
    show_preview(file_path)

    mode = select_mode()
    if st.button("Extract Data", type="primary", key="extract_button"):
        with st.spinner("Processing certificate..."):
            try:
                output = run_pipeline(
                    file_path=file_path,
                    mode=mode,
                    timeout_override=int(timeout) if use_custom_timeout else None,
                    include_debug=show_debug,
                )
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")
                return

        if show_debug:
            result, debug_info = output
        else:
            result = output
            debug_info = None

        st.session_state["extract_last_result"] = result
        st.session_state["extract_last_debug"] = debug_info
        st.session_state["extract_last_file_path"] = file_path
        template_name = ""
        if isinstance(debug_info, dict):
            template_name = str(debug_info.get("template_learning", {}).get("template_name") or "").strip()
        st.session_state["extract_feedback_suggested_template_name"] = template_name
        st.session_state["extract_feedback_full_json"] = json.dumps(result, indent=2, ensure_ascii=False)
        for field in LEARNABLE_FIELDS:
            st.session_state[f"extract_feedback_field_{field}"] = _resolve_field_value(result, field)

    result = st.session_state.get("extract_last_result")
    debug_info = st.session_state.get("extract_last_debug")
    feedback_file_path = st.session_state.get("extract_last_file_path")
    if not isinstance(result, dict):
        return

    show_json(result)
    show_confidence(result)

    if isinstance(feedback_file_path, str) and feedback_file_path:
        _render_feedback_form(engine=engine, file_path=feedback_file_path, extracted_result=result)

    if debug_info is not None:
        with st.expander("Debug: OCR extraction"):
            st.caption(f"Status: {debug_info.get('status', 'unknown')}")
            st.text_area("Text context preview", value=debug_info.get("text_context_preview", ""), height=180)
            st.caption(f"OCR text size: {debug_info.get('ocr_chars', 0)} characters")

        with st.expander("Debug: Template learning"):
            template_debug = debug_info.get("template_learning", {})
            st.json(template_debug)


def _teach_template_tab(
    engine: TemplateLearningEngine,
    timeout: int,
    use_custom_timeout: bool,
) -> None:
    st.subheader("Teach Template")
    st.caption("Upload a labeled certificate, correct values, and save it as template training data.")

    templates = engine.list_templates()
    template_options = ["Create New Template"] + [
        f"{row['template_name']} ({row['examples_count']} examples)" for row in templates
    ]
    selected_option = st.selectbox("Template", options=template_options, index=0, key="teach_template_select")

    selected_template_id = None
    if selected_option == "Create New Template":
        template_name = st.text_input("New template name", value="", key="teach_template_name")
    else:
        selected_index = template_options.index(selected_option) - 1
        selected_template = templates[selected_index]
        selected_template_id = selected_template["template_id"]
        template_name = selected_template["template_name"]
        st.caption(f"Updating existing template: {template_name}")

    uploaded_file = upload_file(label="Upload labeled certificate", key="teach_upload")
    if not uploaded_file:
        st.info("Upload a certificate and fill the fields to create a training example.")
        return

    file_path = save_uploaded_file(uploaded_file)
    show_preview(file_path)

    if st.button("Auto-fill Fields From Extractor", key="teach_autofill"):
        with st.spinner("Generating initial values..."):
            try:
                output = run_pipeline(
                    file_path=file_path,
                    mode="High Accuracy",
                    timeout_override=int(timeout) if use_custom_timeout else None,
                    include_debug=False,
                )
                for field in LEARNABLE_FIELDS:
                    st.session_state[f"teach_field_{field}"] = _resolve_field_value(output, field)
                st.session_state["teach_full_json"] = json.dumps(output, indent=2, ensure_ascii=False)
            except Exception as exc:
                st.error(f"Auto-fill failed: {exc}")

    annotations: dict[str, str] = {}
    for field in LEARNABLE_FIELDS:
        field_key = f"teach_field_{field}"
        if field_key not in st.session_state:
            st.session_state[field_key] = ""
        annotations[field] = st.text_input(_field_label(field), key=field_key)

    include_all_fields = st.checkbox(
        "Train with full labeled JSON (all relevant fields)",
        value=False,
        key="teach_train_full_json",
    )

    full_payload: dict[str, str] | None = None
    if include_all_fields:
        if "teach_full_json" not in st.session_state:
            seed_payload = {field: annotations.get(field) for field in LEARNABLE_FIELDS}
            st.session_state["teach_full_json"] = json.dumps(seed_payload, indent=2, ensure_ascii=False)
        full_json_text = st.text_area("Labeled JSON", key="teach_full_json", height=260)
        try:
            parsed_json = json.loads(full_json_text)
            if isinstance(parsed_json, dict):
                full_payload = parsed_json
            else:
                st.error("Labeled JSON must be a JSON object.")
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc.msg}")

    if st.button("Save Training Example", type="primary", key="teach_save"):
        if not template_name.strip():
            st.error("Enter a template name or choose an existing template.")
            return
        if not any(value.strip() for value in annotations.values()):
            st.error("Provide at least one field value before saving.")
            return
        if include_all_fields and full_payload is None:
            st.error("Fix labeled JSON before saving.")
            return

        with st.spinner("Saving training example..."):
            try:
                summary = engine.add_training_example(
                    file_path=file_path,
                    annotations=annotations,
                    template_name=template_name,
                    template_id=selected_template_id,
                    include_all_fields=include_all_fields,
                    full_payload=full_payload,
                )
            except Exception as exc:
                st.error(f"Failed to save training example: {exc}")
                return

        unresolved = ", ".join(summary.get("unresolved_fields", [])) or "None"
        st.success(
            f"Saved example to '{summary['template_name']}'. "
            f"Learned {summary['learned_fields']}/{summary['total_annotated_fields']} annotated fields."
        )
        st.caption(f"Fields not localized in OCR (still stored as values): {unresolved}")


def main() -> None:
    st.set_page_config(page_title="Certificate Extractor", layout="wide")
    st.title("Intelligent Certificate Processing Platform")
    st.caption("Learn template layouts from annotated certificates and extract fields automatically from similar documents.")

    template_engine = TemplateLearningEngine()
    with st.sidebar:
        st.subheader("OCR Settings")
        use_custom_timeout = st.checkbox("Use custom timeout", value=False)
        timeout = st.number_input("Timeout (seconds)", min_value=30, max_value=1200, value=240)
        show_debug = st.checkbox("Show OCR debug", value=False)
        st.divider()
        st.subheader("Template Learner")
        _show_template_learning_status(template_engine)

    tab_extract, tab_teach = st.tabs(["Extract", "Teach Template"])
    with tab_extract:
        _extract_tab(
            engine=template_engine,
            timeout=int(timeout),
            use_custom_timeout=use_custom_timeout,
            show_debug=show_debug,
        )
    with tab_teach:
        _teach_template_tab(
            engine=template_engine,
            timeout=int(timeout),
            use_custom_timeout=use_custom_timeout,
        )


if __name__ == "__main__":
    main()
