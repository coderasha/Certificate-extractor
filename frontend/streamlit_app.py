import streamlit as st
import json
import re
from typing import Any, Callable

from backend.college_registry import get_college, load_colleges, upsert_college
from backend.pipeline import run_pipeline
from backend.template_learning import LEARNABLE_FIELDS, TemplateLearningEngine
from frontend.components.confidence_display import show_confidence
from frontend.components.file_uploader import upload_file
from frontend.components.json_viewer import show_json
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

FIELD_OPTION_KEYS = list(FIELD_LABELS.keys())


def _field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " ").title())


def _normalize_field_key(raw_field: str) -> str:
    return str(raw_field or "").strip().replace(" ", "_").lower()


def _merge_field_options(*groups: Any) -> list[str]:
    merged = list(FIELD_OPTION_KEYS)
    seen = set(merged)
    for group in groups:
        if not isinstance(group, list):
            continue
        for raw in group:
            field = _normalize_field_key(raw)
            if not field or field in seen:
                continue
            merged.append(field)
            seen.add(field)
    return merged


def _get_by_dotted_path(payload: dict[str, object], dotted_path: str) -> object | None:
    current: object = payload
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            if part in current:
                current = current.get(part)
                continue
            lower_map = {str(key).lower(): key for key in current.keys()}
            mapped = lower_map.get(part.lower())
            if mapped is not None:
                current = current.get(mapped)
                continue
            return None
        if isinstance(current, list) and part.isdigit():
            idx = int(part)
            if 0 <= idx < len(current):
                current = current[idx]
                continue
        return None
    return current


def _find_value_by_leaf_key(payload: object, key: str) -> object | None:
    if isinstance(payload, dict):
        for raw_key, value in payload.items():
            if str(raw_key).lower() == key.lower():
                return value
            nested = _find_value_by_leaf_key(value, key)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _find_value_by_leaf_key(item, key)
            if nested is not None:
                return nested
    return None


def _stringify_value(field: str, value: object) -> str:
    if field.startswith("trimester_") and isinstance(value, dict):
        text = _trimester_to_text(value)
        if text:
            return text
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    text = str(value).strip()
    return text


def _add_custom_fields_control(multiselect_key: str, state_prefix: str) -> None:
    input_key = f"{state_prefix}_custom_fields_input"
    button_key = f"{state_prefix}_custom_fields_add"
    custom_fields_text = st.text_input(
        "Add custom fields (comma separated)",
        key=input_key,
        help="Example: university_details.program, student_details.father_name",
    )
    if st.button("Add fields", key=button_key):
        requested = [_normalize_field_key(item) for item in custom_fields_text.split(",")]
        requested = [item for item in requested if item]
        existing = st.session_state.get(multiselect_key, [])
        current = [_normalize_field_key(item) for item in existing if _normalize_field_key(item)]
        seen = set(current)
        for field in requested:
            if field not in seen:
                current.append(field)
                seen.add(field)
        st.session_state[multiselect_key] = current


def _resolve_field_value(extracted_result: dict[str, object], field: str) -> str:
    normalized_field = _normalize_field_key(field)
    raw = extracted_result.get(normalized_field)
    if raw is None:
        raw = extracted_result.get(field)
    if raw is not None and str(raw).strip():
        return _stringify_value(normalized_field, raw)

    field_paths = FIELD_PATHS.get(normalized_field, [])
    if "." in field and field not in field_paths:
        field_paths = [field, *field_paths]
    elif "." in normalized_field and normalized_field not in field_paths:
        field_paths = [normalized_field, *field_paths]

    for dotted_path in field_paths:
        value = _get_by_dotted_path(extracted_result, dotted_path)
        if value is None:
            continue
        text = _stringify_value(normalized_field, value)
        if text:
            return text
    fallback = _find_value_by_leaf_key(extracted_result, normalized_field)
    if fallback is not None:
        text = _stringify_value(normalized_field, fallback)
        if text:
            return text
    return ""


def _trimester_to_text(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    ordered_keys = ["credits_earned", "marks", "percentage", "gpa"]
    parts = [f"{key}: {value.get(key)}" for key in ordered_keys if value.get(key)]
    if parts:
        return ", ".join(parts)
    return json.dumps(value, ensure_ascii=False)


def _show_template_learning_status(engine: TemplateLearningEngine, college_name: str) -> None:
    templates = engine.list_templates(college_name=college_name)
    if not templates:
        st.caption("Template learner is empty. Add examples in the Teach Template tab.")
        return

    total_examples = sum(int(row.get("examples_count", 0)) for row in templates)
    st.caption(f"Templates: {len(templates)} | Training examples: {total_examples}")


def _get_college_field_defaults(college_name: str | None) -> list[str]:
    profile = get_college(college_name)
    if profile and profile.get("fields"):
        return list(profile["fields"])
    return list(LEARNABLE_FIELDS)


def _get_active_fields(college_name: str) -> list[str]:
    stored = st.session_state.get("active_fields")
    if isinstance(stored, list) and stored:
        return stored
    return _get_college_field_defaults(college_name)


def _get_college_options() -> list[str]:
    names = [entry["name"] for entry in load_colleges()]
    if "Other" not in names:
        names.append("Other")
    return names


def _show_requested_fields(extracted: dict[str, object], fields: list[str]) -> None:
    rows = [{"Field": _field_label(_normalize_field_key(field)), "Value": _resolve_field_value(extracted, field)} for field in fields]
    if not rows:
        return
    st.subheader("Requested fields")
    st.table(rows)


def _render_college_library_widget() -> None:
    st.caption("Add or update colleges and the exact fields you want extracted.")
    options = ["Add new college"] + _get_college_options()
    select_key = "college_library_select"
    cached_key = "college_library_select_cached"
    if select_key not in st.session_state:
        st.session_state[select_key] = options[0]

    selected_option = st.selectbox("Profile", options=options, key=select_key)
    if st.session_state.get(cached_key) != selected_option:
        profile = get_college(selected_option)
        if profile:
            st.session_state["college_library_name"] = profile["name"]
            st.session_state["college_library_fields"] = profile["fields"]
        else:
            st.session_state["college_library_name"] = "" if selected_option == "Add new college" else selected_option
            st.session_state["college_library_fields"] = list(LEARNABLE_FIELDS)
        st.session_state[cached_key] = selected_option

    college_name_input = st.text_input("College name", key="college_library_name")
    selected_fields = st.multiselect(
        "Fields to extract",
        options=_merge_field_options(st.session_state.get("college_library_fields", [])),
        format_func=_field_label,
        key="college_library_fields",
    )
    _add_custom_fields_control(multiselect_key="college_library_fields", state_prefix="college_library")

    if st.button("Save college profile", key="college_library_save"):
        if not college_name_input.strip():
            st.error("Enter a college name before saving.")
            return
        try:
            saved = upsert_college(college_name_input, selected_fields)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success(f"Saved \"{saved['name']}\" with {len(saved['fields'])} fields.")
        st.session_state[select_key] = saved["name"]
        st.session_state[cached_key] = saved["name"]


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


def _render_feedback_form(
    engine: TemplateLearningEngine,
    file_path: str,
    extracted_result: dict[str, str],
    college_name: str,
    selected_fields: list[str] | None = None,
) -> None:
    st.subheader("Improve Accuracy With Corrections")
    st.caption("Correct any fields below and save. Each save becomes a new template-learning example.")
    st.info("Correct the Institute Address field (just below the institute name) to teach the model the precise address text.")

    templates = engine.list_templates(college_name=college_name)
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

    field_list = selected_fields or _get_college_field_defaults(college_name)
    annotations: dict[str, str] = {}
    for field in field_list:
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
    college_name: str,
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
    field_selection_key = "extract_selected_fields"
    college_cache_key = "extract_selected_fields_college"
    base_fields = _get_college_field_defaults(college_name)
    if st.session_state.get(college_cache_key) != college_name or field_selection_key not in st.session_state:
        st.session_state[field_selection_key] = base_fields
        st.session_state[college_cache_key] = college_name

    selected_fields = st.multiselect(
        "Fields to extract",
        options=_merge_field_options(base_fields, st.session_state.get(field_selection_key, [])),
        format_func=_field_label,
        key=field_selection_key,
    )
    _add_custom_fields_control(multiselect_key=field_selection_key, state_prefix="extract")

    if not selected_fields:
        st.warning("Select at least one field to extract before running the pipeline.")

    previous_active = st.session_state.get("active_fields")
    if selected_fields:
        active_fields = selected_fields
        st.session_state["active_fields"] = list(active_fields)
    else:
        active_fields = previous_active or base_fields

    if st.button("Extract Data", type="primary", key="extract_button"):
        if not selected_fields:
            st.error("Choose at least one field before extracting.")
            return
        with st.spinner("Processing certificate..."):
            try:
                output = run_pipeline(
                    file_path=file_path,
                    mode="High Accuracy",
                    college_name=college_name,
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

        transformed_result = (
            _transform_result_for_college(result, college_name)
            if isinstance(result, dict)
            else result
        )

        st.session_state["extract_last_result"] = result
        st.session_state["extract_last_transformed"] = transformed_result
        st.session_state["extract_last_debug"] = debug_info
        st.session_state["extract_last_file_path"] = file_path
        template_name = ""
        if isinstance(debug_info, dict):
            template_name = str(debug_info.get("template_learning", {}).get("template_name") or "").strip()
        st.session_state["extract_feedback_suggested_template_name"] = template_name
        st.session_state["extract_feedback_full_json"] = json.dumps(result, indent=2, ensure_ascii=False)
        for field in _merge_field_options(active_fields):
            st.session_state[f"extract_feedback_field_{field}"] = _resolve_field_value(result, field)

    result = st.session_state.get("extract_last_result")
    debug_info = st.session_state.get("extract_last_debug")
    feedback_file_path = st.session_state.get("extract_last_file_path")
    transformed_result = st.session_state.get("extract_last_transformed")
    if not isinstance(result, dict):
        return

    display_result = transformed_result if isinstance(transformed_result, dict) else result
    show_json(display_result)
    _show_requested_fields(display_result, active_fields)
    st.caption(f"JSON schema tailored for {college_name}")
    show_confidence(result)

    if isinstance(feedback_file_path, str) and feedback_file_path:
        _render_feedback_form(
            engine=engine,
            file_path=feedback_file_path,
            extracted_result=result,
            college_name=college_name,
            selected_fields=active_fields,
        )

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
    college_name: str,
    timeout: int,
    use_custom_timeout: bool,
    selected_fields: list[str] | None = None,
) -> None:
    st.subheader("Teach Template")
    st.caption("Upload a labeled certificate, correct values, and save it as template training data.")

    templates = engine.list_templates(college_name=college_name)
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
                    college_name=college_name,
                    timeout_override=int(timeout) if use_custom_timeout else None,
                    include_debug=False,
                )
                field_pool = _merge_field_options(selected_fields or _get_college_field_defaults(college_name))
                for field in field_pool:
                    st.session_state[f"teach_field_{field}"] = _resolve_field_value(output, field)
                st.session_state["teach_full_json"] = json.dumps(output, indent=2, ensure_ascii=False)
            except Exception as exc:
                st.error(f"Auto-fill failed: {exc}")

    field_list = selected_fields or _get_college_field_defaults(college_name)
    field_list = [_normalize_field_key(field) for field in field_list if _normalize_field_key(field)]
    field_list = list(dict.fromkeys(field_list))
    annotations: dict[str, str] = {}
    for field in field_list:
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
            seed_payload = {field: annotations.get(field) for field in field_list}
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
                    college_name=college_name,
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


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    cleaned = str(value).replace("%", "").replace(",", ".").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _transform_sies_result(result: dict[str, Any]) -> dict[str, Any]:
    student = _safe_dict(result.get("student_details"))
    summary = _safe_dict(result.get("result_summary"))
    final_summary = _safe_dict(result.get("final_summary"))
    declaration = _safe_dict(result.get("result_declaration"))
    course_details_list = _safe_list(result.get("course_details"))
    first_course = _safe_dict(course_details_list[0] if course_details_list else {})
    trimester_rows = _safe_list(result.get("trimester_wise_performance"))
    trimester_vi: dict[str, Any] = {}
    for row in trimester_rows:
        row_dict = _safe_dict(row)
        if str(row_dict.get("trimester") or "").strip().upper() == "VI":
            trimester_vi = row_dict
            break
    if not trimester_vi and trimester_rows:
        trimester_vi = _safe_dict(trimester_rows[-1])

    marks_obtained = summary.get("total_marks_obtained")
    marks_maximum = summary.get("total_maximum_marks")
    if (not marks_obtained or not marks_maximum) and trimester_vi.get("marks"):
        marks_tokens = str(trimester_vi.get("marks")).split("/", 1)
        if len(marks_tokens) == 2:
            marks_obtained = marks_obtained or marks_tokens[0].strip()
            marks_maximum = marks_maximum or marks_tokens[1].strip()
    marks_ratio = f"{marks_obtained}/{marks_maximum}" if marks_obtained and marks_maximum else None
    if not marks_ratio and first_course.get("marks_obtained") and first_course.get("maximum_marks"):
        marks_ratio = f"{first_course.get('marks_obtained')}/{first_course.get('maximum_marks')}"

    course_credits = first_course.get("course_credits") or first_course.get("credits_earned") or trimester_vi.get("credits_earned")
    grade_points = first_course.get("grade_points") or trimester_vi.get("gpa") or summary.get("gpa")
    cxg_value = first_course.get("cxg")
    if cxg_value is None:
        credits_val = _to_float(course_credits)
        grade_points_val = _to_float(grade_points)
        if credits_val is not None and grade_points_val is not None:
            cxg_value = round(credits_val * grade_points_val, 2)

    trimester_results: dict[str, Any] = {}
    for row in trimester_rows:
        row_dict = _safe_dict(row)
        tri = str(row_dict.get("trimester") or "").strip().upper()
        if not tri:
            continue
        trimester_results[f"trimester_{tri}"] = {
            "credits_earned": _to_float(row_dict.get("credits_earned")),
            "marks": row_dict.get("marks"),
            "percentage": row_dict.get("percentage"),
            "gpa": _to_float(row_dict.get("gpa")),
        }

    total_obtained = final_summary.get("total_marks_obtained")
    total_maximum = final_summary.get("total_maximum_marks")
    total_marks_ratio = f"{total_obtained}/{total_maximum}" if total_obtained and total_maximum else None

    return {
        "name": student.get("name") or result.get("student_name"),
        "examination": student.get("examination") or result.get("course_name"),
        "trimester": "VI",
        "held_in": student.get("held_in"),
        "seat_number": student.get("seat_number") or result.get("certificate_id"),
        "specialisation": student.get("specialization"),
        "aicte_number": student.get("aicte_number"),
        "course_details": {
            "course_code": first_course.get("course_code"),
            "course_title": first_course.get("course_title"),
            "maximum_marks": _to_float(first_course.get("maximum_marks") or marks_maximum),
            "minimum_marks": _to_float(first_course.get("minimum_marks")),
            "marks_obtained": _to_float(first_course.get("marks_obtained") or marks_obtained),
            "course_credits": _to_float(course_credits),
            "grade": first_course.get("grade") or summary.get("overall_grade"),
            "credits_earned": _to_float(first_course.get("credits_earned") or course_credits),
            "grade_points": _to_float(grade_points),
            "cxg": _to_float(cxg_value),
        },
        "result_summary": {
            "remark": summary.get("result"),
            "marks_obtained": marks_ratio,
            "percentage": summary.get("percentage"),
            "gpa": _to_float(summary.get("gpa")),
            "overall_grade": summary.get("overall_grade"),
            "range": summary.get("grade_range"),
        },
        "trimester_results": trimester_results,
        "overall_academic_summary": {
            "final_cgpa": _to_float(final_summary.get("final_cgpa")),
            "total_credits": _to_float(final_summary.get("total_credits")),
            "total_grade_points": _to_float(final_summary.get("total_grade_points")),
            "total_marks_obtained": total_marks_ratio,
        },
        "result_declared_on": declaration.get("result_declared_on"),
        "institution": _safe_dict(result.get("institute_details")).get("name") or result.get("issuer"),
    }


def _build_semester_gpa(rows: list[dict[str, Any]]) -> dict[str, float]:
    gpas: dict[str, float] = {}
    for idx, row in enumerate(rows):
        gpa_value = _to_float(_safe_dict(row).get("gpa"))
        if gpa_value is not None:
            gpas[f"semester_{idx + 1}"] = gpa_value
    return gpas


def _is_practical_subject(title: Any) -> bool:
    return bool(re.search(r"\b(lab|project|practical)\b", str(title or ""), flags=re.IGNORECASE))


def _build_semester_subject_blocks(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    existing_third = _safe_dict(result.get("third_semester"))
    existing_fourth = _safe_dict(result.get("fourth_semester"))
    third_theory = _safe_list(existing_third.get("theory"))
    third_practicals = _safe_list(existing_third.get("practicals"))
    fourth_theory = _safe_list(existing_fourth.get("theory"))
    fourth_practicals = _safe_list(existing_fourth.get("practicals"))
    if third_theory or third_practicals or fourth_theory or fourth_practicals:
        return existing_third, existing_fourth

    for subject in _safe_list(result.get("course_details")):
        row = _safe_dict(subject)
        code = str(row.get("subject_code") or row.get("course_code") or "").upper()
        if not code:
            continue
        entry = {
            "subject_code": code,
            "subject_name": row.get("subject_name") or row.get("course_title"),
            "credits": _to_float(row.get("credits") or row.get("course_credits")),
            "grade": row.get("grade"),
        }
        if code.startswith("KCA3"):
            if _is_practical_subject(entry["subject_name"]):
                third_practicals.append(entry)
            else:
                third_theory.append(entry)
        elif code.startswith("KCA4"):
            if _is_practical_subject(entry["subject_name"]):
                fourth_practicals.append(entry)
            else:
                fourth_theory.append(entry)

    trimester_rows = _safe_list(result.get("trimester_wise_performance"))
    third_total = existing_third.get("total_credits")
    fourth_total = existing_fourth.get("total_credits")
    if third_total is None and len(trimester_rows) > 2:
        third_total = _safe_dict(trimester_rows[2]).get("credits_earned")
    if fourth_total is None and len(trimester_rows) > 3:
        fourth_total = _safe_dict(trimester_rows[3]).get("credits_earned")

    third = {
        "theory": third_theory,
        "practicals": third_practicals,
        "total_credits": _to_float(third_total),
    }
    fourth = {
        "theory": fourth_theory,
        "practicals": fourth_practicals,
        "total_credits": _to_float(fourth_total),
    }
    return third, fourth


def _derive_division(percentage: Any) -> str | None:
    score = _to_float(percentage)
    if score is None:
        return None
    if score >= 60:
        return "I-DIV"
    if score >= 50:
        return "II-DIV"
    if score >= 40:
        return "III-DIV"
    return "Pass"


def _transform_inmantec_result(result: dict[str, Any]) -> dict[str, Any]:
    university_details = _safe_dict(result.get("university_details"))
    student_details = _safe_dict(result.get("student_details"))
    institute_details = _safe_dict(result.get("institute_details"))
    result_summary = _safe_dict(result.get("result_summary"))
    final_summary = _safe_dict(result.get("final_summary"))
    declaration = _safe_dict(result.get("result_declaration"))
    trimester_rows = _safe_list(result.get("trimester_wise_performance"))
    third_semester, fourth_semester = _build_semester_subject_blocks(result)

    return {
        "university_details": {
            "university_name": university_details.get("university_name") or institute_details.get("name") or result.get("issuer"),
            "location": university_details.get("location") or institute_details.get("address"),
            "document_type": university_details.get("document_type") or institute_details.get("document_type"),
            "program": university_details.get("program") or student_details.get("examination") or result.get("course_name"),
            "course_duration": university_details.get("course_duration"),
            "year": university_details.get("year"),
            "session": university_details.get("session"),
            "declared_on": university_details.get("declared_on") or declaration.get("result_declared_on"),
            "printed_on": university_details.get("printed_on") or result.get("issue_date"),
        },
        "student_details": {
            "candidate_name": student_details.get("candidate_name") or student_details.get("name") or result.get("student_name"),
            "father_name": student_details.get("father_name"),
            "institution": student_details.get("institution") or institute_details.get("name"),
            "roll_number": student_details.get("roll_number") or result.get("certificate_id"),
            "enrollment_number": student_details.get("enrollment_number") or student_details.get("seat_number"),
            "serial_number": student_details.get("serial_number") or result.get("certificate_id"),
        },
        "third_semester": third_semester,
        "fourth_semester": fourth_semester,
        "semester_performance": {
            "semester_gpa": _build_semester_gpa(trimester_rows),
        },
        "final_result": {
            "cgpa": _to_float(_safe_dict(result.get("final_result")).get("cgpa") or final_summary.get("final_cgpa")),
            "result_division": _safe_dict(result.get("final_result")).get("result_division")
            or _derive_division(result_summary.get("percentage")),
        },
    }


COLLEGE_TRANSFORMERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "SIES School of Business Studies": _transform_sies_result,
    "Inmantec College": _transform_inmantec_result,
    "Inmantec": _transform_inmantec_result,
}


def _transform_result_for_college(result: dict[str, Any], college_name: str) -> dict[str, Any]:
    transformer = COLLEGE_TRANSFORMERS.get(college_name)
    if not transformer:
        return result
    return transformer(result)


def main() -> None:
    st.set_page_config(page_title="Certificate Extractor", layout="wide")
    st.title("Intelligent Certificate Processing Platform")
    st.caption(
        "Learn template layouts from annotated certificates and extract fields using the high-accuracy, bounding-box-first pipeline."
    )

    template_engine = TemplateLearningEngine()
    with st.sidebar:
        st.subheader("College")
        college_name = st.selectbox("Select college", options=_get_college_options(), index=0, key="selected_college")
        st.subheader("OCR Settings")
        use_custom_timeout = st.checkbox("Use custom timeout", value=False)
        timeout = st.number_input("Timeout (seconds)", min_value=30, max_value=1200, value=240)
        show_debug = st.checkbox("Show OCR debug", value=False)
        st.divider()
        st.subheader("Template Learner")
        _show_template_learning_status(template_engine, college_name)
        st.divider()
        st.subheader("College Library")
        _render_college_library_widget()

    tab_extract, tab_teach = st.tabs(["Extract Data", "Teach Template"])
    with tab_extract:
        _extract_tab(
            engine=template_engine,
            college_name=college_name,
            timeout=int(timeout),
            use_custom_timeout=use_custom_timeout,
            show_debug=show_debug,
        )
    with tab_teach:
        _teach_template_tab(
            engine=template_engine,
            college_name=college_name,
            timeout=int(timeout),
            use_custom_timeout=use_custom_timeout,
            selected_fields=_get_active_fields(college_name),
        )


if __name__ == "__main__":
    main()
