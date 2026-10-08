from __future__ import annotations
from pathlib import Path
import tempfile
import streamlit as st
from seminar_ranker import SeminarImageRanker

st.set_page_config(page_title="Seminar Image Ranking System", page_icon="📷", layout="wide")
st.title("Seminar Image Ranking System")
st.caption("Local pretrained computer vision + transparent feature-engineering ranking. No images are sent to an API.")

with st.sidebar:
    st.header("Analysis settings")
    top_k = st.slider("Images to return", 1, 30, 10)
    st.caption("The first run downloads CLIP and YOLO weights; later runs use their local caches.")

folder = st.text_input("Image folder path", placeholder="/path/to/input_images")
uploads = st.file_uploader("Or upload images", type=["jpg", "jpeg", "png", "heic", "heif"], accept_multiple_files=True)

if st.button("Analyze images", type="primary"):
    temp_dir = None
    try:
        if uploads:
            temp_dir = tempfile.TemporaryDirectory()
            for upload in uploads:
                (Path(temp_dir.name) / upload.name).write_bytes(upload.getbuffer())
            source = temp_dir.name
        elif folder.strip():
            source = folder.strip()
            if not Path(source).is_dir():
                raise ValueError("That folder does not exist or is not accessible.")
        else:
            raise ValueError("Select a folder or upload at least one image.")
        bar, status = st.progress(0), st.empty()
        def progress(current, total, name):
            bar.progress(current / total, text=f"Analyzing {current}/{total}: {name}")
        with st.spinner("Loading local computer-vision models and analyzing images…"):
            result = SeminarImageRanker().rank(source, top_k=top_k, progress=progress)
        status.success(f"Eligible images: {result.processed_count}; selected {len(result.selected)} diverse results.")
        for row in result.selected:
            left, right = st.columns([1, 1.2])
            with left: st.image(row["source_path"], caption=f"#{row['rank']} — {row['filename']}", use_container_width=True)
            with right:
                st.subheader(f"#{row['rank']} · {row['final_score']:.1f}")
                st.write(row["predicted_category"].replace("_", " ").title())
                st.write(row["explanation"])
                st.caption(f"Semantic {row['semantic_score']:.1f} · Quality {row['quality_score']:.1f} · Coverage {row['event_coverage_score']:.1f} · People {row['person_count']}")
        csv_path = result.output_dir / "results.csv"
        st.download_button("Download ranking CSV", csv_path.read_bytes(), "results.csv", "text/csv")
        if result.errors:
            st.warning("Some files were skipped:\n\n" + "\n".join(result.errors[:10]))
        if result.rejected:
            st.info(f"Excluded {len(result.rejected)} blurred, tilted, or poorly framed images. "
                    "Adjust `processing.eligibility` in config/config.yaml if needed.")
    except Exception as exc:
        st.error(str(exc))
    finally:
        if temp_dir:
            temp_dir.cleanup()
