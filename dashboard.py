import os
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image
from collections import Counter
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

OUTPUT_DIR = "output"
STATS_DIR = "statistics"

def extract_tags_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'lxml')
        tags = [tag.name for tag in soup.find_all()]
    return Counter(tags)

def list_groups(output_root=OUTPUT_DIR):
    groups = {}
    for tier in os.listdir(output_root):
        tier_path = os.path.join(output_root, tier)
        if not os.path.isdir(tier_path):
            continue
        groups[tier] = {}
        for group in os.listdir(tier_path):
            group_path = os.path.join(tier_path, group)
            if os.path.isdir(group_path):
                files = [f for f in os.listdir(group_path) if f.endswith(".html")]
                groups[tier][group] = files
    return groups

def display_html_preview(tier, group, file):
    file_url = f"http://localhost:8000/{tier}/{group}/{file}"
    st.markdown("### Website Preview")
    
    # White background wrapper
    html_wrapper = f"""
    <div style="background-color: white; padding: 1px; border: 1px solid #ddd;">
        <iframe src="{file_url}" width="100%" height="600px" style="border:none;"></iframe>
    </div>
    """
    components.html(html_wrapper, height=610)

def extract_tags_from_files(file_paths):
    all_tags = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'lxml')
            tags = [tag.name for tag in soup.find_all()]
            all_tags.extend(tags)
    return Counter(all_tags)

def display_heatmap(tier_name):
    heatmap_path = os.path.join(STATS_DIR, tier_name, f"{tier_name}_tag_frequency_heatmap.png")
    if os.path.exists(heatmap_path):
        st.image(Image.open(heatmap_path), caption="Tag Frequency Heatmap", use_container_width=True)
    else:
        st.warning("No heatmap found for this tier.")

def main():
    st.set_page_config(page_title="HTML Group Explorer", layout="wide")
    st.title("HTML Group Explorer")
    groups = list_groups()

    selected_tier = st.sidebar.selectbox("Select a tier", sorted(groups.keys()))
    selected_group = st.sidebar.selectbox("Select a group", sorted(groups[selected_tier].keys()))

    tier_path = os.path.join(OUTPUT_DIR, selected_tier)
    group_path = os.path.join(OUTPUT_DIR, selected_tier, selected_group)
    file_list = sorted(groups[selected_tier][selected_group])
    file_paths = [os.path.join(group_path, f) for f in file_list]

    st.subheader(f"Group: {selected_group} (Tier: {selected_tier})")
    st.markdown(f"**Total files:** {len(file_list)}")

    # Heatmap section
    with st.expander("Show tag heatmap"):
        display_heatmap(selected_tier)

    st.markdown("### Select an HTML file to preview")
    selected_file = st.selectbox("File", file_list)
    selected_file_path = os.path.join(group_path, selected_file)

    # Top tags side-by-side
    with st.sidebar:
        st.markdown("#### 📄 Top 5 tags in selected file")
        tag_counter = extract_tags_from_file(selected_file_path)
        top_tags = tag_counter.most_common(5)
        df_file = pd.DataFrame(top_tags, columns=["Tag", "Occurrences"])[["Occurrences", "Tag"]]
        st.dataframe(df_file, use_container_width=True, height=220)

        st.markdown("#### 📁 Top 5 tags in group")
        tag_counter = extract_tags_from_files(file_paths)
        top_tags = tag_counter.most_common(5)
        df_group = pd.DataFrame(top_tags, columns=["Tag", "Occurrences"])[["Occurrences", "Tag"]]
        st.dataframe(df_group, use_container_width=True, height=220)


    # Display page
    display_html_preview(selected_tier, selected_group, selected_file)

    # Logs section
    st.markdown("### Postprocessing Logs")
    log_path = os.path.join(tier_path, "postprocessing.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            logs = f.read().splitlines()

        with st.expander("Postprocessing Logs"):
            for line in logs:
                st.markdown(f"• `{line}`")

        st.download_button("Download logs", "\n".join(logs), file_name="postprocessing.log")
    else:
        st.info("No postprocessing log found for this group.")

    # Stats section
    stats_path = os.path.join(tier_path, "postprocessing_stats.csv")
    if os.path.exists(stats_path):
        st.markdown("### Postprocessing Statistics")
        df_stats = pd.read_csv(stats_path)

        # Redimensionăm tabelul doar cât trebuie
        st.table(df_stats.style.set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'left')]},
            {'selector': 'td', 'props': [('text-align', 'left')]}
        ]))

        st.download_button(
            label="Download stats",
            data=df_stats.to_csv(index=False),
            file_name="postprocessing_stats.csv",
            mime="text/csv"
        )
    else:
        st.info("No postprocessing statistics file found for this group.")

if __name__ == "__main__":
    main()
