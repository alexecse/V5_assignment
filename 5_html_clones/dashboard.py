import os
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image
from collections import Counter
import numpy as np
import streamlit.components.v1 as components

OUTPUT_DIR = "output"
STATS_DIR = "statistics"

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

    group_path = os.path.join(OUTPUT_DIR, selected_tier, selected_group)
    file_list = sorted(groups[selected_tier][selected_group])
    file_paths = [os.path.join(group_path, f) for f in file_list]

    st.subheader(f"Group: {selected_group} (Tier: {selected_tier})")
    st.markdown(f"**Total files:** {len(file_list)}")

    # Heatmap section
    with st.expander("Show tag heatmap"):
        display_heatmap(selected_tier)

    # Tag stats section
    with st.expander("Top 10 most frequent tags"):
        tag_counter = extract_tags_from_files(file_paths)
        top_tags = tag_counter.most_common(10)
        for tag, count in top_tags:
            st.markdown(f"`{tag}`: {count} occurrences")

    # File selector
    selected_file = st.selectbox("Select an HTML file to preview", file_list)
    selected_file_path = os.path.join(group_path, selected_file)

    display_html_preview(selected_tier, selected_group, selected_file)


if __name__ == "__main__":
    main()
