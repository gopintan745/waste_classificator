"""
Reusable UI components for the Streamlit app.
Each component is a self-contained function that can be tested independently.
"""
import streamlit as st
from typing import Dict, List
import plotly.graph_objects as go
import pandas as pd
from quality_checks import run_all_checks

# ============================================
# RESULT DISPLAY COMPONENTS
# ============================================
def display_classification_result(result: Dict):
    """Display the full classification result with all UX enhancements."""
    top_class = result["top_class"]
    conf = result["confidence"]

    # ============================================
    # Top Section: Material Name + Confidence
    # ============================================
    st.markdown("---")
    col1, col2 = st.columns([3, 1])

    with col1:
        confidence_emoji = (
            "🟢" if conf >= 0.80
            else "🟡" if conf >= 0.60
            else "🔴"
        )
        st.markdown(
            f"### {confidence_emoji} **{top_class.title()}**\n"
            f"Confidence: **{conf:.1%}**"
        )

    with col2:
        if result["is_low_confidence"]:
            st.error("Low confidence")
        elif result["is_ambiguous"]:
            st.warning("Ambiguous")
        else:
            st.success("High confidence")

    # ============================================
    # Warnings Section
    # ============================================
    if result["warnings"]:
        for warning in result["warnings"]:
            st.warning(f"⚠️ {warning}")

    # ============================================
    # Top K Predictions
    # ============================================
    with st.expander("📊 See all top predictions", expanded=False):
        show_top_k_chart(result["top_k"])

    # ============================================
    # Material Properties
    # ============================================
    show_material_properties(result["properties"])

    # ============================================
    # Handling Steps
    # ============================================
    show_handling_steps(result["properties"]["handling_steps"])


def show_top_k_chart(top_k: List[Dict]):
    """Display a horizontal bar chart of top-k predictions."""
    classes = [r["class"].title() for r in top_k]
    probs = [r["confidence"] for r in top_k]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=probs,
        y=classes,
        orientation='h',
        marker=dict(
            color=probs,
            colorscale='Blues',
            cmin=0,
            cmax=1,
        ),
        text=[f"{p:.1%}" for p in probs],
        textposition='auto',
    ))
    fig.update_layout(
        title="Top-3 Predictions",
        xaxis_title="Probability",
        yaxis=dict(autorange="reversed"),
        height=250,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_material_properties(props: Dict):
    """Display material properties as colored metric cards."""
    st.markdown("#### ♻️ Material Properties")

    cols = st.columns(4)
    with cols[0]:
        emoji = "✅" if props["recyclable"] else "❌"
        st.metric(
            label=f"{emoji} Recyclable",
            value="Yes" if props["recyclable"] else "No",
        )
    with cols[1]:
        emoji = "✅" if props["reusable"] else "❌"
        st.metric(
            label=f"{emoji} Reusable",
            value="Yes" if props["reusable"] else "No",
        )
    with cols[2]:
        emoji = "✅" if props["safely_disposed"] else "👀"
        st.metric(
            label=f"{emoji} Safe to dispose",
            value="Yes" if props["safely_disposed"] else "Caution",
        )
    with cols[3]:
        if props["hazardous"]:
            st.markdown(
                "<div style='background-color: #ff4b4b; padding: 12px; "
                "border-radius: 8px; text-align: center; color: white; "
                "font-weight: bold;'>⚠️ HAZARDOUS</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background-color: #00cc66; padding: 12px; "
                "border-radius: 8px; text-align: center; color: white; "
                "font-weight: bold;'>✓ Not Hazardous</div>",
                unsafe_allow_html=True,
            )


def show_handling_steps(steps: List[str]):
    """Display handling steps as a numbered list with copy buttons."""
    if not steps:
        st.info("No specific handling steps for this material.")
        return

    st.markdown("#### 📋 Steps to Handle This Waste")

    for i, step in enumerate(steps, 1):
        st.markdown(
            f"""
            <div style='display: flex; align-items: flex-start; margin-bottom: 8px;'>
                <div style='flex: 0 0 30px; height: 30px; background-color: #1f77b4;
                            color: white; border-radius: 50%; display: flex;
                            align-items: center; justify-content: center;
                            font-weight: bold; margin-right: 12px;'>
                    {i}
                </div>
                <div style='flex: 1; padding-top: 4px;'>{step}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================
# USER FEEDBACK COMPONENTS
# ============================================
def show_feedback_widget(result: Dict, image_key: str = "feedback_image"):
    """Allow users to correct the model if it was wrong."""
    st.markdown("---")
    st.markdown("#### Was this classification correct?")

    cols = st.columns([1, 1, 1])
    with cols[0]:
        if st.button("✅ Correct", key=f"correct_{image_key}"):
            st.success("Thanks for confirming!")
            return True
    with cols[1]:
        if st.button("❌ Wrong", key=f"wrong_{image_key}"):
            st.session_state[f"show_correction_{image_key}"] = True
    with cols[2]:
        if st.button("🤔 Unsure", key=f"unsure_{image_key}"):
            st.info("Take another photo with better lighting/angle.")

    # Show correction UI if user clicked "Wrong"
    if st.session_state.get(f"show_correction_{image_key}", False):
        show_correction_form(result["top_class"], image_key)

    return None


def show_correction_form(predicted_class: str, image_key: str):
    """Let the user select the correct class."""
    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    correct_class = st.selectbox(
        "What was the actual material?",
        classes,
        index=classes.index(predicted_class) if predicted_class in classes else 0,
        key=f"correct_class_{image_key}",
    )

    if st.button("Submit correction", key=f"submit_{image_key}"):
        # In a real app, save this to a feedback database
        st.success(
            f"Recorded: predicted={predicted_class}, actual={correct_class}. "
            f"Thanks! This will help improve the model."
        )
        # Reset
        st.session_state[f"show_correction_{image_key}"] = False


# ============================================
# ANALYTICS COMPONENTS
# ============================================
def show_session_analytics():
    """Display analytics about the user's session classifications."""
    if "history" not in st.session_state:
        return

    history = st.session_state["history"]
    if not history:
        return

    with st.expander(f"📊 Session Statistics ({len(history)} predictions)"):
        df = pd.DataFrame(history)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Classifications by Class:**")
            class_counts = df["class"].value_counts()
            st.bar_chart(class_counts)

        with col2:
            st.markdown("**Confidence Distribution:**")
            st.bar_chart(df["confidence"])


def update_history(result: Dict):
    """Add the current prediction to session history."""
    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.session_state["history"].append({
        "class": result["top_class"],
        "confidence": result["confidence"],
        "is_ambiguous": result["is_ambiguous"],
    })

# In ui_components.py
def show_classification_history():
    """Show all past classifications in this session."""
    if "history" not in st.session_state or not st.session_state["history"]:
        return
    
    st.markdown("### 📜 Classification History")
    for i, item in enumerate(reversed(st.session_state["history"]), 1):
        with st.expander(f"#{i}: {item['class'].title()} ({item['confidence']:.1%})"):
            st.write(item)


def show_quality_report(quality_result: Dict):
    """Display image quality check results."""
    st.markdown("### 📸 Image Quality Check")
    
    # Quality score with color
    score = quality_result["quality_score"]
    if score >= 85:
        score_color = "🟢"
        score_label = "Excellent"
    elif score >= 70:
        score_color = "🟡"
        score_label = "Good"
    elif score >= 50:
        score_color = "🟠"
        score_label = "Fair"
    else:
        score_color = "🔴"
        score_label = "Poor"
    
    st.markdown(
        f"**Quality Score: {score_color} {score:.0f}/100 ({score_label})**"
    )
    
    # Warnings (more prominent)
    if quality_result["warnings"]:
        for warning in quality_result["warnings"]:
            st.warning(warning)
    
    # Info (in expander for less visual clutter)
    if quality_result["info"]:
        with st.expander(f"✅ Passed checks ({len(quality_result['info'])})", 
                         expanded=False):
            for info_msg in quality_result["info"]:
                st.markdown(info_msg)
    
    # Decision
    if not quality_result["should_proceed"]:
        st.error(
            "**⛔ Classification blocked due to critical image quality issues.** "
            "Please upload a better image before proceeding."
        )
    elif quality_result["warnings"]:
        st.info(
            "**⚠️ Classification will proceed, but accuracy may be affected.** "
            "Consider taking a better photo for more reliable results."
        )


def show_quality_tips():
    """Display tips for taking good waste photos."""
    with st.expander("📷 Tips for taking a good photo"):
        st.markdown("""
        **For best results:**
        
        1. **Lighting:** Use even, natural light. Avoid harsh shadows or direct sunlight.
        2. **Background:** Use a plain, contrasting background (white, gray, or solid color).
        3. **Framing:** Center the item in the frame. Fill 60-80% of the image with the object.
        4. **Focus:** Make sure the object is sharp, not blurry.
        5. **Single object:** Photograph one item at a time.
        6. **Distance:** Hold the camera 30-60 cm from the object.
        7. **Angle:** Take the photo from a standard 3/4 view, not extreme angles.
        
        **Examples of good vs. bad photos:**
        - ✅ Good: A single glass jar on a white table, well-lit
        - ❌ Bad: A cluttered drawer with many items
        - ❌ Bad: A blurry photo taken in low light
        - ❌ Bad: A photo from an extreme angle (looking up at the object)
        """)
