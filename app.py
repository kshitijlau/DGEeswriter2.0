import streamlit as st
import pandas as pd
import openai
import io

# This script (v8.0) is the final version, incorporating a three-tiered
# conditional logic for the opening sentence based on the highest score value.

# --- Helper Function to convert DataFrame to Excel in memory ---
def to_excel(df):
    """
    Converts a pandas DataFrame to an Excel file in memory (bytes).
    This function is used to prepare the final output for download.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Summaries')
    processed_data = output.getvalue()
    return processed_data

# --- The RE-ENGINEERED Master Prompt Template (Version 8.0 - Conditional Logic) ---
def create_master_prompt(salutation_name, pronoun, person_data):
    """
    Dynamically creates the new, highly-constrained prompt for the Azure OpenAI API.
    VERSION 8.0: Implements a new, three-tiered conditional logic for the opening
    sentence based on the value of the single highest competency score.

    Args:
        salutation_name (str): The name to be used for the person, including titles (e.g., "Dr. Jonas", "Irene").
        pronoun (str): The correct pronoun for the person (e.g., 'He' or 'She').
        person_data (str): A string representation of the person's scores and competencies.

    Returns:
        str: A fully constructed prompt ready to be sent to the AI model.
    """
    prompt_text = f"""
You are an elite talent management consultant from a top-tier firm. Your writing is strategic, cohesive, and you follow instructions with absolute precision.

## NON-NEGOTIABLE CORE RULES
1.  **Language:** The entire summary MUST be written in **British English**.
2.  **Structure:** The entire summary MUST be a **single, unified paragraph**. There can be no deviation from this rule.
3.  **Competencies:** You MUST NOT use the exact name of a competency (e.g., "Results Driver", "Strategic Thinker") in the narrative. Instead, you MUST describe the behavior using a verb phrase (e.g., "...demonstrated an ability to drive results," or "...showcased strategic thinking.").

## Core Objective
Synthesize the provided competency data for {salutation_name} into a single, cohesive, and integrated narrative paragraph that adheres to all core rules.

## Input Data for {salutation_name}
<InputData>
{person_data}
</InputData>

## ---------------------------------------------
## CRITICAL DIRECTIVES FOR SUMMARY STRUCTURE & TONE
## ---------------------------------------------

1.  **CRITICAL OPENING SENTENCE PROTOCOL (CONDITIONAL LOGIC):**
    * You must first identify the single highest numerical score in the input data and then follow the appropriate rule below.

    * **Rule A: If the highest score is 3.5 or greater (>= 3.5):**
        * The first sentence **MUST** follow one of these four exact formats, reflecting a clear strength:
            1.  `{salutation_name} evidenced a strong ability to [highest scoring competency verb phrase].`
            2.  `{salutation_name} evidenced a strong capacity to [highest scoring competency verb phrase].`
            3.  `{salutation_name} demonstrated a strong ability to [highest scoring competency verb phrase].`
            4.  `{salutation_name} demonstrated a strong capacity to [highest scoring competency verb phrase].`

    * **Rule B: If the highest score is between 2.5 and 3.49 (inclusive):**
        * The first sentence **MUST** follow one of these two formats, reflecting competence:
            1.  `{salutation_name} evidenced the competence to [highest scoring competency verb phrase].`
            2.  `{salutation_name} demonstrated the competence to [highest scoring competency verb phrase].`

    * **Rule C: If the highest score is less than 2.5 (< 2.5):**
        * You **MUST NOT** use any of the formulaic opening sentences from Rule A or B.
        * Instead, the summary must begin immediately by describing the most positive observed behavior, even if it's not a formal strength.
        * **You must learn from and replicate the style of the 'Fatema' example provided below for this specific case.**

2.  **STRUCTURE AFTER OPENING: The Integrated Feedback Loop.**
    * After the opening sentence (or from the beginning, in the case of Rule C), the rest of the paragraph **MUST** address each competency one by one in a logical flow.
    * For each competency, you will first describe the **observed positive behavior** (the strength).
    * Then, **IMMEDIATELY AFTER** describing the behavior, you will provide the **related development area** for that same competency, introduced with a phrase like "As a next step...", "To build on this...", or "Fatema may benefit from...".

3.  **Name and Pronoun Usage:**
    * Use the candidate's full salutation name, **{salutation_name}**, only in the first sentence (if applicable). Thereafter, use the pronoun **{pronoun}**.

## ---------------------------------------------
## ANALYSIS OF GOLD-STANDARD EXAMPLES (INTERNALIZE ALL LOGIC)
## ---------------------------------------------

**Example 1: Khasiba (Highest Score >= 3.5)**
* **Logic:** Follows Rule A. The summary begins with a "strong ability/capacity" sentence based on her highest score. The rest of the paragraph follows the Integrated Feedback Loop.
* **Correct Output:** "Khasiba demonstrated a strong ability to drive results. She consistently evidenced the ability to analyse complex scenarios... To build on this, she could focus on refining her ability to evaluate complex, high-stakes scenarios..."

**Example 2: Fatema (Highest Score < 2.5)**
* **Analysis:** This is the gold standard for Rule C. Notice how it does **not** have a formal opening sentence.
* **Logic:** It begins immediately by describing her most positive observed behavior from her highest-scoring competency (Effective Collaborator: 2.47). It then seamlessly moves into the Integrated Feedback Loop for all other competencies. The tone is constructive and developmental.
* **Correct Output:** "Fatema evidenced collaboration by engaging positively with different parties, internally and externally, offering support and ensuring shared goals were properly achieved. To further develop her skills, Fatema may need to strengthen her self-confidence and the ability to communicate clearly, especially when facing stressful situations. Adapting her communication style, articulating compelling reasoning for her arguments and understanding stakeholders’ diverse needs and perspectives, could help enhance her influence, conflict management and collaboration. She demonstrated customer advocacy through resolving customers’ issues and fulfilling their requirements, and showed awareness of emerging trends and their potential impact on the organisation. Fatema may benefit from identifying customers’ needs and adopting service standards that would enhance quality and improve customer experience..."

## ---------------------------------------------
## FINAL INSTRUCTIONS
## ---------------------------------------------

Now, process the data for {salutation_name}. First, determine which opening sentence rule (A, B, or C) to apply based on the highest score. Then, create a **strict single-paragraph summary** that follows all rules precisely. The total word count should remain between 250-280 words.
"""
    return prompt_text

# --- API Call Function for Azure OpenAI ---
def generate_summary_azure(prompt, api_key, endpoint, deployment_name):
    """
    Calls the Azure OpenAI API to generate a summary.
    """
    try:
        client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-02-01"
        )
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are an elite talent management consultant. Your writing is strategic and cohesive. You follow all instructions with absolute precision, especially the conditional logic for the opening sentence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while contacting Azure OpenAI: {e}")
        return None

# --- Streamlit App Main UI ---
st.set_page_config(page_title="DGE Executive Summary Generator v8.0", layout="wide")

st.title("📄 DGE Executive Summary Generator (V8.0)")
st.markdown("""
This application generates professional executive summaries based on leadership competency scores.
**Version 8.0 uses advanced conditional logic for the opening sentence based on the candidate's score.**
1.  **Set up your secrets**.
2.  **Download the Sample Template**. The format requires a `salutation_name` column.
3.  **Upload your completed Excel file**.
4.  **Click 'Generate Summaries'**.
""")

# --- Create and provide a sample file for download ---
sample_data = {
    'email': ['irene.a@example.com', 'jonas.k@example.com', 'fatema.f@example.com'],
    'salutation_name': ['Irene', 'Dr. Jonas', 'Fatema'],
    'gender': ['F', 'M', 'F'],
    'level': ['Director', 'Manager', 'Specialist'],
    'Strategic Thinker': [3.66, 3.23, 2.05],
    'Impactful Decision Maker': [3.51, 3.52, 1.62],
    'Effective Collaborator': [3.53, 3.28, 2.47],
    'Talent Nurturer': [3.38, 2.9, 1.97],
    'Results Driver': [3.3, 3.06, 2.03],
    'Customer Advocate': [3.29, 3.2, 2.38],
    'Transformation Enabler': [2.97, 3.02, 2.01],
    'Innovation Explorer': [3.42, 3.29, 2.31]
}
sample_df = pd.DataFrame(sample_data)
sample_excel_data = to_excel(sample_df)

st.download_button(
    label="📥 Download Sample Template File (V8.0)",
    data=sample_excel_data,
    file_name="dge_summary_template_v8.0.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.divider()

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your completed Excel file here", type="xlsx")

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("Excel file loaded successfully. Ready to generate summaries.")
        st.dataframe(df.head())

        if st.button("Generate Summaries", key="generate"):
            try:
                azure_api_key = st.secrets["azure_openai"]["api_key"]
                azure_endpoint = st.secrets["azure_openai"]["endpoint"]
                azure_deployment_name = st.secrets["azure_openai"]["deployment_name"]
            except (KeyError, FileNotFoundError):
                st.error("Azure OpenAI credentials not found. Please configure them in your Streamlit secrets.")
                st.stop()
            
            identifier_cols = ['email', 'salutation_name', 'gender', 'level']
            all_known_competencies = [
                'Strategic Thinker', 'Impactful Decision Maker', 'Effective Collaborator',
                'Talent Nurturer', 'Results Driver', 'Customer Advocate',
                'Transformation Enabler', 'Innovation Explorer'
            ]
            competency_columns = [col for col in df.columns if col in all_known_competencies]
            
            if 'salutation_name' not in df.columns:
                st.error("Error: The uploaded file is missing the required 'salutation_name' column. Please download the new template and try again.")
                st.stop()

            generated_summaries = []
            progress_bar = st.progress(0)
            
            for i, row in df.iterrows():
                salutation_name = row['salutation_name']
                gender_input = str(row['gender']).upper()
                pronoun = 'They'
                if gender_input == 'M':
                    pronoun = 'He'
                elif gender_input == 'F':
                    pronoun = 'She'
                else:
                    st.warning(f"Invalid or missing gender '{row['gender']}' for {salutation_name}. Defaulting to pronoun 'They'.")

                st.write(f"Processing summary for: {salutation_name}...")
                
                scores_data = []
                for competency in competency_columns:
                    if competency in row and pd.notna(row[competency]):
                        scores_data.append(f"- {competency}: {float(row[competency])}")
                person_data_str = "\n".join(scores_data)

                prompt = create_master_prompt(salutation_name, pronoun, person_data_str)
                summary = generate_summary_azure(prompt, azure_api_key, azure_endpoint, azure_deployment_name)
                
                if summary:
                    generated_summaries.append(summary)
                    st.success(f"Successfully generated summary for {salutation_name}.")
                else:
                    generated_summaries.append("Error: Failed to generate summary.")
                    st.error(f"Failed to generate summary for {salutation_name}.")

                progress_bar.progress((i + 1) / len(df))

            if generated_summaries:
                st.balloons()
                st.subheader("Generated Summaries (V8.0)")
                
                output_df = df.copy()
                output_df['Executive Summary'] = generated_summaries
                
                st.dataframe(output_df)
                
                results_excel_data = to_excel(output_df)
                st.download_button(
                    label="📥 Download V8.0 Results as Excel",
                    data=results_excel_data,
                    file_name="Generated_Executive_Summaries_V8.0.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
