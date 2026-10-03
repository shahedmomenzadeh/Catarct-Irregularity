"""
prompts.py
Centralized prompt definitions, canonical 3-class ophthalmic irregularity taxonomy,
and output formatting specifications for Cataract-Irregularity evaluation.
Source dataset: Cataract-1K
"""

# =============================================================================
# CANONICAL 3-CLASS IRREGULARITY TAXONOMY (CATARACT-1K)
# =============================================================================

TAXONOMY_3CLASS = {
    "A": "Normal",
    "B": "Lens Irregularity",
    "C": "Pupil Contraction",
}

TAXONOMY_DESCRIPTIONS = {
    "Normal": "Standard routine cataract extraction procedure without significant lens capsular disruption or acute pupil contraction.",
    "Lens Irregularity": "Intraoperative lens irregularity, including capsular tear/defect, zonular laxity/dehiscence, lens subluxation, or abnormal lens morphology.",
    "Pupil Contraction": "Intraoperative pupil contraction / constriction (miosis), floppy iris syndrome (IFIS), iris prolapse, or irregular pupil dynamics.",
}

FOLDER_TO_CLASS = {
    "Normal": {
        "letter": "A",
        "category": "Normal",
        "reference_reasoning": "The surgical video demonstrates a standard routine cataract extraction procedure with normal lens anatomy and pupil dilation, without significant lens capsule compromise or intraoperative pupil constriction."
    },
    "Lens_irregularity": {
        "letter": "B",
        "category": "Lens Irregularity",
        "reference_reasoning": "The surgical video exhibits intraoperative lens irregularity, featuring capsular irregularity, tear/defect, zonular instability, or abnormal lens dynamics during cataract removal."
    },
    "Pupil_Contraction": {
        "letter": "C",
        "category": "Pupil Contraction",
        "reference_reasoning": "The surgical video exhibits intraoperative pupil contraction / constriction (miosis), floppy iris behavior, or irregular pupil margin dynamics requiring adaptive surgical handling."
    }
}

# Standard output format contract enforced across evaluation
FORMAT_INSTRUCTION = """
Respond ONLY with a JSON object formatted exactly as:
{
  "explanation": "<3-10 sentences describing the visible intraoperative evidence, anatomical findings, and rationale>",
  "answer": "<single uppercase letter corresponding to your chosen option: A, B, or C>"
}
Do not include any text outside the JSON object.
"""

# 3-Class Cataract Irregularity Evaluation Prompt Template
IRREGULARITY_PROMPT_TEMPLATE = """You are given a cataract procedure video. Analyze the surgical video carefully, explain the visible intraoperative events, anatomical structures, and any observed irregularities, and then classify the case into one of the following 3 options:

Options:
A) Normal
B) Lens Irregularity
C) Pupil Contraction
""" + FORMAT_INSTRUCTION
