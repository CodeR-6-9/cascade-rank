import logging
from pathlib import Path
from typing import Dict, Any, Optional

from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "phi-3-mini.gguf"


class ReasoningGenerator:
    """
    Generates a 1-2 sentence reasoning explaining why a candidate
    matches a given job description using Phi-3 (GGUF).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 1024,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0
    ):
        target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at {target_path.resolve()}")

        logging.info(f"Loading Phi-3 model from {target_path}...")
        self.llm = Llama(
            model_path=str(target_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        logging.info("Model loaded successfully.")

    def _format_profile(self, candidate_data: Dict[str, Any]) -> str:
        """Helper to extract title, YoE, and summary."""
        profile = candidate_data.get("profile", {})
        title = profile.get("current_title", "Unknown Title")
        yoe = candidate_data.get("years_of_experience") or profile.get("years_of_experience", "Unknown")
        summary = profile.get("summary", "No summary provided.")

        return f"Title: {title}\nYears of Experience: {yoe}\nSummary: {summary}"

    def _format_skills(self, candidate_data: Dict[str, Any], limit: int = 15) -> str:
        """Helper to extract a capped list of skills to save prompt space."""
        raw_skills = candidate_data.get("skills", [])
        skills_list = [s.get("name") for s in raw_skills if isinstance(s, dict) and s.get("name")]

        capped_skills = skills_list[:limit]
        return ", ".join(capped_skills) if capped_skills else "No skills listed."

    def _format_history(self, candidate_data: Dict[str, Any], limit: int = 2) -> str:
        """Helper to extract recent roles, including a truncated description."""
        career_history = candidate_data.get("career_history", [])
        if not isinstance(career_history, list) or len(career_history) == 0:
            return "No recent history."

        recent_jobs = career_history[:limit]
        history_strings = []

        for job in recent_jobs:
            title = job.get("title", "Unknown")
            company = job.get("company", "Unknown")
            desc = job.get("description", "")

            short_desc = (desc[:100] + "...") if len(desc) > 100 else desc
            history_strings.append(f"- {title} at {company}: {short_desc.strip()}")

        return "\n".join(history_strings)

    def _build_reasoning_context(self, candidate_record: Dict[str, Any]) -> str:
        """
        Coordinates the smaller helper methods to build the LLM context.
        """
        raw_candidate = candidate_record.get("candidate", {})

        return (
            f"{self._format_profile(raw_candidate)}\n"
            f"Skills: {self._format_skills(raw_candidate)}\n"
            f"Recent History:\n{self._format_history(raw_candidate)}"
        )

    def _build_prompt(self, candidate_context: str, jd: str) -> str:
        prompt = (
            "<|user|>\n"
            "You are an expert technical recruiter. Your task is to explain the candidate's "
            "relevant experience and skills that align with the job description.\n"
            "Focus only on technical skills and relevant professional experience. "
            "Base your explanation only on the provided candidate information. "
            "Do not invent qualifications.\n\n"
            "Job Description:\n"
            f"{jd}\n\n"
            "Candidate Profile:\n"
            f"{candidate_context}\n\n"
            "Task:\n"
            "Write exactly 1-2 sentences explaining why the candidate is a good fit. "
            "Do not include greetings or conversational filler.\n"
            "<|end|>\n"
            "<|assistant|>\n"
        )
        return prompt

    def generate(self, candidate_record: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """
        Generates reasoning and returns the entire candidate record enriched with the new reasoning.
        """
        candidate_id = candidate_record.get("candidate_id", "Unknown")

        if not job_description or not job_description.strip():
            logging.warning(f"Empty job description for {candidate_id}. Skipping generation.")
            candidate_record["reasoning"] = "No job description provided."
            return candidate_record

        context = self._build_reasoning_context(candidate_record)
        prompt = self._build_prompt(context, job_description)

        try:
            response = self.llm(
                prompt,
                max_tokens=60,
                temperature=0.1,
                stop=["<|end|>"]
            )

            choices = response.get("choices", [])
            if not choices:
                raise ValueError("LLM returned no choices.")

            cleaned_reasoning = choices[0].get("text", "").strip()
            if not cleaned_reasoning:
                cleaned_reasoning = "Could not generate reasoning."

        except Exception as e:
            logging.error(f"Failed to generate reasoning for candidate {candidate_id}: {e}")
            cleaned_reasoning = "Error generating reasoning."

        candidate_record["reasoning"] = cleaned_reasoning
        return candidate_record