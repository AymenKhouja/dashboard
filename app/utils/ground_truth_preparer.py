import random
from typing import Dict, List

import pandas as pd

import select_llm.LLM_inference.data_labeling_utils as data_labeling_utils
from select_llm.LLM_inference.inference import LLMInference
from select_llm.Utilities.logs import BaseLogger, LocalLogger


class DataLabeler:
    """Class responsible for ground truth preparation based on user review."""

    def __init__(self, cv_filter, cv_starring, cv_ranker, llm_inference, logger=None):
        """Initialize Labeler with dependencies."""
        self.cv_filter = cv_filter
        self.cv_starring = cv_starring
        self.cv_ranker = cv_ranker
        self.llm_inference = llm_inference
        self.logger = logger or LocalLogger("data_labeler.log")

    def label_data(
        self, jobs_df, cvs_df, save_path="data/ground truth/ground truth1.csv", N_jobs=10
    ):
        """Labels data by prompting the user to review CVs and provide star ratings."""
        labeled_data = pd.DataFrame()
        self.logger.log("Starting data labeling process.")
        categories = jobs_df["category"].unique()
        experience_levels = jobs_df["experience_level"].unique()
        print("Available Categories:")
        for i, category in enumerate(categories):
            print(f"{i + 1}. {category} (Index: {i})")
        category_index = int(input("Select a category (enter number): ")) - 1
        selected_category = categories[category_index]
        print("Available Experience Levels:")
        for i, level in enumerate(experience_levels):
            print(f"{i + 1}. {level} (Index: {i})")
        experience_level_index = int(input("Select an experience level (enter number): ")) - 1
        selected_experience_level = experience_levels[experience_level_index]
    
        self.logger.log(f"User selected category: {selected_category}, experience level: {selected_experience_level}")
    
        grouped_jobs = jobs_df[
            (jobs_df["category"] == selected_category) &
            (jobs_df["experience_level"] == selected_experience_level)
        ]
    
        selected_jobs = grouped_jobs.sample(n=min(N_jobs, len(grouped_jobs)), random_state=42)
    
        for idx, job in selected_jobs.iterrows():
            job_category_experience = f"Job Category: {selected_category}, Experience Level: {selected_experience_level}"
            job_description = job["Document"]
            index = job["Index"]
            job_content = f"{job_category_experience}\n\n{job_description} (Index: {idx})"
            job_file_path = "selected_job.txt"
            with open(job_file_path, "w") as job_file:
                job_file.write(job_content)
    
            self.logger.log(f"Job file written to: {job_file_path}")
    
            treat_job = input(f"Do you want to treat this job? (Index: {index}) (yes/no): ").strip().lower()
            self.logger.log(f"User decision for job treatment: {treat_job}")
    
            if treat_job != "yes":
                continue
    
            filtered_cvs_dict = self.cv_filter.filter_cvs(cvs_df, selected_category, selected_experience_level)
            self.logger.log(f"Filtered CVs for job: {len(filtered_cvs_dict)}")
    
            requirements = self.job_reqs_extraction(self.llm_inference, job_description)
            self.logger.log("Requirements extracted.")
    
            starred_cvs = self._star_cvs(requirements, filtered_cvs_dict)
            self.logger.log("CVs starred and reviewed.")
    
            matched_cvs, unmatched_cvs = self._separate_cvs(starred_cvs)
            self.logger.log(f"Matched CVs: {matched_cvs}, Unmatched CVs: {unmatched_cvs}")
    
            ranked_matched_cvs = self.cv_ranker.rank_cvs(matched_cvs)
            ranked_unmatched_cvs = self.cv_ranker.rank_cvs(unmatched_cvs)
            self.logger.log("CVs ranked.")
    
            job_result = {
                "Job Description": index,
                "Top 5 CVs": ranked_matched_cvs,
                "Worst 5 CVs": ranked_unmatched_cvs,
                "Sampled CV Indices": filtered_cvs_dict.keys().tolist(),
            }
    
            labeled_data = pd.concat(
                [labeled_data, pd.DataFrame([job_result])], ignore_index=True
            )
            self.logger.log("Job result saved.")
    
            data_labeling_utils.save_labeled_instance(job_result, save_path)
            self.logger.log("Labeled instance saved.")
    
        return labeled_data



    def job_reqs_extraction(self, llm_client: LLMInference, job: str) -> str:
        """Extract minimum & additional requirements from a job description."""
        while True:
            prompt = f"""
You are an expert in job analysis. Your task is to analyze the provided job description and extract the main global requirements for the position, categorizing them into 'Minimum Requirements' and 'Preferred Requirements'.

Context:
The job description includes essential and desirable qualifications and experiences. These qualifications can be categorized based on their necessity and desirability.

Objective:
- Identify 'Minimum Requirements': essential qualifications and experiences (education, skills, experience, traits) a candidate must have for this job.
- Identify 'Preferred Requirements': additional desirable qualifications (further knowledge, certifications, traits) a candidate may have for this job.

Instructions:
1. Summarize each requirement concisely without including job location or gender.
2. Requirements number should be concise and they wrap small related requirements in a single one for simplicity.
3. Number of requirements in each category should be less than five.

Response Format:
Minimum Requirements:
- 

Preferred Requirements:
- 

Job Description:
{job}

Please analyze the job description, infer any additional requirements, and provide the categorized requirements as specified.
"""

            response = llm_client.generate_response(prompt)
            
            with open("selected_job.txt", "a") as job_file:
                job_file.write(f"\n\nExtracted Requirements: {response}")
            user_input = input("Are these requirements acceptable? (yes/no): ").strip().lower()
            if user_input == "yes":
                return response
            else:
                self.logger.log("User requested to regenerate the requirements.")
                print("Regenerating requirements...")

    def resume_formatter(self, llm_client: LLMInference, resume: str) -> str:
        """Format the resume with LLM inference, handling potential errors."""
        try:
            response = llm_client.generate_response(self.prompt_for_resume(resume))
        except Exception as e:
            self.logger.log_error(f"Error generating response from LLM: {e}")
            self.logger.log_error(f"Error formatting resume : {resume}")
            print(f"Error generating response from LLM: {e}")
            response = self.split_and_combine_resume(llm_client, resume)
    
        return response
    
    def split_and_combine_resume(self, llm_client: LLMInference, resume: str) -> str:
        """Split the resume, format each sub-part, and combine the responses."""
        chunks = [resume[i:i+3000] for i in range(0, len(resume), 3000)] 
        formatted_chunks = []
        for chunk in chunks:
            try:
                formatted_chunk = llm_client.generate_response(self.prompt_for_resume(chunk))
                formatted_chunks.append(formatted_chunk)
            except Exception as e:
                self.logger.log_error(f"Error generating response from LLM: {e}")
                formatted_chunks.append(chunk)
                continue
        return ''.join(formatted_chunks)


    def prompt_for_resume(self, resume: str) -> str:
        """Generate the prompt for formatting the resume."""
        return f"""
            You are a professional resume formatter.
            Format the following resume in a clear, and professional manner, highlighting key sections such as 'Contact Information', 'Summary', 'Experience', 'Education', and 'Skills'. 
    
        Resume:
        {resume}
    
        Formatted Resume:
        """

    def _star_cvs(self, job, filtered_cvs_dict):
        """Prompts the user to review each CV and provide a star rating."""
        starred_cvs = {}
        for index, cv in filtered_cvs_dict.items():
            formatted_cv = self.resume_formatter(self.llm_inference, cv)
            with open("review.txt", "w") as review_file:
                review_file.write(f"Job Description:\n{job}\n\n")
                review_file.write(f"({index}) Resume Content:\n{formatted_cv}\n\n")
    
            while True:
                try:
                    match_input = input(f"CV Index: {index} : Does this CV match the job description? (yes/no) ").strip().lower()
                    if match_input not in {"yes", "no"}:
                        raise ValueError("Invalid input. Please enter 'yes' or 'no'.")
                    match = match_input == "yes"
                    break
                except ValueError as e:
                    print(e)
    
            # Prompt for stars
            while True:
                try:
                    stars_input = input("How many stars for this CV? ")
                    stars = int(stars_input)
                    if stars < 1:
                        raise ValueError("Stars must be a positive integer.")
                    break
                except ValueError as e:
                    print(e)
    
            starred_cvs[index] = {"match": match, "stars": stars}
    
        return starred_cvs




    def _separate_cvs(self, starred_cvs):
        """Separates CVs into matched and unmatched based on user input."""
        matched_cvs = {
            index: data["stars"] for index, data in starred_cvs.items() if data["match"]
        }
        unmatched_cvs = {
            index: data["stars"] for index, data in starred_cvs.items() if not data["match"]
        }
        return matched_cvs, unmatched_cvs


class CVStarring:
    """Prompt HR to star CVs based on relevance to the job."""

    def __init__(
        self,
        llm_inference: LLMInference,
        logger: BaseLogger = LocalLogger("ground_truth_preparer.log"),
    ):
        """Initialize the CVStarring."""
        self.llm_inference = llm_inference
        self.logger = logger

    def star_cv(self, job_description: pd.Series, cv: str) -> int:
        """Prompt HR to star CV based on relevance to the job."""
        self.logger.log(f"Extracting job requirements for CV: {cv}")
        extracted_requirements = self.llm_inference.run(
            prompt=job_description["Description"]
        )
        self.logger.log(f"Extracted job requirements: {extracted_requirements}")

        stars = self._prompt_for_stars(cv, extracted_requirements)
        return stars

    def _prompt_for_stars(self, cv: str, requirements: str) -> int:
        """Prompt user to input stars for the CV based on requirements."""
        self.logger.log(f"Prompting for stars for CV: {cv}")
        print(f"Job Requirements: {requirements}")
        stars = int(input("How many stars for this CV? "))
        self.logger.log(f"Stars given: {stars}")
        return stars


class CVFilter:
    """Filter CVs based on job category and experience level."""

    def __init__(self, num_samples: int = 3):
        """Initialize the CVFilter."""
        self.num_samples = num_samples
        self.logger: BaseLogger = LocalLogger("ground_truth_preparer.log")

    def filter_cvs(
        self, cvs_df: pd.DataFrame, job_category: str, experience_level: str
    ) -> Dict[int, str]:
        """Filter CVs based on job category and experience level."""
        filtered_cvs_dict = cvs_df[
            (cvs_df["category"] == job_category)
            & (cvs_df["experience_level"] == experience_level)
        ]
        filtered_cv_dict = {}  
        for idx, row in filtered_cvs_dict.iterrows():
            index = row["Index"]
            filtered_cv_dict[index] = row["Document"]
            
        self.logger.log(f"Number of filtered cvs : {len(filtered_cv_dict)}")
        if self.num_samples is not None and self.num_samples < len(filtered_cv_dict):
            sampled_indices = random.sample(list(filtered_cv_dict.keys()), self.num_samples)
            sampled_cvs = {idx: filtered_cv_dict[idx] for idx in sampled_indices}
            return sampled_cvs
        else:
            return filtered_cv_dict



class CVRanker:
    """Rank CVs based on the number of stars given by HR."""

    def __init__(self, logger: BaseLogger = LocalLogger("cv_ranker.log")):
        """Initialize the CVRanker."""
        self.logger = logger

    def rank_cvs(self, starred_cvs: Dict[str, int]) -> List[str]:
        """Rank CVs based on the number of stars given by HR."""
        self.logger.log(f"Ranking CVs: {starred_cvs}")
        ranked_cvs = sorted(starred_cvs, key=starred_cvs.get, reverse=True)
        self.logger.log(f"Ranked CVs: {ranked_cvs}")
        return ranked_cvs
