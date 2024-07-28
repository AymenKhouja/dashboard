"""Annotation script for the resume & job datasets."""

from pathlib import Path
from typing import Dict

import pandas as pd
from pandas import DataFrame
from pyaml import yaml
from select_llm.Utilities.configs_loader import load_config
from select_llm.data_management.data_loader import (
    CsvDataLoader,
    PdfDataLoader,
    TxtDataLoader,
)
from select_llm.LLM_inference.data_labeler import JobLabeler, ResumeLabeler
from select_llm.LLM_inference.default_prompts import (
    JOB_INSTRUCTIONS,
    RESUME_INSTRUCTIONS,
)
from select_llm.LLM_inference.inference import (
    HuggingFaceInference,
    HuggingFaceLoader,
    LLMInference,
    PromptPreparer,
)
from select_llm.Utilities.data_labeling_models import LabelingConfig
from select_llm.Utilities.data_models import DataConfig
from select_llm.Utilities.inference_models import InferenceConfig
from select_llm.Utilities.logs import BaseLogger, LocalLogger
from select_llm.Utilities.prompts_model import PromptsConfig


def run_resume_annotation(
    resume_data: DataFrame,
    instructions: str,
    fields: Dict[str, str],
    llm_checkpoint: str,
    output_path: Path,
    logger: BaseLogger,
):
    """Run resume annotation and save results."""
    model = HuggingFaceLoader().load_model(llm_checkpoint)
    tokenizer = HuggingFaceLoader().load_tokenizer(llm_checkpoint)
    prompt_preparer = PromptPreparer(tokenizer)
    inference_api = HuggingFaceInference(
        model=model, tokenizer=tokenizer, prompt_preparer=prompt_preparer, logger=logger
    )
    annotator = ResumeLabeler(
        llm_resume_labeler=LLMInference(logger=logger, inference_api=inference_api),
        resume_df=resume_data,
        logger=logger,
        output_file=output_path,
    )
    annotated_df = annotator.run(instructions, fields)
    logger.log("Annotated resumes completed.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_df.to_csv(output_path, index=False)
    logger.log(f"Annotated resumes saved to: {output_path}")


def run_job_annotation(
    job_df: DataFrame,
    instructions: str,
    fields: Dict[str, str],
    llm_checkpoint: str,
    output_path: Path,
    logger: BaseLogger,
):
    """Run job postings annotation and save results."""
    inference_logger = LocalLogger("inference_logger.log")
    model = HuggingFaceLoader().load_model(llm_checkpoint)
    tokenizer = HuggingFaceLoader().load_tokenizer(llm_checkpoint)
    prompt_preparer = PromptPreparer(tokenizer)
    inference_api = HuggingFaceInference(
        model=model,
        tokenizer=tokenizer,
        prompt_preparer=prompt_preparer,
        logger=inference_logger,
    )
    annotator = JobLabeler(
        llm_job_labeler=LLMInference(logger=logger, inference_api=inference_api),
        job_df=job_df,
        logger=inference_logger,
        output_file=output_path,
    )
    annotated_df = annotator.run(instructions, fields)
    logger.log("Annotated job postings completed.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_df.to_csv(output_path, index=False)
    logger.log(f"Annotated job postings saved to: {output_path}")


def annotate_job_postings_csv(
    data_paths,
    output_path,
    instructions,
    fields,
    llm_checkpoint: str,
    logger: BaseLogger,
):
    """Annotate job postings from CSV files."""
    if data_paths.exists():
        job_postings_csv = CsvDataLoader(data_path=data_paths).run()
        if job_postings_csv:
            logger.log(f"Loaded job_postings_csv dataset from directory: {data_paths}")
            job_postings_df = pd.DataFrame(job_postings_csv[0])
            job_postings_df.rename(columns={"description": "Content"}, inplace=True)
            run_job_annotation(
                job_df=job_postings_df[6275:],
                instructions=instructions,
                fields=fields,
                llm_checkpoint=llm_checkpoint,
                output_path=output_path,
                logger=logger,
            )
    else:
        raise NotADirectoryError(f"Data directory : {data_paths} does not exist.")

def annotate_data_science_job_descriptions_csv(
    data_paths,
    output_path,
    instructions,
    fields,
    llm_checkpoint: str,
    logger: BaseLogger,
):
    """Annotate data science job descriptions from CSV files."""
    if data_paths.exists():
        data_science_job_descriptions_csv = CsvDataLoader(data_path=data_paths).run()
        if data_science_job_descriptions_csv:
            logger.log(f"Loaded data_science_job_descriptions dataset from directory: {data_paths}")
            data_science_job_descriptions_df = pd.DataFrame(data_science_job_descriptions_csv[0])
            data_science_job_descriptions_df.rename(columns={"content": "Content"}, inplace=True)
            run_job_annotation(
                job_df=data_science_job_descriptions_df[400:],
                instructions=instructions,
                fields=fields,
                llm_checkpoint=llm_checkpoint,
                output_path=output_path,
                logger=logger,
            )
    else:
        raise NotADirectoryError(f"Data directory : {data_paths} does not exist.")



def annotate_chinese_resume_corpus(
    data_paths: Path,
    output_path: Path,
    instructions,
    fields,
    llm_checkpoint: str,
    logger: BaseLogger,
):
    """Annotate Chinese resume corpus from CSV files."""
    if not data_paths.exists():
        raise NotADirectoryError(f"Data directory : {data_paths} does not exist.")

    chinese_resume_pdf = CsvDataLoader(data_path=data_paths).run()
    if chinese_resume_pdf:
        logger.log(f"Loaded chinese_resume_pdf dataset from directory: {data_paths}")
        chinese_resume_pdf_df = pd.concat(chinese_resume_pdf, ignore_index=True)
        chinese_resume_pdf_df.rename(
            columns={"Translated_Resume": "Content"}, inplace=True
        )
        run_resume_annotation(
            resume_data=chinese_resume_pdf_df,
            instructions=instructions,
            fields=fields,
            llm_checkpoint=llm_checkpoint,
            output_path=output_path,
            logger=logger,
        )


def annotate_curriculum_vitae_pdf(
    data_paths,
    output_path,
    instructions,
    fields,
    llm_checkpoint,
    logger: BaseLogger,
):
    """Annotate curriculum vitae from PDF files."""
    if not data_paths.exists():
        raise NotADirectoryError(f"Data directory : {data_paths} does not exist.")

    curriculum_vitae_pdf = PdfDataLoader(data_path=data_paths).run()
    if curriculum_vitae_pdf:
        logger.log(f"Loaded curriculum_vitae_pdf dataset from directory: {data_paths}")
        curriculum_vitae_pdf_df = pd.DataFrame(curriculum_vitae_pdf, columns=["Content"])
        run_resume_annotation(
            resume_data=curriculum_vitae_pdf_df,
            instructions=instructions,
            fields=fields,
            llm_checkpoint=llm_checkpoint,
            output_path=output_path,
            logger=logger,
        )


def annotate_resume_samples_txt(
    data_paths,
    output_path,
    instructions,
    fields,
    llm_checkpoint: str,
    logger: BaseLogger,
):
    """Annotate resume samples from TXT files."""
    if not data_paths.exists():
        logger.log(f"Data directory : {data_paths} does not exist.")
        raise NotADirectoryError(f"Data directory : {data_paths} does not exist.")

    resume_samples_txt = TxtDataLoader(data_path=data_paths).run()
    if resume_samples_txt:
        logger.log(f"Loaded resume_samples_txt dataset from directory: {data_paths}")
        resume_samples_txt[0].drop([4277, 21415, 25426], axis=0, inplace=True)
        resume_samples_txt = resume_samples_txt[0][0].str.split(":::", expand=True)
        pattern = r"C:\\Workspace\\java\\scrape_indeed\\(.*?)\\"
        resume_samples_txt[0] = resume_samples_txt[0].str.extract(pattern, expand=False)
        resume_samples_txt.columns = ["Category", "Skills", "Content"]
        run_resume_annotation(
            resume_data=resume_samples_txt.iloc[10418:],
            instructions=instructions,
            fields=fields,
            llm_checkpoint=llm_checkpoint,
            output_path=output_path,
            logger=logger,
        )


def annotate_resume_dataset_csv(
    data_paths,
    output_path,
    instructions,
    fields,
    llm_checkpoint: str,
    logger: BaseLogger,
):
    """Annotate resume dataset from CSV files."""
    if not data_paths.exists():
        logger.log(f"Data directory : {data_paths} does not exist.")
        raise NotADirectoryError(f"Data directory : {data_paths} does not exist.")

    resume_dataset_csv = CsvDataLoader(data_path=data_paths).run()
    if resume_dataset_csv:
        logger.log(f"Loaded resume_dataset_csv dataset from directory: {data_paths}")
        resume_dataset_csv[0].columns = ["Category", "Content"]
        run_resume_annotation(
            resume_data=resume_dataset_csv[0],
            instructions=instructions,
            fields=fields,
            llm_checkpoint=llm_checkpoint,
            output_path=output_path,
            logger=logger,
        )


if __name__ == "__main__":

    data_configs = load_config("configs/data_configs.yaml", DataConfig)
    prompt_configs = load_config("configs/llm_prompts_config.yaml", PromptsConfig)

    inference_nested_configs = {
        "prompts": prompt_configs
    }
    inference_configs = load_config("configs/llm_config.yaml", InferenceConfig,
                                    nested_config_attrs=["prompts"],
                                    nested_configs=[inference_nested_configs["prompts"]])

    labeling_config = load_config(
        "configs/labeling_script_config.yaml", 
        LabelingConfig, 
        nested_config_attrs=["data", "inference"],
        nested_configs=[data_configs, inference_configs]
    )

    labeler_script_logger = LocalLogger(str(labeling_config.logger_file))

    resume_fields = labeling_config.resume_fields
    job_fields = labeling_config.job_fields
    data_dir = Path(labeling_config.data.data_dir)
    resumes_path = data_dir / labeling_config.data.resume_dir
    jobs_path = data_dir / labeling_config.data.job_dir

    llm_checkpoint = labeling_config.inference.hugging_face.checkpoint_name
    """
    data_paths = jobs_path / labeling_config.data.data_paths.data_science_job_descriptions_csv
    output_path = (
        data_dir
        / labeling_config.data.annotation_dir
        / labeling_config.data.data_paths.data_science_job_descriptions_csv
    )
    annotate_data_science_job_descriptions_csv(
        data_paths=data_paths,
        output_path=output_path,
        instructions=labeling_config.inference.prompts.job_instructions,
        fields=job_fields,
        llm_checkpoint=llm_checkpoint,
        logger=labeler_script_logger,
    )
    
    data_paths = jobs_path / labeling_config.data.data_paths.job_postings_csv
    output_path = (
        data_dir
        / labeling_config.data.annotation_dir
        / labeling_config.data.data_paths.job_postings_csv
    )
    annotate_job_postings_csv(
        data_paths=data_paths,
        output_path=output_path,
        instructions=JOB_INSTRUCTIONS,
        fields=job_fields,
        llm_checkpoint=llm_checkpoint,
        logger=labeler_script_logger,
    )

    data_paths = resumes_path / labeling_config.data.data_paths.chinese_resume_corpus
    output_path = (
        data_dir
        / labeling_config.data.annotation_dir
        / labeling_config.data.data_paths.chinese_resume_corpus
    )
    annotate_chinese_resume_corpus(
        data_paths=data_paths,
        output_path=output_path,
        instructions=RESUME_INSTRUCTIONS,
        fields=resume_fields,
        llm_checkpoint=llm_checkpoint,
        logger=labeler_script_logger,
    )

    data_paths = resumes_path / labeling_config.data.data_paths.curriculum_vitae_pdf
    output_path = (
        data_dir
        / labeling_config.data.annotation_dir
        / labeling_config.data.data_paths.curriculum_vitae_pdf
    )
    annotate_curriculum_vitae_pdf(
        data_paths=data_paths,
        output_path=output_path,
        instructions=RESUME_INSTRUCTIONS,
        fields=resume_fields,
        llm_checkpoint=llm_checkpoint,
        logger=labeler_script_logger,
    )
    """
    data_paths = resumes_path / labeling_config.data.data_paths.resume_samples_txt
    output_path = (
        data_dir
        / labeling_config.data.annotation_dir
        / labeling_config.data.data_paths.resume_samples_txt
    )
    annotate_resume_samples_txt(
        data_paths=data_paths,
        output_path=output_path,
        instructions=RESUME_INSTRUCTIONS,
        fields=resume_fields,
        llm_checkpoint=llm_checkpoint,
        logger=labeler_script_logger,
    )

    data_paths = resumes_path / labeling_config.data.data_paths.resume_dataset_csv
    output_path = (
        data_dir
        / labeling_config.data.annotation_dir
        / labeling_config.data.data_paths.resume_dataset_csv
    )
    annotate_resume_dataset_csv(
        data_paths=data_paths,
        output_path=output_path,
        instructions=RESUME_INSTRUCTIONS,
        fields=resume_fields,
        llm_checkpoint=llm_checkpoint,
        logger=labeler_script_logger,
    )
